"""
Tutorial: terminal-cost reaching with a cooperative two-player robot arm.

This example demonstrates the frontend API for an iLQ game whose important
objective is a *terminal* cost. Two players jointly move a planar two-link
robot arm so that its end effector reaches a desired position at the end of a
finite horizon.

The tutorial's modeling story is:

    1. define the arm's joint-angle state and joint-rate controls
    2. define nonlinear forward kinematics for the end effector
    3. assign one joint-rate control to each player
    4. give both players the same terminal end-effector objective
    5. give each player a running cost only on its own effort
    6. build and solve the nonlinear game through the frontend API

Problem
-------

The state contains the two arm joint angles in radians:

    x = [theta_1, theta_2]

The joint control vector contains joint angular rates:

    u = [theta_1_dot, theta_2_dot]

The kinematic dynamics are therefore especially simple:

    theta_1_dot = u_1
    theta_2_dot = u_2

The end-effector position is nonlinear in the joint angles. For link lengths
``L1`` and ``L2``, it is

    p_ee(theta) = [
        L1 cos(theta_1) + L2 cos(theta_1 + theta_2),
        L1 sin(theta_1) + L2 sin(theta_1 + theta_2),
    ]

Both players share the terminal objective

    phi_i(x_T) = 0.5 * w_goal * ||p_ee(x_T) - p_goal||^2.

There is deliberately *no* running state cost. The only running costs are
the players' individual control efforts:

    ell_1(u) = 0.5 * r_1 * u_1^2
    ell_2(u) = 0.5 * r_2 * u_2^2.

Consequently, any motion in the solution is caused by the terminal cost. This
makes the example a compact demonstration of how iLQ quadraticizes a nonlinear
terminal objective into the LQ game's ``Qf`` and ``qf`` boundary terms.

Run it with:

    python -m pydgens.examples.terminal_cost_robot_arm
"""

from __future__ import annotations

import jax.numpy as jnp

import pydgens as pdg


def end_effector_position(
    joint_angles: jnp.ndarray,
    *,
    link_lengths: jnp.ndarray,
) -> jnp.ndarray:
    """Return the planar two-link arm end-effector position.

    Parameters
    ----------
    joint_angles:
        Joint-angle state ``[theta_1, theta_2]`` in radians.
    link_lengths:
        Link lengths ``[L1, L2]``.

    Returns
    -------
    jax.Array
        Cartesian end-effector position ``[px, py]``.

    Notes
    -----
    Keeping forward kinematics as a standalone function is useful for both
    the terminal cost and later visualization code. It uses only JAX array
    operations, so iLQ can differentiate it to obtain the terminal gradient
    and Hessian automatically.
    """
    theta_1, theta_2 = joint_angles
    link_1, link_2 = link_lengths
    theta_12 = theta_1 + theta_2

    return jnp.array([
        link_1 * jnp.cos(theta_1) + link_2 * jnp.cos(theta_12),
        link_1 * jnp.sin(theta_1) + link_2 * jnp.sin(theta_12),
    ])


def build_terminal_cost_robot_arm_game(
    *,
    nt: int = 31,
    dt: float = 0.1,
):
    """Build the frontend game and tutorial metadata for the robot arm.

    Parameters
    ----------
    nt:
        Number of state sample nodes. There are ``nt - 1`` control intervals.
    dt:
        Duration of each control interval in seconds.

    Returns
    -------
    tuple
        ``(game, x0, link_lengths, target_position, player_1, player_2)``.
        The player objects are returned so callers can reuse their control
        slices when inspecting or plotting the solution.
    """
    # -----------------------------------------------------------------
    # Step 0: choose geometric and objective data
    # -----------------------------------------------------------------
    #
    # The arm begins in a mildly bent pose, away from the fully extended arm's
    # kinematic singularity. The selected target is reachable and asks the arm
    # to move upward and inward. It is intentionally specified in task space
    # (end-effector coordinates), rather than as a desired pair of joint
    # angles.
    link_lengths = jnp.array([1.0, 0.7])
    x0 = jnp.array([0.3, -0.5])
    target_position = jnp.array([1.62, 0.5])

    # ``goal_weight`` controls how important terminal accuracy is relative to
    # control effort. The effort weights may be different if one joint is more
    # expensive to actuate than the other.
    goal_weight = 800.0
    effort_weight_1 = 0.05
    effort_weight_2 = 0.05

    # -----------------------------------------------------------------
    # Step 1: define the finite time grid
    # -----------------------------------------------------------------
    #
    # ``nt`` indexes state nodes, including the terminal state. With the
    # defaults there are 31 nodes and 30 control intervals, spanning 3 seconds.
    tg = pdg.time_grid(nt=nt, dt=dt)

    # -----------------------------------------------------------------
    # Step 2: define joint-space nonlinear dynamics
    # -----------------------------------------------------------------
    #
    # We use velocity-controlled joints. The dynamics receive the *joint*
    # control vector, not separate player-local vectors:
    #
    #   x_dot = [u[0], u[1]].
    #
    # Although these kinematics are linear in joint space, the task-space
    # terminal objective below is nonlinear because of forward kinematics.
    dynamics = pdg.nonlinear_dynamics(
        nx=2,
        nu=2,
        dynamics=lambda t, x, u: jnp.array([u[0], u[1]]),
    )

    # -----------------------------------------------------------------
    # Step 3: define the shared terminal objective
    # -----------------------------------------------------------------
    #
    # Every nonlinear player cost has a running function ``running(t, x, u)``
    # and may have a terminal function ``terminal(t, x)``. There is no control
    # argument in the terminal function because no control exists at the final
    # state node.
    #
    # Both players receive this same terminal objective. This makes the game
    # cooperative: each player independently chooses its own control to reduce
    # the common final end-effector error, while accounting for its own effort.
    def terminal_end_effector_cost(t, x):
        del t  # The target is time-invariant, but the standard signature has t.
        position_error = end_effector_position(
            x,
            link_lengths=link_lengths,
        ) - target_position
        return 0.5 * goal_weight * jnp.sum(position_error**2)

    # -----------------------------------------------------------------
    # Step 4: define player costs and control ownership
    # -----------------------------------------------------------------
    #
    # Cost functions are written over the entire joint control vector. Each
    # player nevertheless penalizes only the component it owns:
    #
    #   Player 1 owns u[0], the first-joint angular rate.
    #   Player 2 owns u[1], the second-joint angular rate.
    #
    # This produces the block-separable control structure required by iLQ.
    player_1_cost = pdg.player_cost(
        running=lambda t, x, u: 0.5 * effort_weight_1 * u[0] ** 2,
        terminal=terminal_end_effector_cost,
    )
    player_2_cost = pdg.player_cost(
        running=lambda t, x, u: 0.5 * effort_weight_2 * u[1] ** 2,
        terminal=terminal_end_effector_cost,
    )

    player_1 = pdg.player(
        name="shoulder_joint_player",
        cost=player_1_cost,
        joint_ctrl_slice=slice(0, 1),
    )
    player_2 = pdg.player(
        name="elbow_joint_player",
        cost=player_2_cost,
        joint_ctrl_slice=slice(1, 2),
    )

    # -----------------------------------------------------------------
    # Step 5: build the semantic frontend game
    # -----------------------------------------------------------------
    #
    # The frontend factory selects ``NonlinearGame`` because the dynamics and
    # costs are callable nonlinear models. ``pdg.solve(..., method="ilq")``
    # below will lower this semantic object to ``NonlinearGameType1`` and solve
    # the iterative LQ approximations.
    game = pdg.game(
        tg=tg,
        dynamics=dynamics,
        players=[player_1, player_2],
    )

    return game, x0, link_lengths, target_position, player_1, player_2


def main() -> None:
    """Solve the tutorial game and print terminal-reaching diagnostics."""
    (
        game,
        x0,
        link_lengths,
        target_position,
        player_1,
        player_2,
    ) = build_terminal_cost_robot_arm_game()

    # -----------------------------------------------------------------
    # Step 6: solve using iLQ
    # -----------------------------------------------------------------
    #
    # The initial strategy is zero by default, so the initial operating point
    # keeps the arm at its initial joint configuration. iLQ then repeatedly
    # linearizes the dynamics and quadraticizes the nonlinear terminal
    # end-effector error around its current terminal joint-angle state.
    solution = pdg.solve(
        game,
        x0=x0,
        method="ilq",
        max_iters=50,
        converged_max_diff=1e-4,
    )

    # -----------------------------------------------------------------
    # Step 7: inspect the solution
    # -----------------------------------------------------------------
    #
    # ``states`` has one entry per time node. ``joint_controls`` has one fewer
    # entry because controls act over intervals and there is no terminal control.
    states = solution.states
    joint_controls = solution.joint_controls
    terminal_joint_angles = states[-1]
    terminal_position = end_effector_position(
        terminal_joint_angles,
        link_lengths=link_lengths,
    )
    terminal_error = terminal_position - target_position

    print(solution.format_summary("Terminal-Cost Robot Arm"))
    print(f"target end effector:   {target_position}")
    print(f"terminal end effector: {terminal_position}")
    print(f"terminal error norm:   {jnp.linalg.norm(terminal_error):.6f}")
    print(f"terminal joint angles: {terminal_joint_angles}")
    print(
        "mean |joint rate|:      "
        f"[{jnp.mean(jnp.abs(joint_controls[:, player_1.joint_ctrl_slice])):.6f}, "
        f"{jnp.mean(jnp.abs(joint_controls[:, player_2.joint_ctrl_slice])):.6f}]"
    )


if __name__ == "__main__":
    main()
