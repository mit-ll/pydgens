"""
Tutorial: terminal-cost reaching with a cooperative eight-player robot arm.

This example uses the frontend API to solve an iterative linear-quadratic
(iLQ) game with eight players. It deliberately combines two ideas that are
easy to miss in smaller examples:

* iLQ supports terminal costs, including nonlinear terminal costs.
* A game can have more than two or three players: here every arm joint has an
  independent player.

Problem
-------

We model a planar eight-link serial arm. Its state is the eight joint angles
and its joint control vector is the eight joint angular rates:

    x = [theta_1, ..., theta_8]
    u = [theta_1_dot, ..., theta_8_dot]

The kinematic dynamics are simply ``x_dot = u``. The end-effector position,
however, is nonlinear in the joint angles because every link orientation is a
cumulative sum of upstream joint angles.

Each of the eight players owns one joint-rate entry. All players cooperate by
using the same terminal objective: place the end effector at a target point.
They differ only in the running effort term each pays for its own joint:

    phi_i(x_T) = 0.5 * w_goal * ||p_ee(x_T) - p_goal||^2
    ell_i(u)   = 0.5 * r_i * u_i^2

There is no running state cost. Thus motion in the solution is driven solely
by the terminal objective, which iLQ maps to the LQ boundary terms ``Qf`` and
``qf`` at each iteration.

Run this tutorial with:

    python -m pydgens.examples.robot_arm
"""

from __future__ import annotations

import jax.numpy as jnp

import pydgens as pdg


def end_effector_position(
    joint_angles: jnp.ndarray,
    *,
    link_lengths: jnp.ndarray,
) -> jnp.ndarray:
    """Compute the planar end-effector position of a serial arm.

    ``joint_angles[j]`` is the relative rotation at joint ``j``. The absolute
    orientation of link ``j`` is consequently the cumulative sum of joints
    through that link. The calculation uses JAX operations only, so iLQ can
    automatically differentiate this function inside the terminal cost.
    """
    link_orientations = jnp.cumsum(joint_angles)
    return jnp.array([
        jnp.sum(link_lengths * jnp.cos(link_orientations)),
        jnp.sum(link_lengths * jnp.sin(link_orientations)),
    ])


def build_robot_arm_game(
    *,
    n_players: int = 8,
    nt: int = 31,
    dt: float = 0.1,
):
    """Build the cooperative multi-player terminal-cost arm game.

    Parameters
    ----------
    n_players:
        Number of arm joints and players. The tutorial default is eight. One
        player owns each joint-rate control entry.
    nt, dt:
        Number of state nodes and the sample spacing in seconds.

    Returns
    -------
    tuple
        ``(game, x0, link_lengths, target_position, players)``. The returned
        ``players`` tuple is useful to visualization code that wants to label
        the individual joint controls.
    """
    if n_players != 8:
        raise ValueError(
            "This tutorial is parameterized as an eight-player example; "
            f"got n_players={n_players}."
        )

    # -----------------------------------------------------------------
    # Step 0: choose arm geometry, initial pose, and terminal target
    # -----------------------------------------------------------------
    #
    # Equal link lengths make the geometry easy to visualize. The initial arm
    # is gently curved instead of fully straight, avoiding a singular initial
    # end-effector Jacobian. The target is generated from another reachable,
    # gently curved pose. In a real application it would usually come from a
    # task-space planner or user input rather than from a known joint pose.
    link_lengths = jnp.full((n_players,), 0.25)
    x0 = jnp.array([0.10, -0.16, 0.13, -0.11, 0.08, -0.06, 0.04, -0.02])
    target_pose_for_tutorial = jnp.array(
        [0.28, -0.30, 0.25, -0.20, 0.16, -0.12, 0.08, -0.04]
    )
    target_position = end_effector_position(
        target_pose_for_tutorial,
        link_lengths=link_lengths,
    )

    # A strong terminal weight makes reaching the target more important than
    # minimizing motion. Each player gets the same effort weight here, but
    # this could be made joint-specific to represent different actuators.
    goal_weight = 1_000.0
    effort_weights = jnp.full((n_players,), 0.05)

    # -----------------------------------------------------------------
    # Step 1: define the finite time grid
    # -----------------------------------------------------------------
    # ``nt`` indexes state nodes, including the terminal node. Therefore there
    # are ``nt - 1`` joint-rate control intervals. The defaults span 3 seconds.
    tg = pdg.time_grid(nt=nt, dt=dt)

    # -----------------------------------------------------------------
    # Step 2: define the shared joint-space dynamics
    # -----------------------------------------------------------------
    # All players' controls are assembled into one joint vector ``u``. The
    # dynamics do not need to know which player owns which entry:
    #
    #     theta_dot_j = u_j, for j = 1, ..., 8.
    #
    # The nonlinearity of this problem comes from the end-effector terminal
    # cost, not from these velocity-controlled joint dynamics.
    dynamics = pdg.nonlinear_dynamics(
        nx=n_players,
        nu=n_players,
        dynamics=lambda t, x, u: u,
    )

    # -----------------------------------------------------------------
    # Step 3: define the common nonlinear terminal cost
    # -----------------------------------------------------------------
    # The terminal-cost signature is ``terminal(t, x)``. It has no control
    # argument because a finite-horizon trajectory has no control at its final
    # state node. iLQ differentiates this function with respect to the terminal
    # joint-angle vector and supplies the result to the LQ solver as ``Qf/qf``.
    def terminal_end_effector_cost(t, x):
        del t  # The target is fixed in time for this tutorial.
        error = end_effector_position(x, link_lengths=link_lengths) - target_position
        return 0.5 * goal_weight * jnp.sum(error**2)

    # -----------------------------------------------------------------
    # Step 4: define all eight players
    # -----------------------------------------------------------------
    # Player ``i`` owns the one-entry slice ``u[i:i+1]`` and pays effort only
    # on ``u[i]``. Every player receives the same terminal callable, making
    # this a cooperative potential game. The frontend nevertheless represents
    # eight distinct players, so it exercises arbitrary player count, control
    # ownership, and player-aligned terminal ``Qf/qf`` terms.
    players = []
    for i in range(n_players):
        effort_weight_i = effort_weights[i]
        player_cost = pdg.player_cost(
            running=lambda t, x, u, i=i, weight=effort_weight_i: (
                0.5 * weight * u[i] ** 2
            ),
            terminal=terminal_end_effector_cost,
        )
        players.append(
            pdg.player(
                name=f"joint_{i + 1}_player",
                cost=player_cost,
                joint_ctrl_slice=slice(i, i + 1),
            )
        )

    # -----------------------------------------------------------------
    # Step 5: build the frontend nonlinear game
    # -----------------------------------------------------------------
    # ``pdg.game`` selects the frontend ``NonlinearGame`` type. Calling
    # ``pdg.solve(..., method="ilq")`` lowers it to the iLQ IR and solves the
    # repeated linear-quadratic game approximations.
    game = pdg.game(tg=tg, dynamics=dynamics, players=players)
    return game, x0, link_lengths, target_position, tuple(players)


def main() -> None:
    """Solve the eight-player tutorial game and print terminal diagnostics."""
    game, x0, link_lengths, target_position, players = build_robot_arm_game()

    # -----------------------------------------------------------------
    # Step 6: solve with iLQ
    # -----------------------------------------------------------------
    # The zero default strategy leaves the arm at ``x0`` for the first
    # operating trajectory. Each iLQ iteration then quadraticizes the common
    # nonlinear terminal objective about its current terminal configuration.
    solution = pdg.solve(
        game,
        x0=x0,
        method="ilq",
        max_iters=50,
        converged_max_diff=1e-2,
    )

    # -----------------------------------------------------------------
    # Step 7: inspect the terminal configuration and player controls
    # -----------------------------------------------------------------
    states = solution.states
    joint_controls = solution.joint_controls
    terminal_joint_angles = states[-1]
    terminal_position = end_effector_position(
        terminal_joint_angles,
        link_lengths=link_lengths,
    )
    terminal_error = terminal_position - target_position
    mean_absolute_rates = jnp.mean(jnp.abs(joint_controls), axis=0)

    print(solution.format_summary("Eight-Player Terminal-Cost Robot Arm"))
    print(f"players:                 {len(players)}")
    print(f"target end effector:     {target_position}")
    print(f"terminal end effector:   {terminal_position}")
    print(f"terminal error norm:     {jnp.linalg.norm(terminal_error):.6f}")
    print(f"terminal joint angles:   {terminal_joint_angles}")
    print(f"mean |joint rates|:      {mean_absolute_rates}")


if __name__ == "__main__":
    main()
