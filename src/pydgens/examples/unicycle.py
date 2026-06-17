"""
Beginner example: a nonlinear unicycle game solved with iLQ.

This example is the nonlinear companion to ``tug_o_war.py``. Its purpose is
to show the high-level modeling workflow a new user should see when building
an unconstrained nonlinear game:

    1. define a time grid
    2. define joint nonlinear dynamics
    3. define each player's running cost
    4. assign each player a slice of the joint control vector
    5. build the game
    6. solve it with iLQ

Unlike the LQ tug-of-war example, there is no simple closed-form Nash
equilibrium to compare against. Instead, the goal is to understand how the
frontend API maps a modeling problem into a solver call and what parts of the
solution are most useful to inspect.

Problem
-------

We model a simple unicycle with state

    x = [px, py, theta, v]

where:

    - px, py are position coordinates
    - theta is heading angle
    - v is forward speed

The joint control vector is

    u = [omega, a]

where:

    - omega is turning rate
    - a is linear acceleration

The continuous-time dynamics are

    px_dot    = v cos(theta)
    py_dot    = v sin(theta)
    theta_dot = omega
    v_dot     = a

This is a two-player game:

    - Player 1 chooses the turning rate and wants the vehicle near the origin.
    - Player 2 chooses the acceleration and wants the vehicle near 1 m/s speed.

Each player also pays effort on its own control input.

The important modeling idea is that costs are still written in the *joint*
state and *joint* control coordinates, even though each player only owns one
slice of the control vector.
"""

from __future__ import annotations

import jax.numpy as jnp

import pydgens as pdg


def build_unicycle_game(
    *,
    nt: int = 34,
    dt: float = 0.1,
):
    """
    Build the semantic frontend game object for the unicycle example.

    Parameters
    ----------
    nt:
        Number of sampled state nodes along the horizon.

    dt:
        Sample spacing in seconds.

    Returns
    -------
    tuple
        ``(game, x0, player_turn, player_speed)``, where ``game`` is the
        frontend game object, ``x0`` is the initial state, and the player
        objects are returned so their control slices can be reused when
        inspecting the solution.
    """

    # -----------------------------------------------------------------
    # Step 0: choose the initial state
    # -----------------------------------------------------------------
    #
    # The vehicle starts away from the origin, facing along the +x axis,
    # with zero initial speed. This gives both players something useful to
    # do:
    #
    #   - the turning player can influence how the vehicle steers
    #   - the speed player can accelerate the vehicle toward its preferred
    #     cruising speed
    x0 = jnp.array([4.0, 4.0, 0.0, 0.0])

    # -----------------------------------------------------------------
    # Step 1: define the time grid
    # -----------------------------------------------------------------
    #
    # ``nt`` counts state sample nodes, so the number of control intervals is
    #
    #   nsteps = nt - 1
    #
    # Here we use a short finite horizon of 3.3 seconds:
    #
    #   nt = 34
    #   dt = 0.1
    tg = pdg.time_grid(
        nt=nt,
        dt=dt,
    )

    # -----------------------------------------------------------------
    # Step 2: define the nonlinear joint dynamics
    # -----------------------------------------------------------------
    #
    # The nonlinear frontend expects a dynamics function with signature
    #
    #   f(t, x, u) -> dxdt
    #
    # Even though this particular model does not explicitly depend on time,
    # we still include the ``t`` argument because that is the standard
    # continuous-time interface used throughout the nonlinear stack.
    #
    # Notice that the dynamics are written in *joint* coordinates:
    #
    #   u[0] is the turning-rate control
    #   u[1] is the acceleration control
    #
    # The dynamics object itself does not know which player owns which entry.
    dynamics = pdg.nonlinear_dynamics(
        nx=4,
        nu=2,
        dynamics=lambda t, x, u: jnp.array([
            x[3] * jnp.cos(x[2]),
            x[3] * jnp.sin(x[2]),
            u[0],
            u[1],
        ]),
    )

    # -----------------------------------------------------------------
    # Step 3: define each player's cost
    # -----------------------------------------------------------------
    #
    # The generic nonlinear cost factory keeps the same frontend convention
    # as the LQ API: each player's running cost is written over the *joint*
    # state and *joint* control spaces.
    #
    # Player 1 cares about position:
    #
    #   J1 running cost = px^2 + py^2 + omega^2
    #
    # The final ``omega^2`` term says that Player 1 also pays effort on its
    # own control, which lives in the first entry of the joint control vector.
    player_1_cost = pdg.player_cost(
        running=lambda t, x, u: x[0] ** 2 + x[1] ** 2 + u[0] ** 2,
    )

    # Player 2 cares about forward speed:
    #
    #   J2 running cost = (v - 1)^2 + a^2
    #
    # This encourages the vehicle to move at roughly 1 m/s without using
    # excessive acceleration.
    player_2_cost = pdg.player_cost(
        running=lambda t, x, u: (x[3] - 1.0) ** 2 + u[1] ** 2,
    )

    # -----------------------------------------------------------------
    # Step 4: define the players
    # -----------------------------------------------------------------
    #
    # Each player is given:
    #
    #   - a name
    #   - a cost object
    #   - a contiguous slice of the joint control vector
    #
    # Here the ownership is:
    #
    #   player_turn  owns u[0]
    #   player_speed owns u[1]
    player_turn = pdg.player(
        name="turn_player",
        cost=player_1_cost,
        joint_ctrl_slice=slice(0, 1),
    )

    player_speed = pdg.player(
        name="speed_player",
        cost=player_2_cost,
        joint_ctrl_slice=slice(1, 2),
    )

    # -----------------------------------------------------------------
    # Step 5: build the game
    # -----------------------------------------------------------------
    #
    # The game factory chooses the appropriate frontend game type from the
    # semantic ingredients above. In this case, nonlinear dynamics plus
    # generic callable player costs implies the nonlinear frontend game path.
    game = pdg.game(
        tg=tg,
        dynamics=dynamics,
        players=[player_turn, player_speed],
    )

    return game, x0, player_turn, player_speed


def main() -> None:
    # Build the game and keep the player objects so we can use their control
    # slices when inspecting the joint-control solution.
    game, x0, player_turn, player_speed = build_unicycle_game()

    # -----------------------------------------------------------------
    # Step 6: solve the game with iLQ
    # -----------------------------------------------------------------
    #
    # We request ``method="ilq"`` explicitly to make the example's intent
    # obvious. The frontend could infer this automatically from the game type,
    # but spelling it out is useful in a tutorial.
    #
    # The returned object is a small normalized solution bundle. For iLQ, the
    # pieces beginners usually care about most are:
    #
    #   - ``solution.converged``
    #   - ``solution.states``
    #   - ``solution.joint_controls``
    solution = pdg.solve(
        game,
        x0=x0,
        method="ilq",
    )

    # -----------------------------------------------------------------
    # Step 7: inspect a few useful outputs
    # -----------------------------------------------------------------
    #
    # The state trajectory has shape
    #
    #   (nt, nx)
    #
    # and the joint control trajectory has shape
    #
    #   (nsteps, nu)
    #
    # where ``nsteps = nt - 1``.
    states = solution.states
    joint_controls = solution.joint_controls

    turn_rate_controls = joint_controls[:, player_turn.joint_ctrl_slice].squeeze(-1)
    accel_controls = joint_controls[:, player_speed.joint_ctrl_slice].squeeze(-1)

    print()
    print(solution.format_summary("Two-Player Unicycle"))

    print("\nADDED CHECKS:")
    print(f"First 5 turn-rate controls: {turn_rate_controls[:5]}")
    print(f"First 5 acceleration controls: {accel_controls[:5]}")


if __name__ == "__main__":
    main()
