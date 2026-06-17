"""
Beginner example: a one-step tug-of-war game.

The goal of this example is not to teach the internal IR or solver data
structures. Instead, it shows the modeling story a new user should see:

    1. define a time grid
    2. define joint dynamics
    3. define each player's objective
    4. assign each player a slice of the joint control vector
    5. build the game
    6. solve it

Problem
-------

This is the smallest useful dynamic game we can write:

    - one scalar state x
    - two scalar controls u1 and u2
    - two players with opposing target states

We model the continuous-time dynamics

    dx/dt = u1 + u2

over a single interval of length ``dt = 1``. With ``x(0) = x0``, the next
state after one step is

    x1 = x0 + u1 + u2

which makes the example easy to reason about by hand.

Each player wants the terminal state to move toward a different target, while
also paying effort on its own control:

    J1 = (x1 - x_ref_1)^2 + r1 * u1^2
    J2 = (x1 - x_ref_2)^2 + r2 * u2^2

Because the game is tiny, we can compare the numerical solution against a
closed-form Nash equilibrium.
"""

from __future__ import annotations

import jax.numpy as jnp

import pydgens as pdg


def analytic_solution(
    *,
    x0: float,
    target_1: float,
    target_2: float,
    r1: float,
    r2: float,
) -> tuple[float, float, float]:
    """
    Compute the closed-form Nash equilibrium for the one-step game.

    Notes
    -----
    For this example, the state update is

        x1 = x0 + u1 + u2

    and each player minimizes a scalar quadratic objective. The first-order
    optimality conditions can be solved analytically, producing the
    equilibrium controls ``u1_star`` and ``u2_star`` below.

    Returns
    -------
    tuple[float, float, float]
        ``(u1_star, u2_star, x1_star)``
    """

    denom = r1 * r2 + r1 + r2

    u1_star = (
        (1.0 + r2) * (target_1 - x0)
        -
        (target_2 - x0)
    ) / denom

    u2_star = (
        (1.0 + r1) * (target_2 - x0)
        -
        (target_1 - x0)
    ) / denom

    x1_star = x0 + u1_star + u2_star

    return u1_star, u2_star, x1_star


def main() -> None:
    # -----------------------------------------------------------------
    # Step 0: choose simple scalar problem data
    # -----------------------------------------------------------------
    #
    # We start at x0 = 0.
    #
    # Player 1 wants the state to move toward +1.
    # Player 2 wants the state to move toward -1.
    #
    # Player 1 has cheaper control effort, so we expect the equilibrium
    # state to move somewhat toward Player 1's preferred target.
    x0 = jnp.array([0.0])

    target_1 = jnp.array([1.0])
    target_2 = jnp.array([-1.0])

    r1 = 0.5
    r2 = 2.0

    # -----------------------------------------------------------------
    # Step 1: define the time grid
    # -----------------------------------------------------------------
    #
    # ``nt=2`` means there are two state sample points:
    #
    #   t0 = 0
    #   t1 = 1
    #
    # so there is exactly one control interval between them.
    #
    # This is the cleanest possible finite-horizon dynamic game.
    tg = pdg.time_grid(
        nt=2,
        dt=1.0,
    )

    # -----------------------------------------------------------------
    # Step 2: define the joint dynamics
    # -----------------------------------------------------------------
    #
    # The dynamics are written in *joint* coordinates.
    #
    # The model below says:
    #
    #   dx/dt = A x + B u
    #
    # with
    #
    #   A = [0]
    #   B = [1  1]
    #
    # so both controls push on the same scalar state.
    #
    # Notice that the dynamics do *not* know which player owns which
    # control entry. They only see the joint control vector
    #
    #   u = [u1, u2].
    #
    # Ownership is introduced later when we define the players.
    dynamics = pdg.linear_dynamics(
        A=jnp.array([[0.0]]),
        B=jnp.array([[1.0, 1.0]]),
    )

    # -----------------------------------------------------------------
    # Step 3: define player costs
    # -----------------------------------------------------------------
    #
    # The beginner-facing cost API is intentionally semantic.
    #
    # Rather than asking users to manually build canonical affine-quadratic
    # forms, we let them express the idea directly:
    #
    #   - which state dimensions matter
    #   - what target state each player wants
    #   - which control dimensions are penalized
    #   - how strongly those penalties are weighted
    #
    # Player 1:
    #   wants the terminal state x1 near +1
    #   pays effort on control index 0 only
    player_1_cost = pdg.quadratic_cost(
        nx=1,   # dimension of *joint* state space
        nu=2,   # dimension of *joint* (not player-specific) control space
        terminal_state_weights=[1.0],
        terminal_state_target=target_1,
        control_weights=[r1],
        control_indices=[0],
    )

    # Player 2:
    #   wants the terminal state x1 near -1
    #   pays effort on control index 1 only
    player_2_cost = pdg.quadratic_cost(
        nx=1,   # dimension of *joint* state space
        nu=2,   # dimension of *joint* (not player-specific) control space
        terminal_state_weights=[1.0],
        terminal_state_target=target_2,
        control_weights=[r2],
        control_indices=[1],
    )

    # -----------------------------------------------------------------
    # Step 4: define the players
    # -----------------------------------------------------------------
    #
    # A player combines:
    #
    #   - a name
    #   - a cost object
    #   - a slice of the joint control vector
    #
    # Here we make the control variable ownership explicit:
    #
    #   player_1 owns u[0]
    #   player_2 owns u[1]
    #
    # In larger problems, this exact same idea scales to blocks like
    # ``slice(0, 2)`` and ``slice(2, 5)``.
    player_1 = pdg.player(
        name="player_1",
        cost=player_1_cost,
        joint_ctrl_slice=slice(0, 1),
    )

    player_2 = pdg.player(
        name="player_2",
        cost=player_2_cost,
        joint_ctrl_slice=slice(1, 2),
    )

    # -----------------------------------------------------------------
    # Step 5: build the game
    # -----------------------------------------------------------------
    #
    # Compose the game using the sampling times, joint dynamics,
    # and players

    game = pdg.game(
        tg=tg,
        dynamics=dynamics,
        players=[player_1, player_2],
    )

    # -----------------------------------------------------------------
    # Step 6: solve the game
    # -----------------------------------------------------------------
    #
    # For this example we explicitly request the LQ solver, since the model
    # is linear and the player costs are quadratic.
    solution = pdg.solve(
        game,
        x0=x0,
        method="lq",
    )

    # -----------------------------------------------------------------
    # Step 7: compare against the analytic answer
    # -----------------------------------------------------------------
    #
    # This final block does two things:
    #
    #   1. it shows a user what quantities they are likely to inspect
    #      after solving a game
    #   2. it turns the example into a small correctness check
    #
    # In richer examples, this section could be replaced or extended with
    # plotting, rollout, or strategy inspection.
    u1_expected, u2_expected, x1_expected = analytic_solution(
        x0=float(x0[0]),
        target_1=float(target_1[0]),
        target_2=float(target_2[0]),
        r1=r1,
        r2=r2,
    )

    # Extract the equilibrium controls and terminal state
    u1_computed = solution.trajectory.us[0, player_1.joint_ctrl_slice][0]
    u2_computed = solution.trajectory.us[0, player_2.joint_ctrl_slice][0]
    x1_computed = solution.states[1, 0]

    # Assert that the analytical and numerical solutions for 
    # equilibrium match
    assert jnp.allclose(u1_computed, u1_expected, atol=1e-8)
    assert jnp.allclose(u2_computed, u2_expected, atol=1e-8)
    assert jnp.allclose(x1_computed, x1_expected, atol=1e-8)

    print()
    print(solution.format_summary("Tug-of-War"))

    print("\nADDED CHECKS:")
    print("analytic match: True")
    print(f"u1* = {float(u1_computed): .6f}")
    print(f"u2* = {float(u2_computed): .6f}")
    print(f"x1* = {float(x1_computed): .6f}")


if __name__ == "__main__":
    main()
