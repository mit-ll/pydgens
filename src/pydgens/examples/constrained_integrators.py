# Copyright 2026 MIT Lincoln Laboratory
# SPDX-License-Identifier: MIT

"""
Beginner example: constrained two-player single integrators.

This example is the augmented-Lagrangian counterpart to ``tug_o_war``:
it is meant to show the intended *frontend* workflow rather than the lower-
level IR objects.

The modeling story is:

    1. define a time grid
    2. define nonlinear joint dynamics
    3. define one player cost per player
    4. assign each player a slice of the joint control vector
    5. define simple shared bounds
    6. build the constrained game
    7. solve it with the AL frontend

Problem
-------

We model two independent single integrators:

    p1_dot = u1
    p2_dot = u2

with joint state and control vectors

    x = [p1, p2]
    u = [u1, u2].

Each player wants its own position to move toward a goal while also paying
effort on its own control:

    J1 = integral ( q * (p1 - p1_goal)^2 + r * u1^2 ) dt
    J2 = integral ( q * (p2 - p2_goal)^2 + r * u2^2 ) dt

The twist is that both controls are bounded:

    -u_max <= u1 <= u_max
    -u_max <= u2 <= u_max

Those bounds make the problem a natural fit for the augmented Lagrangian
solver path.
"""

from __future__ import annotations

import jax.numpy as jnp

import pydgens as pdg


def main() -> None:
    # -----------------------------------------------------------------
    # Step 0: choose simple problem data
    # -----------------------------------------------------------------
    #
    # We choose a short but nontrivial horizon, asymmetric initial/goal
    # positions, and control bounds that are tight enough to matter.
    #
    # With these settings we expect each player to push toward its own goal,
    # but possibly saturate its control early in the horizon if the bound is
    # active.
    nt = 31
    dt = 0.1
    u_max = 2.0

    q = 2.0
    r = 0.05

    x0 = jnp.array([0.0, 1.0], dtype=jnp.float32)
    x_goal = jnp.array([4.0, -2.0], dtype=jnp.float32)

    # -----------------------------------------------------------------
    # Step 1: define the time grid
    # -----------------------------------------------------------------
    #
    # ``nt`` counts state sample nodes. The number of control intervals is
    #
    #   nsteps = nt - 1.
    #
    # The AL solver works over those control intervals while propagating the
    # joint state across the full set of nodes.
    tg = pdg.time_grid(
        nt=nt,
        dt=dt,
    )

    # -----------------------------------------------------------------
    # Step 2: define the joint nonlinear dynamics
    # -----------------------------------------------------------------
    #
    # Even though this system is very simple, we still model it through the
    # nonlinear frontend to exercise the same API shape used by the AL solver
    # for more general constrained nonlinear games.
    #
    # The dynamics are written in *joint* coordinates:
    #
    #   x = [p1, p2]
    #   u = [u1, u2]
    #
    # so the ODE is simply
    #
    #   x_dot = [u1, u2].
    dynamics = pdg.nonlinear_dynamics(
        nx=2,
        nu=2,
        dynamics=lambda t, x, u: jnp.array([
            u[0],
            u[1],
        ], dtype=x.dtype),
    )

    # -----------------------------------------------------------------
    # Step 3: define player costs
    # -----------------------------------------------------------------
    #
    # The frontend nonlinear-cost API uses joint-control coordinates, just
    # like the quadratic frontend. That means each player cost is written as
    #
    #   running(t, x, u_joint) -> scalar
    #
    # even though each player only *cares* about one entry of the joint
    # control vector.
    #
    # Player 1 cares about:
    #   - position error in x[0]
    #   - effort in u[0]
    #
    # and ignores u[1].
    player_1_cost = pdg.player_cost(
        running=lambda t, x, u: 0.5 * (
            q * (x[0] - x_goal[0]) ** 2
            +
            r * u[0] ** 2
        ),
        # We make the zero terminal cost explicit here so the example reads
        # exactly like the mathematical problem statement.
        terminal=lambda t, x: jnp.asarray(0.0, dtype=x.dtype),
    )

    # Player 2 is symmetric, but targets the second state coordinate and
    # penalizes the second control entry.
    player_2_cost = pdg.player_cost(
        running=lambda t, x, u: 0.5 * (
            q * (x[1] - x_goal[1]) ** 2
            +
            r * u[1] ** 2
        ),
        terminal=lambda t, x: jnp.asarray(0.0, dtype=x.dtype),
    )

    # -----------------------------------------------------------------
    # Step 4: define the players
    # -----------------------------------------------------------------
    #
    # Ownership of control variables is introduced here:
    #
    #   player 1 owns u[0]
    #   player 2 owns u[1]
    #
    # The costs above were still written in joint coordinates. The player
    # objects tell the frontend which slice of the joint control vector
    # belongs to each player.
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
    # Step 5: define shared bounds
    # -----------------------------------------------------------------
    #
    # This is beginner-facing constraint frontend. The control bound
    #
    #   -u_max <= u[i] <= u_max
    #
    # is applied to *all* joint-control coordinates because we omit
    # ``indices``. It is also applied on every control interval because we
    # omit ``steps``.
    #
    # For this first AL tutorial we keep the constraint set simple:
    # control bounds only, with no state bounds or terminal constraints.
    cons = pdg.constraint_set(
        pdg.control_bounds(
            lower=-u_max,
            upper=u_max,
        ),
    )

    # -----------------------------------------------------------------
    # Step 6: build the constrained nonlinear game
    # -----------------------------------------------------------------
    #
    # Passing ``constraints=...`` is what selects the constrained frontend
    # path. Under the hood this produces a ``ConstrainedNonlinearGame``,
    # which the solver frontend will later lower into the AL-specific IR.
    game = pdg.game(
        tg=tg,
        dynamics=dynamics,
        players=[player_1, player_2],
        constraints=cons,
    )

    # -----------------------------------------------------------------
    # Step 7: solve with the AL frontend
    # -----------------------------------------------------------------
    #
    # ``pdg.solve(...)`` handles the frontend-to-IR lowering internally.
    #
    # We pass the initial state ``x0`` and explicitly request ``method="al"``
    # to make the tutorial path easy to read. The frontend will build a
    # default primal-dual initial guess for us.
    result = pdg.solve(
        game,
        x0=x0,
        method="al",
    )

    # -----------------------------------------------------------------
    # Step 8: inspect the result
    # -----------------------------------------------------------------
    #
    # AL returns a primal-dual trajectory plus diagnostics. For a beginner,
    # the most natural quantities to inspect are:
    #
    #   - whether the outer solver reported convergence
    #   - the state trajectory
    #   - the control trajectory
    #   - whether the control bounds were respected
    pdtraj = result.primal_dual_trajectory
    diag = result.diagnostics

    xs = pdtraj.xs
    us = pdtraj.us

    nt = xs.shape[0]
    nsteps = us.shape[0]
    mid = nt // 2

    print("\n=== Constrained Integrators (AL Frontend) ===")
    print(f"converged: {diag.converged}")
    print(f"reason:    {diag.reason}")
    print(f"iters:     {diag.iters}")

    if diag.history:
        print("\nlast outer diagnostics:")
        print(diag.history[-1])

    def state_row(k: int) -> str:
        p1, p2 = map(float, xs[k])
        if k < nsteps:
            u1, u2 = map(float, us[k])
        else:
            u1, u2 = float("nan"), float("nan")
        return (
            f"k={k:02d}  "
            f"p1={p1:+.3f}  p2={p2:+.3f}  "
            f"u1={u1:+.3f}  u2={u2:+.3f}"
        )

    print("\ntrajectory sample:")
    print(state_row(0))
    print(state_row(mid))
    print(state_row(nt - 2))
    print(state_row(nt - 1))

    max_u1 = float(jnp.max(jnp.abs(us[:, 0]))) if us.size else 0.0
    max_u2 = float(jnp.max(jnp.abs(us[:, 1]))) if us.size else 0.0

    print("\nquick checks:")
    print(f"max |u1| = {max_u1:.3f}   (bound {u_max:.3f})")
    print(f"max |u2| = {max_u2:.3f}   (bound {u_max:.3f})")
    print(f"final state = {jnp.asarray(xs[-1])}")


if __name__ == "__main__":
    main()
