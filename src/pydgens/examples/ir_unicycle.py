# Copyright 2026 MIT Lincoln Laboratory
# SPDX-License-Identifier: MIT

"""
Advanced example: the unicycle game built directly with IR objects.

This is the lower-level companion to ``unicycle.py``. It solves the same
two-player nonlinear unicycle game, but skips the beginner-facing frontend
factories and constructs the intermediate representation (IR) expected by the
iLQ solver.

Use this example when you want to understand what the frontend lowers into, or
when you need direct control over the JAX-friendly dataclasses used by solver
internals.

The modeling ingredients are the same as the frontend example:

    1. define a time grid
    2. define joint continuous-time dynamics
    3. define one running cost per player
    4. encode each player's control dimension with ``u_splits``
    5. build the nonlinear IR game
    6. solve it with the iLQ solver

The main difference is Step 4. In the frontend example, player ownership is
written as explicit ``player(..., joint_ctrl_slice=...)`` objects. In the IR,
the same ownership information is encoded compactly as

    u_splits = [1, 1]

meaning Player 1 owns the first scalar control and Player 2 owns the second.
"""

from __future__ import annotations

import jax.numpy as jnp

from pydgens.examples._ir_reporting import format_ir_feedback_summary
from pydgens.ir.costtypes import PlayerCostSpecContinuous
from pydgens.ir.gametypes import NonlinearGameType1
from pydgens.ir.systemtypes import SampledContinuousSystemType1
from pydgens.ir.timetypes import TimeGrid
from pydgens.solvers.ilqsolver import solve_ilqgame_feedback


def unicycle_dynamics(t, x, u):
    """
    Continuous-time unicycle dynamics in joint state/control coordinates.

    State:
        x = [px, py, theta, v]

    Control:
        u = [omega, a]
    """
    return jnp.array([
        x[3] * jnp.cos(x[2]),
        x[3] * jnp.sin(x[2]),
        u[0],
        u[1],
    ])


class Unicycle:
    """
    Build the IR representation of the two-player unicycle game.

    This wrapper is intentionally thin: its main job is to gather the IR
    objects into ``self.game`` while keeping the example readable.
    """

    def __init__(
        self,
        nt: int = 34,
        dt: float = 0.1,
    ):
        """
        Parameters
        ----------
        nt:
            Number of sampled state nodes along the horizon.

        dt:
            Sample spacing in seconds.
        """

        # -----------------------------------------------------------------
        # Step 1: define the time grid
        # -----------------------------------------------------------------
        #
        # ``nt`` counts state sample nodes, so the number of control intervals
        # is ``nt - 1``.
        tg = TimeGrid(
            nt=nt,
            dt=dt,
        )

        # -----------------------------------------------------------------
        # Step 2: define the joint nonlinear dynamics
        # -----------------------------------------------------------------
        #
        # The IR system type stores the time grid, dimensions, and dynamics
        # callable together. The callable still has the same convention used
        # by the frontend:
        #
        #   f(t, x, u) -> dxdt
        cs = SampledContinuousSystemType1(
            tg=tg,
            nx=4,
            nu=2,
            dynamics=unicycle_dynamics,
        )

        # -----------------------------------------------------------------
        # Step 3: define one running cost per player
        # -----------------------------------------------------------------
        #
        # IR costs are plain callable specs. They are written in joint
        # state/control coordinates, just like the frontend ``player_cost``.
        #
        # Player 1 cares about position and turning effort:
        #
        #   J1 running cost = px^2 + py^2 + omega^2
        player_1_cost = PlayerCostSpecContinuous(
            running=lambda t, x, u: x[0] ** 2 + x[1] ** 2 + u[0] ** 2,
        )

        # Player 2 cares about speed tracking and acceleration effort:
        #
        #   J2 running cost = (v - 1)^2 + a^2
        player_2_cost = PlayerCostSpecContinuous(
            running=lambda t, x, u: (x[3] - 1.0) ** 2 + u[1] ** 2,
        )

        # -----------------------------------------------------------------
        # Step 4: encode player ownership of the joint control vector
        # -----------------------------------------------------------------
        #
        # The joint control is
        #
        #   u = [omega, a]
        #
        # so ``u_splits = [1, 1]`` means:
        #
        #   player 1 owns u[0]
        #   player 2 owns u[1]
        u_splits = jnp.asarray([1, 1])

        # -----------------------------------------------------------------
        # Step 5: build the nonlinear IR game
        # -----------------------------------------------------------------
        #
        # The solver consumes this IR object directly. Compared with the
        # frontend game, the IR is more explicit about dimensions and stores
        # player ownership as numeric structure rather than player objects.
        self.game = NonlinearGameType1(
            cs=cs,
            N=2,
            costs=[player_1_cost, player_2_cost],
            u_splits=u_splits,
        )


def main() -> None:
    # -----------------------------------------------------------------
    # Step 0: choose the initial state
    # -----------------------------------------------------------------
    #
    # This matches ``examples/unicycle.py`` so users can compare the frontend
    # and IR versions directly.
    x0 = jnp.array([4.0, 4.0, 0.0, 0.0])

    # Build the IR game.
    unicycle = Unicycle(nt=34, dt=0.1)

    # -----------------------------------------------------------------
    # Step 6: solve the IR game with iLQ
    # -----------------------------------------------------------------
    #
    # Unlike the frontend ``pdg.solve(...)`` wrapper, the low-level solver
    # returns the raw solver tuple:
    #
    #   converged, trajectory, strategy
    converged, trajectory, strategy = solve_ilqgame_feedback(
        unicycle.game,
        x0,
    )

    # -----------------------------------------------------------------
    # Step 7: inspect a few useful outputs
    # -----------------------------------------------------------------
    #
    # The IR trajectory stores the same state and joint-control arrays exposed
    # through the frontend solution convenience properties:
    #
    #   trajectory.xs shape = (nt, nx)
    #   trajectory.us shape = (nt - 1, nu)
    states = trajectory.xs
    joint_controls = trajectory.us

    turn_rate_controls = joint_controls[:, 0]
    accel_controls = joint_controls[:, 1]

    print(
        format_ir_feedback_summary(
            "IR Solve Summary",
            solver="ilq",
            converged=converged,
            trajectory=trajectory,
            strategy=strategy,
        )
    )
    print("\n=== example-specific checks ===")
    print(f"First 5 turn-rate controls: {turn_rate_controls[:5]}")
    print(f"First 5 acceleration controls: {accel_controls[:5]}")


if __name__ == "__main__":
    main()
