# Copyright 2026 MIT Lincoln Laboratory
# SPDX-License-Identifier: MIT

# Frontend solver interfaces for semantic game objects.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import jax.numpy as jnp

from pydgens.frontend.games import (
    ConstrainedNonlinearGame,
    LQGame,
    NonlinearGame,
)
from pydgens.ir.altypes import (
    JointAugmentedLagrangianState,
    init_joint_augmented_lagrangian_state,
)
from pydgens.ir.gametypes import (
    LinearQuadraticGameType1,
    NonlinearGameType1,
    NonlinearGameType2,
)
from pydgens.ir.strategytypes import FixedStepAffineStrategies
from pydgens.ir.systemtypes import propagate_system_trajectory
from pydgens.ir.trajectorytypes import (
    FixedStepPrimalDualTrajectory,
    FixedStepSystemTrajectory,
)
from pydgens.solvers.alsolver import al_solve_autodiff
from pydgens.solvers.ilqsolver import solve_ilqgame_feedback
from pydgens.solvers.lqsolver import solve_lqgame_feedback

SolveMethod = Literal["auto", "lq", "ilq", "al"]


@dataclass(frozen=True)
class SolveResult:
    """
    Minimal frontend solution bundle returned by ``solve(...)``.

    This object is intentionally plain for the first frontend prototype.
    Different solver families currently return different low-level objects:

    - LQ returns a feedback strategy
    - iLQ returns a convergence flag, trajectory, and strategy
    - AL returns a primal-dual trajectory, AL state, and diagnostics

    Rather than expose those raw tuples directly at the frontend, this
    dataclass provides a small normalized container with optional fields.

    Attributes
    ----------
    method:
        Solver family that was actually used.

    converged:
        Whether the solver reported convergence. For direct LQ solves this
        is recorded as ``True`` because the current LQ solver is a direct
        computation rather than an iterative one.

    strategy:
        Affine feedback strategy returned by LQ or iLQ solves, if available.

    trajectory:
        State/control trajectory returned by iLQ, or propagated from an LQ
        strategy when the caller supplies ``x0``.

    primal_dual_trajectory:
        Primal-dual trajectory returned by the AL solver, if available.

    al_state:
        Augmented Lagrangian state returned by the AL solver, if available.

    diagnostics:
        Solver-specific diagnostics object, if available.

    raw:
        The raw low-level solver output. This is included so power users can
        inspect solver-family-specific details while the frontend solution
        schema is still evolving.
    """

    method: Literal["lq", "ilq", "al"]
    converged: bool | None = None
    strategy: FixedStepAffineStrategies | None = None
    trajectory: FixedStepSystemTrajectory | None = None
    primal_dual_trajectory: FixedStepPrimalDualTrajectory | None = None
    al_state: JointAugmentedLagrangianState | None = None
    diagnostics: Any = None
    raw: Any = None

    @property
    def states(self):
        """
        Convenience access to the state trajectory array, when available.
        """
        if self.trajectory is not None:
            return self.trajectory.xs

        if self.primal_dual_trajectory is not None:
            return self.primal_dual_trajectory.xs

        return None

    @property
    def joint_controls(self):
        """
        Convenience access to the joint control trajectory array, when available.
        """
        if self.trajectory is not None:
            return self.trajectory.us

        if self.primal_dual_trajectory is not None:
            return self.primal_dual_trajectory.us

        return None


def solve(
    game,
    *,
    x0=None,
    method: SolveMethod = "auto",
    op0: FixedStepPrimalDualTrajectory | None = None,
    al_state0: JointAugmentedLagrangianState | None = None,
    **solver_kwargs,
) -> SolveResult:
    """
    Solve a frontend or IR game object using the appropriate solver family.

    This is the intended beginner-facing solver entry point. It accepts a
    semantic frontend game when available, lowers to solver IR when needed,
    chooses a solver family, and returns a small normalized solution bundle.

    Current dispatch rules
    ----------------------
    - ``LQGame`` or ``LinearQuadraticGameType1`` -> LQ solver
    - ``NonlinearGame`` or ``NonlinearGameType1`` -> iLQ solver
    - ``ConstrainedNonlinearGame`` or ``NonlinearGameType2`` ->
      Augmented Lagrangian solver

    Parameters
    ----------
    game:
        Frontend or IR game object to solve.

    x0:
        Optional initial joint state. This is required for iLQ solves, and
        for AL solves unless the caller provides ``op0`` directly. For LQ,
        this is optional; if provided, the frontend will propagate the
        returned feedback strategy to produce a trajectory.

    method:
        Solver family selection. ``"auto"`` infers the method from the game
        type. Explicit choices are currently ``"lq"``, ``"ilq"``, and ``"al"``.

    op0:
        Optional initial primal-dual trajectory for the AL solver.

    al_state0:
        Optional initial augmented Lagrangian state for the AL solver.

    **solver_kwargs:
        Additional keyword arguments forwarded to the selected low-level solver.

    Returns
    -------
    SolveResult
        Minimal normalized frontend solution bundle.

    Notes
    -----
    This frontend wrapper is intentionally lightweight. It focuses on:

    - method dispatch
    - lowering frontend games to IR when needed
    - simple default initialization for AL solves
    - returning a consistent top-level container

    Solver-family structural assumptions still matter:

    - iLQ currently expects unconstrained nonlinear games whose running
      costs are defined over the JOINT control vector and whose declared
      control structure is compatible with the LQ approximation solved
      inside iLQ. In practice, that means no ``GENERAL`` mixed-control
      block structure across player-owned control partitions.
    - AL currently expects constrained nonlinear games whose running costs
      are expressed in each player's LOCAL control variables and are
      ``LOCAL_ONLY`` by control structure.

    It does not yet attempt to provide a polished high-level solution API
    for named player controls, plotting hooks, rich diagnostics, or solver
    logs. Those can be layered on top later without changing the basic
    dispatch contract.
    """

    resolved_method = _resolve_solve_method(
        game=game,
        method=method,
    )

    if resolved_method == "lq":
        return _solve_lq_frontend(
            game=game,
            x0=x0,
            **solver_kwargs,
        )

    if resolved_method == "ilq":
        return _solve_ilq_frontend(
            game=game,
            x0=x0,
            **solver_kwargs,
        )

    if resolved_method == "al":
        return _solve_al_frontend(
            game=game,
            x0=x0,
            op0=op0,
            al_state0=al_state0,
            **solver_kwargs,
        )

    raise RuntimeError(
        f"Unhandled solve method {resolved_method!r}."
    )


def _resolve_solve_method(
    *,
    game,
    method: SolveMethod,
) -> Literal["lq", "ilq", "al"]:
    """
    Resolve an explicit or inferred solver family from the supplied game.
    """

    if method not in {"auto", "lq", "ilq", "al"}:
        raise ValueError(
            "`method` must be one of "
            "{'auto', 'lq', 'ilq', 'al'}."
        )

    if isinstance(game, (LQGame, LinearQuadraticGameType1)):
        inferred = "lq"
    elif isinstance(game, (NonlinearGame, NonlinearGameType1)):
        inferred = "ilq"
    elif isinstance(game, (ConstrainedNonlinearGame, NonlinearGameType2)):
        inferred = "al"
    else:
        raise NotImplementedError(
            f"No frontend solver rule for game type {type(game)}."
        )

    if method == "auto":
        return inferred

    if method != inferred:
        raise ValueError(
            f"`method={method}` is incompatible with game type "
            f"{type(game).__name__}. Expected `{inferred}` or `auto`."
        )

    return method


def _solve_lq_frontend(
    *,
    game,
    x0,
    **solver_kwargs,
) -> SolveResult:
    """
    Solve an LQ game and optionally propagate the resulting strategy.
    """

    # Frontend LQ games lower themselves to IR; direct IR inputs are used
    # as-is so power users can bypass the semantic layer if they want.
    if isinstance(game, LQGame):
        lqgame = game.to_ir()
    else:
        lqgame = game

    strategy = solve_lqgame_feedback(
        lqgame,
        **solver_kwargs,
    )

    trajectory = None

    # If the caller supplies an initial condition, turn the feedback
    # strategy into a concrete trajectory on the sampled grid.
    if x0 is not None:
        trajectory = propagate_system_trajectory(
            lqgame.cs,
            x0=jnp.asarray(x0),
            strategy=strategy,
        )

    return SolveResult(
        method="lq",
        converged=True,
        strategy=strategy,
        trajectory=trajectory,
        raw=strategy,
    )


def _solve_ilq_frontend(
    *,
    game,
    x0,
    **solver_kwargs,
) -> SolveResult:
    """
    Solve an unconstrained nonlinear game using the iLQ solver.

    This frontend path assumes running costs are provided in JOINT-control
    form and satisfy the control-structure assumptions required by the
    linear-quadratic approximation solved inside iLQ.
    """

    if x0 is None:
        raise ValueError(
            "`x0` is required when solving a nonlinear game with `method='ilq'`."
        )

    if isinstance(game, NonlinearGame):
        nlgame = game.to_ir()
    else:
        nlgame = game

    converged, trajectory, strategy = solve_ilqgame_feedback(
        nlgame,
        x0=jnp.asarray(x0),
        **solver_kwargs,
    )

    return SolveResult(
        method="ilq",
        converged=bool(converged),
        strategy=strategy,
        trajectory=trajectory,
        raw=(converged, trajectory, strategy),
    )


def _solve_al_frontend(
    *,
    game: ConstrainedNonlinearGame | NonlinearGameType2,
    x0,
    op0: FixedStepPrimalDualTrajectory | None,
    al_state0: JointAugmentedLagrangianState | None,
    **solver_kwargs,
) -> SolveResult:
    """
    Solve a constrained nonlinear game using the augmented Lagrangian solver.

    Frontend constrained nonlinear games lower themselves to the AL-facing
    IR. Direct ``NonlinearGameType2`` inputs are used as-is so power users
    can bypass the frontend layer when needed.
    """
    if isinstance(game, ConstrainedNonlinearGame):
        game = game.to_ir()

    if op0 is None:
        if x0 is None:
            raise ValueError(
                "`x0` or `op0` is required when solving a constrained "
                "nonlinear game with `method='al'`."
            )

        op0 = _default_al_op0(
            game=game,
            x0=jnp.asarray(x0),
        )

    if al_state0 is None:
        al_state0 = init_joint_augmented_lagrangian_state(
            nc_ineq=game.constraints.nc_ineq,
            nc_eq=game.constraints.nc_eq,
        )

    primal_dual_trajectory, al_state, diagnostics = al_solve_autodiff(
        game,
        op0,
        al_state0,
        **solver_kwargs,
    )

    return SolveResult(
        method="al",
        converged=None,
        primal_dual_trajectory=primal_dual_trajectory,
        al_state=al_state,
        diagnostics=diagnostics,
        raw=(primal_dual_trajectory, al_state, diagnostics),
    )


def _default_al_op0(
    *,
    game: NonlinearGameType2,
    x0: jnp.ndarray,
) -> FixedStepPrimalDualTrajectory:
    """
    Build a zero initial primal-dual trajectory for the AL solver.
    """

    if x0.shape != (game.nx,):
        raise ValueError(
            f"`x0` must have shape ({game.nx},). Got {x0.shape}."
        )

    dtype = x0.dtype

    xs = jnp.zeros(
        (game.nt, game.nx),
        dtype=dtype,
    ).at[0].set(x0)

    us = jnp.zeros(
        (game.nt - 1, game.nu),
        dtype=dtype,
    )

    ls = jnp.zeros(
        (game.nt - 1, game.N, game.nx),
        dtype=dtype,
    )

    return FixedStepPrimalDualTrajectory(
        tg=game.tg,
        xs=xs,
        us=us,
        ls=ls,
    )
