# Copyright 2026 MIT Lincoln Laboratory
# SPDX-License-Identifier: MIT

# Functions for iterative linear-quadratic game solver

# Ref
# - Fridovich-Keil, David, et al. 
#   "Efficient iterative linear-quadratic approximations for nonlinear multi-player general-sum differential games." 
#   2020 IEEE international conference on robotics and automation (ICRA). IEEE, 2020.
#   https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9197129
# C++ Implementation: https://github.com/HJReachability/ilqgames/tree/master
# Julia Implementation: https://github.com/JuliaGameTheoreticPlanning/iLQGames.jl

import logging
import jax
import jax.numpy as jnp

from dataclasses import dataclass
from typing import Tuple
from copy import deepcopy

from pydgens.ir.trajectorytypes import (
    FixedStepSystemTrajectory,
    _as_xs_tolerance,
    are_xs_close,
)
from pydgens.ir.timetypes import compute_ts
from pydgens.ir.strategytypes import FixedStepAffineStrategies
from pydgens.ir.systemtypes import propagate_system_trajectory
from pydgens.ir.gametypes import (
    LinearQuadraticGameType1,
    NonlinearGameType1,
    approx_linear_quadratic_game,
)
from pydgens.ir.diagnostictypes import (
    DiagnosticsLevel,
    SolverDiag,
    validate_diagnostics_level,
)
from pydgens.solvers.lqsolver import solve_lqgame_feedback


@dataclass(frozen=True)
class ILQModelAgreementDiag:
    """Per-player agreement between an iLQ quadratic model and rollout.

    Attributes
    ----------
    nonlinear_cost_before
        Per-player nonlinear trajectory costs at the operating trajectory used
        to construct the local LQ game.
    nonlinear_cost_after
        Per-player nonlinear trajectory costs after the selected backtracking
        rollout.
    nonlinear_cost_change
        Per-player nonlinear cost changes, equal to ``after - before``.
    lq_predicted_cost_change
        Per-player cost changes predicted by the LQ approximation evaluated at
        the selected state and control update. The zero-update LQ baseline is
        zero because the local model omits constant cost terms.
    cost_reduction_ratio
        Per-player ratio of actual to LQ-predicted cost reduction:
        ``(before - after) / (-lq_predicted_cost_change)``. ``None`` when the
        LQ model does not predict a strictly positive reduction, since a ratio
        would not be an interpretable agreement measure in that case.
    """

    nonlinear_cost_before: Tuple[float, ...]
    nonlinear_cost_after: Tuple[float, ...]
    nonlinear_cost_change: Tuple[float, ...]
    lq_predicted_cost_change: Tuple[float, ...]
    cost_reduction_ratio: Tuple[float | None, ...]


@dataclass(frozen=True)
class ILQSolverIterDiag:
    """Compact outcome of one iLQ outer iteration.

    Attributes
    ----------
    iteration
        Zero-based outer iLQ iteration number.
    state_update_inf
        Infinity norm of the accepted state-trajectory update:
        ``max(abs(current.xs - previous.xs))``.
    state_update_ratio_inf
        Maximum componentwise state update normalized by the corresponding
        ``converged_max_diff`` tolerance. A value strictly below one passes
        the componentwise convergence criterion.
    state_update_worst_time_node
        Time-node index of the coordinate attaining
        ``state_update_ratio_inf``. Ties use the first coordinate in
        row-major trajectory order.
    state_update_worst_index
        Joint-state index of the coordinate attaining
        ``state_update_ratio_inf``.
    backtrack_iters
        Number of backtracking rollout attempts made for this iteration.
    accepted_alpha_scale
        Feedforward scaling factor accepted by backtracking, or ``None`` if
        no candidate passed the backtracking check.
    backtracking_succeeded
        Whether the backtracking rollout satisfied its state-deviation bound.
    model_agreement
        Per-player nonlinear-versus-LQ cost comparison collected only at
        ``diagnostics_level="detailed"``; otherwise ``None``.
    """

    iteration: int
    state_update_inf: float
    state_update_ratio_inf: float
    state_update_worst_time_node: int
    state_update_worst_index: int
    backtrack_iters: int
    accepted_alpha_scale: float | None
    backtracking_succeeded: bool
    model_agreement: ILQModelAgreementDiag | None


@dataclass(frozen=True)
class ILQSolverDiag(SolverDiag):
    """Diagnostics for an iLQ solve, matching the AL termination contract."""

    history: Tuple[ILQSolverIterDiag, ...]


def scale_strategy(strategy: FixedStepAffineStrategies, alpha_scale: float) -> FixedStepAffineStrategies:
    """
    Return a new affine strategy with the same stage-indexed feedback gains
    but feedforward term scaled by ``alpha_scale`` on each control interval.

    Args:
        strategy (FixedStepAffineStrategies): Input strategy with
            ``P.shape == (nsteps, nu, nx)`` and
            ``alpha.shape == (nsteps, nu)``.
        alpha_scale (float): Scalar to multiply each ``alpha[k]``.

    Returns:
        FixedStepAffineStrategies: New strategy with scaled feedforward term.
    
    Raises:
        ValueError: If alpha_scale is not in the range [0, 1].
    """

    if not (0.0 <= alpha_scale <= 1.0):
        raise ValueError(f"`alpha_scale` must be in [0, 1], got {alpha_scale}")
    
    return FixedStepAffineStrategies(
        tg = strategy.tg, 
        P = strategy.P, 
        alpha = strategy.alpha * alpha_scale
    )

def backtrack_scale_strategy(
    strat_del: FixedStepAffineStrategies,
    op: FixedStepSystemTrajectory,
    nlgame: NonlinearGameType1,
    max_iters: int,
    alpha_scale_init: float,
    alpha_scale_step: float,
    max_elwise_diff: float | jnp.ndarray,
    return_info: bool = False,
) -> (
    tuple[FixedStepAffineStrategies, FixedStepSystemTrajectory, bool]
    | tuple[
        FixedStepAffineStrategies,
        FixedStepSystemTrajectory,
        bool,
        tuple[int, float | None],
    ]
):
    """
    Perform backtracking line search to stabilize strategy updates in iLQ solvers.

    This function iteratively scales the feedforward term (alpha) of a newly computed
    strategy toward a previous strategy, and tests whether the resulting system
    trajectory diverges too far from the previous operating point. If it does,
    the step size is decreased and the process is retried. The purpose is to ensure
    smooth updates and convergence when applying iLQ-style iterative feedback algorithms.

    Args:
    - strat_del : FixedStepAffineStrategies
        Candidate strategy in ``(delx, delu)`` coordinates, indexed by control
        interval.
    - op : FixedStepSystemTrajectory
        Operating trajectory about which the ``(delx, delu)`` strategy is
        defined. ``op.xs`` is node-indexed with length ``nt`` while ``op.us``
        is stage-indexed with length ``nsteps = nt - 1``.
    - nlgame : NonlinearGameType1
        The nonlinear game definition (provides dynamics).
    - max_iters : int
        Maximum number of backtracking steps to try.
    - alpha_scale_init : float
        Initial step size toward the new strategy (in [0, 1]).
        e.g., 1.0 attempts to step all the way to the candidate strategy
    - alpha_scale_step : float
        Multiplicative decay rate for candidate strategy feedforward term (alpha) during backtracking 
        e.g., 0.5 halves the step size at each attempt
    - max_elwise_diff : float or jax.Array, shape ``(nx,)``
        Strict scalar or per-state-component bounds on trajectory deviation
        before declaring divergence. A scalar applies to every state
        component; a vector supports mixed-unit state scales.
    - return_info : bool, optional
        When true, append ``(attempts, accepted_alpha_scale)`` to the return
        tuple. This is intended for solver diagnostics.

    Returns:
    - scaled_strategy : FixedStepAffineStrategies
        The stabilized strategy (may be unscaled if no backtracking succeeded).
    - new_trajectory : FixedStepSystemTrajectory
        The resulting trajectory using the stabilized strategy.
    - success : bool
        Whether a sufficiently small divergence was achieved.

    Raises:
    - ValueError: If `initial_alpha_scale` or `alpha_scale_step` are not in (0, 1].
    """
    if not (0 < alpha_scale_init <= 1.0):
        raise ValueError("alpha_scale_init must be in the interval (0, 1].")
    if not (0 < alpha_scale_step <= 1.0):
        raise ValueError("alpha_scale_step must be in the interval (0, 1].")

    for i in range(max_iters):

        # rescale the (delx, delu) strategy
        scale = alpha_scale_init if i == 0 else alpha_scale_step ** i
        scaled_strat_del = scale_strategy(strat_del, scale)

        # Map the rescaled delta-strategy from (delx, delu) coordinates into
        # absolute (x, u) coordinates. This mapping is stage-indexed: each
        # strategy slice k pairs with the operating point control u[k] and the
        # state at the start of that control interval, x[k].
        new_strat = FixedStepAffineStrategies(
            tg = nlgame.tg,
            P=scaled_strat_del.P, 
            alpha=(
                scaled_strat_del.alpha
                - op.us
                - jax.vmap(lambda P_t, x_t: P_t @ x_t)(
                    scaled_strat_del.P,
                    op.xs[:-1],
                )
            )
        )

        # propagate a new operating point trajectory from rescaled
        # strategy that has been mapped into absolut (x, u) space
        new_op = propagate_system_trajectory(nlgame.cs,
            x0 = op.xs[0],
            strategy = new_strat
        )

        # check for nearnest to original operating point
        if are_xs_close(op, 
            traj2=new_op, 
            max_elwise_diff=max_elwise_diff
        ):
            if return_info:
                return new_strat, new_op, True, (i + 1, scale)
            return new_strat, new_op, True

    # Return the last tested values with failure flag
    if return_info:
        return new_strat, new_op, False, (max_iters, None)
    return new_strat, new_op, False


def solve_ilqgame_feedback(
    nlgame: NonlinearGameType1,
    x0: jnp.ndarray,
    init_traj: FixedStepSystemTrajectory = None,
    init_strat: FixedStepAffineStrategies = None,
    max_iters: int = 50,
    converged_max_diff: float | jnp.ndarray = 5e-2,   # Ref: https://github.com/JuliaGameTheoreticPlanning/iLQGames.jl/blob/v0.2.7/src/ilq_solver.jl#L17
    backtrack_max_iters: int = 20,
    backtrack_scale_init: float = 0.5,
    backtrack_scale_step: float = 0.5,
    backtrack_scale_max_diff: float | jnp.ndarray = 30 * 5e-2, # Ref: https://github.com/JuliaGameTheoreticPlanning/iLQGames.jl/blob/v0.2.7/src/ilq_solver.jl#L20
    logger = None,
    diagnostics_level: DiagnosticsLevel = "off",
) -> Tuple[
    bool,
    FixedStepSystemTrajectory,
    FixedStepAffineStrategies,
    ILQSolverDiag | None,
]:
    """
    Solve a nonlinear dynamic game using iterative linear-quadratic (iLQ) feedback Nash strategy.

    This function implements an iterative procedure to approximate a feedback Nash equilibrium
    of a nonlinear, finite horizon, unconstrained dynamic game by repeatedly linearizing the 
    system dynamics, quadraticizing the players' cost functions, solving the resulting 
    linear-quadratic game, and updating the strategy with a backtracking line search to 
    ensure stable convergence.

    Parameters
    ----------
    - nlgame : NonlinearGameType1
        Instance of the nonlinear, finite horizon, unconstrained game
        including dynamics, cost functions, and game dimensions.
    - x0 : jnp.ndarray
        Initial joint state of game, vector of size n
    - init_traj : FixedStepSystemTrajectory, optional
        Initial trajectory guess used to start the iteration. ``xs`` is
        node-indexed with shape ``(nt, nx)`` and ``us`` is stage-indexed with
        shape ``(nsteps, nu)``. If omitted, a zero trajectory of matching shape
        is used.
    - init_strat : FixedStepAffineStrategies, optional
        Initial affine feedback strategy of the form
        ``u[k] = -P[k] @ x[k] - alpha[k]`` on each control interval. If
        omitted, a zero strategy of matching shape is used.
    - max_iters : int, optional
        Maximum number of outer iterations before termination.
    - converged_max_diff : float or jax.Array, shape ``(nx,)``, optional
        Strict scalar or per-state-component bounds for convergence of the
        state trajectory. A vector supports different absolute tolerances for
        mixed-unit state components.
    - backtrack_max_iters : int, optional
        Maximum number of backtracking steps when scaling the strategy toward the LQ solution.
    - backtrack_scale_init : float, optional
        Initial step size used in scaling the candidate strategy during backtracking.
        e.g., 1.0 attempts to step all the way to the candidate strategy
    - backtrack_scale_step : float, optional
        Multiplicative decay rate for alpha during backtracking 
        e.g., 0.5 halves the step size at each attempt
    - backtrack_scale_max_diff : float or jax.Array, shape ``(nx,)``, optional
        Strict scalar or per-state-component bounds on the trajectory update
        allowed during backtrack scaling.
    - logger : Logger
        logger object to manage logs of solver
    - diagnostics_level : {"off", "basic", "detailed"}, optional
        Collection level for :class:`ILQSolverDiag`. ``"basic"`` retains
        lightweight per-iteration records. ``"detailed"`` additionally
        evaluates per-player nonlinear costs and compares their changes with
        the local LQ model's predicted changes.

    Returns
    -------
    converged : bool
        True if iterative linear-quadratic algorithm converged to local feedback Nash equilibrium
    trajectory : SystemTrajectory
        Operating point of local feedback Nash equilibrium, similar to tha open-loop equilibrium
    strategy : FixedStepAffineStrategies
        Converged feedback strategy that approximates a local Nash equilibrium of the nonlinear game.
    diagnostics : ILQSolverDiag or None
        Per-iteration diagnostics when ``diagnostics_level`` is not ``"off"``;
        otherwise ``None``.

    Notes
    -----
    - Convergence and backtracking both use the same scalar-or-componentwise
      absolute state-deviation check.
    - Cost quadraticizations are computed per player, allowing heterogeneous objectives.
    - The algorithm follows the general structure of iLQ or iLQGames algorithms, 
    with backtracking to ensure numerical stability.
    """

    logger = logger or logging.getLogger(__name__)
    diagnostics_level = validate_diagnostics_level(diagnostics_level)
    collect_basic_diagnostics = diagnostics_level != "off"
    collect_detailed_diagnostics = diagnostics_level == "detailed"
    history: list[ILQSolverIterDiag] | None = (
        [] if collect_basic_diagnostics else None
    )

    if init_traj is None:
        init_traj = FixedStepSystemTrajectory(
            tg = nlgame.tg,
            xs = jnp.zeros((nlgame.nt, nlgame.nx)),
            us = jnp.zeros((nlgame.nsteps, nlgame.nu))
        )

    if init_strat is None:
        init_strat = FixedStepAffineStrategies(
            tg = nlgame.tg,
            P = jnp.zeros((nlgame.nsteps, nlgame.nu, nlgame.nx)),
            alpha = jnp.zeros((nlgame.nsteps, nlgame.nu))
        )

    prev_traj = deepcopy(init_traj) # used for checking deviation during backtrack scaling
    curr_strat = deepcopy(init_strat)

    # compute operating point for first iteration
    curr_traj = propagate_system_trajectory(
        nlgame.cs,
        x0 = x0,
        strategy = curr_strat
    )
    nonlinear_costs = (
        _nonlinear_player_costs(nlgame, curr_traj)
        if collect_detailed_diagnostics
        else None
    )

    for iteration in range(max_iters):

        # approximate the nonlinear game as linear-quadratic about the operating point
        # The LQgame is formulated as the second order Taylor expansion
        # around the current operating point because the linearization and
        # quadratization compute the jacobians and hessians at that points
        # Therefore, the solution to the LQ game is expressed in transformed
        # stage-indexed coordinates delx[k] = x[k] - x_op[k] and
        # delu[k] = u[k] - u_op[k].
        lq_game_del = approx_linear_quadratic_game(nlgame, op=curr_traj)

        # solve for nash feedback strategy of the linear-quadrate game in (delx, delu) space.
        # Note that lq approximate game is not checked for block-diagonal quadratic cost
        # matrix, R, because it should be so by design of the approx_linear_quadratic_game
        lq_strat_del = solve_lqgame_feedback(lq_game_del, check_block_diag=False)

        # Step 5: Backtrack scale to step the current strategy (in absolute x,u space)
        # toward the candidate lq strategy (expressed in delx, delu space) while avoiding large 
        # trajectory deviations from prev_traj that can cause algorithm to diverge
        prev_traj = deepcopy(curr_traj)
        backtrack_result = backtrack_scale_strategy(
            strat_del=lq_strat_del,
            op=prev_traj,
            nlgame=nlgame,
            max_iters=backtrack_max_iters,
            alpha_scale_init=backtrack_scale_init,
            alpha_scale_step=backtrack_scale_step,
            max_elwise_diff=backtrack_scale_max_diff,
            return_info=collect_basic_diagnostics,
        )
        if collect_basic_diagnostics:
            curr_strat, curr_traj, success, (backtrack_iters, accepted_alpha) = backtrack_result
            (
                state_update_inf,
                state_update_ratio_inf,
                state_update_worst_time_node,
                state_update_worst_index,
            ) = _state_update_metrics(
                curr_traj, prev_traj, converged_max_diff
            )
            model_agreement = None
            if collect_detailed_diagnostics:
                assert nonlinear_costs is not None
                new_nonlinear_costs = _nonlinear_player_costs(nlgame, curr_traj)
                model_agreement = _model_agreement_diagnostics(
                    nonlinear_costs,
                    new_nonlinear_costs,
                    lq_game_del,
                    curr_traj,
                    prev_traj,
                )
                nonlinear_costs = new_nonlinear_costs
            assert history is not None
            iteration_diag = ILQSolverIterDiag(
                iteration=iteration,
                state_update_inf=state_update_inf,
                state_update_ratio_inf=state_update_ratio_inf,
                state_update_worst_time_node=state_update_worst_time_node,
                state_update_worst_index=state_update_worst_index,
                backtrack_iters=backtrack_iters,
                accepted_alpha_scale=accepted_alpha,
                backtracking_succeeded=bool(success),
                model_agreement=model_agreement,
            )
            history.append(iteration_diag)
            _log_ilq_iteration(logger, iteration_diag)
        else:
            curr_strat, curr_traj, success = backtrack_result
        if not success:
            diagnostics = (
                ILQSolverDiag(
                    converged=False, iters=iteration + 1,
                    reason="backtracking_failed", history=tuple(history),
                )
                if collect_basic_diagnostics
                else None
            )
            _log_ilq_result(logger, diagnostics)
            return False, curr_traj, curr_strat, diagnostics

        # Step 6: Check for convergence
        if are_xs_close(curr_traj, traj2=prev_traj, max_elwise_diff=converged_max_diff):
            diagnostics = (
                ILQSolverDiag(
                    converged=True, iters=iteration + 1,
                    reason="converged", history=tuple(history),
                )
                if collect_basic_diagnostics
                else None
            )
            _log_ilq_result(logger, diagnostics)
            return True, curr_traj, curr_strat, diagnostics

    diagnostics = (
        ILQSolverDiag(
            converged=False, iters=max_iters, reason="max_iters", history=tuple(history),
        )
        if collect_basic_diagnostics
        else None
    )
    _log_ilq_result(logger, diagnostics)
    return False, curr_traj, curr_strat, diagnostics


def _log_ilq_iteration(logger, diag: ILQSolverIterDiag) -> None:
    """Render an iLQ iteration diagnostic without recomputing solver metrics."""
    if logger.isEnabledFor(logging.DEBUG):
        message = (
            "iLQ iter=%d alpha=%s backtrack_iters=%d success=%s "
            "state_update_inf=%.6g state_update_ratio_inf=%.6g "
            "state_update_worst=(%d, %d)"
        )
        values = (
            diag.iteration,
            diag.accepted_alpha_scale,
            diag.backtrack_iters,
            diag.backtracking_succeeded,
            diag.state_update_inf,
            diag.state_update_ratio_inf,
            diag.state_update_worst_time_node,
            diag.state_update_worst_index,
        )
        if diag.model_agreement is not None:
            message += (
                " nonlinear_cost_change=%s lq_predicted_cost_change=%s "
                "cost_reduction_ratio=%s"
            )
            values += (
                diag.model_agreement.nonlinear_cost_change,
                diag.model_agreement.lq_predicted_cost_change,
                diag.model_agreement.cost_reduction_ratio,
            )
        logger.debug(message, *values)


def _state_update_metrics(
    current: FixedStepSystemTrajectory,
    previous: FixedStepSystemTrajectory,
    converged_max_diff: float | jnp.ndarray,
) -> tuple[float, float, int, int]:
    """Return raw and tolerance-normalized state-update diagnostics."""
    tolerance = _as_xs_tolerance(converged_max_diff, current.nx)
    abs_update = jnp.abs(current.xs - previous.xs)

    # A zero tolerance permits only an exactly unchanged coordinate. Avoid a
    # ``0 / 0`` diagnostic for that valid case while reporting any nonzero
    # update as an infinite normalized ratio.
    safe_tolerance = jnp.where(tolerance > 0.0, tolerance, 1.0)
    update_ratio = jnp.where(
        tolerance > 0.0,
        abs_update / safe_tolerance,
        jnp.where(abs_update == 0.0, 0.0, jnp.inf),
    )
    worst_flat_index = int(jnp.argmax(update_ratio))
    worst_time_node, worst_state_index = divmod(worst_flat_index, current.nx)
    return (
        float(jnp.max(abs_update)),
        float(jnp.max(update_ratio)),
        worst_time_node,
        worst_state_index,
    )


def _nonlinear_player_costs(
    nlgame: NonlinearGameType1, trajectory: FixedStepSystemTrajectory
) -> Tuple[float, ...]:
    """Evaluate the iLQ running-plus-terminal cost for each player."""
    ts = compute_ts(trajectory.tg)
    costs = []
    for player_cost in nlgame.costs:
        running_cost = sum(
            player_cost.running(ts[k], trajectory.xs[k], trajectory.us[k])
            for k in range(trajectory.nsteps)
        )
        terminal_cost = (
            player_cost.terminal(ts[-1], trajectory.xs[-1])
            if player_cost.terminal is not None
            else 0.0
        )
        costs.append(float(running_cost + terminal_cost))
    return tuple(costs)


def _lq_predicted_player_cost_changes(
    lq_game: LinearQuadraticGameType1,
    current: FixedStepSystemTrajectory,
    previous: FixedStepSystemTrajectory,
) -> Tuple[float, ...]:
    """Evaluate local LQ cost changes for a selected trajectory update."""
    state_update = current.xs - previous.xs
    control_update = current.us - previous.us
    running_state_change = (
        0.5 * jnp.einsum(
            "ki,kpij,kj->p", state_update[:-1], lq_game.Q, state_update[:-1]
        )
        + jnp.einsum("kpi,ki->p", lq_game.q, state_update[:-1])
    )
    running_control_change = (
        0.5 * jnp.einsum(
            "ki,kpij,kj->p", control_update, lq_game.R, control_update
        )
        + jnp.einsum("kpi,ki->p", lq_game.r, control_update)
    )
    terminal_change = (
        0.5 * jnp.einsum(
            "i,pij,j->p", state_update[-1], lq_game.Qf, state_update[-1]
        )
        + jnp.einsum("pi,i->p", lq_game.qf, state_update[-1])
    )
    return tuple(
        float(cost)
        for cost in running_state_change + running_control_change + terminal_change
    )


def _model_agreement_diagnostics(
    nonlinear_cost_before: Tuple[float, ...],
    nonlinear_cost_after: Tuple[float, ...],
    lq_game: LinearQuadraticGameType1,
    current: FixedStepSystemTrajectory,
    previous: FixedStepSystemTrajectory,
) -> ILQModelAgreementDiag:
    """Compare per-player nonlinear rollout and local LQ cost changes."""
    predicted_change = _lq_predicted_player_cost_changes(
        lq_game, current, previous
    )
    nonlinear_change = tuple(
        after - before
        for before, after in zip(nonlinear_cost_before, nonlinear_cost_after)
    )
    reduction_ratio = tuple(
        (before - after) / -predicted
        if predicted < 0.0
        else None
        for before, after, predicted in zip(
            nonlinear_cost_before, nonlinear_cost_after, predicted_change
        )
    )
    return ILQModelAgreementDiag(
        nonlinear_cost_before=nonlinear_cost_before,
        nonlinear_cost_after=nonlinear_cost_after,
        nonlinear_cost_change=nonlinear_change,
        lq_predicted_cost_change=predicted_change,
        cost_reduction_ratio=reduction_ratio,
    )


def _log_ilq_result(logger, diag: ILQSolverDiag | None) -> None:
    """Render the final iLQ diagnostic without recomputing solver metrics."""
    if diag is not None and logger.isEnabledFor(logging.INFO):
        logger.info(
            "iLQ finished converged=%s iters=%d reason=%s",
            diag.converged,
            diag.iters,
            diag.reason,
        )
