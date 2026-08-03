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

from pydgens.ir.trajectorytypes import FixedStepSystemTrajectory, are_xs_close
from pydgens.ir.strategytypes import FixedStepAffineStrategies
from pydgens.ir.systemtypes import propagate_system_trajectory
from pydgens.ir.gametypes import NonlinearGameType1, approx_linear_quadratic_game
from pydgens.ir.diagnostictypes import (
    DiagnosticsLevel,
    SolverDiag,
    validate_diagnostics_level,
)
from pydgens.solvers.lqsolver import solve_lqgame_feedback


@dataclass(frozen=True)
class ILQSolverIterDiag:
    """Compact outcome of one iLQ outer iteration."""

    iteration: int
    trajectory_delta_inf: float
    backtrack_iters: int
    accepted_alpha_scale: float | None
    backtracking_succeeded: bool


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
        lightweight per-iteration records; ``"detailed"`` currently has the
        same fields and reserves room for future expensive metrics.

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
    history: list[ILQSolverIterDiag] | None = (
        [] if diagnostics_level != "off" else None
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
            return_info=history is not None,
        )
        if history is not None:
            curr_strat, curr_traj, success, (backtrack_iters, accepted_alpha) = backtrack_result
            delta_inf = float(jnp.max(jnp.abs(curr_traj.xs - prev_traj.xs)))
            assert history is not None
            iteration_diag = ILQSolverIterDiag(
                iteration=iteration,
                trajectory_delta_inf=delta_inf,
                backtrack_iters=backtrack_iters,
                accepted_alpha_scale=accepted_alpha,
                backtracking_succeeded=bool(success),
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
                if history is not None
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
                if history is not None
                else None
            )
            _log_ilq_result(logger, diagnostics)
            return True, curr_traj, curr_strat, diagnostics

    diagnostics = (
        ILQSolverDiag(
            converged=False, iters=max_iters, reason="max_iters", history=tuple(history),
        )
        if history is not None
        else None
    )
    _log_ilq_result(logger, diagnostics)
    return False, curr_traj, curr_strat, diagnostics


def _log_ilq_iteration(logger, diag: ILQSolverIterDiag) -> None:
    """Render an iLQ iteration diagnostic without recomputing solver metrics."""
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "iLQ iter=%d alpha=%s backtrack_iters=%d success=%s delta_inf=%.6g",
            diag.iteration,
            diag.accepted_alpha_scale,
            diag.backtrack_iters,
            diag.backtracking_succeeded,
            diag.trajectory_delta_inf,
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
