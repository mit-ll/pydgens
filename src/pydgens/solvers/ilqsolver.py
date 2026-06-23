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

from typing import Tuple
from copy import deepcopy

from pydgens.ir.trajectorytypes import FixedStepSystemTrajectory, are_xs_close
from pydgens.ir.strategytypes import FixedStepAffineStrategies
from pydgens.ir.systemtypes import propagate_system_trajectory
from pydgens.ir.gametypes import NonlinearGameType1, approx_linear_quadratic_game
from pydgens.solvers.lqsolver import solve_lqgame_feedback


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
    max_elwise_diff: float
) -> tuple[FixedStepAffineStrategies, FixedStepSystemTrajectory, bool]:
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
    - max_elwise_diff : float
        Maximum allowed element-wise trajectory deviation before declaring divergence.

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
            return new_strat, new_op, True

    # Return the last tested values with failure flag
    return new_strat, new_op, False


def solve_ilqgame_feedback(
    nlgame: NonlinearGameType1,
    x0: jnp.ndarray,
    init_traj: FixedStepSystemTrajectory = None,
    init_strat: FixedStepAffineStrategies = None,
    max_iters: int = 50,
    converged_max_diff: float = 5e-2,   # Ref: https://github.com/JuliaGameTheoreticPlanning/iLQGames.jl/blob/v0.2.7/src/ilq_solver.jl#L17
    backtrack_max_iters: int = 20,
    backtrack_scale_init: float = 0.5,
    backtrack_scale_step: float = 0.5,
    backtrack_scale_max_diff: float = 30 * 5e-2, # Ref: https://github.com/JuliaGameTheoreticPlanning/iLQGames.jl/blob/v0.2.7/src/ilq_solver.jl#L20
    logger = None
) -> Tuple[bool, FixedStepSystemTrajectory, FixedStepAffineStrategies]:
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
    - converged_max_diff : float, optional
        Threshold for convergence of the state trajectory (infinity norm).
    - backtrack_max_iters : int, optional
        Maximum number of backtracking steps when scaling the strategy toward the LQ solution.
    - backtrack_scale_init : float, optional
        Initial step size used in scaling the candidate strategy during backtracking.
        e.g., 1.0 attempts to step all the way to the candidate strategy
    - backtrack_scale_step : float, optional
        Multiplicative decay rate for alpha during backtracking 
        e.g., 0.5 halves the step size at each attempt
    - backtrack_scale_max_diff : float, optional
        maximum infnorm of trajectories allowed during backtrack scaling
    - logger : Logger
        logger object to manage logs of solver

    Returns
    -------
    converged : bool
        True if iterative linear-quadratic algorithm converged to local feedback Nash equilibrium
    trajectory : SystemTrajectory
        Operating point of local feedback Nash equilibrium, similar to tha open-loop equilibrium
    strategy : FixedStepAffineStrategies
        Converged feedback strategy that approximates a local Nash equilibrium of the nonlinear game.

    Notes
    -----
    - The convergence check is based on the elementwise infinity norm of state deviations.
    - Cost quadraticizations are computed per player, allowing heterogeneous objectives.
    - The algorithm follows the general structure of iLQ or iLQGames algorithms, 
    with backtracking to ensure numerical stability.
    """

    logger = logger or logging.getLogger("solve_ilqgame_feedback")

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

        logger.debug(f"iter {iteration}: op.xs = \n{curr_traj.xs}")
        logger.debug(f"iter {iteration}: op.us = \n{curr_traj.us}")

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
        curr_strat, curr_traj, success = backtrack_scale_strategy(
            strat_del = lq_strat_del,
            op = prev_traj,
            nlgame = nlgame,
            max_iters = backtrack_max_iters,
            alpha_scale_init = backtrack_scale_init,
            alpha_scale_step = backtrack_scale_step,
            max_elwise_diff = backtrack_scale_max_diff
        )
        # logger.debug(f"iter {iteration}: nl_strat = \n{curr_strat}")

        if not success:
            logger.warning(f"Backtracking failed at iteration {iteration}: exiting early.")
            return False, curr_traj, curr_strat

        # Step 6: Check for convergence
        if are_xs_close(curr_traj, traj2=prev_traj, max_elwise_diff=converged_max_diff):
            logger.info(f"Converged at iteration {iteration}")
            return True, curr_traj, curr_strat

    logger.warning("Max iterations reached without convergence.")
    return False, curr_traj, curr_strat
