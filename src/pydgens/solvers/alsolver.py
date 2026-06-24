# Copyright 2026 MIT Lincoln Laboratory
# SPDX-License-Identifier: MIT

# Augmented Lagrangian solver for constrained dynamic games
#
# Ref: 
# - https://arxiv.org/pdf/2104.08452
# - https://github.com/RoboticExplorationLab/Algames.jl
import jax
import math
import jax.numpy as jnp
import numpy as np
import logging
import time

from typing import Tuple, Literal, Callable, List, get_args
from functools import singledispatch

import pydgens.ir.trajectorytypes as trajtypes
import pydgens.ir.systemtypes as systypes
import pydgens.ir.costtypes as costtypes
import pydgens.ir.constrainttypes as contypes
import pydgens.ir.gametypes as gametypes
import pydgens.ir.altypes as altypes

# module-level logger
logger = logging.getLogger(__name__)

ResidualNormKind = Literal["l1", "l2", "l1_mean", "l2_rms", "linf"]

_DEBUG_CALL_COUNTS: dict[str, int] = {}


def _next_debug_call_id(name: str) -> int:
    _DEBUG_CALL_COUNTS[name] = _DEBUG_CALL_COUNTS.get(name, 0) + 1
    return _DEBUG_CALL_COUNTS[name]


def _should_emit_sampled_debug(call_id: int, *, first_n: int = 5, every: int = 25) -> bool:
    return call_id <= first_n or (every > 0 and call_id % every == 0)


def _debug_shape(x) -> tuple[int, ...] | None:
    shape = getattr(x, "shape", None)
    if shape is None:
        return None
    return tuple(int(s) for s in shape)

def pack_decision_vars_no_checks(op:trajtypes.FixedStepPrimalDualTrajectory) -> jnp.ndarray:
    """
    Pack decision variables z from a FixedStepPrimalDualTrajectory into a 1D vector.

    Args
    ----
    - op : FixedStepPrimalDualTrajectory
        Structured trajectory data to be packed into a flat vector
        xs: (nt,nx), joint state trajectory
        us: (nsteps,nu), joint control trajectory
        ls: (nsteps,N,nx), player-specific Lagrange multipliers for dynamics constraints

    Returns
    -------
    - z : jnp.ndarray shape ( nsteps*nx + nsteps*nu + nsteps*N*nx, )
        flattened decision variables from trajectory with packing convention
        z = [ vec(xs[1:]) ; vec(us) ; vec(ls) ]
        Note that x0 is excluded from decision variables as it is assumed fixed
    """
    return jnp.concatenate([
        jnp.ravel(op.xs[1:, :]),
        jnp.ravel(op.us),
        jnp.ravel(op.ls),
    ])


def unpack_decision_vars_no_checks(
    z: jnp.ndarray,
    template: trajtypes.FixedStepPrimalDualTrajectory
) -> trajtypes.FixedStepPrimalDualTrajectory:
    """
    Unpack z into a new FixedStepPrimalDualTrajectory using `template` for context.

    Args
    ----
    - z : jnp.ndarray shape ( nsteps*nx + nsteps*nu + nsteps*N*nx, )
        flattened decision variables to be unpacked into structured trajectory with packing convention
        z = [ vec(xs[1:]) ; vec(us) ; vec(ls) ]
        Note that x0 is excluded from decision variables as it is assumed fixed
    - template : FixedStepPrimalDualTrajectory
        template of trajectory to be created, used to identify shapes of
        state, control, and lagrange multipliers trajectories
        Also used to set time grid and initial state x0

    Returns
    -------
    - op : FixedStepPrimalDualTrajectory
        Structured trajectory data unpacked from flat z vector
        xs: (nt,nx), joint state trajectory
        us: (nsteps,nu), joint control trajectory
        ls: (nsteps,N,nx), player-specific Lagrange multipliers for dynamics constraints

    Notes
    -----
    - z does NOT contain x0; x0 is copied from template.xs[0]
    - tg is copied from template.tg
    - Assumes z has correct length for template shapes (no checks performed).
    """
    nt, nx = template.xs.shape
    K = template.nsteps
    nu = template.us.shape[1]
    N = template.ls.shape[1]

    # block sizes
    dim_x = K * nx
    dim_u = K * nu
    dim_l = K * N * nx

    zx = z[:dim_x]
    zu = z[dim_x:dim_x + dim_u]
    zl = z[dim_x + dim_u:dim_x + dim_u + dim_l]

    xs_tail = zx.reshape((K, nx))
    us = zu.reshape((K, nu))
    ls = zl.reshape((K, N, nx))

    xs = jnp.concatenate([template.xs[0:1, :], xs_tail], axis=0)

    return trajtypes.FixedStepPrimalDualTrajectory(
        tg=template.tg,
        xs=xs,
        us=us,
        ls=ls,
    )

def pack_decision_vars_1d(
    op: trajtypes.FixedStepPrimalDualTrajectory, 
    *, 
    check_shapes: bool = True
) -> jnp.ndarray:
    """
    Checked wrapper around pack_decision_vars_no_checks.
    """
    if check_shapes:
        _, nx = op.xs.shape
        K = op.nsteps
        if op.us.ndim != 2:
            raise ValueError(f"us must be 2D (nsteps,nu), got shape {op.us.shape}")
        if op.us.shape[0] != K:
            raise ValueError(f"us first dim must be nsteps={K}, got {op.us.shape[0]}")
        if op.ls.ndim != 3:
            raise ValueError(f"ls must be 3D (nsteps,N,nx), got shape {op.ls.shape}")
        if op.ls.shape[0] != K or op.ls.shape[2] != nx:
            raise ValueError(f"ls must have shape (nsteps,N,nx)=({K},N,{nx}), got {op.ls.shape}")
    return pack_decision_vars_no_checks(op)


def unpack_decision_vars(
    z: jnp.ndarray, 
    template: trajtypes.FixedStepPrimalDualTrajectory, 
    *, 
    check_length: bool = True
) -> trajtypes.FixedStepPrimalDualTrajectory:
    """
    Checked wrapper around unpack_decision_vars_no_checks.
    """
    z = jnp.asarray(z)
    if z.ndim != 1:
        raise ValueError(f"z must be 1D, got shape {z.shape}")

    if check_length:
        _, nx = template.xs.shape
        K = template.nsteps
        nu = template.us.shape[1]
        N = template.ls.shape[1]
        expected = K*nx + K*nu + K*N*nx
        if int(z.shape[0]) != expected:
            raise ValueError(f"z has wrong length: expected {expected}, got {z.shape[0]}")

    return unpack_decision_vars_no_checks(z, template)

@singledispatch
def gradient_aug_lagrangian_trajectory(nlgame, *args, **kwargs):
    raise NotImplementedError(f"No implementation for {type(nlgame)}")

@gradient_aug_lagrangian_trajectory.register(gametypes.NonlinearGameType2)
def _gradient_aug_lagrangian_trajectory(
    nlgame: gametypes.NonlinearGameType2,
    op: trajtypes.FixedStepPrimalDualTrajectory,
    alstate: altypes.JointAugmentedLagrangianState,
    *,
    discretize_method: str, # = "rk2",
    ineq_activation: str, # = "altro",
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Assemble stationarity gradients of the augmented Lagrangian for *all* players
    along a primal-dual trajectory.

    For each player i, the augmented Lagrangian is (Le Cleac'h et al., ALGAMES):
        L_i(X, U_i, μ_i; λ, ρ)
          = J_i(X,U) + Σ_k μ_{k,i}ᵀ D_k(X,U)
            + λᵀ C(X,U) + 1/2 * C(X,U)ᵀ diag(ρ) C(X,U),

    where:
      - X is the joint state trajectory (nt, nx)
      - U is the joint control trajectory (nsteps, nu)
      - μ_i are player-specific dynamics multipliers (stored in op.ls[:, i, :])
      - λ, ρ are shared auxiliary-constraint AL parameters (alstate)

    This function computes, for each player i, the joint-state and joint-control gradients:
        dL_dX[i] : (nt, nx)
        dL_dU[i] : (nsteps, nu)

    Design choice
    ------------
    The auxiliary constraint terms (linear λᵀC and quadratic penalty) are shared across players,
    so they are computed once and added to each player's gradients. Player-specific terms
    (cost gradient and dynamics-multiplier terms) are computed per player.

    Returns
    -------
    dL_dX_all : jnp.ndarray, shape (N, nt, nx)
        Player-stacked joint-state gradients.
    dL_dU_all : jnp.ndarray, shape (N, nsteps, nu)
        Player-stacked joint-control gradients (joint control dimension).
        Cost contributions are inserted only into the owning player's control slice; shared
        terms contribute to all players in joint coordinates.
    """
    # ---- basic checks ----
    if nlgame.tg != op.tg:
        raise ValueError(f"game and trajectory must share TimeGrid; got {nlgame.tg} vs {op.tg}")
    if nlgame.nx != op.nx:
        raise ValueError(f"game and trajectory must share joint state dimension; got {nlgame.nx} vs {op.nx}")
    if nlgame.nu != op.nu:
        raise ValueError(f"game and trajectory must share joint control dimension; got {nlgame.nu} vs {op.nu}")

    N = nlgame.N
    nt, nx, nu = nlgame.nt, nlgame.nx, nlgame.nu
    K = nt - 1

    # ---- shared auxiliary constraint gradients (compute once) ----
    # NOTE: both of these functions (*_constraints and *_penalty) independently 
    # call build_constraint_step_linearizations with the same inputs,
    # thus this is redundant work. Instead, we should call this once
    # and pass the results to both the constraints and penalty computations
    dLC_dX, dLC_dU = gradient_aug_lagrangian_trajectory_constraints(
        constraints=nlgame.constraints,
        alstate=alstate,
        op=op,
    )
    dLP_dX, dLP_dU = gradient_aug_lagrangian_trajectory_penalty(
        constraints=nlgame.constraints,
        alstate=alstate,
        op=op,
        ineq_activation=ineq_activation,
    )

    dC_X = dLC_dX + dLP_dX          # (nt, nx)
    dC_U = dLC_dU + dLP_dU          # (nt-1, nu)

    # ---- allocate outputs ----
    dL_dX_all = jnp.zeros((N, nt, nx), dtype=dC_X.dtype)
    dL_dU_all = jnp.zeros((N, K,  nu), dtype=dC_U.dtype)

    # broadcast shared constraint contributions to all players
    dL_dX_all = dL_dX_all + dC_X[None, :, :]
    dL_dU_all = dL_dU_all + dC_U[None, :, :]

    # ---- player-specific terms ----
    # For each player:
    #   add cost gradient (state + local control inserted into joint control)
    #   add dynamics-multiplier gradient (state + joint control)
    u_splits = nlgame.u_splits

    # The dynamics Jacobians depend on the shared trajectory, not on the player.
    # Compute them once here and pass them into the playerwise helper so multi-player
    # games do not repeat the same trajectory linearization N times.
    dfd_dx = None
    dfd_du = None
    if isinstance(nlgame.cs, systypes.SampledContinuousSystemType1):
        dfd_dx, dfd_du = systypes.jacobian_discrete_dynamics_trajectory(
            nlgame.cs, op=op, method=discretize_method
        )

    # helper: compute slice for player i
    def _u_slice(i: int) -> slice:
        start = int(jnp.sum(u_splits[:i]))
        stop = start + int(u_splits[i])
        return slice(start, stop)

    for i in range(N):
        # cost gradient wrt state and player-local control
        # see Le Cleac'h Eqn (3) for reference to player-local control
        # assumption of cost function
        dLJ_dX_i, dLJ_dU_i = costtypes.gradient_cost_local_ctrl_playerwise_trajectory(
            costfn_i=nlgame.costs[i].running,
            termfn_i=nlgame.costs[i].terminal,
            op=op,
            player_i=i,
            u_splits=u_splits,
        )  # (nt,nx), (K,nu_i)

        # dynamics-multiplier term gradient (depends on player i's μ stored in op.ls)
        dLD_dX_i, dLD_dU_i = gradient_aug_lagrangian_playerwise_trajectory_dynamics(
            cs=nlgame.cs,
            player_i=i,
            op=op,
            discretize_method=discretize_method,
            dfd_dx=dfd_dx,
            dfd_du=dfd_du,
        )  # (nt,nx), (K,nu)

        # add to outputs
        dL_dX_all = dL_dX_all.at[i].add(dLJ_dX_i + dLD_dX_i)

        # cost control gradient is local => insert into player's joint slice
        sl = _u_slice(i)
        dL_dU_all = dL_dU_all.at[i, :, sl].add(dLJ_dU_i)

        # dynamics control gradient is joint => add full
        dL_dU_all = dL_dU_all.at[i].add(dLD_dU_i)

    return dL_dX_all, dL_dU_all

def gradient_aug_lagrangian_playerwise_trajectory_dynamics(
    cs: systypes.SampledContinuousSystemType1,
    player_i: int,
    op: trajtypes.FixedStepPrimalDualTrajectory,
    discretize_method: str,
    *,
    dfd_dx: jnp.ndarray | None = None,
    dfd_du: jnp.ndarray | None = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Contribution to ∇_X L_i and ∇_U L_i from the dynamics-constraint term, where L_i
    is the augmented lagrangian function for player i, X is the joint state trajectory
    and U is the joint control trajectory. Note that this function is "playerwise"
    in that it multiplies a specific player's lagrange multipliers, even though
    it evaluates jacobian of joint control (not just player's local controls)

    The continuous-time dynamics are discretized using an integration
    method such as Euler, RK2, RK3, RK4
        continuous: dx/dt = f(t, x, u)
        discretized: x_{k+1} = f_d(t_k, x_k, u_k)
    
    Then discrete dynamics residual per step are given as:
        D_k(x_k, u_k, x_{k+1}) = f_d(t_k, x_k, u_k) - x_{k+1}

    Dynamics part of player i Lagrangian:
        Σ_k μ_{k,i}^T D_k
    where μ_{k,i} is stored in op.ls[k, player_i, :].

    Returns contributions:
      State:
        at x_k:     +A_k^T μ_k
        at x_{k+1}: -μ_k
      Control (JOINT control u_k):
        at u_k:     +B_k^T μ_k

    Parameters
    ----------
    cs : SampledContinuousSystemType1
        control system definition containing time grid, joint state dims, joint control dims, and dynamics
    player_i : int
        Player index (0-based).
    op : FixedStepPrimalDualTrajectory
        Primal-dual trajectory about which the gradient is evaluated. Must contain
        joint state trajectory `xs`, joint control trajectory `us`, and per-player
        dynamics multipliers `ls` (μ_i).
    discretize_method : str
        Integration method used to construct f_d ("euler", "rk2", "rk3", "rk4", ...).
        Should be treated as static for JIT purposes.
    dfd_dx, dfd_du : jnp.ndarray, optional
        Precomputed trajectory Jacobians of the discrete dynamics. When omitted, this
        function computes them itself for backwards-compatible direct use. Passing them
        avoids recomputing the same trajectory Jacobians once per player.

    Returns
    -------
    dLD_dX : jnp.ndarray, shape (nt, nx)
        State-gradient contributions from Σ_k ls_{k,i}^T D_k.
    dLD_dU : jnp.ndarray, shape (nt-1, nu)
        Joint-control-gradient contributions from dynamics term.
        (Slice to player-local later using u_splits.)
    """

    # Player-specific dynamics multipliers μ_k: (nt-1, nx)
    mu_i = op.ls[:, player_i, :]

    # Jacobians along trajectory:
    # dfd_dx: (nt-1, nx, nx)  A_k
    # dfd_du: (nt-1, nx, nu)  B_k
    if dfd_dx is None or dfd_du is None:
        dfd_dx, dfd_du = systypes.jacobian_discrete_dynamics_trajectory(
            cs, op=op, method=discretize_method
        )

    # State contribution: term_x[k] = A_k^T @ μ_k   -> (nt-1, nx)
    term_x = jax.vmap(lambda A, lam: A.T @ lam)(dfd_dx, mu_i)

    # Control contribution: term_u[k] = B_k^T @ μ_k -> (nt-1, nu)
    dLD_dU = jax.vmap(lambda B, lam: B.T @ lam)(dfd_du, mu_i)

    # Assemble into (nt, nx): x_k gets +term_x[k], x_{k+1} gets -μ_k
    nt = op.nt
    nx = op.nx
    dLD_dX = jnp.zeros((nt, nx), dtype=mu_i.dtype)
    dLD_dX = dLD_dX.at[:-1].add(term_x)
    dLD_dX = dLD_dX.at[1:].add(-mu_i)

    return dLD_dX, dLD_dU


def gradient_aug_lagrangian_trajectory_constraints(
    constraints: contypes.GameConstraintGridMap,
    alstate: altypes.JointAugmentedLagrangianState,
    op: trajtypes.FixedStepPrimalDualTrajectory,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Gradient contribution from the *linear* auxiliary-constraint term in the augmented Lagrangian:
        L_C(X,U) = λᵀ C(X,U).

    This computes joint gradients w.r.t. the discretized trajectories X and U at the
    operating point `op`, using the constraint blocks in `constraints` and multipliers
    in `alstate`.

    Stacking / shapes
    -----------------
    Constraints are stacked in the canonical order:
      1) all inequality blocks (in order), expanded over each block's active_steps,
      2) then all equality blocks similarly.

    Therefore this function requires:
        alstate.lam_ineq.shape == (constraints.nc_ineq,)
        alstate.lam_eq.shape   == (constraints.nc_eq,)

    Returns
    -------
    dLC_dX : jnp.ndarray, shape (nt, nx)
        Gradient of λᵀC w.r.t. the joint state trajectory X.
    dLC_dU : jnp.ndarray, shape (nt-1, nu)
        Gradient of λᵀC w.r.t. the joint control trajectory U.

    Notes
    -----
    - This is the auxiliary (non-dynamics) constraint term only.
    - Terminal-only constraint blocks contribute only to dL_dX at k=nt-1.
    """
    # Build per-step constraint values/Jacobians once (canonical stacking order)
    ineq_lins, eq_lins = contypes.build_constraint_step_linearizations(constraints=constraints, op=op)

    # Validate multiplier shapes match the stacked constraint dimensions
    if alstate.lam_ineq.shape != (constraints.nc_ineq,):
        raise ValueError(
            f"alstate.lam_ineq must have shape ({constraints.nc_ineq},), got {alstate.lam_ineq.shape}"
        )
    if alstate.lam_eq.shape != (constraints.nc_eq,):
        raise ValueError(
            f"alstate.lam_eq must have shape ({constraints.nc_eq},), got {alstate.lam_eq.shape}"
        )

    # Accumulate Jᵀ λ into trajectory-shaped arrays
    dX_i, dU_i = contypes.accumulate_Jt_weighted_vector(
        lins=ineq_lins,
        w_flat=alstate.lam_ineq,
        nt=op.nt,
        nx=op.nx,
        nu=op.nu,
        dtype=alstate.lam_ineq.dtype,
    )
    dX_e, dU_e = contypes.accumulate_Jt_weighted_vector(
        lins=eq_lins,
        w_flat=alstate.lam_eq,
        nt=op.nt,
        nx=op.nx,
        nu=op.nu,
        dtype=alstate.lam_eq.dtype,
    )

    return dX_i + dX_e, dU_i + dU_e


def gradient_aug_lagrangian_trajectory_penalty(
    constraints: contypes.GameConstraintGridMap,
    alstate: altypes.JointAugmentedLagrangianState,
    op: trajtypes.FixedStepPrimalDualTrajectory,
    *,
    ineq_activation: Literal["altro", "none"] = "altro",
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Gradient contribution from the *quadratic penalty* auxiliary-constraint term:
        L_ρ(X,U) = 1/2 * C(X,U)ᵀ diag(ρ) C(X,U)

    where C(X,U) is the trajectory-stacked auxiliary constraint vector (excluding dynamics),
    and ρ are per-component penalty weights stored in `alstate` as (rho_ineq, rho_eq).

    Stacking / shapes
    -----------------
    Constraints are stacked in the canonical order:
      1) all inequality blocks (in order), expanded over each block's active_steps,
      2) then all equality blocks similarly.

    Therefore this function requires:
        alstate.rho_ineq.shape == (constraints.nc_ineq,)
        alstate.rho_eq.shape   == (constraints.nc_eq,)

    Returned shapes
    ---------------
    dLP_dX : jnp.ndarray, shape (nt, nx)
        Gradient of the penalty term w.r.t. the joint state trajectory X.
    dLP_dU : jnp.ndarray, shape (nt-1, nu)
        Gradient of the penalty term w.r.t. the joint control trajectory U.

    Inequality activation (optional)
    -------------------------------
    For inequality constraints, some AL implementations apply an "active rule" to avoid
    penalizing constraints that are safely satisfied. If `ineq_activation="altro"`,
    we apply Altro/ALGAMES' common rule componentwise:
        a = (c >= 0) OR (λ > 0)
        c_eff = a * c
    and treat `a` as constant via stop_gradient.

    If `ineq_activation="none"`, all inequality penalty components are always active
    (c_eff = c).

    Notes
    -----
    - This routine assumes the same stacking convention as
      `build_constraint_step_linearizations`, since it uses the slices stored in the
      resulting ConstraintStepLinearization objects.
    - Terminal-only blocks contribute only to dL_dX at k=nt-1.
    """
    if ineq_activation not in ("altro", "none"):
        raise ValueError("ineq_activation must be one of {'altro','none'}")

    # Build per-step constraint values/Jacobians once (canonical stacking order)
    ineq_lins, eq_lins = contypes.build_constraint_step_linearizations(constraints=constraints, op=op)

    # Validate penalty shapes match stacked constraint dimensions
    if alstate.rho_ineq.shape != (constraints.nc_ineq,):
        raise ValueError(
            f"alstate.rho_ineq must have shape ({constraints.nc_ineq},), got {alstate.rho_ineq.shape}"
        )
    if alstate.rho_eq.shape != (constraints.nc_eq,):
        raise ValueError(
            f"alstate.rho_eq must have shape ({constraints.nc_eq},), got {alstate.rho_eq.shape}"
        )

    # Build weight vectors w = rho ⊙ c_eff (componentwise) for each stack,
    # then accumulate Jᵀ w into trajectory-shaped arrays.
    w_ineq = jnp.zeros_like(alstate.rho_ineq)
    for li in ineq_lins:
        rho_step = alstate.rho_ineq[li.sl]
        c = li.c

        if ineq_activation == "altro":
            # Need lambda for activation rule; enforce shape once up-front if you want
            if alstate.lam_ineq.shape != (constraints.nc_ineq,):
                raise ValueError(
                    f"alstate.lam_ineq must have shape ({constraints.nc_ineq},) for ineq_activation='altro', "
                    f"got {alstate.lam_ineq.shape}"
                )
            lam_step = alstate.lam_ineq[li.sl]
            a = (c >= 0) | (lam_step > 0)
            a = jax.lax.stop_gradient(a.astype(c.dtype))
            c_eff = a * c
        else:
            c_eff = c

        w_ineq = w_ineq.at[li.sl].set(rho_step * c_eff)

    w_eq = jnp.zeros_like(alstate.rho_eq)
    for li in eq_lins:
        rho_step = alstate.rho_eq[li.sl]
        w_eq = w_eq.at[li.sl].set(rho_step * li.c)

    dX_i, dU_i = contypes.accumulate_Jt_weighted_vector(
        lins=ineq_lins,
        w_flat=w_ineq,
        nt=op.nt,
        nx=op.nx,
        nu=op.nu,
        dtype=w_ineq.dtype,
    )
    dX_e, dU_e = contypes.accumulate_Jt_weighted_vector(
        lins=eq_lins,
        w_flat=w_eq,
        nt=op.nt,
        nx=op.nx,
        nu=op.nu,
        dtype=w_eq.dtype,
    )

    return dX_i + dX_e, dU_i + dU_e

def compute_al_residual_struct_from_traj(
    nlgame: gametypes.NonlinearGameType2,
    op: trajtypes.FixedStepPrimalDualTrajectory,
    alstate: altypes.JointAugmentedLagrangianState,
    *,
    discretize_method: str, # = "rk2",
    ineq_activation: str, # = "altro",
) -> altypes.ALResidualStruct:
    """
    Compute the structured augmented-Lagrangian Newton residual for a game.

    Returns the block form of:
        G = [ ∇_{X[1:],U} L_1 ;
              ...
              ∇_{X[1:],U} L_N ;
              D(X,U) ]

    but stored as:
      - dLdX : (N,K,nx)  for stationarity w.r.t. X[1:]
      - dLdU : (N,K,nu)  for stationarity w.r.t. joint U
      - dyn_res : (K,nx) feasibility residuals
    """
    if nlgame.tg != op.tg:
        raise ValueError(f"TimeGrid mismatch: nlgame.tg={nlgame.tg} vs op.tg={op.tg}")

    # Stationarity gradients for all players (joint control coordinates)
    dL_dX_all, dL_dU_all = gradient_aug_lagrangian_trajectory(
        nlgame,
        op,
        alstate,
        discretize_method=discretize_method,
        ineq_activation=ineq_activation,
    )

    # Exclude x0 since it is not a decision variable
    dLdX = dL_dX_all[:, 1:, :]      # (N,K,nx)
    dLdU = dL_dU_all                # (N,K,nu)

    # Dynamics feasibility residual
    dyn_res = systypes.residual_discrete_dynamics_trajectory(
        nlgame.cs,
        op,
        method=discretize_method,
    )                               # (K,nx)

    return altypes.ALResidualStruct(dLdX=dLdX, dLdU=dLdU, dyn_res=dyn_res)


def pack_al_residual_1d(
    r: altypes.ALResidualStruct,
    u_splits: jnp.ndarray,
    *,
    return_layout: bool = False,
) -> jnp.ndarray | Tuple[jnp.ndarray, altypes.ALResidualPackLayout]:
    """
    Pack ALResidualStruct into a 1D residual vector for Newton solves.

    Packing order
    -------------
    For i = 0..N-1:
      1) vec(dLdX[i])                         shape (K*nx,)
      2) vec(dLdU[i][:, u_i_slice])           shape (K*nu_i,)   (local controls only)
    Then:
      3) vec(dyn_res)                         shape (K*nx,)

    This function also optionally returns a layout object that records where the
    stationarity block ends and the dynamics block begins, so downstream code does
    not need to assume dyn_res has size K*nx or that it sits at the end by convention.

    Parameters
    ----------
    r : ALResidualStruct
        Residual with fields:
          - dLdX : (N, K, nx)
          - dLdU : (N, K, nu)
          - dyn_res : (K, nx)   (may be empty if K==0)
    u_splits : jnp.ndarray, shape (N,)
        Player control partition sizes summing to nu (joint control dimension).
    return_layout : bool, default=False
        If True, return (g_flat, layout) where layout provides sta/dyn slices.

    Returns
    -------
    g_flat : jnp.ndarray, shape (ng,)
        Packed residual vector.
    layout : ALResidualPackLayout (only if return_layout=True)
        Slice bookkeeping for stationarity vs dynamics portions.
    """
    # --- shape sanity ---
    if r.dLdX.ndim != 3:
        raise ValueError(f"dLdX must be 3D (N,K,nx), got {r.dLdX.shape}")
    if r.dLdU.ndim != 3:
        raise ValueError(f"dLdU must be 3D (N,K,nu), got {r.dLdU.shape}")
    if r.dyn_res.ndim != 2:
        raise ValueError(f"dyn_res must be 2D (K,nx), got {r.dyn_res.shape}")

    N, K, nx = r.dLdX.shape
    N2, K2, nu = r.dLdU.shape
    if (N2, K2) != (N, K):
        raise ValueError(f"dLdU leading dims must match dLdX: {(N2,K2)} vs {(N,K)}")

    # dyn_res can be empty; if non-empty, must match (K,nx)
    if r.dyn_res.size != 0 and r.dyn_res.shape != (K, nx):
        raise ValueError(f"dyn_res must have shape (K,nx)=({K},{nx}), got {r.dyn_res.shape}")

    splits = np.asarray(u_splits, dtype=int)
    if splits.shape != (N,):
        raise ValueError(f"u_splits must have shape (N,)={(N,)}, got {splits.shape}")
    if splits.sum() != nu:
        raise ValueError(f"u_splits must sum to nu={nu}, got {splits.sum()}")

    starts = np.cumsum(np.concatenate(([0], splits[:-1])))

    parts = []
    for i in range(N):
        parts.append(jnp.ravel(r.dLdX[i]))  # (K*nx,)
        sl = slice(int(starts[i]), int(starts[i] + splits[i]))
        parts.append(jnp.ravel(r.dLdU[i, :, sl]))  # (K*nu_i,)

    sta_len = int(sum(p.size for p in parts))
    parts.append(jnp.ravel(r.dyn_res))  # (dyn_len,)

    g_flat = jnp.concatenate(parts) if parts else jnp.zeros((0,), dtype=r.dLdX.dtype)

    if not return_layout:
        return g_flat

    dyn_len = int(r.dyn_res.size)
    layout = altypes.ALResidualPackLayout(
        sta_slice=slice(0, sta_len),
        dyn_slice=slice(sta_len, sta_len + dyn_len),
    )
    return g_flat, layout

def compute_al_residual_flat_from_decision_vars(
    nlgame: gametypes.NonlinearGameType2,
    z: jnp.ndarray,
    template_op: trajtypes.FixedStepPrimalDualTrajectory,
    alstate: altypes.JointAugmentedLagrangianState,
    *,
    discretize_method: str = "rk2",
    ineq_activation: str = "altro",
) -> jnp.ndarray:
    """
    Compute the packed Newton residual G(z) as a flat 1D vector.

    Parameters
    ----------
    nlgame : NonlinearGameType2
        Must provide u_splits and fields used by compute_al_residual_struct.
    z : jnp.ndarray, shape (nz,)
        Packed decision variables in the canonical order:
            z = [ vec(xs[1:]); vec(us); vec(ls) ].
    template_op : FixedStepPrimalDualTrajectory
        Provides tg and x0 (and shapes) needed to unpack z.
    alstate : JointAugmentedLagrangianState
        Augmented Lagrangian multipliers/penalties for auxiliary constraints.
    discretize_method, ineq_activation : str
        Passed through to the residual builder.

    Returns
    -------
    G_flat : jnp.ndarray, shape (ng,)
        Packed augmented lagrangian residual vector consistent with pack_al_residual ordering.
    """
    log_debug = logger.isEnabledFor(logging.DEBUG)
    call_id = _next_debug_call_id("compute_al_residual_flat_from_decision_vars")
    emit = log_debug and _should_emit_sampled_debug(call_id)
    if emit:
        t0 = time.perf_counter()
        logger.debug(
            "AL trace residual_flat start call=%d z_shape=%s x_shape=%s u_shape=%s l_shape=%s",
            call_id,
            _debug_shape(z),
            _debug_shape(template_op.xs),
            _debug_shape(template_op.us),
            _debug_shape(template_op.ls),
        )

    op = unpack_decision_vars(z, template_op, check_length=True)
    r = compute_al_residual_struct_from_traj(
        nlgame,
        op,
        alstate,
        discretize_method=discretize_method,
        ineq_activation=ineq_activation,
    )
    g_flat = pack_al_residual_1d(r, u_splits=nlgame.u_splits)

    if emit:
        dt = time.perf_counter() - t0
        logger.debug(
            "AL trace residual_flat done call=%d dt=%.3fs residual_shape=%s",
            call_id,
            dt,
            _debug_shape(g_flat),
        )

    return g_flat


def jacobian_al_residual_flat_autodiff(
    nlgame: gametypes.NonlinearGameType2,
    z: jnp.ndarray,
    template_op: trajtypes.FixedStepPrimalDualTrajectory,
    alstate: altypes.JointAugmentedLagrangianState,
    *,
    discretize_method: str = "rk2",
    ineq_activation: str = "altro",
    mode: Literal["jacfwd", "jacrev"] = "jacfwd",
) -> jnp.ndarray:
    """
    Compute the Jacobian H = ∂G/∂z of the packed (1D) residual wrt decision vars (z) using JAX autodiff.

    Intended use
    ------------
    - Validation / golden tests on small problems.
    - Not expected to scale well to large horizons.

    Returns
    -------
    H : jnp.ndarray, shape (ng, nz)
    """
    z = jnp.asarray(z)
    log_debug = logger.isEnabledFor(logging.DEBUG)
    call_id = _next_debug_call_id("jacobian_al_residual_flat_autodiff")
    emit = log_debug and _should_emit_sampled_debug(call_id, first_n=10, every=10)
    if emit:
        t0 = time.perf_counter()
        logger.debug(
            "AL trace jacobian start call=%d mode=%s z_shape=%s",
            call_id,
            mode,
            _debug_shape(z),
        )

    def G_of_z(z_):
        return compute_al_residual_flat_from_decision_vars(
            nlgame,
            z_,
            template_op,
            alstate,
            discretize_method=discretize_method,
            ineq_activation=ineq_activation,
        )

    if mode == "jacfwd":
        H = jax.jacfwd(G_of_z)(z)
    elif mode == "jacrev":
        H = jax.jacrev(G_of_z)(z)
    else:
        raise ValueError("mode must be 'jacfwd' or 'jacrev'")

    if emit:
        dt = time.perf_counter() - t0
        logger.debug(
            "AL trace jacobian done call=%d dt=%.3fs H_shape=%s",
            call_id,
            dt,
            _debug_shape(H),
        )
    return H

def _validate_reg_params(
    *,
    reg0: float,
    reg1_min: float,
    reg_increase: float,
    reg_max: float,
    reg_max_iters: int,
) -> None:
    if not isinstance(reg_max_iters, int) or reg_max_iters <= 0:
        raise ValueError(f"reg_max_iters must be a positive int, got {reg_max_iters}")
    for name, v in [("reg0", reg0), ("reg1_min", reg1_min), ("reg_increase", reg_increase), ("reg_max", reg_max)]:
        if not isinstance(v, (float, int)):
            raise TypeError(f"{name} must be float-like, got {type(v)}")
        if not math.isfinite(float(v)):
            raise ValueError(f"{name} must be finite, got {v}")

    if float(reg0) < 0.0:
        raise ValueError(f"reg0 must be >= 0, got {reg0}")
    if float(reg1_min) < 0.0:
        raise ValueError(f"reg1_min must be >= 0, got {reg1_min}")
    # allow reg_increase==1 in theory (then reg won't grow), but it's almost always a bug
    if float(reg_increase) <= 1.0:
        raise ValueError(f"reg_increase must be > 1 to guarantee growth, got {reg_increase}")
    if float(reg_max) <= 0.0:
        raise ValueError(f"reg_max must be > 0, got {reg_max}")
    if float(reg_max) < float(reg1_min):
        raise ValueError(f"reg_max must be >= reg1_min, got reg_max={reg_max}, reg1_min={reg1_min}")

def solve_newton_system_tikhonov(
    H: jnp.ndarray,
    g: jnp.ndarray,
    *,
    reg0: float, # = 0.0,
    reg1_min: float, # = 1e-12,
    reg_increase: float, # = 10.0,
    reg_max: float, # = 1e8,
    max_iters: int, # = 64,
) -> altypes.RegularizedSolveResult:
    """
    Solve the Newton linear system with diagonal (Tikhonov) regularization.

    We want a Newton step dz that solves:
        H dz = -g
    where H is the Jacobian of the residual and g is the residual.

    If H is singular or ill-conditioned, the solve may fail or produce unstable steps.
    We instead attempt a *regularized* solve:
        (H + reg I) dz = -g

    and increase reg geometrically until the solve succeeds or reg exceeds reg_max.

    Parameters
    ----------
    H : jnp.ndarray, shape (nz, nz)
        Jacobian matrix of the residual (Newton system matrix).
    g : jnp.ndarray, shape (nz,)
        Residual vector. The right-hand side is -g.
    reg0 : float
        Initial regularization parameter reg >= 0.
    reg1_min : float
        Minimum regularization parameter after first failure > 0
        enables reg0=0 while ensuring non-zero reg after first failure
    reg_increase : float
        Multiplicative factor applied to reg after each failed attempt.
        Example: reg *= reg_increase.
    reg_max : float
        Maximum regularization allowed. If exceeded, the function returns ok=False.
    max_iters : int
        Maximum number of attempts to solve linear system

    Returns
    -------
    RegularizedSolveResult
        dz is the step if ok=True, else dz=None.

    Notes
    -----
    - This is a simple, robust “damped Newton” primitive.
    - It does *not* guarantee the step will reduce ||g|| (that’s handled by line search).
    - In large-scale code you’d typically use sparse/structured solves; this is a dense
      prototype intended for correctness and small problems.
    - NOTE: Not jit-safe due to control flow
    """
    _validate_reg_params(reg0=reg0, reg1_min=reg1_min, reg_increase=reg_increase, reg_max=reg_max, reg_max_iters=max_iters)
    nz = int(g.shape[0])
    I = jnp.eye(nz, dtype=H.dtype)

    reg = float(reg0)

    for _ in range(max_iters):
        try:
            dz = jnp.linalg.solve(H + reg * I, -g)
            if not bool(jnp.all(jnp.isfinite(dz))):
                raise FloatingPointError("non-finite dz")
            return altypes.RegularizedSolveResult(dz=dz, reg=reg, ok=True)
        except Exception:
            # escape reg==0 and grow geometrically
            reg = max(reg1_min, reg * reg_increase)
            if reg > reg_max:
                return altypes.RegularizedSolveResult(dz=None, reg=reg, ok=False)

    # safety net if max_iters triggers before reg_max
    return altypes.RegularizedSolveResult(dz=None, reg=reg, ok=False)


def residual_norm(g: jnp.ndarray, kind: ResidualNormKind) -> float:
    """
    Compute a scalar merit norm for a residual, treating `g` as a flat vector.

    Parameters
    ----------
    g : jnp.ndarray
        Residual array of any shape. This function treats it as a vector by
        flattening it to shape (n,) where n = g.size.
    kind : {"l1", "l2", "l1_mean", "l2_rms"}
        - "l1":      ||g||_1  (sum(abs(g)))
        - "l2":      ||g||_2  (sqrt(sum(g^2)))
        - "l1_mean": ||g||_1 / n
        - "l2_rms":  ||g||_2 / sqrt(n)
        - "linf":    ||g||_∞  (max(abs(g)))

    Returns
    -------
    float
        The requested norm value.

    Notes
    -----
    - Flattening avoids matrix-norm semantics when `g` is not 1D.
    - If g.size == 0, returns 0.0.
    """
    g = jnp.asarray(g)
    n = int(g.size)
    if n == 0:
        return 0.0

    gv = jnp.ravel(g)  # ensure vector norm semantics

    if kind == "l1":
        return float(jnp.linalg.norm(gv, ord=1))
    if kind == "l2":
        return float(jnp.linalg.norm(gv, ord=2))
    if kind == "l1_mean":
        return float(jnp.linalg.norm(gv, ord=1) / n)
    if kind == "l2_rms":
        return float(jnp.linalg.norm(gv, ord=2) / jnp.sqrt(n))
    if kind == "linf":
        return float(jnp.max(jnp.abs(gv)))

    raise ValueError(f"Unknown residual norm kind: {kind}")


def optimality_violation_inf(
    nlgame: gametypes.NonlinearGameType2,
    op: trajtypes.FixedStepPrimalDualTrajectory,
    alstate: altypes.JointAugmentedLagrangianState,
    *,
    discretize_method: str,
    ineq_activation: str,
) -> float:
    """
    Compute ALGAMES-style optimality/stationarity violation (L-infinity norm).

    This corresponds to an L-infinity norm over the stationarity residual components
    (i.e., the augmented-Lagrangian gradients w.r.t. decision variables), i.e. the per-player 
    gradients w.r.t. X[1:] and *local* controls U_i, excluding dynamics feasibility residuals.

    Computes:
        opt_vio = max( max(|dLdX|), max(|dLdU_local|) )

    where dLdU_local refers to each player's local control block within the joint-control
    gradient (consistent with packing conventions used by Newton solves).

    Parameters
    ----------
    nlgame : NonlinearGameType2
        Game definition (dimensions, costs, constraints).
    op : FixedStepPrimalDualTrajectory
        Current primal-dual trajectory (X, U, μ).
    alstate : JointAugmentedLagrangianState
        Current auxiliary AL parameters (λ, ρ).
    discretize_method : str
        Discretization method used when building the residual.
    ineq_activation : str
        Inequality activation rule used in penalty terms.

    Returns
    -------
    float
        The maximum absolute stationarity residual component (L-infinity metric).

    Notes
    -----
    - Uses the structured residual directly (does not rely on packed layout).
    - Ignores dynamics feasibility residuals by construction.
    """
    res = compute_al_residual_struct_from_traj(
        nlgame,
        op,
        alstate,
        discretize_method=discretize_method,
        ineq_activation=ineq_activation,
    )

    # dLdX: (N,K,nx) includes only X[1:] gradients already (by your builder)
    vio_x = 0.0 if res.dLdX.size == 0 else float(jnp.max(jnp.abs(res.dLdX)))

    # dLdU is joint; restrict to each player's local control slice
    vio_u = 0.0
    if res.dLdU.size != 0:
        splits = np.asarray(nlgame.u_splits, dtype=int)
        starts = np.cumsum(np.concatenate(([0], splits[:-1])))
        for i in range(int(res.dLdU.shape[0])):  # N
            sl = slice(int(starts[i]), int(starts[i] + splits[i]))
            block = res.dLdU[i, :, sl]
            if block.size:
                vio_u = max(vio_u, float(jnp.max(jnp.abs(block))))

    return max(vio_x, vio_u)


def _validate_linesearch_params(
    *,
    alpha0: float,
    tau: float,
    beta: float,
    max_iters: int,
    norm: ResidualNormKind,
) -> None:
    
    if not isinstance(alpha0, (float, int)) or not math.isfinite(float(alpha0)) or float(alpha0) <= 0.0:
        raise ValueError(f"alpha0 must be finite and > 0, got {alpha0}")
    if not isinstance(tau, (float, int)) or not math.isfinite(float(tau)) or not (0.0 < float(tau) < 1.0):
        raise ValueError(f"tau must be in (0,1), got {tau}")
    if not isinstance(beta, (float, int)) or not math.isfinite(float(beta)) or not (0.0 < float(beta) < 0.5):
        raise ValueError(f"beta must be in (0, 1/2), got {beta}")
    if not isinstance(max_iters, int) or max_iters <= 0:
        raise ValueError(f"max_iters must be a positive int, got {max_iters}")
    if norm not in get_args(ResidualNormKind):
        raise ValueError(f"norm must be one of {get_args(ResidualNormKind)}, got {norm!r}")
    

def backtracking_linesearch_armijo(
    G_of_z: Callable[[jnp.ndarray], jnp.ndarray],
    z0: jnp.ndarray,
    dz: jnp.ndarray,
    g0: jnp.ndarray,
    *,
    alpha0: float,
    tau: float,
    beta: float,
    max_iters: int,
    normkind: ResidualNormKind,
) -> altypes.LineSearchResult:
    """
    Backtracking line search on the residual norm, matching Le Cleac'h et al. Alg. 1 / ALGames.jl.

    We seek a step size α that satisfies an Armijo-style sufficient decrease condition on the
    residual norm merit function:

        ||G(z0 + α dz)||  <=  (1 - α * beta) * ||G(z0)||

    where `beta ∈ (0, 1/2)` and the backtracking schedule is:

        α <- tau * α,   with tau ∈ (0, 1)

    Notes
    -----
    - This differs from a fixed accept_ratio rule (||G_trial|| <= accept_ratio * ||G0||) in that
      the required decrease becomes weaker as α shrinks, which prevents pathological rejection
      near convergence due to floating-point noise.
    - The paper uses the 1-norm; ALGames.jl uses an L1 norm scaled by vector length. Here we
      support "l1" and "l2". Use "l1" to match the reference implementations most closely.

    Parameters
    ----------
    G_of_z : Callable[[jnp.ndarray], jnp.ndarray]
        Function returning the packed residual vector G(z).
    z0 : jnp.ndarray, shape (nz,)
        Current decision vector.
    dz : jnp.ndarray, shape (nz,)
        Proposed step direction (e.g. Newton step).
    g0 : jnp.ndarray, shape (ng,)
        Residual vector at z0 (passed in to avoid recomputation).
    alpha0 : float
        Initial step size to try (typically 1.0).
    tau : float
        Backtracking shrink factor in (0,1). Each rejection sets α <- tau * α.
        (This is τ in Le Cleac'h Alg. 1; corresponds to opts.α_decrease in ALGames.jl.)
    beta : float
        Armijo slope parameter in (0, 1/2). (This is β in Le Cleac'h Alg. 1; corresponds
        to opts.β in ALGames.jl.)
    max_iters : int
        Maximum number of trial evaluations.
    norm : {"l1","l2","l1_mean","l2_rms"}
        Norm used for the merit function. "l1_mean" best matches the reference algorithms.

    Returns
    -------
    LineSearchResult
        accepted : bool
            Whether an α satisfying the Armijo condition was found.
        alpha : float
            Accepted α if accepted, else 0.0.
        g_norm_trial : float
            Norm at accepted point if accepted, else last tried.
        ls_iters : int
            Number of trial evaluations performed.
        best_alpha : float
            α that produced the smallest norm among tried steps (even if not accepted).
        best_norm : float
            Smallest norm encountered.
    """
    # Basic validation (keep runtime cheap; this is not intended to be jitted)
    _validate_linesearch_params(
        alpha0=alpha0, tau=tau, beta=beta, max_iters=max_iters, norm=normkind,
    )

    g_norm0 = residual_norm(g0,kind=normkind)

    alpha = float(alpha0)
    best_alpha = 0.0
    best_norm = g_norm0
    accepted = False
    g_norm_trial = g_norm0
    ls_iters = 0

    for j in range(max_iters):
        ls_iters = j + 1
        z_trial = z0 + alpha * dz
        g_trial = G_of_z(z_trial)
        g_norm_trial = residual_norm(g_trial, kind=normkind)

        # Armijo condition: ||G(z+αdz)|| <= (1 - αβ) ||G(z)||
        rhs = max(0.0, (1.0 - alpha*beta)) * g_norm0    # prevent negative right-hand side
        if g_norm_trial <= rhs:
            accepted = True
            best_alpha = alpha
            best_norm = g_norm_trial
            break

        if g_norm_trial < best_norm:
            best_norm = g_norm_trial
            best_alpha = alpha

        alpha *= float(tau)

    return altypes.LineSearchResult(
        accepted=accepted,
        alpha=best_alpha if accepted else 0.0,
        g_norm_trial=best_norm if accepted else g_norm_trial,
        ls_iters=ls_iters,
        best_alpha=best_alpha,
        best_norm=best_norm,
    )

def backtracking_linesearch_fixedratio(
    G_of_z: Callable[[jnp.ndarray], jnp.ndarray],
    z0: jnp.ndarray,
    dz: jnp.ndarray,
    g0: jnp.ndarray,
    *,
    alpha0: float, # = 1.0,
    beta: float, # = 0.5,
    max_iters: int, # = 20,
    accept_ratio: float, # = 0.99,
) -> altypes.LineSearchResult:
    """
    Backtracking line search using residual norm as a merit function.

    We seek a step size alpha such that the residual norm decreases sufficiently:
        ||G(z0 + alpha * dz)||_2 <= accept_ratio * ||G(z0)||_2.

    The algorithm tries:
        alpha = alpha0, alpha0*beta, alpha0*beta^2, ..., up to max_iters trials.

    Parameters
    ----------
    G_of_z : Callable[[jnp.ndarray], jnp.ndarray]
        Function returning the packed residual vector G(z).
    z0 : jnp.ndarray, shape (nz,)
        Current decision vector.
    dz : jnp.ndarray, shape (nz,)
        Proposed step direction (e.g., Newton step).
    g0 : jnp.ndarray, shape (ng,)
        Residual vector at z0, i.e., g0 = G_of_z(z0). Passed in to avoid recomputation.
    alpha0 : float
        Initial step size to try (typically 1.0).
    beta : float
        Backtracking shrink factor in (0,1). Common values: 0.5, 0.8.
    max_iters : int
        Maximum number of trial evaluations.
    accept_ratio : float
        Sufficient decrease threshold. Accept if trial norm is <= accept_ratio * current norm.
        Example: 0.99 requires at least 1% reduction.

    Returns
    -------
    LineSearchResult
        Contains whether an acceptable alpha was found and diagnostic information.

    Notes
    -----
    - This line search is designed for root-finding merit functions (||G||), not for
      minimizing a scalar objective.
    - If accepted=False, the caller can either reject the step or choose to accept
      best_alpha as a fallback (more aggressive behavior).
    """

    assert alpha0 > 0
    assert beta > 0
    assert beta < 1.0
    assert max_iters > 0
    assert accept_ratio < 1.0
    assert accept_ratio > 0.0

    g_norm0 = float(jnp.linalg.norm(g0))

    alpha = float(alpha0)
    best_alpha = 0.0
    best_norm = g_norm0
    accepted = False
    g_norm_trial = g_norm0
    ls_iters = 0

    for j in range(max_iters):
        ls_iters = j + 1
        z_trial = z0 + alpha * dz
        g_trial = G_of_z(z_trial)
        g_norm_trial = float(jnp.linalg.norm(g_trial))

        if g_norm_trial <= accept_ratio * g_norm0:
            accepted = True
            best_alpha = alpha
            best_norm = g_norm_trial
            break

        if g_norm_trial < best_norm:
            best_norm = g_norm_trial
            best_alpha = alpha

        alpha *= beta

    return altypes.LineSearchResult(
        accepted=accepted,
        alpha=best_alpha if accepted else 0.0,
        g_norm_trial=best_norm if accepted else g_norm_trial,
        ls_iters=ls_iters,
        best_alpha=best_alpha,
        best_norm=best_norm,
    )


def _validate_newton_step_params(
    *,
    step_rtol: float,
    step_atol: float,
) -> None:
    for name, v in [("step_rtol", step_rtol), ("step_atol", step_atol)]:
        if not isinstance(v, (float, int)):
            raise TypeError(f"{name} must be float-like, got {type(v)}")
        if not math.isfinite(float(v)):
            raise ValueError(f"{name} must be finite, got {v}")
        if float(v) <= 0.0:
            raise ValueError(f"{name} must be > 0, got {v}")


def newton_step_autodiff(
    nlgame: gametypes.NonlinearGameType2,
    op: trajtypes.FixedStepPrimalDualTrajectory,
    alstate: altypes.JointAugmentedLagrangianState,
    *,
    step_rtol: float, # = 1e-7,
    step_atol: float, # = 1e-8,
    discretize_method: str, # = "rk2",
    ineq_activation: str, # = "altro",
    reg0: float, # = 0.0,
    reg1_min: float, # = 1e-12,
    reg_increase: float, # = 10.0,
    reg_max: float, # = 1e8,
    reg_max_iters: int, # = 64,
    ls_alpha0: float, # = 1.0,
    ls_beta: float, # = 0.25,
    ls_tau: float, # = 0.5,
    ls_max_iters: int, # = 20,
    normkind: ResidualNormKind,
) -> Tuple[trajtypes.FixedStepPrimalDualTrajectory, altypes.NewtonStepDiag]:
    """
    Perform one Newton step on the packed augmented-Lagrangian residual system G(z)=0.

    Take one Newton step for the augmented-Lagrangian residual system G(z)=0.

    This routine forms the packed decision vector z from the current primal-dual trajectory,
    computes the residual g = G(z) and its Jacobian H = ∂G/∂z via autodiff, solves the
    damped linear system for a Newton direction, then applies an Armijo-style backtracking
    line search on the residual norm to select a step size.

    High-level procedure
    --------------------
    1) Pack decision variables: z0 = pack_decision_vars(op)
    2) Evaluate residual:        g0 = G(z0)
    3) Autodiff Jacobian:        H0 = dG/dz |_{z0}
    4) Solve for direction:      (H0 + reg I) dz = -g0  (Tikhonov regularization)
    5) Backtracking line search: choose alpha to reduce ||G(z0 + alpha dz)||_2
    6) If accepted: update z and unpack op_new
    
    Parameters
    ----------
    nlgame : NonlinearGameType2
        Game definition (dynamics, costs, auxiliary constraints, player partitioning).
    op : FixedStepPrimalDualTrajectory
        Current primal-dual operating point (X, U, μ) about which the Newton step is computed.
    alstate : JointAugmentedLagrangianState
        Auxiliary-constraint augmented-Lagrangian state (λ, ρ) held fixed during this step.

    step_rtol, step_atol : float
        If ||dz||₂ < step_rtol ||z0||₂ + step_atol, treat the direction as a no-op (converged/stalled) and return
        without line search.

    discretize_method : str
        Integration/discretization method used when evaluating discrete dynamics inside G.
        Examples: "euler", "rk2", "rk3", "rk4".
    ineq_activation : str
        Inequality activation rule used in augmented-Lagrangian penalty terms (e.g. "altro", "none").

    reg0 : float
        Initial Tikhonov regularization coefficient for the linear solve.
    reg1_min : float
        Minimum regularization to use after the first solve failure (useful when reg0=0).
    reg_increase : float
        Multiplicative growth factor for regularization upon solve failure.
    reg_max : float
        Maximum allowed regularization before declaring the linear solve failed.
    reg_max_iters : int
        Maximum number of regularization escalation attempts.

    ls_alpha0 : float
        Initial line-search step size α (trial point z_trial = z + α dz).
    ls_beta : float
        Armijo slope parameter β in (0, 1/2) for the acceptance test:
            ||G(z + α dz)|| <= (1 - α β) ||G(z)||.
    ls_tau : float
        Backtracking shrink factor τ in (0, 1): on rejection α <- τ α.
    ls_max_iters : int
        Maximum number of line-search trial evaluations.
    ls_norm : {"l1", "l2"}
        Norm used for the residual merit function during line search.
        - "l1" uses ||g||₁ / len(g) (matches ALGames.jl convention),
        - "l2" uses ||g||₂.


    Returns
    -------
    op_new : FixedStepPrimalDualTrajectory
        Updated primal-dual trajectory if the line search accepts a step; otherwise typically
        returns the input `op` unchanged.
    diag : NewtonStepDiag
        Diagnostics for this Newton step (acceptance, step size, residual norms, regularization,
        line-search iterations, etc.).

    Notes
    -----
    - This implementation uses autodiff Jacobians and dense linear algebra and is intended for
      correctness/validation on small problems.
    - A production implementation typically exploits block structure/sparsity and uses specialized
      linear solves.
    """
    debug_enabled = logger.isEnabledFor(logging.DEBUG)
    step_call_id = _next_debug_call_id("newton_step_autodiff")
    step_t0 = time.perf_counter() if debug_enabled else 0.0

    # --- parameter validation (cheap; this function is not intended to be jitted) ---
    _validate_reg_params(
        reg0=reg0, reg1_min=reg1_min, reg_increase=reg_increase, reg_max=reg_max, reg_max_iters=reg_max_iters
    )
    _validate_linesearch_params(
        alpha0=ls_alpha0, tau=ls_tau, beta=ls_beta, max_iters=ls_max_iters, norm=normkind,
    )
    _validate_newton_step_params(
        step_rtol=step_rtol, step_atol=step_atol
    )

    # Ensure time grids align (prevents subtle bugs)
    if nlgame.tg != op.tg:
        raise ValueError(f"TimeGrid mismatch: nlgame.tg={nlgame.tg} vs op.tg={op.tg}")
    
    # --- augmented lagrangian residual as function of decision variables packed as 1D vector ---
    z0 = pack_decision_vars_1d(op)
    z0 = jnp.asarray(z0)
    if debug_enabled:
        logger.debug(
            "Newton step start call=%d z_shape=%s x_shape=%s u_shape=%s l_shape=%s",
            step_call_id,
            _debug_shape(z0),
            _debug_shape(op.xs),
            _debug_shape(op.us),
            _debug_shape(op.ls),
        )

    def G_of_z(z: jnp.ndarray) -> jnp.ndarray:
        return compute_al_residual_flat_from_decision_vars(
            nlgame,
            z,
            op,
            alstate,
            discretize_method=discretize_method,
            ineq_activation=ineq_activation,
        )

    g0 = G_of_z(z0)
    # Guard against NaN/Inf residuals early (otherwise line search can behave pathologically)
    if not bool(jnp.all(jnp.isfinite(g0))):
        diag = altypes.NewtonStepDiag(
            accepted=False, alpha=0.0,
            g_norm0=float("nan"), g_norm_trial=float("nan"),
            step_norm=0.0, reg=float(reg0),
            ls_iters=0, solve_ok=False
        )
        if debug_enabled:
            logger.debug(
                "Newton step abort call=%d reason=nonfinite_g0 total_dt=%.3fs",
                step_call_id,
                time.perf_counter() - step_t0,
            )
        return op, diag

    g_norm0 = residual_norm(g0, kind=normkind)
    if not math.isfinite(g_norm0):
        diag = altypes.NewtonStepDiag(
            accepted=False, alpha=0.0,
            g_norm0=float("nan"), g_norm_trial=float("nan"),
            step_norm=0.0, reg=float(reg0),
            ls_iters=0, solve_ok=False
        )
        if debug_enabled:
            logger.debug(
                "Newton step abort call=%d reason=nonfinite_gnorm total_dt=%.3fs",
                step_call_id,
                time.perf_counter() - step_t0,
            )
        return op, diag


    # --- jacobian of augmented lagrangian residual used for root finding ---
    H0 = jacobian_al_residual_flat_autodiff(
        nlgame,
        z0,
        op,
        alstate,
        discretize_method=discretize_method,
        ineq_activation=ineq_activation,
        mode="jacfwd",
    )

    # --- solve for step direction of decision vars for finding root of AL residual ---
    sol = solve_newton_system_tikhonov(H0, g0, 
        reg0=reg0,
        reg1_min=reg1_min,
        reg_increase=reg_increase,
        reg_max=reg_max,
        max_iters=reg_max_iters)
    
    # immediately return non-accepted newton step if solution of step direction failed outright
    if (not sol.ok) or (sol.dz is None):
        diag = altypes.NewtonStepDiag(
            accepted=False, alpha=0.0,
            g_norm0=g_norm0, g_norm_trial=g_norm0,
            step_norm=0.0, reg=sol.reg,
            ls_iters=0, solve_ok=False
        )
        if debug_enabled:
            logger.debug(
                "Newton step reject call=%d reason=linear_solve_failed g_norm0=%.6g reg=%.6g total_dt=%.3fs",
                step_call_id,
                g_norm0,
                sol.reg,
                time.perf_counter() - step_t0,
            )
        return op, diag

    # immediately return accepted newton step if step magnitude is within step size tolerance
    # NOTE: this hard-codes a different norm than the aug lagrangian jacobian residual 
    # (i.e. "residual") norm because they need not be the same. However, it is a 
    # slight "abuse of notation" to be calling the "residual_norm" function
    # when z and dz relate to the decision variables, not the residual vector
    dz = sol.dz
    step_norm = residual_norm(dz, kind="l2")
    z0_norm = residual_norm(z0, kind="l2")
    if step_norm < z0_norm * step_rtol + step_atol:
        diag = altypes.NewtonStepDiag(
            accepted=True, alpha=0.0,
            g_norm0=g_norm0, g_norm_trial=g_norm0,
            step_norm=step_norm, reg=sol.reg,
            ls_iters=0, solve_ok=True
        )
        if debug_enabled:
            logger.debug(
                "Newton step stall call=%d g_norm0=%.6g step_norm=%.6g reg=%.6g total_dt=%.3fs",
                step_call_id,
                g_norm0,
                step_norm,
                sol.reg,
                time.perf_counter() - step_t0,
            )
        return op, diag

    # --- backtrack newton step magnitude along solved direction ---
    # prevents stepping too far from the approximately linear region around current decision vars
    # while also ensuring that AL residual (G) magnitude is decreasing (i.e. moving toward zero)
    def G_of_z_safe(z: jnp.ndarray) -> jnp.ndarray:
        g = G_of_z(z)
        # If non-finite, return a vector of +inf to force rejection by norm comparisons.
        # (same shape, avoids NaN propagation in norms)
        return jnp.where(jnp.isfinite(g), g, jnp.inf)

    ls = backtracking_linesearch_armijo(
        G_of_z_safe, z0, dz, g0,
        alpha0 = ls_alpha0,
        tau = ls_tau,
        beta = ls_beta,
        max_iters = ls_max_iters,
        normkind = normkind
    )

    if not ls.accepted:
        diag = altypes.NewtonStepDiag(
            accepted=False, alpha=ls.best_alpha,
            g_norm0=g_norm0, g_norm_trial=ls.best_norm,
            step_norm=step_norm, reg=sol.reg,
            ls_iters=ls.ls_iters, solve_ok=True
        )
        if debug_enabled:
            logger.debug(
                "Newton step reject call=%d reason=line_search g_norm0=%.6g best_norm=%.6g alpha=%.6g step_norm=%.6g reg=%.6g ls_iters=%d total_dt=%.3fs",
                step_call_id,
                g_norm0,
                ls.best_norm,
                ls.best_alpha,
                step_norm,
                sol.reg,
                ls.ls_iters,
                time.perf_counter() - step_t0,
            )
        return op, diag

    # --- update decision variables based on newton step and unpack them to a new prime-dual trajectory ---
    z_new = z0 + ls.alpha * dz
    op_new = unpack_decision_vars(z_new, op, check_length=True)

    diag = altypes.NewtonStepDiag(
        accepted=True, alpha=ls.alpha,
        g_norm0=g_norm0, g_norm_trial=ls.g_norm_trial,
        step_norm=step_norm, reg=sol.reg,
        ls_iters=ls.ls_iters, solve_ok=True
    )
    if debug_enabled:
        logger.debug(
            "Newton step accept call=%d g_norm0=%.6g g_trial=%.6g alpha=%.6g step_norm=%.6g reg=%.6g ls_iters=%d total_dt=%.3fs",
            step_call_id,
            g_norm0,
            ls.g_norm_trial,
            ls.alpha,
            step_norm,
            sol.reg,
            ls.ls_iters,
            time.perf_counter() - step_t0,
        )
    return op_new, diag


def newton_solve_stationarity_autodiff(
    nlgame: gametypes.NonlinearGameType2,
    op0: trajtypes.FixedStepPrimalDualTrajectory,
    alstate: altypes.JointAugmentedLagrangianState,
    *,
    discretize_method: str,
    ineq_activation: str,
    # stationarity termination
    opt_tol: float,  # = 1e-3,
    dyn_tol: float,  # = 1e-3,
    max_iters: int,  # = 25,
    max_rejects: int,  # = 5,
    # forwarded to newton_step_autodiff
    step_rtol: float,  # = 1e-7,
    step_atol: float,  # = 1e-8,
    reg0: float,  # = 0.0,
    reg1_min: float,  # = 1e-12,
    reg_increase: float,  # = 10.0,
    reg_max: float,  # = 1e8,
    reg_max_iters: int,  # = 64,
    ls_alpha0: float,  # = 1.0,
    ls_tau: float,  # = 0.5,
    ls_beta: float,  # = 0.25,
    ls_max_iters: int,  # = 20,
    normkind: ResidualNormKind,  # = "l1_mean",
    # bookkeeping
    return_last_accepted: bool = True,
) -> Tuple[trajtypes.FixedStepPrimalDualTrajectory, altypes.StationarityNewtonDiag]:
    """
    Newton-like inner solve for the AL inner loop that targets *stationarity + dynamics feasibility*,
    in the spirit of ALGames.jl.

    The Newton step direction is computed from the packed residual system G(z)=0 (via autodiff),
    but termination is based on two L∞ metrics evaluated at the current trajectory:
        opt_vio_inf = optimality_violation_inf(...)  <= opt_tol
        dyn_vio_inf = max(abs(D(X,U)))               <= dyn_tol

    where:
      - opt_vio_inf measures stationarity/optimality of the augmented Lagrangian w.r.t. the
        decision variables (X[1:], U_i local slices), excluding dynamics residual components.
      - dyn_vio_inf measures feasibility of the discrete dynamics constraints.

    A merit norm of the *full* packed residual G is still computed each iteration and is used
    by the Armijo line search inside `newton_step_autodiff`.

    Evolution from earlier inner solves
    -----------------------------------
    This function supersedes `newton_solve_autodiff`, which terminated on a relative/absolute
    norm of the full packed residual `G(z)`. That residual stacks stationarity terms and
    dynamics feasibility terms together, so a single scalar norm made it difficult to tell
    whether the inner AL solve had made the progress the outer loop actually needs. In
    practice, it could also keep iterating after the ALGAMES-style inner-loop conditions were
    already good enough, because small residual components and scaling choices dominated the
    global norm.

    The first stationarity-oriented draft changed the stopping rule to `opt_tol` plus
    `dyn_tol`, while still computing the full residual norm as a merit metric. The current
    version keeps that contract and tightens the implementation around it:
      - `_compute_metrics` is the single place that evaluates stationarity, dynamics feasibility,
        and the full residual merit norm for diagnostics/line-search consistency.
      - `dyn_vios` is recorded alongside `opt_vios`, which makes it possible to distinguish
        a stationarity stall from a dynamics-feasibility stall.
      - all early exits flow through `_finalize`, so termination reasons and history arrays
        stay consistent across non-finite residuals, rejected steps, stalls, and max-iteration
        exits.

    The Newton step itself is still computed from the same autodiff Jacobian of `G(z)`.
    The important change is the meaning of "inner convergence": it is no longer "make the
    whole packed residual tiny"; it is "satisfy the stationarity and dynamics-feasibility
    conditions that the AL outer loop consumes."

    Parameters
    ----------
    nlgame : NonlinearGameType2
        Game definition (dynamics, costs, constraints, u_splits).
    op0 : FixedStepPrimalDualTrajectory
        Initial primal-dual trajectory guess (X,U,μ).
    alstate : JointAugmentedLagrangianState
        Current auxiliary AL parameters (λ, ρ) held fixed for this inner solve.
    discretize_method : str
        Discretization/integration method used for dynamics residual/Jacobians.
    ineq_activation : str
        Inequality activation rule used in penalty terms.
    opt_tol : float
        Stationarity tolerance (L∞ optimality violation threshold).
    dyn_tol : float
        Dynamics feasibility tolerance (L∞ dynamics residual threshold).
    max_iters : int
        Maximum Newton iterations.
    max_rejects : int
        Maximum consecutive rejected steps before terminating.
    normkind : ResidualNormKind
        Merit norm used for tracking/line-searching (e.g., "l1_mean" to match ALGames).

    Other parameters are forwarded to `newton_step_autodiff`.

    Returns
    -------
    op_best : FixedStepPrimalDualTrajectory
        Final trajectory (last accepted if return_last_accepted=True).
    diag : StationarityNewtonDiag
        Iteration history and termination reason.
    """

    # ---- basic validation ----
    if nlgame.tg != op0.tg:
        raise ValueError(f"TimeGrid mismatch: nlgame.tg={nlgame.tg} vs op0.tg={op0.tg}")
    if not isinstance(opt_tol, (float, int)) or not math.isfinite(float(opt_tol)) or float(opt_tol) <= 0.0:
        raise ValueError(f"opt_tol must be finite and > 0, got {opt_tol}")
    if not isinstance(dyn_tol, (float, int)) or not math.isfinite(float(dyn_tol)) or float(dyn_tol) <= 0.0:
        raise ValueError(f"dyn_tol must be finite and > 0, got {dyn_tol}")
    if not isinstance(max_iters, int) or max_iters < 0:
        raise ValueError(f"max_iters must be int >= 0, got {max_iters}")
    if not isinstance(max_rejects, int) or max_rejects < 0:
        raise ValueError(f"max_rejects must be int >= 0, got {max_rejects}")

    debug_enabled = logger.isEnabledFor(logging.DEBUG)
    solve_call_id = _next_debug_call_id("newton_solve_stationarity_autodiff")
    solve_t0 = time.perf_counter() if debug_enabled else 0.0
    if debug_enabled:
        logger.debug(
            "Newton solve start call=%d max_iters=%d max_rejects=%d opt_tol=%.3g dyn_tol=%.3g norm=%s",
            solve_call_id,
            max_iters,
            max_rejects,
            float(opt_tol),
            float(dyn_tol),
            normkind,
        )

    op = op0
    op_last_accepted = op0

    # history buffers
    opt_vios: List[float] = []
    dyn_vios: List[float] = []
    merit_norms: List[float] = []
    step_norms: List[float] = []
    alphas: List[float] = []
    regs: List[float] = []
    accepted_hist: List[bool] = []
    solve_ok_hist: List[bool] = []

    # Closure that finalizes the output/diagnostics of 
    # of this function, capturing all history buffers above
    def _finalize(*, converged: bool, iters: int, reason: str) -> altypes.StationarityNewtonDiag:
        return altypes.StationarityNewtonDiag(
            converged=converged,
            iters=iters,
            opt_vios=tuple(opt_vios),
            dyn_vios=tuple(dyn_vios),
            merit_norms=tuple(merit_norms),
            step_norms=tuple(step_norms),
            alphas=tuple(alphas),
            regs=tuple(regs),
            accepted=tuple(accepted_hist),
            solve_ok=tuple(solve_ok_hist),
            reason=reason,
        )

    def _compute_metrics(cur_op: trajtypes.FixedStepPrimalDualTrajectory) -> Tuple[float, float, float, bool]:
        """
        Compute the metrics that define the current inner-loop contract.

        `opt` and `dyn` are the convergence checks consumed by this stationarity-oriented
        solve. `merit` remains the full packed residual norm so the diagnostics stay
        comparable to the older residual-norm solve and to the line-search objective used
        inside `newton_step_autodiff`.
        """
        # -- optimality/stationarity metric --
        opt = optimality_violation_inf(
            nlgame, cur_op, alstate, discretize_method=discretize_method, ineq_activation=ineq_activation
        )

        # -- dynamics feasibility metric --
        # NOTE: this is also likely to be a misuse/inefficient use of dynamics residual.
        # This should have already been computed elsewhere and could be passed on for constraint checking
        D = systypes.residual_discrete_dynamics_trajectory(nlgame.cs, cur_op, method=discretize_method)
        dyn = float(jnp.max(jnp.abs(D))) if D.size else 0.0

        if not (bool(jnp.isfinite(opt)) and bool(jnp.all(jnp.isfinite(dyn)))):
            return float("nan"), float("nan"), float("nan"), False

        # -- Lagrangian gradient residual merit metric (i.e. opt+dyn) -- 
        z = pack_decision_vars_1d(cur_op)
        g = compute_al_residual_flat_from_decision_vars(
            nlgame, z, cur_op, alstate, discretize_method=discretize_method, ineq_activation=ineq_activation
        )
        if not bool(jnp.all(jnp.isfinite(g))):
            return float(opt), float(dyn), float("nan"), False

        merit = residual_norm(g, kind=normkind)
        return float(opt), float(dyn), float(merit), True

    # ---- iteration 0 metrics ----
    opt0, dyn0, merit0, ok0 = _compute_metrics(op)

    # check if should be exited immediately due to 
    # non-finite, not-a-number metrics
    if not ok0:
        opt_vios.append(float("nan"))
        dyn_vios.append(float("nan"))
        merit_norms.append(float("nan"))
        diag = _finalize(converged=False, iters=0, reason="nonfinite_residual_at_start")
        if debug_enabled:
            logger.debug(
                "Newton solve abort call=%d reason=%s total_dt=%.3fs",
                solve_call_id,
                diag.reason,
                time.perf_counter() - solve_t0,
            )
        return op0, diag

    # update solver history
    opt_vios.append(opt0)
    dyn_vios.append(dyn0)
    merit_norms.append(merit0)

    # check if convergence is already achieved at 
    # iteration 0
    if (opt0 <= opt_tol) and (dyn0 <= dyn_tol):
        diag = _finalize(converged=True, iters=0, reason="opt_dyn_tol_at_start")
        if debug_enabled:
            logger.debug(
                "Newton solve converged call=%d iters=0 opt=%.6g dyn=%.6g merit=%.6g total_dt=%.3fs",
                solve_call_id,
                opt0,
                dyn0,
                merit0,
                time.perf_counter() - solve_t0,
            )
        return op, diag

    # start counter for number of reject newton_step calls
    reject_streak = 0

    for k in range(max_iters):

        # attempt to compute the next step in decision variables
        op_new, step_diag = newton_step_autodiff(
            nlgame,
            op,
            alstate,
            step_rtol=step_rtol,
            step_atol=step_atol,
            discretize_method=discretize_method,
            ineq_activation=ineq_activation,
            reg0=reg0,
            reg1_min=reg1_min,
            reg_increase=reg_increase,
            reg_max=reg_max,
            reg_max_iters=reg_max_iters,
            ls_alpha0=ls_alpha0,
            ls_tau=ls_tau,
            ls_beta=ls_beta,
            ls_max_iters=ls_max_iters,
            normkind=normkind,
        )

        # update solver history
        step_norms.append(step_diag.step_norm)
        alphas.append(step_diag.alpha)
        regs.append(step_diag.reg)
        accepted_hist.append(step_diag.accepted)
        solve_ok_hist.append(step_diag.solve_ok)

        # check if the step was accept to
        # record new operating point trajector or
        # record a rejected step
        if step_diag.accepted:
            op = op_new
            op_last_accepted = op_new
            reject_streak = 0
        else:
            reject_streak += 1

        # ---- compute metrics at current op ----
        optk, dynk, meritk, okk = _compute_metrics(op)

        if debug_enabled:
            logger.debug(
                "Newton iter call=%d iter=%d/%d accepted=%s solve_ok=%s alpha=%.6g reg=%.6g step_norm=%.6g opt=%.6g dyn=%.6g merit=%.6g reject_streak=%d",
                solve_call_id,
                k + 1,
                max_iters,
                step_diag.accepted,
                step_diag.solve_ok,
                step_diag.alpha,
                step_diag.reg,
                step_diag.step_norm,
                optk,
                dynk,
                meritk,
                reject_streak,
            )
        
        # at each iteration, check if should be exited 
        # immediately based upon non-finite metrics
        if not okk:
            opt_vios.append(float("nan"))
            dyn_vios.append(float("nan"))
            merit_norms.append(float("nan"))
            diag = _finalize(converged=False, iters=k + 1, reason="nonfinite_residual")
            if debug_enabled:
                logger.debug(
                    "Newton solve abort call=%d iters=%d reason=%s total_dt=%.3fs",
                    solve_call_id,
                    k + 1,
                    diag.reason,
                    time.perf_counter() - solve_t0,
                )
            return (op_last_accepted if return_last_accepted else op), diag

        # update solver history
        opt_vios.append(optk)
        dyn_vios.append(dynk)
        merit_norms.append(meritk)

        # check for solver convergence at iteration k
        if (optk <= opt_tol) and (dynk <= dyn_tol):
            diag = _finalize(converged=True, iters=k + 1, reason="opt_dyn_tol")
            if debug_enabled:
                logger.debug(
                    "Newton solve converged call=%d iters=%d opt=%.6g dyn=%.6g merit=%.6g total_dt=%.3fs",
                    solve_call_id,
                    k + 1,
                    optk,
                    dynk,
                    meritk,
                    time.perf_counter() - solve_t0,
                )
            return op, diag

        # Stall signal: newton_step decided dz is "too small" and returned accepted=True, alpha==0
        if step_diag.accepted and step_diag.alpha == 0.0:
            converged = (optk <= opt_tol) and (dynk <= dyn_tol)
            reason = "opt_dyn_tol_and_step_stall" if converged else "step_stall_before_opt_dyn_tol"
            diag = _finalize(converged=converged, iters=k + 1, reason=reason)
            op_ret = op if converged else (op_last_accepted if return_last_accepted else op)
            if debug_enabled:
                logger.debug(
                    "Newton solve stop call=%d iters=%d reason=%s opt=%.6g dyn=%.6g merit=%.6g total_dt=%.3fs",
                    solve_call_id,
                    k + 1,
                    reason,
                    optk,
                    dynk,
                    meritk,
                    time.perf_counter() - solve_t0,
                )
            return op_ret, diag

        # Break if too many steps have been rejected to prevent endless loop
        if reject_streak >= max_rejects:
            diag = _finalize(converged=False, iters=k + 1, reason="too_many_rejected_steps")
            if debug_enabled:
                logger.debug(
                    "Newton solve stop call=%d iters=%d reason=%s opt=%.6g dyn=%.6g merit=%.6g total_dt=%.3fs",
                    solve_call_id,
                    k + 1,
                    diag.reason,
                    optk,
                    dynk,
                    meritk,
                    time.perf_counter() - solve_t0,
                )
            return (op_last_accepted if return_last_accepted else op), diag

    diag = _finalize(converged=False, iters=max_iters, reason="max_iters")
    if debug_enabled:
        merit_last = merit_norms[-1] if merit_norms else float("nan")
        opt_last = opt_vios[-1] if opt_vios else float("nan")
        dyn_last = dyn_vios[-1] if dyn_vios else float("nan")
        logger.debug(
            "Newton solve stop call=%d iters=%d reason=%s opt=%.6g dyn=%.6g merit=%.6g total_dt=%.3fs",
            solve_call_id,
            max_iters,
            diag.reason,
            opt_last,
            dyn_last,
            merit_last,
            time.perf_counter() - solve_t0,
        )
    return (op_last_accepted if return_last_accepted else op), diag


def dual_ascent_update(
    constraints: contypes.GameConstraintGridMap,
    op: trajtypes.FixedStepPrimalDualTrajectory,
    alstate: altypes.JointAugmentedLagrangianState,
    *,
    validate_shapes: bool = True,
) -> altypes.JointAugmentedLagrangianState:
    """
    Dual-ascent update for auxiliary-constraint multipliers (λ) in the Augmented Lagrangian outer loop.

    This updates λ for the *auxiliary (non-dynamics)* constraint stacks:
        C(X,U) = [C_ineq(X,U); C_eq(X,U)],

    using a standard augmented-Lagrangian multiplier update:
      - Inequality constraints (projected):
            λ_ineq^+ = max(0, λ_ineq + ρ_ineq ⊙ c_ineq)
        where c_ineq = C_ineq(X,U) and max is componentwise.
      - Equality constraints:
            λ_eq^+   = λ_eq   + ρ_eq   ⊙ c_eq

    Here ρ are penalty weights (often increased by a separate schedule) and ⊙ denotes
    elementwise multiplication.

    Parameters
    ----------
    constraints : GameConstraintGridMap
        Auxiliary constraint specification as blocks. Defines the *canonical stacking order*
        of constraint values across time:
          - block order, then active_steps order, then within-step vector components.
    op : FixedStepPrimalDualTrajectory
        Current operating-point trajectory (X,U,μ) about which constraints are evaluated.
        Uses op.tg for time stamps, op.xs for states, and op.us for controls.
    alstate : JointAugmentedLagrangianState
        Current AL parameters shared across players:
            alstate.lam_ineq : (nc_ineq,)
            alstate.rho_ineq : (nc_ineq,)
            alstate.lam_eq   : (nc_eq,)
            alstate.rho_eq   : (nc_eq,)
        These dimensions must match the total scalar constraints implied by `constraints`.
    validate_shapes : bool
        If True, perform sanity checks that the evaluated constraint stacks match the
        shapes stored in `alstate`. Useful for debugging and preventing silent shape bugs.

    Returns
    -------
    alstate_new : JointAugmentedLagrangianState
        New AL state with updated multipliers (λ). Penalties (ρ) are unchanged.

    Notes
    -----
    - This function updates only λ. Penalty weights ρ are typically updated by a separate
      "increasing schedule" (e.g., ρ <- min(ρ * ρ_increase, ρ_max)).
    - Constraint *values* are extracted in the same stacking order used elsewhere in
      your solver (constraint gradients and penalty gradients). To guarantee this,
      we reuse `build_constraint_step_linearizations`, which already enforces the
      canonical ordering and normalizes kernel outputs to 1D vectors of known length.
    - This is written for correctness and clarity, not JIT-compatibility. It iterates
      over Python callables and tuples, which is expected in the outer AL loop.
    - Reference: Le Cleac'h et al. Eqn 10
    """
    # --- Evaluate constraints in canonical stacking order ---
    # build_constraint_step_linearizations returns a per-site list/tuple where each entry
    # contains:
    #   - lin.c : constraint value at that site, shape (cdim_out_step,)
    #   - lin.sl: slice into the FLAT stack for that constraint kind (ineq or eq)
    #
    # The slice bookkeeping (lin.sl) is *exactly* what defines the stacking order used
    # by your constraint-gradient routines. By filling c_ineq/c_eq using these slices,
    # we ensure the dual update is consistent with the rest of the solver.
    # NOTE: this is inefficient as it does the wasted work of computing jacobians that are
    # unused here. Need to refactor to reduce wasted work of build_constraint_step_linearizations
    ineq_lins, eq_lins = contypes.build_constraint_step_linearizations(constraints, op)

    # Use alstate dtype to avoid unintended dtype mixing (float32 vs float64).
    dtype = alstate.lam_ineq.dtype

    # Allocate flat stacks matching constraints.nc_* (the total scalar constraints implied by blocks).
    c_ineq = jnp.zeros((constraints.nc_ineq,), dtype=dtype)
    c_eq   = jnp.zeros((constraints.nc_eq,), dtype=dtype)

    # Fill stacks by slice. This is safe because:
    #  - lin.sl increments by cdim_out_step each active step
    #  - lin.c has shape (cdim_out_step,)
    for lin in ineq_lins:
        c_ineq = c_ineq.at[lin.sl].set(lin.c.astype(dtype))
    for lin in eq_lins:
        c_eq = c_eq.at[lin.sl].set(lin.c.astype(dtype))

    if validate_shapes:
        # These checks prevent silent mismatch between:
        #   - what constraints imply (constraints.nc_*)
        #   - what alstate expects (shape of lam/rho arrays)
        #
        # If these fail, either:
        #   - constraint blocks were changed without reinitializing alstate, or
        #   - a block kernel output dim doesn't match its declared cdim_out_step.
        if c_ineq.shape != alstate.lam_ineq.shape:
            raise ValueError(f"c_ineq shape {c_ineq.shape} != lam_ineq shape {alstate.lam_ineq.shape}")
        if c_eq.shape != alstate.lam_eq.shape:
            raise ValueError(f"c_eq shape {c_eq.shape} != lam_eq shape {alstate.lam_eq.shape}")
        if alstate.rho_ineq.shape != alstate.lam_ineq.shape:
            raise ValueError("rho_ineq and lam_ineq must have same shape")
        if alstate.rho_eq.shape != alstate.lam_eq.shape:
            raise ValueError("rho_eq and lam_eq must have same shape")

    # --- Dual ascent updates ---
    # Inequality multipliers are projected onto the nonnegative orthant.
    lam_ineq_new = jnp.maximum(0.0, alstate.lam_ineq + alstate.rho_ineq * c_ineq)

    # Equality multipliers are unconstrained.
    lam_eq_new = alstate.lam_eq + alstate.rho_eq * c_eq

    # Flax dataclass supports .replace(...) to return updated immutable instance.
    return alstate.replace(lam_ineq=lam_ineq_new, lam_eq=lam_eq_new)

def rho_increase_schedule(
    alstate: altypes.JointAugmentedLagrangianState,
    *,
    rho_increase: float, # = 10.0,
    rho_max: float, # = 1e8,
    validate: bool = True,
) -> altypes.JointAugmentedLagrangianState:
    """
    Increase the augmented-Lagrangian penalty weights ρ by a multiplicative schedule.

    This implements a simple "increasing schedule" used in many AL methods:
        ρ^+ = min(ρ * rho_increase, rho_max)
    applied componentwise to both inequality and equality penalty vectors.

    Parameters
    ----------
    alstate : JointAugmentedLagrangianState
        Current AL state containing penalty vectors `rho_ineq` and `rho_eq`.
        Multipliers `lam_ineq` and `lam_eq` are left unchanged.
    rho_increase : float
        Multiplicative penalty growth factor. Must be >= 1.
        Typical values: 2, 5, 10.
    rho_max : float
        Maximum allowed penalty (cap), applied componentwise. Must be > 0.
    validate : bool
        If True, validate scalar inputs and require rho_increase >= 1, rho_max > 0.

    Returns
    -------
    alstate_new : JointAugmentedLagrangianState
        New AL state with updated `rho_ineq` and `rho_eq`, other fields unchanged.

    Notes
    -----
    - This function does not implement any adaptive logic (e.g., increase only if
      constraint violation stagnates). It is a simple deterministic schedule.
    - For inequalities, some algorithms also maintain ρ >= 0; this schedule preserves
      nonnegativity if the input ρ is nonnegative.
    """
    if validate:
        if not isinstance(rho_increase, (float, int)):
            raise TypeError(f"rho_increase must be float-like, got {type(rho_increase)}")
        if not isinstance(rho_max, (float, int)):
            raise TypeError(f"rho_max must be float-like, got {type(rho_max)}")
        if not math.isfinite(float(rho_increase)) or float(rho_increase) < 1.0:
            raise ValueError(f"rho_increase must be finite and >= 1, got {rho_increase}")
        if not math.isfinite(float(rho_max)) or float(rho_max) <= 0.0:
            raise ValueError(f"rho_max must be finite and > 0, got {rho_max}")

    rho_max_arr = jnp.asarray(rho_max, dtype=alstate.rho_ineq.dtype)

    rho_ineq_new = jnp.minimum(alstate.rho_ineq * rho_increase, rho_max_arr)
    rho_eq_new = jnp.minimum(alstate.rho_eq * rho_increase, rho_max_arr)

    return alstate.replace(rho_ineq=rho_ineq_new, rho_eq=rho_eq_new)


def _constraint_violation_metrics(
    c_ineq: jnp.ndarray,
    c_eq: jnp.ndarray,
) -> Tuple[float, float]:
    """
    Compute simple infinity-norm feasibility metrics for auxiliary constraints.

    Args
    ----
    c_ineq : jnp.ndarray, shape (nc_ineq,)
        Stacked inequality constraint values (feasible if c_ineq <= 0 componentwise).
    c_eq : jnp.ndarray, shape (nc_eq,)
        Stacked equality constraint values (feasible if c_eq == 0 componentwise).

    Returns
    -------
    ineq_vio_inf : float
        max(max(c_ineq, 0))  (0 if nc_ineq == 0).
    eq_vio_inf : float
        max(abs(c_eq))       (0 if nc_eq == 0).
    """
    if c_ineq.size == 0:
        ineq_vio = 0.0
    else:
        ineq_vio = float(jnp.max(jnp.maximum(c_ineq, 0.0)))
    if c_eq.size == 0:
        eq_vio = 0.0
    else:
        eq_vio = float(jnp.max(jnp.abs(c_eq)))
    return ineq_vio, eq_vio


def _collect_constraint_stacks_from_linearizations(
    constraints: contypes.GameConstraintGridMap,
    ineq_lins: Tuple[contypes.ConstraintStepLinearization, ...],
    eq_lins: Tuple[contypes.ConstraintStepLinearization, ...],
    *,
    dtype,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Assemble flat constraint-value stacks from per-step linearizations.

    This fills c_ineq and c_eq using each linearization's slice `lin.sl` and value `lin.c`.
    The resulting stacks match the canonical ordering implied by `constraints` and used
    throughout the AL solver.

    Args
    ----
    constraints : GameConstraintGridMap
        Constraint container providing total sizes `nc_ineq` and `nc_eq`.
    ineq_lins : iterable of ConstraintStepLinearization
        Linearizations for inequality constraints (each provides lin.c and lin.sl).
    eq_lins : iterable of ConstraintStepLinearization
        Linearizations for equality constraints (each provides lin.c and lin.sl).
    dtype : jnp dtype
        dtype used for the returned stacks.

    Returns
    -------
    c_ineq : jnp.ndarray, shape (constraints.nc_ineq,)
        Stacked inequality constraint values.
    c_eq : jnp.ndarray, shape (constraints.nc_eq,)
        Stacked equality constraint values.
    """
    c_ineq = jnp.zeros((constraints.nc_ineq,), dtype=dtype)
    c_eq = jnp.zeros((constraints.nc_eq,), dtype=dtype)
    for lin in ineq_lins:
        c_ineq = c_ineq.at[lin.sl].set(lin.c.astype(dtype))
    for lin in eq_lins:
        c_eq = c_eq.at[lin.sl].set(lin.c.astype(dtype))
    return c_ineq, c_eq


def al_solve_autodiff(
    nlgame: gametypes.NonlinearGameType2,
    op0: trajtypes.FixedStepPrimalDualTrajectory,
    alstate0: altypes.JointAugmentedLagrangianState,
    *,
    discretize_method: str = "rk2",
    ineq_activation: str = "altro",
    # outer loop controls
    max_iters: int = 8,
    rho_increase: float = 10.0,
    rho_max: float = 1e8,
    # termination tolerances
    opt_tol: float = 1e-3,        # optimality/stationarity tolerance
    dyn_tol: float = 1e-3,        # dynamics constraint feasibility
    ineq_tol: float = 1e-3,       # inequality constraint feasibility
    eq_tol: float = 1e-3,         # equality constraint feasibility
    # inner solve controls (forwarded)
    newton_max_iters: int = 16,
    newton_max_rejects: int = 4,
    newton_step_rtol: float = 1e-7,
    newton_step_atol: float = 1e-8,
    reg_init: float = 0.0,
    reg_min_on_fail: float = 1e-12,
    reg_increase: float = 10.0,
    reg_max: float = 1e8,
    reg_max_iters: int = 64,
    ls_alpha0: float = 1.0,
    ls_tau: float = 0.5,
    ls_beta: float = 0.25,
    ls_max_iters: int = 20,
    normkind: ResidualNormKind = "l1_mean",
) -> Tuple[trajtypes.FixedStepPrimalDualTrajectory, altypes.JointAugmentedLagrangianState, altypes.ALSolverDiag]:
    """
    Solve a constrained dynamic game using an Augmented Lagrangian (AL) outer loop with a
    Newton-based inner solve.

    At a high level, this routine solves for a root of the AL stationarity/feasibility
    system for fixed auxiliary-constraint multipliers (λ) and penalties (ρ), then updates
    (λ, ρ) in an outer loop:

    Outer iteration k:
    1) Inner solve (Newton): approximately solve G(z; λ,ρ) = 0 for decision variables
            z := (X[1:], U, μ)
        where G stacks per-player stationarity conditions (i.e. 
        gradient of augmented Lagrangian) and the discrete dynamics residual.
    2) Evaluate auxiliary constraint values C(X,U) and compute feasibility metrics.
    3) Dual ascent update on λ (projected for inequalities).
    4) Increase schedule update on ρ.

    For reference on default input argument values, see:
    https://github.com/RoboticExplorationLab/Algames.jl/blob/5c779ca3cebb9b3b31ebb7414331b479cc6c3f6e/src/struct/options.jl

    Parameters
    ----------
    nlgame : NonlinearGameType2
        Continuous-time game definition. Must include:
        - continuous dynamics in nlgame.cs (discretized internally via `discretize_method`)
        - per-player costs
        - auxiliary constraints (non-dynamics, e.g. state and control bounds) in nlgame.constraints
        - control partition info in nlgame.u_splits (used by cost gradients)

    op0 : FixedStepPrimalDualTrajectory
        Initial primal-dual operating point for the *inner* Newton solve:
        - xs : (nt, nx) joint state trajectory
        - us : (nsteps, nu) joint control trajectory
        - ls : (nsteps, N, nx) player-specific dynamics multipliers μ (one μ per player per step)
        Note: auxiliary-constraint AL state (λ, ρ) is *not* stored in op0; it is provided
        separately via `alstate0`.

    alstate0 : JointAugmentedLagrangianState
        Initial AL state for auxiliary constraints C(X,U):
        - lam_ineq, rho_ineq for inequality constraints
        - lam_eq,   rho_eq   for equality constraints
        These parameters are shared across players and updated only in the *outer* loop.

    discretize_method : str, default="rk2"
        Discretization/integration method used to form the one-step discrete dynamics map
        f_d(t_k, x_k, u_k) and its Jacobians inside the inner solve.
        Expected values match your integrator registry (e.g. "euler", "rk2", "rk3", "rk4").

    ineq_activation : str, default="altro"
        Activation rule for inequality constraints used in the quadratic penalty-gradient/Hessian
        contributions (e.g. Altro-style active-set indicator). This affects the inner Newton
        system when inequality constraints are present.
        See Le Cleac'h et al. Eqn 5 for reference of Altro-style

    Outer-loop controls
    -------------------
    max_iters : int, default=10
        Maximum number of outer AL iterations (dual ascent + penalty increases).

    rho_increase : float, default=10.0
        Multiplicative penalty growth factor for ρ applied after each outer iteration:
            ρ <- min(ρ * rho_increase, rho_max)
        Must be >= 1.0 for a monotone increase schedule.

    rho_max : float, default=1e8
        Componentwise cap for ρ to avoid unbounded penalty growth.

    Termination tolerances (outer loop)
    -----------------------------------
    opt_tol : float, default=1e-6
        Target tolerance for stationarity/optimality conditions, measured as a 
        an infinity-norm

    dyn_tol : float, default=1e-6
        Dynamics feasibility tolerance, measured as an infinity norm on the discrete dynamics
        residual D(X,U) (e.g. max(abs(D))).

    ineq_tol : float, default=1e-6
        Inequality feasibility tolerance for auxiliary constraints C_ineq(X,U) <= 0, measured as:
            max(max(C_ineq, 0))

    eq_tol : float, default=1e-6
        Equality feasibility tolerance for auxiliary constraints C_eq(X,U) == 0, measured as:
            max(abs(C_eq))

    Inner Newton solve controls (forwarded)
    ---------------------------------------
    newton_max_iters : int, default=25
        Maximum number of Newton iterations per outer iteration.

    newton_max_rejects : int, default=5
        Maximum number of consecutively rejected Newton steps (via line search) allowed before
        aborting the inner solve.

    newton_residual_rtol, newton_residual_atol : float, default=1e-7, 1e-8
        If ||g||₂ < residual_rtol ||g0||₂ + residual_atol, than the newton
        root finding algorithm is treated as converged

    newton_step_rtol, newton_step_atol : float, default=1e-7, 1e-8
        If ||dz||₂ < step_rtol ||z0||₂ + step_atol, If ||dz|| is below this,
        the inner solve may treat the iterate as converged/stalled.

    Tikhonov regularization controls (inner linear solve)
    -----------------------------------------------------
    reg_init : float, default=0.0
        Initial Tikhonov regularization for the Newton linear system solve:
            (H + reg * I) Δz = -G
        This can be 0.0; if the solve fails, the solver escalates to at least `reg_min_on_fail`.

    reg_min_on_fail : float, default=1e-12
        Minimum regularization to use after the first linear-solve failure (useful when reg_init=0).

    reg_increase : float, default=10.0
        Multiplicative growth factor for regularization when retrying a failed linear solve.

    reg_max : float, default=1e8
        Maximum allowed regularization before declaring the linear solve failed.

    reg_max_iters : int, default=64
        Maximum number of regularization escalation attempts in a single Newton step.

    Backtracking line search controls (inner step acceptance)
    ---------------------------------------------------------
    ls_alpha0 : float, default=1.0
        Initial step size α for the line search (trial point z_trial = z + αΔz).

    ls_tau : float
        Backtracking shrink factor τ in (0, 1): on rejection α <- τ α.

    ls_beta : float
        Armijo slope parameter β in (0, 1/2) for the acceptance test:
            ||G(z + α dz)|| <= (1 - α β) ||G(z)||.

    ls_max_iters : int, default=20
        Maximum number of line-search trial evaluations per Newton step.

    normkind : ResidualNormKind
        Merit norm used for line search and recorded diagnostics (e.g. "l1_mean" to match ALGAMES).

    Returns
    -------
    op : FixedStepPrimalDualTrajectory
        Final primal-dual trajectory (X,U,μ) returned by the outer loop.

    alstate : JointAugmentedLagrangianState
        Final auxiliary-constraint AL state (λ, ρ) after outer iterations.

    diag : ALSolverDiag
        Solver diagnostics, including per-outer-iteration history with:
        - inner Newton outcome (iters, reason, residual norm)
        - feasibility metrics (dyn/ineq/eq)
        - λ/ρ magnitude summaries

    Notes
    -----
    - This implementation uses autodiff Jacobians and dense linear solves, so it is intended
    for correctness and small problems. A production implementation typically exploits
    block sparsity and structured solves.
    - Because the inner solve uses line search on ||G||, it is possible to satisfy feasibility
    while stalling slightly above `residual_tol` in float32; choose tolerances accordingly.
    """

    if nlgame.tg != op0.tg:
        raise ValueError(f"TimeGrid mismatch: nlgame.tg={nlgame.tg} vs op0.tg={op0.tg}")

    debug_enabled = logger.isEnabledFor(logging.DEBUG)
    al_call_id = _next_debug_call_id("al_solve_autodiff")
    al_t0 = time.perf_counter() if debug_enabled else 0.0
    if debug_enabled:
        logger.debug(
            "AL outer solve start call=%d max_outer_iters=%d newton_max_iters=%d rho_increase=%.6g rho_max=%.6g",
            al_call_id,
            max_iters,
            newton_max_iters,
            rho_increase,
            rho_max,
        )

    op = op0
    alstate = alstate0
    hist: List[altypes.ALSolverOuterIterDiag] = []

    for k in range(max_iters):
        if debug_enabled:
            logger.debug(
                "AL outer iter start call=%d iter=%d/%d rho_ineq_max=%.6g rho_eq_max=%.6g lam_ineq_max=%.6g lam_eq_max=%.6g",
                al_call_id,
                k + 1,
                max_iters,
                float(jnp.max(alstate.rho_ineq)) if alstate.rho_ineq.size else 0.0,
                float(jnp.max(alstate.rho_eq)) if alstate.rho_eq.size else 0.0,
                float(jnp.max(alstate.lam_ineq)) if alstate.lam_ineq.size else 0.0,
                float(jnp.max(alstate.lam_eq)) if alstate.lam_eq.size else 0.0,
            )

        # ---- 1) inner solve: newton root finding of G=0 with fixed (λ,ρ) ----
        op, newton_diag = newton_solve_stationarity_autodiff(
            nlgame, op, alstate,
            discretize_method = discretize_method,
            ineq_activation = ineq_activation,
            opt_tol = opt_tol,
            dyn_tol = dyn_tol,
            max_iters = newton_max_iters,
            max_rejects = newton_max_rejects,
            step_rtol = newton_step_rtol,
            step_atol = newton_step_atol,
            reg0 = reg_init,
            reg1_min = reg_min_on_fail,
            reg_increase = reg_increase,
            reg_max = reg_max,
            reg_max_iters = reg_max_iters,
            ls_alpha0 = ls_alpha0,
            ls_tau = ls_tau,
            ls_beta = ls_beta,
            ls_max_iters = ls_max_iters,
            normkind = normkind,
            return_last_accepted = True,
        )

        # extract norm of aug lagrange gradient (G) and optimality/stationarity violation from
        # final iteration of newton root finding
        g_norm = float(newton_diag.merit_norms[-1]) if newton_diag.merit_norms else float("nan")
        opt_vio = float(newton_diag.opt_vios[-1]) if newton_diag.opt_vios else float("nan")
        dyn_vio = float(newton_diag.dyn_vios[-1]) if newton_diag.dyn_vios else float("nan")

        # ---- 2) evaluate feasibility conditions (dynamics, constraints) ----

        # # -- stationarity/optimality metric (e.g. gradL = 0) --
        # # NOTE: this is a misuse-of-convience of the compute_al_residual_from_traj function
        # # that inefficiently computes gradients for things that would have already been computed
        # # during the newton_solve_autodiff
        # # NOTE: this also computes dyn_vio inadvertantly
        # opt_vio = optimality_violation_inf(
        #     nlgame, op, alstate, 
        #     discretize_method=discretize_method, 
        #     ineq_activation=ineq_activation
        # )

        # # -- dynamics feasibility metric --
        # # Prefer your existing function:
        # # D has shape (K, nx) where K=nt-1
        # # NOTE: this is also likely to be a misuse/inefficient use of dynamics residual.
        # # This should have already been computed elsewhere and could be passed on for constraint checking
        # D = systypes.residual_discrete_dynamics_trajectory(nlgame.cs, op, method=discretize_method)
        # dyn_vio = float(jnp.max(jnp.abs(D))) if D.size else 0.0

        # -- inequality & equlaity constraint feasibility (e.g. state and control constraints) --
        # NOTE: this is another "convenient misuse" of build_constraint_step_linearizations.
        # The jacobians are not needed here and are extra work to compute
        ineq_lins, eq_lins = contypes.build_constraint_step_linearizations(nlgame.constraints, op)
        c_ineq, c_eq = _collect_constraint_stacks_from_linearizations(
            nlgame.constraints, ineq_lins, eq_lins, dtype=alstate.lam_ineq.dtype
        )
        ineq_vio, eq_vio = _constraint_violation_metrics(c_ineq, c_eq)

        # record diagnostics
        hist.append(altypes.ALSolverOuterIterDiag(
            outer_iter=k,
            newton_converged=newton_diag.converged,
            newton_iters=newton_diag.iters,
            newton_reason=newton_diag.reason,
            residual_norm_final=g_norm,
            opt_vio_inf=opt_vio,
            dyn_vio_inf=dyn_vio,
            ineq_vio_inf=ineq_vio,
            eq_vio_inf=eq_vio,
            rho_ineq_max=float(jnp.max(alstate.rho_ineq)) if alstate.rho_ineq.size else 0.0,
            rho_eq_max=float(jnp.max(alstate.rho_eq)) if alstate.rho_eq.size else 0.0,
            lam_ineq_max=float(jnp.max(alstate.lam_ineq)) if alstate.lam_ineq.size else 0.0,
            lam_eq_max=float(jnp.max(alstate.lam_eq)) if alstate.lam_eq.size else 0.0,
        ))

        if debug_enabled:
            logger.debug(
                "AL outer iter done call=%d iter=%d/%d newton_reason=%s newton_iters=%d newton_converged=%s residual=%.6g opt=%.6g dyn=%.6g ineq=%.6g eq=%.6g",
                al_call_id,
                k + 1,
                max_iters,
                newton_diag.reason,
                newton_diag.iters,
                newton_diag.converged,
                g_norm,
                opt_vio,
                dyn_vio,
                ineq_vio,
                eq_vio,
            )

        # ---- 5) convergence check ----
        if (
            (dyn_vio <= dyn_tol)    and 
            (ineq_vio <= ineq_tol)  and 
            (eq_vio <= eq_tol)      and 
            (opt_vio <= opt_tol)
        ):
            diag = altypes.ALSolverDiag(converged=True, iters=k + 1, reason="converged", history=tuple(hist))
            if debug_enabled:
                logger.debug(
                    "AL outer solve converged call=%d iters=%d total_dt=%.3fs",
                    al_call_id,
                    k + 1,
                    time.perf_counter() - al_t0,
                )
            return op, alstate, diag

        # ---- 3) Dual ascent update on λ (uses values in canonical order) ----
        alstate = dual_ascent_update(nlgame.constraints, op, alstate, validate_shapes=True)

        # ---- 4) Increase schedule update on ρ ----
        alstate = rho_increase_schedule(alstate, rho_increase=rho_increase, rho_max=rho_max, validate=True)

    diag = altypes.ALSolverDiag(converged=False, iters=max_iters, reason="max_outer_iters", history=tuple(hist))
    if debug_enabled:
        logger.debug(
            "AL outer solve stop call=%d iters=%d reason=%s total_dt=%.3fs",
            al_call_id,
            max_iters,
            diag.reason,
            time.perf_counter() - al_t0,
        )
    return op, alstate, diag
