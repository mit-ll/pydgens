# Copyright 2026 MIT Lincoln Laboratory
# SPDX-License-Identifier: MIT

# Classes and datatypes for Augmented Lagrangian (AL) solver
import jax
import jax.numpy as jnp
import flax.struct

from dataclasses import dataclass
from typing import Optional, Tuple

@flax.struct.dataclass
class JointAugmentedLagrangianState:
    """
    Augmented Lagrangian parameters for the joint constraint mapping vector C.

    This stores multipliers and penalties for inequality and equality constraint
    components separately, matching the convention:
        c = [c_ineq; c_eq]

    Note that a flax dataclass is used instead of stdlib dataclass because
    the AL state vars are all jax-tracable jnp arrays

    Attributes
    ----------
    lam_ineq : jnp.ndarray, shape (nc_ineq,)
        Lagrange multipliers for inequality constraints.
    rho_ineq : jnp.ndarray, shape (nc_ineq,)
        Penalty weights for inequality constraints (often kept >= 0).
    lam_eq : jnp.ndarray, shape (nc_eq,)
        Lagrange multipliers for equality constraints.
    rho_eq : jnp.ndarray, shape (nc_eq,)
        Penalty weights for equality constraints.
    """
    lam_ineq: jnp.ndarray
    rho_ineq: jnp.ndarray
    lam_eq: jnp.ndarray
    rho_eq: jnp.ndarray

    def __post_init__(self):
        # --- Type + rank checks (tracer-safe) ---
        fields = ("lam_ineq", "rho_ineq", "lam_eq", "rho_eq")
        for name in fields:
            arr = getattr(self, name)

            # Accept JAX arrays; jnp.ndarray is typically an alias of jax.Array in modern JAX
            if not isinstance(arr, (jax.Array,)):
                raise TypeError(f"{name} must be a jax.Array, got {type(arr)}")

            if arr.ndim != 1:
                raise ValueError(f"{name} must be 1D, got shape {arr.shape}")

        # --- Shape consistency ---
        if self.lam_ineq.shape != self.rho_ineq.shape:
            raise ValueError(
                f"lam_ineq and rho_ineq must have same shape, got "
                f"{self.lam_ineq.shape} vs {self.rho_ineq.shape}"
            )
        if self.lam_eq.shape != self.rho_eq.shape:
            raise ValueError(
                f"lam_eq and rho_eq must have same shape, got "
                f"{self.lam_eq.shape} vs {self.rho_eq.shape}"
            )

        # --- Dtype consistency (helps avoid mixed precision surprises) ---
        d0 = self.lam_ineq.dtype
        for name in fields[1:]:
            d = getattr(self, name).dtype
            if d != d0:
                raise TypeError(
                    f"All AL arrays must share the same dtype. "
                    f"lam_ineq.dtype={d0}, {name}.dtype={d}"
                )

    @property
    def nc_ineq(self) -> int:
        return int(self.lam_ineq.shape[0])

    @property
    def nc_eq(self) -> int:
        return int(self.lam_eq.shape[0])
    
    @property
    def nc_all(self) -> int:
        return self.nc_ineq + self.nc_eq


def init_joint_augmented_lagrangian_state(
    nc_ineq: int,
    nc_eq: int,
    lam0: float = 0.0,
    rho0: float = 1.0,
    dtype=jnp.float32,
) -> JointAugmentedLagrangianState:
    """
    Initialize AL state for a given constraint dimension split.

    Parameters
    ----------
    nc_ineq, nc_eq : int
        Dimensions of inequality and equality constraint stacks.
    lam0 : float
        Initial multiplier value (often 0).
    rho0 : float
        Initial penalty value (>0).

    Returns
    -------
    AugmentedLagrangianState
    """
    lam_ineq = jnp.full((nc_ineq,), lam0, dtype=dtype)
    lam_eq   = jnp.full((nc_eq,), lam0, dtype=dtype)

    rho_ineq = jnp.full((nc_ineq,), rho0, dtype=dtype)
    rho_eq   = jnp.full((nc_eq,), rho0, dtype=dtype)

    return JointAugmentedLagrangianState(
        lam_ineq=lam_ineq,
        rho_ineq=rho_ineq,
        lam_eq=lam_eq,
        rho_eq=rho_eq,
    )

@dataclass(frozen=True)
class ALResidualPackLayout:
    """
    Slice bookkeeping for a packed AL residual vector.

    Attributes
    ----------
    sta_slice : slice
        Slice covering stationarity components (all players' dLdX and local dLdU blocks).
    dyn_slice : slice
        Slice covering dynamics feasibility residual components.
    """
    sta_slice: slice
    dyn_slice: slice

@flax.struct.dataclass
class ALResidualStruct:
    """
    Structured residual for the augmented-Lagrangian game Newton system.

    This corresponds to the stacked vector often denoted G in Le Cleac'h et al.,
    but stored in a structured (block) form for clarity and debugging.

    Blocks
    ------
    dLdX : jnp.ndarray, shape (N, K, nx)
        Player-stacked stationarity residuals w.r.t. the *decision* state variables X[1:].
        Here K = nt-1, and the first state x0 is treated as fixed (not a decision variable).

    dLdU : jnp.ndarray, shape (N, K, nu)
        Player-stacked stationarity residuals w.r.t. the *joint* control variables U.
        Although each player only optimizes its local control U_i, storing this in joint
        coordinates is convenient because:
          - shared constraint terms naturally live in joint coordinates
          - dynamics Jacobians are w.r.t. joint controls
        A packing step can later slice this into local blocks per player if desired.

    dyn_res : jnp.ndarray, shape (K, nx)
        Discrete dynamics feasibility residuals:
            dyn_res[k] = f_d(t_k, x_k, u_k) - x_{k+1}.

    Notes
    -----
    - This object is a PyTree (flax.struct.dataclass), so it can be returned from
      JAX-transformed functions if needed. It is still fine to treat it as a plain
      container in non-jitted code.
    """

    dLdX: jnp.ndarray   # (N, K, nx)
    dLdU: jnp.ndarray   # (N, K, nu)
    dyn_res: jnp.ndarray  # (K, nx)

    @property
    def N(self) -> int:
        return int(self.dLdX.shape[0])

    @property
    def K(self) -> int:
        return int(self.dLdX.shape[1])

    @property
    def nx(self) -> int:
        return int(self.dLdX.shape[2])

    @property
    def nu(self) -> int:
        return int(self.dLdU.shape[2])

def validate_al_residual_struct(r: ALResidualStruct) -> None:
    """
    Lightweight shape sanity checks for ALResidualStruct.
    Intended for debugging; avoid calling inside jitted code.
    """
    if r.dLdX.ndim != 3:
        raise ValueError(f"dLdX must be (N,K,nx), got {r.dLdX.shape}")
    if r.dLdU.ndim != 3:
        raise ValueError(f"dLdU must be (N,K,nu), got {r.dLdU.shape}")
    if r.dyn_res.ndim != 2:
        raise ValueError(f"dyn_res must be (K,nx), got {r.dyn_res.shape}")

    N, K, nx = r.dLdX.shape
    N2, K2, nu = r.dLdU.shape
    K3, nx3 = r.dyn_res.shape

    if N2 != N or K2 != K:
        raise ValueError(f"dLdU must share (N,K) with dLdX: {r.dLdU.shape} vs {r.dLdX.shape}")
    if K3 != K or nx3 != nx:
        raise ValueError(f"dyn_res must be (K,nx)=({K},{nx}), got {r.dyn_res.shape}")
    
@dataclass(frozen=True)
class RegularizedSolveResult:
    """
    Result of attempting a (possibly regularized) Newton linear solve.

    Attributes
    ----------
    dz : Optional[jnp.ndarray], shape (nz,)
        The computed Newton step Δz satisfying:
            (H + reg * I) dz ≈ -g
        If the solve fails for all attempted reg values, dz is None.

    reg : float
        The diagonal Tikhonov regularization used in the final attempt.
        - If ok=True: reg is the value that produced a solution.
        - If ok=False: reg is the last attempted value (typically > reg_max).

    ok : bool
        True if a solution dz was successfully computed.
    """
    dz: Optional[jnp.ndarray]
    reg: float
    ok: bool

@dataclass(frozen=True)
class LineSearchResult:
    """
    Result of a backtracking line search along a proposed step direction.

    Attributes
    ----------
    accepted : bool
        True if an alpha was found that satisfied the acceptance criterion.

    alpha : float
        The accepted step size alpha in (0, alpha0]. If accepted=False, alpha is 0.0.

    g_norm_trial : float
        The residual norm ||G(z0 + alpha*dz)|| for the accepted alpha (if accepted=True).
        If accepted=False, this may be the norm from the last attempted alpha.

    ls_iters : int
        Number of trial evaluations performed (<= max_iters).

    best_alpha : float
        The alpha that produced the smallest residual norm among the tried candidates.
        This is useful for diagnostics even when accepted=False.

    best_norm : float
        The smallest residual norm observed among tried candidates.
    """
    accepted: bool
    alpha: float
    g_norm_trial: float
    ls_iters: int
    best_alpha: float
    best_norm: float

@dataclass(frozen=True)
class NewtonStepDiag:
    """
    Diagnostics for a single Newton step on the root-finding system G(z)=0.

    Attributes
    ----------
    accepted : bool
        Whether the step was accepted by the line search and applied to produce op_new.
        If False, the returned op is usually unchanged.

    alpha : float
        Accepted step size alpha in [0, 1]. If accepted=False, alpha may be 0.0
        (or you may store best_alpha separately in line search diagnostics).

    g_norm0 : float
        Norm of the residual at the starting point: ||G(z0)||_2.

    g_norm_trial : float
        Norm of the residual at the accepted trial point (if accepted=True),
        else the best or last tried value depending on your policy.

    step_norm : float
        Norm of the proposed Newton direction: ||Δz||_2.

    reg : float
        Regularization used in the linear solve, i.e. solves (H + reg I)Δ = -G.

    ls_iters : int
        Number of line-search trial evaluations performed.

    solve_ok : bool
        Whether the linear solve succeeded (including any regularization escalation).
    """
    accepted: bool
    alpha: float
    g_norm0: float
    g_norm_trial: float
    step_norm: float
    reg: float
    ls_iters: int
    solve_ok: bool


@dataclass(frozen=True)
class NewtonSolveDiag:
    """
    Diagnostics for a multi-iteration Newton solve.

    Attributes
    ----------
    converged : bool
        True if termination met the residual-norm criterion.
    iters : int
        Number of Newton iterations executed.
    g_norms : Tuple[float, ...]
        Residual norms ||G|| at each iteration start (including iteration 0).
    step_norms : Tuple[float, ...]
        Norms ||Δz|| of proposed steps per iteration (length == iters).
    alphas : Tuple[float, ...]
        Accepted step sizes per iteration (length == iters). If step rejected, alpha is 0.
    regs : Tuple[float, ...]
        Regularization used per iteration (length == iters).
    accepted : Tuple[bool, ...]
        Whether each iteration’s step was accepted (length == iters).
    solve_ok : Tuple[bool, ...]
        Whether each iteration’s linear solve succeeded (length == iters).
    reason : str
        Human-readable termination reason.
    """
    converged: bool
    iters: int
    g_norms: Tuple[float, ...]
    step_norms: Tuple[float, ...]
    alphas: Tuple[float, ...]
    regs: Tuple[float, ...]
    accepted: Tuple[bool, ...]
    solve_ok: Tuple[bool, ...]
    reason: str


@dataclass(frozen=True)
class StationarityNewtonDiag:
    """
    Diagnostics for a Newton-like inner solve that targets stationarity (optimality)
    rather than full residual root finding.

    Attributes
    ----------
    converged : bool
        True if opt_vio_inf <= opt_tol was achieved, i.e. the optimality/stationarity
        condition has been achieved (dLdX, dLdU are within a bounded region of zero),
        it does not imply that the residual G has a norm close to zero (for example,
        G also includes dynamic feasibility constraints)
    iters : int
        Number of iterations executed.
    opt_vios : Tuple[float, ...]
        Optimality violation (L∞ of stationarity blocks) at each iteration start,
        including iteration 0.
    dyn_vios : Tuple[float, ...]
        Dynamics feasibility violation (L∞ of stationarity blocks) at each iteration start,
        including iteration 0.
    merit_norms : Tuple[float, ...]
        Merit norm of the full packed residual used for line search (e.g. l1_mean),
        at each iteration start, including iteration 0.
    step_norms : Tuple[float, ...]
        Norms of proposed steps ||dz|| under the step norm used by newton_step_autodiff.
        Length == iters.
    alphas : Tuple[float, ...]
        Accepted step sizes per iteration. Length == iters.
    regs : Tuple[float, ...]
        Regularization used per iteration. Length == iters.
    accepted : Tuple[bool, ...]
        Whether each step was accepted by line search. Length == iters.
    solve_ok : Tuple[bool, ...]
        Whether the linear solve succeeded. Length == iters.
    reason : str
        Termination reason string.
    """
    converged: bool
    iters: int
    opt_vios: Tuple[float, ...]
    dyn_vios: Tuple[float, ...]
    merit_norms: Tuple[float, ...]
    step_norms: Tuple[float, ...]
    alphas: Tuple[float, ...]
    regs: Tuple[float, ...]
    accepted: Tuple[bool, ...]
    solve_ok: Tuple[bool, ...]
    reason: str


@dataclass(frozen=True)
class ALSolverOuterIterDiag:
    """
    Diagnostics recorded for a single Augmented Lagrangian (AL) *outer* iteration.

    One AL outer iteration typically performs:
      1) an inner Newton/root-finding solve for the primal-dual variables (X, U, μ) with
         auxiliary-constraint AL parameters (λ, ρ) held fixed,
      2) evaluation of feasibility metrics for dynamics and auxiliary constraints,
      3) dual-ascent update of λ and penalty update of ρ.

    This dataclass stores a compact summary of those outcomes.

    Attributes
    ----------
    outer_iter : int
        Zero-based index of the AL outer iteration.

    newton_converged : bool
        Whether the inner Newton solve reported convergence for the residual system
        (e.g. ||G|| <= residual_tol), under the current (λ, ρ).

    newton_iters : int
        Number of Newton iterations executed in the inner solve for this outer iteration.

    newton_reason : str
        Termination reason reported by the inner Newton solve, such as:
          - "g_tol" (residual tolerance met),
          - "converged_at_start",
          - "too_many_rejected_steps",
          - "max_iters",
        or other solver-specific labels.

    residual_norm_final : float
        Final residual norm from the inner solve (typically ||G||_2), evaluated at the
        returned (X, U, μ) for this outer iteration. This is the primary stationarity/
        feasibility metric for the inner root-finding system.

    opt_vio_inf : float
        Infinity-norm optimality/stationarity metric (i.e. violation "vio"), i.e. how close dLdX and dLdU
        are to zero where L is the augmented Lagrangian

    dyn_vio_inf : float
        Infinity-norm dynamics feasibility metric computed after the inner solve, e.g.:
            max(abs(D(X,U)))
        where D(X,U) stacks the discrete dynamics residuals (violations "vio") over time.

    ineq_vio_inf : float
        Infinity-norm inequality feasibility metric (violations "vio") for auxiliary 
        constraints C_ineq(X,U) <= 0:
            max(max(C_ineq(X,U), 0))

    eq_vio_inf : float
        Infinity-norm equality feasibility metric (violations "vio") for auxiliary 
        constraints C_eq(X,U) == 0:
            max(abs(C_eq(X,U)))

    rho_ineq_max : float
        Maximum value in the inequality penalty vector ρ_ineq before the outer-iteration
        penalty update (useful to detect saturation at rho_max).

    rho_eq_max : float
        Maximum value in the equality penalty vector ρ_eq before the outer-iteration
        penalty update.

    lam_ineq_max : float
        Maximum value in the inequality multiplier vector λ_ineq before the outer-iteration
        dual ascent update (typically with projection onto λ_ineq >= 0).

    lam_eq_max : float
        Maximum value in the equality multiplier vector λ_eq before the outer-iteration
        dual ascent update.
    """
    outer_iter: int
    # inner solve info
    newton_converged: bool
    newton_iters: int
    newton_reason: str
    residual_norm_final: float

    # feasibility metrics
    opt_vio_inf: float
    dyn_vio_inf: float
    ineq_vio_inf: float
    eq_vio_inf: float

    # AL state summary
    rho_ineq_max: float
    rho_eq_max: float
    lam_ineq_max: float
    lam_eq_max: float


@dataclass(frozen=True)
class ALSolverDiag:
    """
    Diagnostics for the full Augmented Lagrangian outer solve.

    Attributes
    ----------
    converged : bool
        Whether termination tolerances were met.
    iters : int
        Number of outer iterations executed.
    reason : str
        Termination reason (e.g. "converged", "max_outer_iters").
    history : tuple[ALSolverOuterIterDiag, ...]
        Per-outer-iteration diagnostics.
    """
    converged: bool
    iters: int
    reason: str
    history: Tuple[ALSolverOuterIterDiag, ...]