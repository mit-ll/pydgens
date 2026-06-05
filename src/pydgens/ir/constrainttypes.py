# Copyright 2026 MIT Lincoln Laboratory
# SPDX-License-Identifier: MIT

# Class and function definitions for various state and control constraints
import jax
import jax.numpy as jnp

from dataclasses import dataclass, field
from typing import Callable, Tuple, Union, Optional, Literal

from pydgens.ir.timetypes import TimeGrid, compute_ts
from pydgens.ir.trajectorytypes import FixedStepPrimalDualTrajectory


ConstraintKind = Literal["ineq", "eq"]

# Step constraint kernel: c(t, x, u) -> (cdim_out_step,) or scalar
StepConstraintFn = Callable[[float, jnp.ndarray, jnp.ndarray], jnp.ndarray]

# Terminal-only kernel: c(t, x) -> (cdim_out_step,) or scalar
TerminalConstraintFn = Callable[[float, jnp.ndarray], jnp.ndarray]


@dataclass(frozen=True)
class ConstraintBlockGridMap:
    """
    Block of constraints to be applied at selected grid indices in a TimeGrid.

    This object is a *static problem specification* (holds Python callables),
    so it is intentionally a stdlib frozen dataclass.

    Kernel contract
    ---------------
    If terminal == False:
        func(t, x, u) -> c
    If terminal == True:
        func(t, x) -> c
        and the constraint is enforced only at the terminal node ``x[nt-1]``.

    In both cases, c must be a scalar or 1D array of length `cdim_out_step`.
    Scalars are interpreted as (1,).

    Attributes
    ----------
    tg : TimeGrid
        Time grid that defines the discrete node indexing ``k = 0, ..., nt - 1``.
        Used to validate `active_steps` and to enforce alignment with trajectories/games.
    func : callable
        Constraint kernel (either step or terminal signature depending on `terminal`).
    cdim_out_step : int
        Output dimension per enforced step.
    active_steps : tuple[int, ...] | None
        Grid indices where the constraint is enforced. Must be strictly increasing
        and unique. For non-terminal blocks these must lie on the stage grid
        ``0, ..., nsteps - 1`` because a control value ``u[k]`` is required.
        Terminal blocks are enforced only at the terminal node ``k = nt - 1``.
        If terminal == True and active_steps is None, defaults to ``(tg.nt - 1,)``.
    iseq : bool
        False => inequality: c(...) <= 0
        True  => equality:   c(...) == 0
    terminal : bool
        If True: func signature is ``(t, x)`` and the constraint is enforced
        only at the terminal node.

    Notes
    -----
    - Each instance of this class must implicitly "know" the correct joint-state and joint-control
      vector indexing; e.g. which state variables correspond to which index in the joint state, etc.
      However, since such indexing is problem/game-specific, we do not attempt to
      encode or enforce this knowledge explicitly within this lower-level dataclass 
    """
    tg: TimeGrid
    func: Union[StepConstraintFn, TerminalConstraintFn]
    cdim_out_step: int
    active_steps: Optional[Tuple[int, ...]] = None
    iseq: bool = False
    terminal: bool = False

    def __post_init__(self):
        # tg
        if not isinstance(self.tg, TimeGrid):
            raise TypeError(f"tg must be TimeGrid, got {type(self.tg)}")

        # func
        if not callable(self.func):
            raise TypeError(f"func must be callable, got {type(self.func)}")

        # cdim_out_step
        if not isinstance(self.cdim_out_step, int) or self.cdim_out_step <= 0:
            raise ValueError(f"cdim_out_step must be a positive int, got {self.cdim_out_step}")

        # active_steps defaulting + validation
        if self.terminal:
            # terminal constraints are terminal-node-only by design
            if self.active_steps is None:
                object.__setattr__(self, "active_steps", (self.tg.nt - 1,))
            if self.active_steps != (self.tg.nt - 1,):
                raise ValueError(
                    f"terminal=True implies active_steps must be exactly the terminal node "
                    f"(nt-1,) = "
                    f"({self.tg.nt - 1},), got {self.active_steps}"
                )
        else:
            if self.active_steps is None:
                # default: active everywhere on the stage grid where u[k] exists
                object.__setattr__(self, "active_steps", tuple(range(self.tg.nsteps)))

        assert self.active_steps is not None  # for type checkers

        if not isinstance(self.active_steps, tuple):
            raise TypeError("active_steps must be a tuple[int, ...] or None")

        prev = -1
        for k in self.active_steps:
            if not isinstance(k, int):
                raise TypeError(f"active_steps entries must be int, got {type(k)}")
            if k < 0 or k >= self.tg.nt:
                raise ValueError(f"active_steps entry {k} out of range [0, {self.tg.nt-1}]")
            if (not self.terminal) and k >= self.tg.nsteps:
                raise ValueError(
                    f"Non-terminal active_steps entry {k} must lie on the stage grid "
                    f"[0, {self.tg.nsteps - 1}] because u[{k}] must exist."
                )
            if k <= prev:
                raise ValueError("active_steps must be strictly increasing (sorted, unique).")
            prev = k

    @property
    def n_active_steps(self) -> int:
        return len(self.active_steps)  # type: ignore[arg-type]

    @property
    def nc_block(self) -> int:
        """Total scalar constraints contributed by this block across the trajectory."""
        return self.cdim_out_step * self.n_active_steps


@dataclass(frozen=True)
class GameConstraintGridMap:
    """
    Container for all auxiliary (non-dynamics) constraints, as constraint blocks.

        C(X,U) = [ C_ineq(X,U) ; C_eq(X,U) ]

    Notes
    -----
    - Reference: Le Cleac'h et al. ALGAMES. Sec IV.A
    - This is not intended to hold discrete-time dynamics constraints; that 
    is generally treated as a different constraint type within a game definition;
    See systemtypes.py and gametypes.py
    - This container is intentionally "paper-like": it does not encode player ownership,
      or state/control-specific bookkeeping.
    - Each block of this class must implicitly "know" the correct joint-state and joint-control
      vector indexing, and, by extension, that implicit knowledge of indexing must be 
      consistent across all blocks. However, since such indexing is problem/game-specific, 
      we do not attempt to encode or enforce this knowledge explicitly within this 
      lower-level dataclass
    - It is "joint" both in the sense that it holds inequality and equality constraints, as
      well as the fact that the constraints are joint to all players within a game;
      i.e. any player-specific constraints are encoded via the constraint functions
      access to particular components of the joint state, x, and/or joint control u
    - This class is defined as a standard `@dataclass(frozen=True)` (stdlib) rather than a
      `flax.struct.dataclass`. The intent is for this object to act as *static problem
      specification* (it holds Python callables and names), not as JAX-traceable state.
    - Using `frozen=True` makes the instance immutable, which helps avoid accidental mutation
      of the constraint set during solves and makes it safer to share across functions.
    """
    ineq_blocks: Tuple[ConstraintBlockGridMap, ...] = ()
    eq_blocks: Tuple[ConstraintBlockGridMap, ...] = ()

    # cached variable / derived from other input arguments
    tg: Optional[TimeGrid] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        for b in self.ineq_blocks:
            if b.iseq:
                raise ValueError("All blocks in ineq_blocks must have iseq=False")
        for b in self.eq_blocks:
            if not b.iseq:
                raise ValueError("All blocks in eq_blocks must have iseq=True")

        # enforce identical tg across all blocks + cache it
        tgs = [b.tg for b in self.ineq_blocks] + [b.tg for b in self.eq_blocks]
        if not tgs:
            object.__setattr__(self, "tg", None)
            return

        tg0 = tgs[0]
        for tg in tgs[1:]:
            if tg != tg0:
                raise ValueError("All blocks must share the same TimeGrid")

        object.__setattr__(self, "tg", tg0) 

    @property
    def nc_blocks_ineq(self) -> int:
        return len(self.ineq_blocks)
    
    @property
    def nc_blocks_eq(self) -> int:
        return len(self.eq_blocks)

    @property
    def nc_ineq(self) -> int:
        return sum(b.nc_block for b in self.ineq_blocks)

    @property
    def nc_eq(self) -> int:
        return sum(b.nc_block for b in self.eq_blocks)

    @property
    def nc_all(self) -> int:
        return self.nc_ineq + self.nc_eq


@dataclass(frozen=True)
class ConstraintStepLinearization:
    """
    One constraint-kernel instance evaluated and linearized at a single grid index.

    This represents a single slice of the *trajectory-stacked* constraint vector
    ``C(X,U)``, i.e. a specific block ``b`` at a specific active grid index ``k``,
    including:
      - c_k     : constraint value (cdim,)
      - Jx_k    : ∂c/∂x at that step (cdim, nx)
      - Ju_k    : ∂c/∂u at that step (cdim, nu) or None for terminal-only kernels
      - sl      : slice into the flat multiplier/penalty vectors (λ or ρ) corresponding
                 to this kernel instance in the chosen stacking order.

    It is “linearization” because it stores (c, Jx, Ju), which is exactly what you need
    to assemble gradients/hessians of augmented-Lagrangian terms without re-diffing.

    Fields
    ------
    kind : {"ineq","eq"}
        Which stack this instance belongs to (aligns with λ_ineq/ρ_ineq or λ_eq/ρ_eq).
    k : int
        Time grid index where the kernel is enforced/linearized.
    terminal : bool
        True if kernel is terminal-only c(t,x) (no control Jacobian); False for c(t,x,u).
    cdim : int
        Output dimension of this kernel instance (number of scalar constraints).
    c : jnp.ndarray, shape (cdim,)
        Constraint value at (t_k, x_k, u_k) or (t_k, x_k) if terminal.
    Jx : jnp.ndarray, shape (cdim, nx)
        Jacobian ∂c/∂x evaluated at the operating point.
    Ju : Optional[jnp.ndarray], shape (cdim, nu) or None
        Jacobian ∂c/∂u evaluated at the operating point (None if terminal=True).
    sl : slice
        Slice into the corresponding flattened λ/ρ vector selecting the (cdim,) entries
        for this instance in the canonical stacking order.
    """
    kind: ConstraintKind
    k: int
    terminal: bool
    cdim: int
    c: jnp.ndarray
    Jx: jnp.ndarray
    Ju: Optional[jnp.ndarray]
    sl: slice


def _normalize_constraint_output_1d(c_val: jnp.ndarray, expected_dim: int) -> jnp.ndarray:
    """
    Normalize a constraint-kernel output to a 1D vector of fixed length.

    Parameters
    ----------
    c_val : jnp.ndarray
        Raw output of a constraint kernel. May be a scalar (0D) or a 1D array.
    expected_dim : int
        Required output length after normalization.

    Returns
    -------
    c : jnp.ndarray, shape (expected_dim,)
        1D constraint value vector. Scalar outputs are promoted to shape (1,).

    Raises
    ------
    ValueError
        If `c_val` is not scalar/1D, or if its length does not equal `expected_dim`.
    """
    c_val = jnp.asarray(c_val)
    if c_val.ndim == 0:
        c_val = c_val[None]
    if c_val.ndim != 1:
        raise ValueError(f"Constraint kernel must return scalar or 1D array, got shape {c_val.shape}")
    if int(c_val.shape[0]) != expected_dim:
        raise ValueError(f"Constraint kernel output dim mismatch: expected {expected_dim}, got {c_val.shape[0]}")
    return c_val


def _linearize_step_constraint_kernel(
    func: StepConstraintFn,
    t: float,
    x: jnp.ndarray,
    u: jnp.ndarray,
    cdim_out_step: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Evaluate and linearize a non-terminal constraint kernel c(t, x, u).

    Parameters
    ----------
    func : StepConstraintFn
        Constraint kernel with signature func(t, x, u) returning a scalar or 1D array.
    t : float
        Time at which the kernel is evaluated (treated as constant for differentiation).
    x : jnp.ndarray, shape (nx,)
        Joint state at the operating point.
    u : jnp.ndarray, shape (nu,)
        Joint control at the operating point.
    cdim_out_step : int
        Expected output dimension of the kernel after normalization.

    Returns
    -------
    c : jnp.ndarray, shape (cdim_out_step,)
        Constraint value at (t, x, u), normalized to 1D.
    Jx : jnp.ndarray, shape (cdim_out_step, nx)
        Jacobian ∂c/∂x evaluated at (t, x, u).
    Ju : jnp.ndarray, shape (cdim_out_step, nu)
        Jacobian ∂c/∂u evaluated at (t, x, u).

    Raises
    ------
    ValueError
        If the kernel output is not scalar/1D or does not match `cdim_out_step`.
    """
    def c_of_xu(x_, u_):
        return _normalize_constraint_output_1d(func(t, x_, u_), cdim_out_step)

    c = c_of_xu(x, u)
    Jx, Ju = jax.jacfwd(c_of_xu, argnums=(0, 1))(x, u)
    return c, Jx, Ju


def _linearize_terminal_constraint_kernel(
    func: TerminalConstraintFn,
    t: float,
    x: jnp.ndarray,
    cdim_out_step: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Evaluate and linearize a terminal-only constraint kernel c(t, x).

    Parameters
    ----------
    func : TerminalConstraintFn
        Terminal constraint kernel with signature func(t, x) returning a scalar or 1D array.
    t : float
        Time at which the kernel is evaluated (treated as constant for differentiation).
    x : jnp.ndarray, shape (nx,)
        Joint state at the operating point.
    cdim_out_step : int
        Expected output dimension of the kernel after normalization.

    Returns
    -------
    c : jnp.ndarray, shape (cdim_out_step,)
        Constraint value at (t, x), normalized to 1D.
    Jx : jnp.ndarray, shape (cdim_out_step, nx)
        Jacobian ∂c/∂x evaluated at (t, x).

    Raises
    ------
    ValueError
        If the kernel output is not scalar/1D or does not match `cdim_out_step`.
    """
    def c_of_x(x_):
        return _normalize_constraint_output_1d(func(t, x_), cdim_out_step)

    c = c_of_x(x)
    Jx = jax.jacfwd(c_of_x)(x)
    return c, Jx


def build_constraint_step_linearizations(
    constraints: GameConstraintGridMap, 
    op: FixedStepPrimalDualTrajectory,
) -> Tuple[Tuple[ConstraintStepLinearization, ...], Tuple[ConstraintStepLinearization, ...]]:
    """
    Expand constraint blocks into per-step linearizations at a given operating point.

    Parameters
    ----------
    constraints : GameConstraintGridMap
        Constraint specification containing inequality and equality blocks, each with
        `active_steps`, `cdim_out_step`, and a kernel (step or terminal).
    op : FixedStepPrimalDualTrajectory
        Operating point providing the time grid and trajectories:
          - ``op.tg`` (TimeGrid)
          - ``op.xs`` with shape ``(nt, nx)``
          - ``op.us`` with shape ``(nsteps, nu)``

    Returns
    -------
    ineq_lins : tuple[ConstraintStepLinearization, ...]
        Linearizations for the inequality stack, in canonical stacking order
        (block order, then active_steps order).
    eq_lins : tuple[ConstraintStepLinearization, ...]
        Linearizations for the equality stack, in canonical stacking order.

    Raises
    ------
    ValueError
        If constraint and trajectory TimeGrids do not match, or if a non-terminal block
        includes the terminal node index ``nt - 1``.
    """
    nt = op.nt
    xs, us = op.xs, op.us

    if constraints.tg is not None and constraints.tg != op.tg:
        raise ValueError(f"constraints TimeGrid != trajectory TimeGrid: {op.tg} vs {constraints.tg}")

    ts = compute_ts(op.tg)

    def build(kind: ConstraintKind, blocks: Tuple) -> Tuple[ConstraintStepLinearization, ...]:
        out = []
        ptr = 0
        for b in blocks:
            for k in b.active_steps:
                sl = slice(ptr, ptr + b.cdim_out_step)
                ptr += b.cdim_out_step

                if b.terminal:
                    c, Jx = _linearize_terminal_constraint_kernel(b.func, ts[k], xs[k], b.cdim_out_step)
                    out.append(ConstraintStepLinearization(kind, k, True, b.cdim_out_step, c, Jx, None, sl))
                else:
                    if k == nt - 1:
                        raise ValueError("Non-terminal block has active terminal node nt-1; use terminal=True instead.")
                    c, Jx, Ju = _linearize_step_constraint_kernel(b.func, ts[k], xs[k], us[k], b.cdim_out_step)
                    out.append(ConstraintStepLinearization(kind, k, False, b.cdim_out_step, c, Jx, Ju, sl))
        return tuple(out)

    return build("ineq", constraints.ineq_blocks), build("eq", constraints.eq_blocks)


def accumulate_Jt_weighted_vector(
    lins: Tuple[ConstraintStepLinearization, ...],
    w_flat: jnp.ndarray,
    nt: int,
    nx: int,
    nu: int,
    dtype,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Accumulate trajectory-shaped gradients from a stack of constraint linearizations.

    For each linearization instance `li` at step k with Jacobians (Jx, Ju) and a
    corresponding weight vector w = w_flat[li.sl] of shape (li.cdim,), this adds:
        dX[k] += Jxᵀ @ w
        dU[k] += Juᵀ @ w     (only if li.terminal is False)

    Parameters
    ----------
    lins : tuple[ConstraintStepLinearization, ...]
        Per-step constraint values/Jacobians in canonical stacking order.
    w_flat : jnp.ndarray, shape (total_dim,)
        Flattened weight vector aligned with the same stacking order as `lins`.
        Typically λ (linear term) or ρ ⊙ c (quadratic penalty term).
    nt : int
        Number of state nodes (``op.xs.shape[0]``).
    nx : int
        Joint state dimension (op.xs.shape[1]).
    nu : int
        Joint control dimension (op.us.shape[1]).
    dtype : any
        Dtype used to initialize the output arrays.

    Returns
    -------
    dX : jnp.ndarray, shape (nt, nx)
        Accumulated contributions to the state gradient at each state node.
    dU : jnp.ndarray, shape (nt-1, nu)
        Accumulated contributions to the control gradient at each control interval.
    """
    K = nt - 1
    dX = jnp.zeros((nt, nx), dtype=dtype)
    dU = jnp.zeros((K,  nu), dtype=dtype)

    for li in lins:
        w = w_flat[li.sl]
        dX = dX.at[li.k].add(li.Jx.T @ w)
        if not li.terminal:
            dU = dU.at[li.k].add(li.Ju.T @ w)  # type: ignore[union-attr]
    return dX, dU


# ----------------------------
# ARCHIVED, UNUSED SOURCE CODE
# ----------------------------

# @dataclass(frozen=True)
# class BasicConstraint:
#     """
#     Minimal constraint definition for continuous-time games.

#     The constraint is defined by a callable:
#         c = func(t, x, u)

#     where `c` is either a scalar or a vector of shape (cdim,) of constraint values.

#     Conventions
#     -----------
#     - If `iseq == False` (default): this is an inequality constraint of the form
#           c(t, x, u) <= 0
#       (componentwise if vector-valued).
#     - If `iseq == True`: this is an equality constraint of the form
#           c(t, x, u) == 0

#     Notes
#     -----
#     - This is not intended to hold discrete-time dynamics constraints; that 
#     is generally treated as a different constraint type within a game definition;
#     See systemtypes.py and gametypes.py
#     - This is a stdlib dataclass (not a flax struct) because it holds Python
#       callables and is intended as static problem specification, not JAX-traceable state.
#     - The callable should be JAX-compatible (pure, side-effect free) and return
#       a fixed-shape output for all valid inputs.
#     - The output must have a fixed shape for all valid inputs. Scalar outputs
#       are normalized to shape (1,) via `evaluate()`.
#     """
#     func: Callable[[float, jnp.ndarray, jnp.ndarray], jnp.ndarray]
#     iseq: bool = False

# def evaluate_constraint_step_no_checks(
#     con: BasicConstraint, 
#     t: float, 
#     x: jnp.ndarray, 
#     u: jnp.ndarray
# ) -> jnp.ndarray:
#     """
#     JAX-safe constraint evaluation at a single step.
#     Always returns shape (q,) with scalar normalized to (1,).
#     No Python-side validation (safe for jit/vmap/scan).

#     Args
#     ----
#     con : BasicConstraint
#         constraint object to be evalutated
#     t : float
#         time (absolute, not index in series) at which constraint function is to be evaluated
#     x : jnp.ndarray, shape (nx,)
#         joint system state at which constraint function is to be evaluated
#     u : jnp.ndarray, shape (nu,)
#         joint system control at which constraint function is to be evaluted

#     Returns
#     -------
#     c : jnp.ndarray, shape (cdim,)
#         vector value of constraint function evaluted at tuple (t,x,u), 
#         where cdim is the number of constraints encode by the con.func
#     """
#     return jnp.atleast_1d(jnp.asarray(con.func(t, x, u)))

# def evaluate_constraint_step(
#     con: BasicConstraint, 
#     t: float, 
#     x: jnp.ndarray, 
#     u: jnp.ndarray
# ) -> jnp.ndarray:
#     """
#     Evaluate constraint values at a single step.
#     Always returns shape (cdim,) (scalar constraint is normalized to array shape (1,)).
#     """
#     c = evaluate_constraint_step_no_checks(con=con, t=t, x=x, u=u)
#     if c.ndim == 0:
#         c = c[None]
#     elif c.ndim != 1:
#         raise ValueError(f"Constraint func must return scalar or 1D array, got shape {c.shape}")
#     return c

# def constraint_jacobian_step_no_checks(
#     con: BasicConstraint,
#     t: float,
#     x: jnp.ndarray,
#     u: jnp.ndarray,
# ) -> Tuple[jnp.ndarray, jnp.ndarray]:
#     """
#     Compute Jacobians of constraint values c(t,x,u) w.r.t. x and u at a single step.
#     JAX-safe but not performing any conditional argument checks

#     Args
#     ----
#     con : BasicConstraint
#         constraint object for which jacobian is to be evalutated
#     t : float
#         time (absolute, not index in series) at which constraint jacobian is to be evaluated
#     x : jnp.ndarray, shape (nx,)
#         joint system state at which constraint jacobian is to be evaluated
#     u : jnp.ndarray, shape (nu,)
#         joint system control at which constraint jacobian is to be evaluted

#     Returns
#     -------
#     dc_dx : jnp.ndarray, shape (cdim, nx)
#         partial derivative of constraint with respect to state, evaluated at tuple (t, x, u)
#     dc_du : jnp.ndarray, shape (cdim, nu)
#         partial derivative of constraint with respect to control, evaluated at tuple (t, x, u)


#     Notes
#     -----
#     - Uses forward-mode autodiff (jacfwd), typically a good default for constraints
#       where cdim is often <= nx+nu.
#     - Does not discretize anything; this differentiates the constraint function directly.
#     """
#     # define constraint as function of (x,u) with t fixed
#     def c_of_xu(x_, u_):
#         return evaluate_constraint_step_no_checks(con, t, x_, u_)  # ensures shape (q,)

#     dc_dx, dc_du = jax.jacfwd(c_of_xu, argnums=(0, 1))(x, u)
#     return dc_dx, dc_du

# def constraint_cost_expansion_equality(
#     c: jnp.ndarray,      # (cdim,)
#     J: jnp.ndarray,      # (cdim, nz)
#     lam: jnp.ndarray,    # (cdim,)
#     rho: float,          # scalar penalty
# ) -> Tuple[jnp.ndarray, jnp.ndarray]:
#     """
#     Equality constraint augmented lagrangian expansion:
#       λbar = λ + ρ c
#       grad = J^T λbar
#       hess = ρ J^T J

#     Args
#     ----
#     c : jnp.ndarray, shape (cdim,)
#         numerical values of equality constraint 
#         i.e. not the constraint function in terms of (t, x, u), but rather
#         the value of the constraint function at some particular but here-unspecified
#         (t,x,u)
#     J : jnp.ndarray, shape (cdim, nz)
#         numerical values of equality constraint jacobian with respect to x or u
#         if J = dc_dx, then nz = nx
#         if J = dc_du, then nz = nu
#     lam : jnp.ndarray, shape (cdim,)
#         Lagrange multipliers associated with equality contstraint in AL scheme
#     rho : float
#         scalar penalty of constraint in augmented lagrange formulation
#         Note that Le Cleac'h et al Eqn 5 implies scalar penalty weight for 
#         this constraint block at this step

#     Returns
#     -------
#     grad : jnp.ndarray, shape (nz,)
#         gradient of augmented lagrangian constraint term
#     hess : jnp.ndarray, shape (nz, nz)
#         hessian of augmented lagrangian constraint term

#     References
#     ----------
#     - Le Cleac'h et al. "ALGAMES: a Fast Solver for Constrained Dynamic Games" (2020). Section IV.A, IV.B
#     - Algames.jl: https://github.com/RoboticExplorationLab/Algames.jl/blob/5c779ca3cebb9b3b31ebb7414331b479cc6c3f6e/src/constraints/constraints_methods.jl#L287
#     - Altro.jl [v0.3.0]: https://github.com/RoboticExplorationLab/Altro.jl/blob/v0.3.0/src/augmented_lagrangian/alcosts.jl#L48 
#     """
#     c   = jnp.atleast_1d(c)
#     lam = jnp.atleast_1d(lam)

#     lam_bar = lam + rho * c     # (q,)
#     grad = J.T @ lam_bar        # (nz,)
#     hess = rho * (J.T @ J)      # (nz, nz)
#     return grad, hess


# def constraint_cost_expansion_inequality(
#     c: jnp.ndarray,      # (cdim,)
#     J: jnp.ndarray,      # (cdim, nz)
#     lam: jnp.ndarray,    # (cdim,)
#     rho: float,           # scalar penalty
# ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
#     """
#     Inequality constraint augmented lagrangian expansion with Altro's active rule:
#       a = (c >= 0) | (λ > 0)
#       λbar = λ + ρ (a * c)
#       grad = J^T λbar
#       hess = ρ J^T diag(a) J

#     Args
#     ----
#     c : jnp.ndarray, shape (cdim,)
#         numerical values of inequality constraint 
#         i.e. not the constraint function in terms of (t, x, u), but rather
#         the value of the constraint function at some particular but here-unspecified
#         (t,x,u)
#     J : jnp.ndarray, shape (cdim, nz)
#         numerical values of inequality constraint jacobian with respect to x or u
#         if J = dc_dx, then nz = nx
#         if J = dc_du, then nz = nu
#     lam : jnp.ndarray, shape (cdim,)
#         Lagrange multipliers associated with inequality contstraint in AL scheme
#     rho : float
#         scalar penalty of constraint in augmented lagrange formulation
#         Note that Le Cleac'h et al Eqn 5 implies scalar penalty weight for 
#         this constraint block at this step

#     Returns
#     -------
#     grad : jnp.ndarray, shape (nz,)
#         gradient of augmented lagrangian constraint term
#     hess : jnp.ndarray, shape (nz, nz)
#         hessian of augmented lagrangian constraint term
#     a_float : jnp.ndarray
#         constraint activation indicator vector
#         a[i] == 1 implies constraint i is active, else a[i]==0
#         Inequality constraints assumed in negative orthant form c <= 0, the “violated” region is c > 0
#         active if inquality constraint is violated (c >= 0) OR multiplier already positive (λ > 0)

#     References
#     ----------
#     - Le Cleac'h et al. "ALGAMES: a Fast Solver for Constrained Dynamic Games" (2020). Section IV.A, IV.B
#     - Algames.jl: https://github.com/RoboticExplorationLab/Algames.jl/blob/5c779ca3cebb9b3b31ebb7414331b479cc6c3f6e/src/constraints/constraints_methods.jl#L287
#     - Altro.jl [v0.3.0]: https://github.com/RoboticExplorationLab/Altro.jl/blob/v0.3.0/src/augmented_lagrangian/alcosts.jl#L48
#     """
#     c   = jnp.atleast_1d(c)
#     lam = jnp.atleast_1d(lam)

#     a = jnp.logical_or(c >= 0.0, lam > 0.0)       # (cdim,) bool
#     a_f = a.astype(J.dtype)                       # (cdim,) float

#     lam_bar = lam + rho * (a_f * c)                # (cdim,)
#     grad = J.T @ lam_bar                          # (cdim,)

#     # Avoid forming diag(a): row-scale J by a_f
#     J_active = J * a_f[:, None]                   # (cdim, nz)
#     hess = rho * (J.T @ J_active)                  # (nz, nz)

#     return grad, hess, a_f
