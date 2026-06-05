# Copyright 2026 MIT Lincoln Laboratory
# SPDX-License-Identifier: MIT

# Class and function definitions for various player cost functions

import jax
import jax.numpy as jnp
import numpy as np

from dataclasses import dataclass
from enum import Enum, auto
from functools import partial
from typing import Callable, Optional, List

from pydgens.ir.timetypes import compute_ts
from pydgens.utils.utils import is_positive_semidefinite
from pydgens.ir.trajectorytypes import (
    FixedStepSystemTrajectory, 
    FixedStepPrimalDualTrajectory,
    get_player_control_trajectory
)

# ---- Enums describing "domain" vs "structure" ----

class ControlDomain(Enum):
    """What the user-provided running-cost callable expects for its control argument."""
    JOINT = auto()   # running(t, x, u_joint)
    LOCAL = auto()   # running(t, x, u_i)


class ControlStructure(Enum):
    """
    Declared player-block structure of a running cost with respect to control.

    This enum describes how a player's running cost depends on player-owned
    control blocks. It is more specific than the vague word "coupling":
    the important question for current solvers is whether the quadratic
    approximation with respect to control is block-diagonal by player
    ownership, and in the AL case, whether the cost depends only on the
    owning player's local control.

    - LOCAL_ONLY:
      The running cost depends only on player i's own control block. This is
      the structural assumption required by the current AL formulation.

    - BLOCK_SEPARABLE:
      The running cost may depend on other players' control blocks, but not
      through cross terms between different player-owned blocks. Equivalently,
      the Hessian with respect to the JOINT control vector is block-diagonal
      by player ownership. This is compatible with the current iLQ/LQ stack.

    - GENERAL:
      The running cost may contain mixed terms between different player-owned
      control blocks, so the Hessian with respect to the JOINT control vector
      is not block-diagonal by player ownership.

    - UNKNOWN:
      No structural declaration has been made.
    """
    LOCAL_ONLY = auto()
    BLOCK_SEPARABLE = auto()
    GENERAL = auto()
    UNKNOWN = auto()


# ---- Callable type aliases (informal contracts) ----
PlayerCostFnCtrlJoint = Callable[[float, jnp.ndarray, jnp.ndarray], float]  # (t, x, u_joint)->scalar
PlayerCostFnCtrlLocal = Callable[[float, jnp.ndarray, jnp.ndarray], float]  # (t, x, u_i)->scalar
PlayerCostFnTerminal = Callable[[float, jnp.ndarray], float]                # (t, x)->scalar


@dataclass(frozen=True, init=False)
class PlayerCostSpecContinuous:
    """
    Continuous-time cost specification for a single player.

    This is intended as a *problem-definition* object: it stores Python callables that
    define a player's objective in continuous time. Downstream solvers may discretize
    time "under the hood" and evaluate these functions at sampled times along a
    trajectory.

    Fields
    ------
    running:
        Running cost (or cost-rate). The expected control argument depends on `control_domain`:
        - JOINT: running(t, x, u_joint) -> scalar
        - LOCAL: running(t, x, u_i) -> scalar
    terminal:
        Optional terminal cost terminal(t, x) -> scalar (no control dependence).
    control_domain:
        Whether `running` expects u_joint or u_i.
    control_structure:
        Declared player-block structure of the running cost with respect to
        control. Downstream solvers may require ``LOCAL_ONLY`` or
        ``BLOCK_SEPARABLE`` depending on how they use the quadratic control
        terms.

    Notes
    -----
    - We do NOT try to enforce callable signatures via introspection; it’s brittle.
    - Instead, solvers can validate scalar outputs at runtime (outside jit).
    - `running` and `terminal` must be JAX-compatible callables (pure, side-effect free).
    - These functions should return a scalar (0-d JAX array or Python float).
    - This is a frozen stdlib dataclass, not a flax struct dataclass, because it holds functions
      and not strictly jax arrays
    - This spec is *continuous-time* by intention; if you later introduce discrete-time
      per-step cost objects, prefer a separate class name (e.g. PlayerCostSpecDiscrete).
    """
    running: Callable  # PlayerCostFnCtrlJoint or PlayerCostFnCtrlLocal (domain described below)
    terminal: Optional[PlayerCostFnTerminal] = None
    control_domain: ControlDomain = ControlDomain.JOINT
    control_structure: ControlStructure = ControlStructure.UNKNOWN

    def __init__(
        self,
        running: Callable,
        terminal: Optional[PlayerCostFnTerminal] = None,
        control_domain: ControlDomain = ControlDomain.JOINT,
        control_structure: ControlStructure = ControlStructure.UNKNOWN,
        control_coupling: ControlStructure | None = None,
    ):
        if control_coupling is not None:
            if control_structure is not ControlStructure.UNKNOWN:
                raise ValueError(
                    "Use only one of `control_structure` or deprecated "
                    "`control_coupling`."
                )
            control_structure = control_coupling

        object.__setattr__(self, "running", running)
        object.__setattr__(self, "terminal", terminal)
        object.__setattr__(self, "control_domain", control_domain)
        object.__setattr__(self, "control_structure", control_structure)
        self.__post_init__()

    def __post_init__(self):
        if not callable(self.running):
            raise TypeError(f"running must be callable, got {type(self.running)}")
        if self.terminal is not None and not callable(self.terminal):
            raise TypeError(f"terminal must be callable or None, got {type(self.terminal)}")

    @property
    def control_coupling(self) -> ControlStructure:
        """
        Deprecated compatibility alias for ``control_structure``.
        """
        return self.control_structure
        

# ---- Utilities ----

def _player_slices(u_splits: jnp.ndarray) -> List[slice]:
    """Convert u_splits into per-player slices into u_joint."""
    if u_splits.ndim != 1:
        raise ValueError(f"u_splits must be 1D, got shape {u_splits.shape}")
    if not np.issubdtype(np.asarray(u_splits).dtype, np.integer):
        raise TypeError(f"u_splits must be integer dtype, got {u_splits.dtype}")

    # host-side prefix sums for slicing (static metadata)
    splits = np.asarray(u_splits, dtype=int)
    if np.any(splits < 0):
        raise ValueError("u_splits must be nonnegative")
    offsets = np.concatenate([[0], np.cumsum(splits)])
    return [slice(int(offsets[i]), int(offsets[i + 1])) for i in range(len(splits))]


def get_player_control_vector(u_joint: jnp.ndarray, player_i: int, u_splits: jnp.ndarray) -> jnp.ndarray:
    """Slice player_i's control block u_i from u_joint."""
    sls = _player_slices(u_splits)
    if not (0 <= player_i < len(sls)):
        raise IndexError(f"player_i out of range: {player_i}")
    return u_joint[sls[player_i]]
  
def make_running_cost_joint(
    spec: PlayerCostSpecContinuous,
    player_i: int,
    u_splits: jnp.ndarray,
) -> PlayerCostFnCtrlJoint:
    """
    Return a JOINT-control running cost g(t, x, u_joint) -> scalar, regardless of whether
    `spec.running` was provided in JOINT or LOCAL control domain.
    """
    if spec.control_domain == ControlDomain.JOINT:
        return spec.running  # type: ignore[return-value]

    def g_joint(t: float, x: jnp.ndarray, u_joint: jnp.ndarray) -> float:
        u_i = get_player_control_vector(u_joint, player_i, u_splits)
        return spec.running(t, x, u_i)  # type: ignore[misc]

    return g_joint

def _assert_scalar_output(y, name: str):
    y_arr = jnp.asarray(y)
    if y_arr.ndim != 0:
        raise ValueError(f"{name} must return a scalar (0-d), got shape {y_arr.shape}")


def validate_player_cost_spec_continuous(
    spec: PlayerCostSpecContinuous,
    *,
    t: float,
    x: jnp.ndarray,
    u_joint: jnp.ndarray,
    player_i: int,
    u_splits: jnp.ndarray,
):
    """
    Runtime validation (outside jit) to catch mis-specified cost functions early.
    """
    g_joint = make_running_cost_joint(spec, player_i=player_i, u_splits=u_splits)
    _assert_scalar_output(g_joint(t, x, u_joint), "running")

    if spec.terminal is not None:
        _assert_scalar_output(spec.terminal(t, x), "terminal")


def detect_control_structure(
    running_cost_joint: PlayerCostFnCtrlJoint,
    *,
    t: float,
    x: jnp.ndarray,
    u_joint: jnp.ndarray,
    u_splits: jnp.ndarray,
    tol: float = 1e-6,
) -> ControlStructure:
    """
    Empirically detect joint-control block structure of a running cost.

    We compute the Hessian H = d²/d(u_joint)² running_cost_joint(t,x,u_joint), then check
    whether off-diagonal *player blocks* have any magnitude > tol.

    This is a heuristic check at a sample point ``(t, x, u_joint)``, not a
    global proof. The result is best interpreted as a structural hint for
    solver compatibility rather than a mathematically complete guarantee.

    Returns
    -------
    ControlStructure
        ``BLOCK_SEPARABLE`` if the joint-control Hessian is block-diagonal by
        player ownership, otherwise ``GENERAL``.

    Notes
    -----
    This detector cannot distinguish ``LOCAL_ONLY`` from the broader
    ``BLOCK_SEPARABLE`` case because it only analyzes a JOINT-control running
    cost without knowing which player owns the cost or whether dependence on
    other players' control blocks is present only linearly/nonlinearly within
    each block.
    """
    sls = _player_slices(u_splits)
    if u_joint.ndim != 1:
        raise ValueError(f"u_joint must be 1D, got shape {u_joint.shape}")
    if u_joint.shape[0] != sum((s.stop - s.start) for s in sls):
        raise ValueError("u_joint length must match sum(u_splits)")

    def g_u(u):
        return running_cost_joint(t, x, u)

    H = jax.hessian(g_u)(u_joint)  # shape (nu, nu)

    # check off-diagonal blocks by player
    for a in range(len(sls)):
        for b in range(len(sls)):
            if a == b:
                continue
            block = H[sls[a], sls[b]]
            if jnp.max(jnp.abs(block)) > tol:
                return ControlStructure.GENERAL
    return ControlStructure.BLOCK_SEPARABLE


def compute_quadratic_cost(Q, q, R, r, x, u, *, validate: bool = True):
    """
    Compute a single player's quadratic stage cost:

        0.5 * xᵀ Q x + qᵀ x + 0.5 * uᵀ R u + rᵀ u

    Args:
        Q: (nx, nx) array_like — state quadratic term.
        q: (nx,)    array_like — state linear term.
        R: (nu, nu) array_like — control quadratic term.
        r: (nu,)    array_like — control linear term.
        x: (nx,)    array_like — state vector at this timestep.
        u: (nu,)    array_like — control vector at this timestep.
        validate: if True, perform shape/rank checks (default: True).

    Returns:
        cost (scalar jnp.ndarray): The computed cost.

    Raises:
        ValueError: on shape/rank mismatches when `validate=True`.
        TypeError : if `dt`/`dim`-like inputs can’t be converted (not used here).
    """
    # Convert to JAX arrays
    x = jnp.asarray(x); u = jnp.asarray(u)
    Q = jnp.asarray(Q); q = jnp.asarray(q)
    R = jnp.asarray(R); r = jnp.asarray(r)

    if validate:
        if x.ndim != 1: raise ValueError(f"x must be 1-D, got {x.shape}")
        if u.ndim != 1: raise ValueError(f"u must be 1-D, got {u.shape}")
        nx, nu = x.shape[0], u.shape[0]
        if Q.shape != (nx, nx): raise ValueError(f"Q must be {(nx, nx)}, got {Q.shape}")
        if q.shape != (nx,):    raise ValueError(f"q must be {(nx,)}, got {q.shape}")
        if R.shape != (nu, nu): raise ValueError(f"R must be {(nu, nu)}, got {R.shape}")
        if r.shape != (nu,):    raise ValueError(f"r must be {(nu,)}, got {r.shape}")
        if Q.ndim != 2 or R.ndim != 2:
            raise ValueError("Q and R must be 2-D")

    # Common dtype for numeric stability/consistency
    dtype = jnp.result_type(Q, q, R, r, x, u)
    x = x.astype(dtype); u = u.astype(dtype)
    Q = Q.astype(dtype); q = q.astype(dtype)
    R = R.astype(dtype); r = r.astype(dtype)

    return 0.5 * (x @ (Q @ x)) + (q @ x) + 0.5 * (u @ (R @ u)) + (r @ u)
    
def goal_cost_quadratic(xg, Q=None):
    """
    Constructs Q and q such that the cost function 
    x.T @ Q @ x + 2 * q.T @ x is minimized at x_g.
    
    Args:
        xg: goal state (n-dimensional vector)
        Q: Optional (n x n) positive semidefinite weight matrix (defaults to identity)

    Returns:
        Q: (n x n) matrix
        q: (n,) vector
    """
    xg = jnp.atleast_1d(xg)
    n = xg.shape[0]

    if Q is None:
        Q = jnp.eye(n)
    else:
        if not is_positive_semidefinite(Q):
            raise ValueError("quadratic cost matrix Q must be positive semidefinite")


    q = -Q @ xg

    return Q, q

def _blockdiag_mask(u_splits: jnp.ndarray) -> jnp.ndarray:
    """Boolean mask with ones on the block-diagonal defined by u_splits.
    """
    nu = int(sum(u_splits))
    mask = jnp.zeros((nu, nu), dtype=bool)
    start = 0
    for sz in u_splits:
        end = start + int(sz)
        mask = mask.at[start:end, start:end].set(True)
        start = end
    return mask

def quadraticize_cost_joint_ctrl_no_checks(
        g_i: PlayerCostFnCtrlJoint, 
        t: float,
        x: jnp.ndarray,
        u: jnp.ndarray, 
        mask: jnp.ndarray = None
):
    """Compute Q_i, q_i, R_i, r_i at (t,x,u), using mask to ignore mixed u partials across players 
        with no checks on inputs to avoid jit errors

    Compute jacobian and hessians of player cost g_i at (t, x, u), with respect to x and u_j blocks.

    The gradients and hessians can be used to form a quadratic approximation of the function g_i 
    about point (x0, u0) by forming the second-order Taylor series in the form:
    g_i(x,u) ~= g_i(x0,u0) + 0.5*(x-x0).T @ Q @ (x-x0) + q.T @ (x-x0) + 
                0.5*(u-u0).T @ R @ (u-u0) + r.T @ (u-u0)
    where q is the gradient (Jacobian) of g_i with respect to x, Q is the hessian wrt x,
    r is the jacobian wrt u, and R is the hessian wrt u

    Note that the use of a Taylor series approximation about point (x0, u0) makes for a 
    subtle distinction between the approximation function and the original. For example, 
    if the underlying function g_i happens to be quadratic to begin with and has form:
    g_i(x,u) = 0.5 * x.T @ Qa @ x + qa.T @ x + 0.5 * u.T @ Ra @ u + ra.T @ u
    then the terms of the quadratic approximation are not identical to the terms of the
    underlying actual quadratic function, even though the approximation is exact. 
    In particular q != qa and r != ra. This can be seen if you equate the underlying
    quadratic function with it's taylor series quadratic approximation
    g_i(x,u) = 0.5 * x.T @ Qa @ x + qa.T @ x + 0.5 * u.T @ Ra @ u + ra.T @ u = 
        g_i(x0,u0) + 0.5*(x-x0).T @ Q @ (x-x0) + q.T @ (x-x0) + 0.5*(u-u0).T @ R @ (u-u0) + r.T @ (u-u0)
    Rearranging this equality and matching terms of (constants, x, x^2, u, u^2) you find that:
    Q = Qa
    R = Ra
    q = qa + Qa @ x0
    r = ra + Ra @ u0

    Of course, this function would be oblivious to the underlying form of g_i if it were in fact quadratic
    to begin with, so you can't know Qa, Ra, qa, ra in general (as they may not even exist for nonquadratic g_i),
    but this distinction is important with unit testing this function given a known g_i; i.e. don't expect
    q and r to match the terms qa and ra

    Parameters:
    - g_i : PlayerCostFnCtrlJoint
        cost function of player i in form function(t, x, u) -> scalar
        where u is presumed to be the joint control vector
        and the cost function is presumed to be separable by 
        player controls (i.e. no mixed partials of player controls in hessian)
    - t : float 
        time at which cost is quadraticized
    - x : jnp.ndarray of size (nx,)
        joint state vector at which point cost is quadraticized
    - u : jnp.ndarray of size (nu,)
        joint control vector at which point cost is quadraticized
    - mask : jnp.ndarray of size (nu, nu)
        mask to ignore mixed partials in R_i and produce block-diagonal matrix

    Returns:
    - Q_i : jnp.ndarray of size (nx,nx)
        hessian of player i's cost with respect to state: ∂²g_i/∂x²
        which describes the quadratic term of player-i's cost 
        function with respect to joint state x (size (nx,))
    - q_i : jnp.ndarray of size (nx,)
        gradient of player i's cost func with respect to state: ∂g_i/∂x
        which describes the linear term of player-i's cost 
        function with respect to joint state x (size (nx,))
    - R_i : jnp.ndarray of size (nu, nu)
        hessians of player i's cost wrt to each player j's control: ∂²g_i/∂u_j²
        which is a block-diagnonal matrix describing the quadratic term of player-i's cost 
        function with respect to joint control vector u (size (nu,))
    - r_i : jnp.ndarray of size (nu,) 
        gradients of player i's cost wrt to each player j's control:  ∂g_i/∂u_j
        which describes the linear term of player-i's cost 
        function with respect to joint control vector u (size (nu,))

    Notes:
    - mixed partials are ignored. See Sec IV of https://arxiv.org/pdf/1909.04694
    """
    # q_i = ∂g/∂x ; Q_i = ∂²g/∂x²
    g_wrt_x = lambda x_: g_i(t, x_, u)
    q_i = jax.grad(g_wrt_x)(x)
    Q_i = jax.hessian(g_wrt_x)(x)  # hessian == jacfwd(jacrev)

    # r_i = ∂g/∂u ; R_full = ∂²g/∂u² (includes mixed blocks)
    g_wrt_u = lambda u_: g_i(t, x, u_)
    r_full = jax.grad(g_wrt_u)(u)
    R_full = jax.hessian(g_wrt_u)(u)

    # keep only block-diagonal pieces per u_splits
    R_i = jnp.where(mask, R_full, 0)
    r_i = r_full  # no need to split/concat

    return Q_i, q_i, R_i, r_i

def quadraticize_cost_joint_ctrl_playerwise(
        g_i: PlayerCostFnCtrlJoint, 
        t: float,
        x: jnp.ndarray,
        u: jnp.ndarray, 
        u_splits: jnp.ndarray,
        mask: jnp.ndarray = None
):
    """Wrapper of quadraticize_no_checks that applies input checking
    """
    
    # check input shapes
    assert sum(u_splits) == u.shape[0] 

    # produce mask if none given
    # Note: computationally more efficient to compute mask once and pass is input
    if mask is None:
        mask = _blockdiag_mask(u_splits)

    return quadraticize_cost_joint_ctrl_no_checks(g_i=g_i, t=t, x=x, u=u, mask=mask)

# jitted function takes g_i explicitly as arg 0 (static), arrays as dynamic args
@partial(jax.jit, static_argnums=(0,))
def _quad_traj_jit(g_i, ts, xs, us, mask):
    def quad_one(t, x, u):
        return quadraticize_cost_joint_ctrl_no_checks(g_i=g_i, t=t, x=x, u=u, mask=mask)
    return jax.vmap(quad_one, in_axes=(0, 0, 0))(ts, xs, us)

def quadraticize_cost_joint_ctrl_playerwise_trajectory(
    g_i: PlayerCostFnCtrlJoint, 
    op: FixedStepSystemTrajectory,
    u_splits: jnp.ndarray
):
    """Compute running-cost quadratic approximations along a trajectory.

    This routine is intentionally stage-indexed: it evaluates the running cost
    on each control interval using ``(t_k, x_k, u_k)`` for
    ``k = 0, ..., nsteps - 1``. It does not attempt to encode a terminal-only
    state cost at ``x[-1]``.
    
    Parameters:
    - g_i : PlayerCostFnCtrlJoint
        cost function of player i in form function(t, x, u) -> scalar
        where u is presumed to be the joint control vector
        and the cost function is presumed to be separable by 
        player controls (i.e. no mixed partials of player controls in hessian)
    - op : FixedStepSystemTrajectory 
        trajectory about which quadratic approximated
    - u_splits: jnp.ndarray of length N. 
        Lengths of each u_j block defining each player's portion of the joint control vector

    Returns:
    - Q_i : jnp.ndarray of size (nsteps,nx,nx)
        hessian of player i's cost with respect to state: ∂²g_i/∂x²
        at each running-cost evaluation point along trajectory op
    - q_i : jnp.ndarray of size (nsteps,nx)
        gradient of player i's cost func with respect to state: ∂g_i/∂x
        at each running-cost evaluation point along trajectory op
    - R_i : jnp.ndarray of size (nsteps,nu,nu)
        hessians of player i's cost wrt to each player j's control: ∂²g_i/∂u_j²
        at each running-cost evaluation point along trajectory op
        Note that mixed partials are ignored. 
        See Sec IV of https://arxiv.org/pdf/1909.04694
    - r_i : jnp.ndarray of size (nsteps,nu) 
        gradients of player i's cost wrt to each player j's control:  ∂g_i/∂u_j
        at each running-cost evaluation point along trajectory op
    """
    # (light checks outside jit)
    nu_exp = sum(int(s) for s in u_splits)
    nu_act = int(op.us.shape[-1])
    if nu_act != nu_exp:
        raise ValueError(f"u_splits sum ({nu_exp}) != control dim ({nu_act})")

    mask = _blockdiag_mask(u_splits)         # (nu, nu) JAX array
    ts   = compute_ts(op.tg)                 # (nt,)
    Q_i, q_i, R_i, r_i = _quad_traj_jit(g_i, ts[:-1], op.xs[:-1], op.us, mask)
    return Q_i, q_i, R_i, r_i

def gradient_cost_local_ctrl_no_checks(
    g_i: PlayerCostFnCtrlLocal, 
    t: float,
    x: jnp.ndarray,
    u_i: jnp.ndarray, 
):
    """Compute gradient of cost function with respect to joint-state and player-control at (t,x,u_i)
        with no checks on inputs to avoid jit errors

    q_i is the gradient (jacobian) of g_i with respect to joint-state evaluated at x,
    r_i is the jacobian of g_i with respect to player-control evaluated at u_i

    Note that this differs from functions like quadraticize_cost_no_checks in that
    it is assumed that player cost g_i depends on the joint-state, but not on the 
    joint control vector u; rather it only depends on player i's control u_i.
    See https://www.roboticsproceedings.org/rss16/p091.pdf Section III

    Parameters:
    - g_i : PlayerCostFnCtrlLocal
        cost function of player i in form function(t, x, u_i) -> scalar
        assumes cost function only depends upon player i's control input
    - t : float 
        time at which cost gradient is evaluated
    - x : jnp.ndarray of size (nx,)
        joint state vector at which point cost gradient is evaluated
    - u_i : jnp.ndarray of size (nu,)
        player i's control vector at which point cost gradient is evaluated

    Returns:
    - q_i : jnp.ndarray of size (nx,)
        gradient of cost func with respect to joint state: ∂g/∂x
        evaluated at joint state x (size (nx,))
    - r_i : jnp.ndarray of size (nu,) 
        gradient of cost func wrt to player i's control:  ∂g/∂u_i
        evaluated at player i's control u_i (size (nu,))

    """

    # q = ∂g/∂x 
    g_wrt_x = lambda x_: g_i(t, x_, u_i)
    q_i = jax.grad(g_wrt_x)(x)

    # r = ∂g/∂u
    g_wrt_u_i = lambda u_i_: g_i(t, x, u_i_)
    r_i = jax.grad(g_wrt_u_i)(u_i)

    return q_i, r_i

def gradient_terminal_cost_no_checks(
    gterm_i: PlayerCostFnTerminal, 
    t: float,
    x: jnp.ndarray,
):
    """Compute gradient of terminal cost function with respect to joint state at (t,x)
        with no checks on inputs to avoid jit errors

    qterm_i is the gradient (jacobian) of player i's terminal cost
    with respect to state evaluated at joint state x

    Parameters:
    - gterm_i : PlayerCostFnTerminal
        terminal cost function in form function(t, x) -> scalar
        Note: by definition of terminal cost, there is no dependence upon control u
    - t : float 
        time at which terminal cost gradient is evaluated
    - x : jnp.ndarray of size (nx,)
        joint state vector at which point terminal cost gradient is evaluated

    Returns:
    - qterm_i : jnp.ndarray of size (nx,)
        gradient of cost func with respect to joint state: ∂g/∂x
        evaluated at joint state x (size (nx,))

    """

    # q = ∂g/∂x 
    gterm_wrt_x = lambda x_: gterm_i(t, x_)
    qterm_i = jax.grad(gterm_wrt_x)(x)

    return qterm_i

# jitted function takes g_i explicitly as arg 0 (static), arrays as dynamic args
@partial(jax.jit, static_argnums=(0,))
def _grad_cost_traj_jit(g_i, ts, xs, us):
    def grad_one(t, x, u_i):
        return gradient_cost_local_ctrl_no_checks(g_i=g_i, t=t, x=x, u_i=u_i)
    return jax.vmap(grad_one, in_axes=(0, 0, 0))(ts, xs, us)

def gradient_cost_local_ctrl_playerwise_trajectory(
        costfn_i: PlayerCostFnCtrlLocal, 
        termfn_i: PlayerCostFnTerminal,
        op: FixedStepPrimalDualTrajectory,
        player_i: int,
        u_splits: jnp.ndarray 
        ):
    """Compute gradient of a single players cost function at all points along trajectory

    gradients are computed with respect to joint state and control variables
    
    Note that this function is currently only intended to work with a 
    FixedStepPrimalDualTrajectory which has unequal state and control 
    trajectories of length nt and nt-1, respectively. This accounts
    for the fact that the final state in the trajectory depends 
    upon the state and control at the previous step, but control 
    at the final state does not have a meaning since there are no
    further states to effect. 
    
    Parameters:
    - costfn_i : PlayerCostFnCtrlLocal
        running cost function of player i in form function(t, x, u_i) -> scalar
        assumes cost function only depends upon player i's control input
    - termfn_i : PlayerCostFnTerminal
        terminal cost function of player i in form function(t, x) -> scalar
        Note: by definition of terminal cost, there is no dependence upon control u
    - op : FixedStepPrimalDualTrajectory
        trajectory for which gradient is computed at each sample point
    - player_i : int
        player index i for parsing player i's controls from joint control trajectory (0-indexed)
    - u_splits: jnp.ndarray of length N. 
        Lengths of each u_j block defining each player's portion of the joint control vector

    Returns:
    - qs_i : jnp.ndarray of size (nt, nx)
        gradient of player i cost func with respect to joint state: ∂g/∂x
        at each of the nt points along trajectory op
    - rs_i : jnp.ndarray of size (nt-1, nui) 
        gradient of player i cost func wrt to player-i's local control:  ∂g/∂ui
        at each of the nt points along trajectory op
    """

    # arg checks outside of jit
    # enforce only FixedStepPrimalDualTrajectory to ensure proper shaping 
    # of state and control trajectories
    if not isinstance(op, FixedStepPrimalDualTrajectory):
        raise ValueError(f"Invalid trajectory type op. Expected FixedStepPrimalDualTrajectory, got {type(op)}")

    # compute the time vector of the trajectory
    ts   = compute_ts(op.tg)    # (nt,)

    # parse player-i's control trajectory from joint control trajectory
    us_i = get_player_control_trajectory(op, player_i=player_i, u_splits=u_splits)
    
    # initialize jacobian (gradient) vectors for across trajectory
    qs_i = jnp.zeros_like(op.xs)

    # compute cost gradient at all steps except final
    qs_i_part, rs_i = _grad_cost_traj_jit(costfn_i, ts[:-1], op.xs[:-1], us_i)
    qs_i = qs_i.at[:-1].set(qs_i_part)

    # compute cost gradient at final step only wrt to state
    qs_i = qs_i.at[-1].set(gradient_terminal_cost_no_checks(
        gterm_i=termfn_i, 
        t=ts[-1],
        x=op.xs[-1])
    )

    return qs_i, rs_i
