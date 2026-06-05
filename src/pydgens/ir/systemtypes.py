# Copyright 2026 MIT Lincoln Laboratory
# SPDX-License-Identifier: MIT

# class definitions for various control systems that define system dynamics
from __future__ import annotations

import jax
import jax.numpy as jnp
import warnings

from functools import partial
from jax import lax
from typing import Callable, Tuple
from functools import singledispatch
from flax import struct

import pydgens.utils.utils as U
from pydgens.ir.timetypes import TimeGrid, compute_ts, disc2cont
from pydgens.ir.strategytypes import FixedStepAffineStrategies
from pydgens.ir.trajectorytypes import FixedStepSystemTrajectory, FixedStepPrimalDualTrajectory

# Example registry of integrators you already have / will define
_INTEGRATORS = {
    "euler": U.euler_step,
    "rk2":   U.rk2_step,
    "rk3":   U.rk3_step,
    "rk4":   U.rk4_step,
}


@struct.dataclass
class SampledContinuousSystemType1:
    """
    Control system with continuous-time dynamics sampled at a fixed rate `dt` over a finite horizon.

    Attributes:
        tg (TimeGrid): time characteristics (nt, dt, t0)
        nx (int): Number of state dimensions.
        nu (int): Number of control input dimensions.
        dynamics (Callable[[float, jnp.ndarray, jnp.ndarray], jnp.ndarray]): Function of the form dynamics(t, x, u) that returns dx/dt.
    """
    tg: TimeGrid
    nx: int
    nu: int
    dynamics: Callable[[float, jnp.ndarray, jnp.ndarray], jnp.ndarray]

    def __post_init__(self):
        if not isinstance(self.tg, TimeGrid):
            raise TypeError(f"tg must be TimeGrid, got type {type(self.tg)}")
        if not isinstance(self.nx, int):
            raise TypeError(f"nx must be integer, got type {type(self.nx)}")
        if self.nx <= 0:
            raise ValueError(f"nx must be positive, got {self.nx}")
        if not isinstance(self.nu, int):
            raise TypeError(f"nu must be integer, got type {type(self.nu)}")
        if self.nu <= 0:
            raise ValueError(f"nu must be positive, got {self.nu}")
        if not callable(self.dynamics):
            raise TypeError(f"dynamics must be callable. got type {type(self.dynamics)}")
        
    # convience properties to raise time characteristics to top-level

    @property
    def nt(self):
        # number of time nodes
        return self.tg.nt
    
    @property
    def nsteps(self):
        # number of time steps
        return self.tg.nsteps
    
    @property
    def dt(self):
        # length of time step, [s] by default
        return self.tg.dt
    
    @property
    def t0(self):
        # initial time, [s] by default
        return self.tg.t0

@struct.dataclass
class LinearDiscreteSystemType1:
    """
    Discrete-time linear control system with time-varying dynamics matrices.

    Attributes:
        tg (TimeGrid): time characteristics (nt, dt, t0)
        nx (int): Number of state dimensions (inherited).
        nu (int): Number of control input dimensions (inherited).
        A (jnp.ndarray): State transition matrices of shape (nsteps, nx, nx).
        B (jnp.ndarray): Control matrices of shape (nsteps, nx, nu).

    Notes:
        This class represents a time-varying linear system where the state evolves
        according to: x_{t+1} = A_t x_t + B_t u_t.
    """
    tg: TimeGrid
    nx: int
    nu: int
    A: jnp.ndarray
    B: jnp.ndarray

    def __post_init__(self):
        if not isinstance(self.tg, TimeGrid):
            raise TypeError(f"tg must be TimeGrid, got type {type(self.tg)}")
        if not isinstance(self.nx, int):
            raise TypeError(f"nx must be integer, got type {type(self.nx)}")
        if self.nx <= 0:
            raise ValueError(f"nx must be positive, got {self.nx}")
        if not isinstance(self.nu, int):
            raise TypeError(f"nu must be integer, got type {type(self.nu)}")
        if self.nu <= 0:
            raise ValueError(f"nu must be positive, got {self.nu}")
        if self.A.shape != (self.nsteps, self.nx, self.nx):
            raise ValueError(f"Inconsistent shape for A. Expected {(self.nsteps, self.nx, self.nx)}, got {self.A.shape}")
        if self.B.shape != (self.nsteps, self.nx, self.nu):
            raise ValueError(f"Inconsistent shape for B. Expected {(self.nsteps, self.nx, self.nu)}, got {self.B.shape}")
        
    # convience properties to raise time characteristics to top-level

    @property
    def nt(self):
        # number of time nodes
        return self.tg.nt
    
    @property
    def nsteps(self):
        # number of time steps between time nodes
        return self.tg.nsteps
    
    @property
    def dt(self):
        # length of time step, [s] by default
        return self.tg.dt
    
    @property
    def t0(self):
        # initial time, [s] by default
        return self.tg.t0


@singledispatch
def next_x(cs, t, x, u, nt_int) -> jnp.ndarray:
    """
    Compute next state of system

    Args:
        cs (): control system, this is the type that is dispatched over
        t (float): Current time.
        x (jnp.ndarray): Current state vector of shape (nx,).
        u (jnp.ndarray): Constant control input vector of shape (nu,).
        nt_int (int): number of intermediate timesteps to use in integration 


    """
    raise NotImplementedError

@next_x.register(SampledContinuousSystemType1)
def _next_x(
    cs: SampledContinuousSystemType1, 
    k: int, 
    x: jnp.ndarray, 
    u: jnp.ndarray, 
    nt_int: int = 2
) -> jnp.ndarray:
    """
    DEPRECATED: this function has been directly integrated to propogate_system_trajectory to improve performance
    
    Compute the next state on sampling time grid using RK4 integration of the continuous-time dynamics.

    Note that this integrates forward to the next point on the time grid; from k to k+1

    Args:
        cs (SampledContinuousSystem): control system to integrate
        k (int): Current discrete time step .
        x (jnp.ndarray): Current state vector of shape (nx,).
        u (jnp.ndarray): Constant control input vector of shape (nu,).
        nt_int (int): number of intermediate timesteps to use in integration 

    Returns:
        jnp.ndarray: Next state vector after timestep dt.
    """
    warnings.warn("Deprecation Warning: next_x has been deprecated in favor of an optimized propogate_system_trajectory function")
    if nt_int <= 0:
        raise ValueError("nt_int must be positive.")
    
    if not isinstance(k, int):
        raise TypeError(f"k must be an integer, got {type(k)}")
     
        
    f = cs.dynamics
    dt = cs.tg.dt / nt_int

    # compute the continuous floating point time at discrete time k
    t = disc2cont(k=k, tg=cs.tg)

    x_next = x
    for i in range(nt_int):
        ti = t + i * dt
        k1 = f(ti, x_next, u)
        k2 = f(ti + dt/2, x_next + dt/2*k1, u)
        k3 = f(ti + dt/2, x_next + dt/2*k2, u)
        k4 = f(ti + dt, x_next + dt*k3, u)
        x_next = x_next + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    return x_next

@singledispatch
def propagate_system_trajectory(
    cs,
    x0: jnp.ndarray,
    strategy: FixedStepAffineStrategies,
) -> FixedStepSystemTrajectory:
    """
    Propagates a trajectory from the given initial state using an affine feedback control strategy.

    Args:
        cs (): control system to be propagated
        x0 (jnp.ndarray): Initial state vector of shape (nx,).
        strategy (FixedStepAffineStrategies): An AffineStrategy with time-varying
            feedback matrices `P` of shape (nsteps, nu, nx) and `alpha`
            vectors of shape (nsteps, nu), defining control as
            u_t = -P_t @ x_t - alpha_t

    Returns:
        SystemTrajectory
    """
    raise NotImplementedError

def _integrate_over_substeps(f, t0, x0, u, dt, nt_int: int):
    """
    Integrate a continuous-time system forward over several fixed substeps
    using the RK4 integration scheme.

    NOTE: function and docstring based largely on LLM code optimization 

    This performs multiple RK4 substeps starting at time `t0`
    and returns the state at the end of the interval
    (typically corresponding to one discrete-time sample of the system).

    Mathematically, it integrates:

        x_{k+1} = x_k + ∫_{t_k}^{t_k + nt_int*sub_dt} f(t, x(t), u) dt
    using `nt_int` equally spaced RK4 substeps of size `sub_dt`.

    Parameters
    ----------
    f : Callable[[float, jnp.ndarray, jnp.ndarray], jnp.ndarray]
        Continuous-time dynamics function `f(t, x, u)` returning ẋ of shape (nx,).
    t0 : float
        Starting continuous time for this integration window.
    x0 : jnp.ndarray of shape (nx,)
        Initial state at time `t0`.
    u : jnp.ndarray of shape (nu,)
        Constant control input maintained during all substeps.
    sub_dt : float
        Integration substep duration (seconds).
    nt_int : int
        Number of RK4 substeps per discrete integration window.
        Must be a positive integer known at trace time (static in JIT).

    Returns
    -------
    jnp.ndarray of shape (nx,)
        State vector at the end of the integration window after all substeps.

    Notes
    -----
    - Implemented using `jax.lax.fori_loop` for JIT-compatibility and fusion.
    - The control input `u` is held constant during integration.
    - Ensures identical input/output types for use inside `lax.scan`.
    """
    def body(i, x_curr):
        ti = t0 + i * dt
        return U.rk4_step(f, ti, x_curr, u, dt)
    return lax.fori_loop(0, nt_int, body, x0)

@partial(jax.jit, static_argnums=(0, 6))  # f and nt_int static
def _prop_scan(f, cs_t0, dt_grid, P, alpha, x0, nt_int: int):
    """
    Propagate the state of a continuous-time control system over a fixed time grid
    using a time-varying affine feedback control strategy.

    This function performs the full forward rollout of a system defined by
    ẋ = f(t, x, u)
    with control law
    u_t = -P_t @ x_t - alpha_t,
    where P_t and alpha_t vary over discrete time indices t = 0, ..., nsteps-1.

    Integration between grid points uses `nt_int` RK4 substeps per step.
    The function is compiled with JIT and uses `jax.lax.scan` to unroll
    all discrete time steps efficiently in a single XLA computation.

    Parameters
    ----------
    f : Callable[[float, jnp.ndarray, jnp.ndarray], jnp.ndarray]
        Continuous-time dynamics function `f(t, x, u)` returning ẋ (shape (nx,)).
        Treated as static at compile time.
    t0 : float
        Starting time of the trajectory.
    dt_grid : float
        Sampling interval of the discrete time grid.
    P : jnp.ndarray of shape (nsteps, nu, nx)
        Sequence of feedback gain matrices, one per control interval.
    alpha : jnp.ndarray of shape (nsteps, nu)
        Sequence of bias vectors, one per control interval.
    x0 : jnp.ndarray of shape (nx,)
        Initial state at time `t0`.
    nt_int : int
        Number of RK4 substeps per discrete grid step (static).

    Returns
    -------
    xs : jnp.ndarray of shape (nt, nx)
        State trajectory samples on the discrete time grid.
    us : jnp.ndarray of shape (nsteps, nu)
        Control inputs applied over each control interval.

    Notes
    -----
    - Internally uses `jax.lax.scan` to propagate over time steps:
          (x_{t+1}, (x_t, u_t)) = step(x_t, t)
      where `step` computes the control, integrates, and returns the next state.
    - Marked with `@jax.jit(static_argnums=(0, 6))` so that `f` (the Python callable)
      and `nt_int` are treated as static compile-time arguments.
      Each distinct `f` or `nt_int` will trigger a separate compilation.
    - Designed for reuse in multiple trajectory-propagation utilities; safe for benchmarking.
    """
    sub_dt = dt_grid / nt_int
    nsteps = P.shape[0]
    nu = P.shape[1]

    # Even though P/alpha are not "static arguments" in the static_argnums
    # sense, standard JAX tracing still knows their array shapes. That means
    # P.shape[0] is available as a Python integer during tracing, so this
    # shape-based branch is resolved before the scan body is traced. In other
    # words, this relies on JAX's usual specialization on input shapes, not on
    # static_argnums for P/alpha.
    if nsteps == 0:
        dtype = jnp.result_type(x0, P, alpha)
        return x0[None, :], jnp.empty((0, nu), dtype=dtype)

    def one_step(carry, t):
        x = carry
        # time at step t on the sampling grid
        t_grid = cs_t0 + t * dt_grid
        # control from affine strategy
        u = -P[t] @ x - alpha[t]
        # integrate RK4 over nt_int substeps
        x_next = _integrate_over_substeps(f, t_grid, x, u, sub_dt, nt_int)
        # record (x,u) at grid instant t (before stepping)
        return x_next, (x, u)

    xT, (xs, us) = lax.scan(one_step, x0, jnp.arange(nsteps))
    xs = jnp.concatenate([xs, xT[None, :]], axis=0)
    return xs, us

@propagate_system_trajectory.register(SampledContinuousSystemType1)
def _propagate_system_trajectory(cs: SampledContinuousSystemType1,
      x0: jnp.ndarray,
      strategy: FixedStepAffineStrategies,
      nt_int: int = 2) -> FixedStepSystemTrajectory:
    """
    Propagates a trajectory from the given initial state using an affine feedback control strategy.

    Args:
        cs (SampledContinuousSystemType1): A SampledContinuousSystem representing the control system dynamics.
        x0 (jnp.ndarray): Initial state vector of shape (nx,).
        strategy (FixedStepAffineStrategies): An AffineStrategy with time-varying
            feedback matrices `P` of shape (nsteps, nu, nx) and `alpha`
            vectors of shape (nsteps, nu), defining control as
            u_t = -P_t @ x_t - alpha_t

    Returns:
        FixedStepSystemTrajectory

    Notes: 
    - This uses an RK4 discretization scheme to propagate the continuous-time system. 
        To more fully integrate, this could be generalized to use an arbitrary discretization scheme
        from make_discrete_dynamics_step_map

    """

    if strategy.tg != cs.tg:
        raise ValueError(f"Inconsistent time characteristics. cs.tg={cs.tg}, strategy.tg={strategy.tg}")
    if strategy.nx != cs.nx or strategy.nu != cs.nu:
        raise ValueError("Inconsistent state/control dimensions.")
    if x0.shape != (cs.nx,):
        raise ValueError(f"Inconsistent initial state dimensions. Expected {(cs.nx,)}, got {x0.shape}")
    if nt_int <= 0:
        raise ValueError("nt_int must be positive.")

    xs, us = _prop_scan(
        cs.dynamics,                    # static arg
        cs.tg.t0,
        cs.tg.dt,
        strategy.P,                     # (nsteps, nu, nx)
        strategy.alpha,                 # (nsteps, nu)
        x0,
        nt_int                          # static arg
    )
    return FixedStepSystemTrajectory(tg=cs.tg, xs=xs, us=us)

@propagate_system_trajectory.register(LinearDiscreteSystemType1)
def _propagate_system_trajectory(cs: LinearDiscreteSystemType1,
      x0: jnp.ndarray,
      strategy: FixedStepAffineStrategies) -> FixedStepSystemTrajectory:
    """
    Propagates a trajectory from the given initial state using an affine feedback control strategy.

    Args:
        cs (LinearDiscreteSystemType1): The discrete-time system to be propagated
        x0 (jnp.ndarray): Initial state vector of shape (nx,).
        strategy (FixedStepAffineStrategies): An AffineStrategy with time-varying
            feedback matrices `P` of shape (nsteps, nu, nx) and `alpha`
            vectors of shape (nsteps, nu), defining control as
            u_t = -P_t @ x_t - alpha_t

    Returns:
        FixedStepSystemTrajectory

    Notes: Based on LLM-optimized code recommendations
    """

    if strategy.tg != cs.tg:
        raise ValueError(f"Inconsistent time characteristics. cs.tg={cs.tg}, strategy.tg={strategy.tg}")
    if strategy.nx != cs.nx or strategy.nu != cs.nu:
        raise ValueError("Inconsistent state/control dimensions.")
    if x0.shape != (cs.nx,):
        raise ValueError(f"Inconsistent initial state dimensions. Expected {(cs.nx,)}, got {x0.shape}")
    
    tg = cs.tg
    nsteps = tg.nsteps
    nx = cs.nx
    nu = cs.nu

    xs = []
    us = []

    x = x0
    xs.append(x)

    for k in range(nsteps):
        Pk = strategy.P[k]        # (nu,nx)
        ak = strategy.alpha[k]    # (nu,)
        u = -(Pk @ x) - ak
        us.append(u)

        Ak = cs.A[k]
        Bk = cs.B[k]
        x = Ak @ x + Bk @ u
        xs.append(x)


    xs = jnp.stack(xs, axis=0)
    if nsteps == 0:
        us = jnp.empty((0, nu), dtype=x0.dtype)
    else:
        us = jnp.stack(us, axis=0)
    
    return FixedStepSystemTrajectory(tg=cs.tg, xs=xs, us=us)

# f is a Python function; mark it static so it goes into the cache key ONCE.
@partial(jax.jit, static_argnums=(0,))
def _linearize_batch(f, ts, xs, us):
    def jac_xu(t, x, u):
        df_dx, df_du = jax.jacfwd(lambda x_, u_: f(t, x_, u_), argnums=(0, 1))(x, u)
        return df_dx, df_du
    return jax.vmap(jac_xu, in_axes=(0, 0, 0))(ts, xs, us)

def linearize_dynamics(
    f: Callable[[float, jnp.ndarray, jnp.ndarray], jnp.ndarray], 
    op: FixedStepSystemTrajectory, 
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute jacobians matrices A and B of nonlinear dynamics at each point along a trajectory

    Parameters:
    - f : Callable
        nonlinear dynamics function taking (t, x, u)
    - op : FixedStepSystemTrajectory
        operating point (i.e. trajectory) about which sytem is linearized

    Returns:
    - A: Jacobians w.r.t. x, shape (nsteps, nx, nx)
    - B: Jacobians w.r.t. u, shape (nsteps, nx, nu)
    """

    # check input type, this wasn't designed for Primal-Dual trajectories
    if not isinstance(op, FixedStepSystemTrajectory):
        raise TypeError(f"linearize_dynamics function not defined for trajectory type {type(op)}")

    ts = compute_ts(op.tg)      # shape (nt,)
    # Only pass **arrays** into the jitted function:
    return _linearize_batch(f, ts[:-1], op.xs[:-1], op.us)

@singledispatch
def make_discrete_dynamics_step_map(
    cs, 
    method:str,
) -> Callable[[float, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """
    Create a integrator step function for continuous dynamics at a single time step

    Given a continuous-time system of form:
        dx/dt = f(t, x, u)
    
    This function returns the discrete-step approximator function
        x_{t+1} = f_d(t, x_t, u_t)

    Note that this returns just a dynamics function, not a whole control system, because
    the discrete dynamics are only valid at the particular time step whereas a control system
    is meant to be a time-extended description of a system, potentially with time-varying
    dynamics

    Note that this does NOT bake in the exact (time, state, control), rather it is a 
    wrapper to pass back a very general, single-step integrator like Runge-Kutta 4. 
    It does "bake in" the time step size dt taken from the control system object. This
    layer serves the purpose of clarifying that these integrators are meant for use
    in a dynamics discretization process, something that is not obvious with the name
    "integrator". It also serves to make different integrators interchangeable, easily
    selecting between euler, rk2, rk3, rk4, etc.

    Parameters
    ----------
    cs : Any
        Continuous-time system object; dispatch is based on its type.
    method : str
        Name of integration method used ("euler", "rk2", "rk3", "rk4", ...).

    Returns
    -------
    f_d : Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
        Discrete dynamics function at this time step:
            x_next = f_d(x, u)

    """
    raise NotImplementedError(f"No make_discrete_dynamics_step_map implementation for {type(cs)}")

@make_discrete_dynamics_step_map.register(SampledContinuousSystemType1)
def _make_discrete_dynamics_step_map(
    cs: SampledContinuousSystemType1,
    method:str,
) -> Callable[[float, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """
    Discretize a SampledContinuousSystemType1 at a single time step.

    Uses the system's continuous dynamics f(t, x, u) and its time grid dt
    together with the chosen integration method.
    """
    # Continuous dynamics
    f = cs.dynamics          # f(t, x, u) -> xdot
    dt = cs.dt     # or cs.dt, depending on your design

    # Pick integrator
    try:
        step = _INTEGRATORS[method]
    except KeyError as e:
        raise ValueError(
            f"Unknown integration method '{method}'. "
            f"Available: {list(_INTEGRATORS.keys())}"
        ) from e

    # Build one-step discrete map with f and dt method baked in
    def f_d(t_: float, x_: jnp.ndarray, u_: jnp.ndarray) -> jnp.ndarray:
        return step(f, t_, x_, u_, dt)

    return f_d

@singledispatch
def residual_discrete_dynamics_trajectory(cs, op: FixedStepPrimalDualTrajectory, method: str) -> jnp.ndarray:
    """
    Compute the discrete dynamics residuals D(X,U) along a fixed-step trajectory.
    """
    raise NotImplementedError

@residual_discrete_dynamics_trajectory.register(SampledContinuousSystemType1)
def _residual_discrete_dynamics_trajectory(
    cs: SampledContinuousSystemType1, 
    op: FixedStepPrimalDualTrajectory, 
    method: str) -> jnp.ndarray:
    """
    Mathematical definition
    -----------------------
    Let the discrete-time dynamics induced by a continuous-time system and an
    integration rule be:

        x_{k+1} = f_d(t_k, x_k, u_k),

    where k = 0,...,K-1 and K = nt-1.

    The per-step dynamics constraint residuals are:

        D_k(X,U) := f_d(t_k, x_k, u_k) - x_{k+1}.

    This function evaluates D_k at each step along the operating point `op` and
    returns the stacked residual array D with shape (K, nx).

    Parameters
    ----------
    cs : SampledContinousSystemType1
        Continuous-time control system object. Must be compatible with
        `make_discrete_dynamics_step_map(cs, method=...)` and must share the same
        TimeGrid as `op` (cs.tg == op.tg).
    op : FixedStepPrimalDualTrajectory
        Operating point trajectory providing:
          - op.tg (TimeGrid)
          - op.xs of shape (nt, nx)
          - op.us of shape (nt-1, nu)
        Only the primal variables (xs, us) are used here; op.ls is ignored.
    method : str
        Discretization method used to construct f_d ("euler", "rk2", "rk3", "rk4", ...).

    Returns
    -------
    D : jnp.ndarray, shape (K, nx)
        Discrete dynamics residuals per step, where K = nt-1.

    Raises
    ------
    ValueError
        If cs.tg != op.tg.

    Notes
    -----
    - D is the feasibility block appended at the end of the Newton residual vector:
          G = [∇L_1; ...; ∇L_N; D]
    - This routine uses the same discretized dynamics map as the rest of the solver;
      it should match whatever `jacobian_discrete_dynamics_trajectory` is using.
    """
    if cs.tg != op.tg:
        raise ValueError("cs and op must share TimeGrid")
    fd = make_discrete_dynamics_step_map(cs, method=method)  # fd(t,x,u)->x_next
    ts = compute_ts(op.tg)
    # predict next states
    x_next_pred = jax.vmap(fd, in_axes=(0, 0, 0))(ts[:-1], op.xs[:-1], op.us)  # (K,nx)
    return x_next_pred - op.xs[1:]  # (K,nx)

@singledispatch
def jacobian_discrete_dynamics_step(
    cs, 
    t:float, 
    x:jnp.ndarray, 
    u:jnp.ndarray, 
    method:str,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute jacobian of discretized system dynamics at a single time step

    Note that this is related/similar to the approx_linear_discrete_system
    in that it both discretizes and linearizes a system's dynamics. It is different
    in that it
    1. performs the operations in a different order. This function discretizes and 
        the dynamics first and then computes the jacobian of the discrete dynamics. 
        approx_linear_discrete_system linearize first and then discretizes
    2. This function works on a single time step, whereas approx_linear_discrete_system
        performs the linearize-then-discretize process over an entire trajectory

    Parameters
    ----------
    cs : Any
        Continuous-time system object; dispatch is based on its type.
    t : float
        Time at which the system is to be discretized.
    x : jnp.ndarray
        Joint state at this time step.
    u : jnp.ndarray
        Joint control at this time step.
    method : str
        Name of integration method used ("euler", "rk2", "rk3", "rk4", ...).

    Returns
    -------
    dfd_dx : jnp.ndarrary shape (nx, nx)
        Jacobian of system dynamics w.r.t. joint state evaluated at (t,x,u)
    dfd_du : jnp.ndarray shape (nx, nu)
        Jacobian of system dynamics w.r.t. joint control evaluated at (t,x,u)

    Notes
    -----
    - Note that this discretize-first, then take jacobian, is needed because the
        General Nash Equilibrium Problem (GNEP, AlGames Sec III) is posed in terms
        of discrete dynamics of form x_{k+1} = f(t_k, x_k, u_k), not in terms of 
        continuous dynamics of form dx_dt = f(t,x,u); however, it is often
        more convinient and intuitive to define the continuous-time dynamics
        as the "entry point" of the problem definition and leave the 
        discretization under the hood.
    - This computes the jacobian with respect to the JOINT control vector,
      not the LOCAL control vector of a particular agent. In fact, the 
      function is not even labeled as "JOINT" becasue control systems
      are mostly ignorant/agnostic to player-ID's and control subvectors.
      This player-agnostic design is on purpose because the higher-level
      game objects (gametypes.py) are responsible for handling player
      information 

    """
    raise NotImplementedError(f"No discrete_jacobian_system_dynamics_step implementation for {type(cs)}")

@jacobian_discrete_dynamics_step.register(SampledContinuousSystemType1)
def _jacobian_discrete_dynamics_step(
    cs:SampledContinuousSystemType1, 
    t:float, 
    x:jnp.ndarray, 
    u:jnp.ndarray, 
    method:str,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute jacobian of discretized dynamics of SampledContinuousSystemType1 at a single time step
    """
    
    # make dynamics discretization function
    fd = make_discrete_dynamics_step_map(cs, method=method)

    # compute jacobian of discrete dynamics at time step
    dfd_dx, dfd_du = jax.jacfwd(lambda x_, u_: fd(t, x_, u_), argnums=(0, 1))(x, u)

    return dfd_dx, dfd_du

@singledispatch
def jacobian_discrete_dynamics_trajectory(
    cs,
    op: FixedStepPrimalDualTrajectory,
    method: str,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute discrete dynamics Jacobians along a trajectory.

    For each step k, define the one-step discrete dynamics map:
        x_{k+1} = f_d(t_k, x_k, u_k)

    This returns per-step Jacobians:
        A_k = ∂f_d/∂x evaluated at (t_k, x_k, u_k)   shape (nx, nx)
        B_k = ∂f_d/∂u evaluated at (t_k, x_k, u_k)   shape (nx, nu)

    Parameters
    ----------
    cs : Any
        Continuous-time system object (dispatch based on type).
    op : FixedStepPrimalDualTrajectory
        Operating point trajectory along which jacobian is computed
    method : str
        Integration method used to construct f_d ("euler", "rk2", "rk3", "rk4", ...).
        Should be treated as static for JIT purposes.

    Returns
    -------
    As : jnp.ndarray, shape (nt-1, nx, nx)
        Per-step Jacobians wrt joint state.
    Bs : jnp.ndarray, shape (nt-1, nx, nu)
        Per-step Jacobians wrt joint control.
    """
    raise NotImplementedError(f"No discrete_dynamics_traj_jacobian implementation for {type(cs)}")


@jacobian_discrete_dynamics_trajectory.register(SampledContinuousSystemType1)
def _jacobian_discrete_dynamics_trajectory(
    cs: SampledContinuousSystemType1,
    op: FixedStepPrimalDualTrajectory,
    method: str,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    
    # ensure time grid alignment between control system and trajectory
    if not cs.tg == op.tg:
        raise ValueError(f"control system and operating point trajectory must have equal time grids. Got {cs.tg} and {op.tg}, respectively")

    # Make the discrete step map once
    fd = make_discrete_dynamics_step_map(cs, method=method)  # fd(t, x, u) -> x_next

    # Jacobian wrt joint (x,u) for a *single* step (t,x,u)
    jac_xu = jax.jacfwd(lambda t, x, u: fd(t, x, u), argnums=(1, 2))

    # Create time stamps from time grid
    ts = compute_ts(cs.tg)

    # Vectorize over the trajectory
    # Inputs are batched over leading axis: ts[k], xs[k], us[k]
    As, Bs = jax.vmap(jac_xu, in_axes=(0, 0, 0))(ts[:-1], op.xs[:-1], op.us)
    return As, Bs

def discretize_extended_linear_dynamics_euler(Ac: jnp.ndarray, Bc: jnp.ndarray, dt: float):
    """
    Discretizes a continuous-time linear dynamics using forward Euler integration.
    This is a special subcase of discretizing system dynamics with a specific integrator
    and across the entire stage horizon nsteps

    Given a continuous-time linear system:
        dx/dt = Ac_t @ x + Bc_t @ u

    This function returns the discrete-time approximation:
        x_{t+1} ≈ A_t @ x_t + B_t @ u_t
    where:
        A_t = I + dt * Ac_t
        B_t = dt * Bc_t

    Args:
    - Ac (jnp.ndarray): Array of shape (nsteps, nx, nx) representing
        time-varying system matrices.
    - Bc (jnp.ndarray): Array of shape (nsteps, nx, nu) representing
        time-varying input matrices.
    - dt (float): Discretization time step.

    Returns:
    - A (jnp.ndarray): Discrete-time system matrix array of shape (nsteps, nx, nx).
    - B (jnp.ndarray): Discrete-time input matrix array of shape (nsteps, nx, nu).
    """
    if dt <= 0:
        raise ValueError(f"time step dt must be positive, got {dt}")
    
    nt, nx, _ = Ac.shape
    _, _, nu = Bc.shape

    I = jnp.eye(nx)
    A = I[None, :, :] + dt * Ac  # shape (T, n, n)
    B = dt * Bc                  # shape (T, n, m)

    return A, B

@singledispatch
def approx_linear_discrete_system(cs, *args, **kwargs):
    """
    Approximate a linear-discrete system from a continuous one.
    Generic entry point; specialized by type.
    """
    raise NotImplementedError(f"No approx_linear_discrete_system implementation for {type(cs)}")

@approx_linear_discrete_system.register(SampledContinuousSystemType1)
def _approx_linear_discrete_system(
    cs: SampledContinuousSystemType1, 
    op: FixedStepSystemTrajectory,
) -> LinearDiscreteSystemType1:
    """
    Convert a sampled continuous-time system into a linear discrete-time system
    by linearizing and discretizing at each time step.

    Note that this approximation creates a subtle, yet meaningful, re-definition
    of the state and control varaible. If the state and control variables of 
    the SampledContinuousSystemType1 are (x, u); then the state and control 
    variables of LinearDiscreteSystemType1 approximation are (delx, delu) where
    delx = x - op.x
    delu = u - op.u
    delx_(t+1) = A @ delx_(t) + B @ delu_(t)

    This distinction is important for correctly interpreting the nash feedback 
    strategies of linear-quadratic games based upon the LinearDiscreteSystemType1
    approximate dynamics, and thus, correctly propagating trajectories based 
    upon these strategies. The distinction is subtle because misinterpreting
    these variables mostly leads to silent errors as they are all of
    consistent shapes

    Args:
    - cs : SampledContinuousSystemType1
        continuous system to be converted to linear-discrete system
    - op : SystemTrajectory
        operating point about which sytem is linearized and discretized
    """
    nx, nu, nt, dt = cs.nx, cs.nu, cs.nt, cs.dt
    dynamics = cs.dynamics

    # linearize system about operating point into continuous linear system
    Ac, Bc = linearize_dynamics(f=dynamics, op=op)

    # discretize continuous linear system using euler integration
    A, B = discretize_extended_linear_dynamics_euler(Ac=Ac, Bc=Bc, dt=dt)

    return LinearDiscreteSystemType1(tg=cs.tg, nx=nx, nu=nu, A=A, B=B)
