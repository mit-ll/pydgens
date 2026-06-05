# Copyright 2026 MIT Lincoln Laboratory
# SPDX-License-Identifier: MIT

# Collection of data/function generators that can, for example,
# be used to deterministically generate functions and data for 
# pytesting

import jax
import jax.numpy as jnp

from typing import Callable, Dict, Tuple

from pydgens.ir.timetypes import TimeGrid
from pydgens.ir.trajectorytypes import FixedStepSystemTrajectory
from pydgens.ir.strategytypes import FixedStepAffineStrategies

def make_random_dynamics(
    nx: int,
    nu: int,
    seed: int = 0,
    lin_scale: float = 0.8,     # strength of linear A,B parts
    nonlin_scale: float = 0.2,  # strength of smooth nonlinearities
    damping: float = 0.3,       # shifts A's spectrum left: A - damping*I
) -> Tuple[Callable[[float, jnp.ndarray, jnp.ndarray], jnp.ndarray], Dict[str, jnp.ndarray]]:
    """
    Returns a continuous-time dynamics function f_ct(t, x, u) -> xdot and its params.
    The dynamics are smooth (C^∞) and mildly nonlinear:
        xdot = A x + B u
               + nonlin_scale * [ P1 @ sin(Mx @ x) + P2 @ tanh(Mu @ u) + x ⊙ (G @ u) ]
    """
    key = jax.random.PRNGKey(seed)
    kA, kB, kMx, kMu, kP1, kP2, kG = jax.random.split(key, 7)

    # Linear terms
    A0 = jax.random.normal(kA, (nx, nx)) * (lin_scale / jnp.sqrt(nx))
    A  = A0 - damping * jnp.eye(nx)  # encourage stability
    B  = jax.random.normal(kB, (nx, nu)) * (lin_scale / jnp.sqrt(nu))

    # Smooth nonlinear feature projections
    Mx = jax.random.normal(kMx, (nx, nx)) * (1.0 / jnp.sqrt(nx))
    Mu = jax.random.normal(kMu, (nx, nu)) * (1.0 / jnp.sqrt(nu))
    P1 = jax.random.normal(kP1, (nx, nx)) * (1.0 / jnp.sqrt(nx))
    P2 = jax.random.normal(kP2, (nx, nx)) * (1.0 / jnp.sqrt(nx))
    G  = jax.random.normal(kG,  (nx, nu)) * (1.0 / jnp.sqrt(nu))

    def f_ct(t: float, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        lin = A @ x + B @ u
        z_x = jnp.sin(Mx @ x)      # smooth in x
        z_u = jnp.tanh(Mu @ u)     # smooth in u
        cross = x * (G @ u)        # elementwise interaction
        nl = P1 @ z_x + P2 @ z_u + cross
        return lin + nonlin_scale * nl

    params = dict(A=A, B=B, Mx=Mx, Mu=Mu, P1=P1, P2=P2, G=G)
    return f_ct, params

def make_random_cost_fn(nx, nu, u_splits, seed=0):
    """
    Returns a list [g_0, ..., g_{N-1}] of callables g_i(t, x, u) -> scalar,
    each sharing the same random-but-fixed weights (deterministic via seed).
    """
    key = jax.random.PRNGKey(seed)
    N = len(u_splits)

    # random but fixed weights; scale to keep values tame
    k1, k2, k3, k4 = jax.random.split(key, 4)
    C = jax.random.normal(k1, (N, nx)) * 0.2          # player-specific state projections
    M = jax.random.normal(k2, (N, nx, nx)) * 0.05      # mild x coupling in sin()
    W_u = jax.random.normal(k3, (N, nu)) * 0.2         # player-specific control weights
    v_u = jax.random.normal(k4, (N, nu)) * 0.1         # phases in tanh

    # precompute control index slices per player
    starts = []
    s = 0
    for m in u_splits:
        starts.append((s, s + m))
        s += m

    def g_i_factory(i):
        s0, s1 = starts[i]

        def g_i(t, x, u):
            # state terms
            x_proj = x @ C[i]                     # (scalar)
            smooth_x = jnp.sin((x @ (M[i] @ x)))  # smooth nonlinearity in x
            # control terms (player i’s block emphasized, but depend on full u for masking tests)
            u_i = u[s0:s1]
            # smooth control nonlinearity; softplus^2 is C^2 and cheap
            smooth_u = jnp.sum(jax.nn.softplus(u + 0.1)**2)
            # mild saturation-like term to avoid blow-ups
            tanh_u = jnp.sum(jnp.tanh(u + v_u[i])**2)

            # combine with gentle weights (pick scales so magnitudes are ~O(1))
            return (
                0.5 * (x_proj**2)                # quadratic in x
                + 0.05 * smooth_x                # smooth nonlinearity in x
                + 0.5 * jnp.sum((W_u[i, s0:s1] * u_i)**2)     # per-player quadratic effort
                + 0.02 * smooth_u + 0.02 * tanh_u             # smooth, non-quadratic in u
            )

        return g_i

    return [g_i_factory(i) for i in range(N)]

def make_random_trajectory(nt, nx, nu, dt=0.1, seed=0, scale_x=1.0, scale_u=0.5):
    """
    Returns a FixedStepSystemTrajectory containing randomized data (deterministic by seed),
    i.e. does not adhere to any dynamics or control strategy propagation
    """
    key = jax.random.PRNGKey(seed)
    kx, ku = jax.random.split(key)
    xs = jax.random.normal(kx, (nt, nx)) * scale_x
    us = jax.random.normal(ku, (nt - 1, nu)) * scale_u
    tg = TimeGrid(nt=nt, dt=dt)
    return FixedStepSystemTrajectory(tg=tg, xs=xs, us=us)

def make_random_strategy(
    nt: int,
    nx: int,
    nu: int,
    dt: float=0.1,
    seed: int = 0,
    p_scale: float = 0.1,
    alpha_scale: float = 0.1,
) -> FixedStepAffineStrategies:
    """
    Generates a random affine feedback strategy u = -P x - alpha.

    Parameters
    ----------
    nt : int
        Number of time-grid nodes.
    nx : int
        Dimension of the joint state.
    nu : int
        Dimension of the joint control.
    dt : float, optional
        Time step size in time grid (default 0.1)
    seed : int, optional
        Random seed for reproducibility (default 0).
    p_scale : float, optional
        Scaling factor for feedback gains P.
    alpha_scale : float, optional
        Scaling factor for bias terms alpha.

    Returns
    -------
    FixedStepAffineStrategies
        Dataclass containing random strategy tensors P and alpha.
    """
    key = jax.random.PRNGKey(seed)
    kP, kA = jax.random.split(key)
    tg = TimeGrid(nt=nt, dt=dt)

    # Random feedback matrices and biases
    P = jax.random.normal(kP, (tg.nsteps, nu, nx)) * p_scale
    alpha = jax.random.normal(kA, (tg.nsteps, nu)) * alpha_scale

    return FixedStepAffineStrategies(tg=tg, P=P, alpha=alpha)
