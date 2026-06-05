# Copyright 2026 MIT Lincoln Laboratory
# SPDX-License-Identifier: MIT

import jax.numpy as jnp
from typing import Callable

def is_block_diagonal(R: jnp.ndarray, u_splits: list[int], atol=1e-8) -> bool:
    """
    Check if a matrix R is block diagonal with block sizes given by u_split.

    Args:
        R (jnp.ndarray): square matrix of shape (m, m)
        u_splits (list[int]): list of block sizes that sum to m
        atol (float): absolute tolerance for zero comparison

    Returns:
        bool: True if R is block diagonal with the given block sizes, False otherwise.
    """
    if R.ndim != 2:
        raise ValueError("R must be a 2D matrix")
    m, n = R.shape
    if m != n:
        raise ValueError("R must be square")
    if sum(u_splits) != m:
        raise ValueError("Sum of u_split must match matrix size")

    start_i = 0
    for size_i in u_splits:
        start_j = 0
        for size_j in u_splits:
            block = R[start_i:start_i+size_i, start_j:start_j+size_j]
            if start_i != start_j:
                if not jnp.allclose(block, 0.0, atol=atol):
                    return False
            start_j += size_j
        start_i += size_i

    return True

def is_positive_semidefinite(A, tol=1e-8):
    """Check if A is positive semidefinite via eigenvalues."""
    if not jnp.allclose(A, A.T, atol=tol):
        return False  # Must be symmetric

    eigvals = jnp.linalg.eigvalsh(A)  # Use eigvalsh for symmetric/hermitian matrices
    return jnp.all(eigvals >= -tol)  # Allow small numerical negative values

def euler_step(
    f: Callable[[float, jnp.ndarray, jnp.ndarray], jnp.ndarray],
    t: float,
    x: jnp.ndarray,
    u: jnp.ndarray,
    dt: float,
) -> jnp.ndarray:
    """
    Perform a single forward Euler integration step with zero-order hold control.

    Advances the continuous-time dynamics
        ẋ = f(t, x, u)
    forward by one integration substep of duration `dt`:

        x₊ = x + dt * f(t, x, u)

    Parameters
    ----------
    f : Callable[[float, jnp.ndarray, jnp.ndarray], jnp.ndarray]
        Continuous-time dynamics function `f(t, x, u)` returning ẋ of shape (nx,).
    t : float
        Current continuous time.
    x : jnp.ndarray of shape (nx,)
        Current state vector.
    u : jnp.ndarray of shape (nu,)
        Constant control input applied during this substep.
    dt : float
        Substep integration duration.

    Returns
    -------
    jnp.ndarray of shape (nx,)
        State vector advanced by one Euler substep, i.e. `x₊`.
    """
    return x + dt * f(t, x, u)

def rk2_step(
    f: Callable[[float, jnp.ndarray, jnp.ndarray], jnp.ndarray],
    t: float,
    x: jnp.ndarray,
    u: jnp.ndarray,
    dt: float,
) -> jnp.ndarray:
    """
    Perform a single second-order Runge–Kutta (RK2) integration step with zero-order hold control
    using the classical midpoint method.

    Advances the continuous-time dynamics
        ẋ = f(t, x, u)
    forward by one integration substep of duration `dt`:

        k₁ = f(t,          x,              u)
        k₂ = f(t + dt/2,   x + dt/2 * k₁,  u)

        x₊ = x + dt * k₂

    Parameters
    ----------
    f : Callable[[float, jnp.ndarray, jnp.ndarray], jnp.ndarray]
        Continuous-time dynamics function `f(t, x, u)` returning ẋ of shape (nx,).
    t : float
        Current continuous time.
    x : jnp.ndarray of shape (nx,)
        Current state vector.
    u : jnp.ndarray of shape (nu,)
        Constant control input applied during this substep.
    dt : float
        Substep integration duration.

    Returns
    -------
    jnp.ndarray of shape (nx,)
        State vector advanced by one RK2 substep, i.e. `x₊`.

    Notes
    -----
    - This is the classical midpoint variant of RK2.
    - Purely functional and JAX-compatible.
    """
    k1 = f(t, x, u)
    k2 = f(t + 0.5 * dt, x + 0.5 * dt * k1, u)
    return x + dt * k2

def rk3_step(
    f: Callable[[float, jnp.ndarray, jnp.ndarray], jnp.ndarray],
    t: float,
    x: jnp.ndarray,
    u: jnp.ndarray,
    dt: float,
) -> jnp.ndarray:
    """
    Perform a single third-order Runge–Kutta (RK3) integration step with zero-order hold control.

    Advances the continuous-time dynamics
        ẋ = f(t, x, u)
    forward by one integration substep of duration `dt` using the classical
    third-order explicit Runge–Kutta scheme:

        k1 = f(t,               x,                    u)
        k2 = f(t + dt/2,        x + dt/2 * k1,        u)
        k3 = f(t + dt,          x - dt * k1 + 2*dt*k2, u)

        x₊ = x + dt/6 * (k1 + 4*k2 + k3)

    Parameters
    ----------
    f : Callable[[float, jnp.ndarray, jnp.ndarray], jnp.ndarray]
        Continuous-time dynamics function `f(t, x, u)` returning ẋ of shape (nx,).
        Must be JAX-differentiable and side-effect free.
    t : float
        Current continuous time.
    x : jnp.ndarray of shape (nx,)
        Current state vector.
    u : jnp.ndarray of shape (nu,)
        Constant control input applied during this substep.
    dt : float
        Substep integration duration.

    Returns
    -------
    jnp.ndarray of shape (nx,)
        State vector advanced by one RK3 substep, i.e. `x₊`.

    Notes
    -----
    - This function performs no bounds or stability checks.
    - Purely functional: does not modify `x` in place.
    - Safe to use within `jax.jit` or `lax.scan`; all operations are JAX-compatible.
    """
    k1 = f(t, x, u)
    k2 = f(t + 0.5 * dt, x + 0.5 * dt * k1, u)
    k3 = f(t + dt, x - dt * k1 + 2.0 * dt * k2, u)
    return x + (dt / 6.0) * (k1 + 4.0 * k2 + k3)

def rk4_step(
        f: Callable[[float, jnp.ndarray, jnp.ndarray], jnp.ndarray], 
        t: float,
        x: jnp.ndarray,
        u: jnp.ndarray,
        dt: float):
    """
    Perform a single fourth-order Runge–Kutta (RK4) integration step with zero-order hold control.

    NOTE: function and docstring based largely on LLM code optimization 

    This function advances the continuous-time system dynamics
    ẋ = f(t, x, u)
    forward by one integration substep of duration `dt`.

    The classical RK4 method computes intermediate slopes k₁–k₄ and combines them as:

        k₁ = f(t, x, u)
        k₂ = f(t + dt/2, x + dt/2 * k₁, u)
        k₃ = f(t + dt/2, x + dt/2 * k₂, u)
        k₄ = f(t + dt,   x + dt * k₃,   u)

        x₊ = x + dt/6 * (k₁ + 2k₂ + 2k₃ + k₄)

    Parameters
    ----------
    f : Callable[[float, jnp.ndarray, jnp.ndarray], jnp.ndarray]
        Continuous-time dynamics function `f(t, x, u)` returning ẋ of shape (nx,).
        Must be JAX-differentiable and side-effect free.
    t : float
        Current continuous time.
    x : jnp.ndarray of shape (nx,)
        Current state vector.
    u : jnp.ndarray of shape (nu,)
        Constant control input applied during this substep.
    dt : float
        Substep integration duration.

    Returns
    -------
    jnp.ndarray of shape (nx,)
        State vector advanced by one RK4 substep, i.e. `x₊`.

    Notes
    -----
    - This function performs no bounds or stability checks.
    - Purely functional: does not modify `x` in place.
    - Safe to use within `jax.jit` or `lax.scan`; all operations are JAX-compatible.
    """
    k1 = f(t, x, u)
    k2 = f(t + 0.5 * dt, x + 0.5 * dt * k1, u)
    k3 = f(t + 0.5 * dt, x + 0.5 * dt * k2, u)
    k4 = f(t + dt,       x + dt * k3,       u)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
