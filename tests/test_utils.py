# Copyright 2026 MIT Lincoln Laboratory
# SPDX-License-Identifier: MIT

import pytest
import jax.numpy as jnp
from pydgens.utils.utils import (
    is_block_diagonal, 
    is_positive_semidefinite,
    euler_step,
    rk2_step,
    rk3_step,
    rk4_step,
)

def test_block_diagonal_true():
    R = jnp.block([
        [jnp.eye(2), jnp.zeros((2, 3)), jnp.zeros((2, 1))],
        [jnp.zeros((3, 2)), 2 * jnp.eye(3), jnp.zeros((3, 1))],
        [jnp.zeros((1, 2)), jnp.zeros((1, 3)), jnp.eye(1)]
    ])
    u_splits = [2, 3, 1]
    assert is_block_diagonal(R, u_splits)

def test_block_diagonal_not_square():
    R = jnp.ones((3, 4))
    u_splits = [3]
    with pytest.raises(ValueError, match="R must be square"):
        is_block_diagonal(R, u_splits)

def test_block_diagonal_bad_usplit_sum():
    R = jnp.eye(4)
    u_splits = [2, 1]  # sum is 3, not 4
    with pytest.raises(ValueError, match="Sum of u_split must match matrix size"):
        is_block_diagonal(R, u_splits)

def test_block_diagonal_false():
    R = jnp.block([
        [jnp.eye(2), jnp.ones((2, 1))],
        [jnp.zeros((1, 2)), jnp.eye(1)]
    ])
    u_splits = [2, 1]
    assert not is_block_diagonal(R, u_splits)

def test_positive_definite_matrix_2x2():
    A = jnp.array([[2.0, -1.0],
                   [-1.0, 2.0]])
    assert is_positive_semidefinite(A)

def test_positive_semidefinite_matrix_2x2():
    A = jnp.array([[1.0, 0.0],
                   [0.0, 0.0]])
    assert is_positive_semidefinite(A)

def test_indefinite_matrix_2x2():
    A = jnp.array([[0.0, 1.0],
                   [1.0, 0.0]])
    assert not is_positive_semidefinite(A)

def test_negative_definite_matrix_2x2():
    A = jnp.array([[-2.0, 0.0],
                   [0.0, -3.0]])
    assert not is_positive_semidefinite(A)

def test_non_symmetric_matrix_2x2():
    A = jnp.array([[1.0, 2.0],
                   [0.0, 1.0]])
    assert not is_positive_semidefinite(A)

def test_positive_definite_matrix_3x3():
    A = jnp.array([[4.0, 1.0, 1.0],
                   [1.0, 3.0, 0.0],
                   [1.0, 0.0, 2.0]])
    assert is_positive_semidefinite(A)

def test_positive_semidefinite_matrix_3x3_with_zero_eig():
    A = jnp.array([[2.0, -2.0, 0.0],
                   [-2.0, 2.0, 0.0],
                   [0.0,  0.0, 0.0]])
    assert is_positive_semidefinite(A)

def test_indefinite_matrix_3x3():
    A = jnp.array([[0.0, 1.0, 0.0],
                   [1.0, 0.0, 1.0],
                   [0.0, 1.0, 0.0]])
    assert not is_positive_semidefinite(A)

def test_positive_definite_matrix_5x5():
    Q = jnp.array([[5., -2., 0., 0., 0.],
                   [-2., 5., -2., 0., 0.],
                   [0., -2., 5., -2., 0.],
                   [0., 0., -2., 5., -2.],
                   [0., 0., 0., -2., 5.]])
    assert is_positive_semidefinite(Q)

def test_symmetric_but_indefinite_5x5():
    A = jnp.diag(jnp.array([4.0, 3.0, -1.0, 2.0, 0.0]))
    assert not is_positive_semidefinite(A)

@pytest.mark.parametrize("stepfn", [euler_step, rk2_step, rk3_step, rk4_step])
def test_integrator_step_preserves_shape(stepfn):
    """integrator (e.g. rk3, rk4, etc.) step should return an array with the same shape as x."""
    def f(t, x, u):
        # simple linear dynamics: dx/dt = x + 2u
        return x + 2.0 * u

    t = 0.0
    x = jnp.array([1.0, 2.0, 3.0])
    u = jnp.array([0.5, -1.0, 0.0])
    dt = 0.1

    x_next = stepfn(f, t, x, u, dt)

    assert x_next.shape == x.shape

@pytest.mark.parametrize("stepfn", [euler_step, rk2_step, rk3_step, rk4_step])
def test_integrator_step_constant_derivative_is_exact(stepfn):
    """
    For f(t, x, u) = c (constant), integrator (e.g. rk3, rk4) should produce
    x_next = x + dt * c exactly.
    """
    c = jnp.array([0.5, -1.0, 2.0])

    def f(t, x, u):
        return c

    t = 1.23
    x = jnp.array([1.0, 2.0, 3.0])
    u = jnp.array([0.0])  # unused
    dt = 0.7

    x_next = stepfn(f, t, x, u, dt)
    expected = x + dt * c

    assert jnp.allclose(x_next, jnp.array(expected), atol=1e-12, rtol=1e-12)

@pytest.mark.parametrize("stepfn,tol", [
    (euler_step, 1e-2), 
    (rk2_step, 1e-4), 
    (rk3_step, 1e-6), 
    (rk4_step, 1e-7)
    ])
def test_integrator_step_scalar_linear_system_matches_exponential(stepfn, tol):
    """
    For scalar ODE x' = a x, the exact solution is x_next = exp(a*dt) * x.
    integrator (e.g. rk3, rk4, etc.) should approximate this very closely for reasonable dt.
    """
    a = -0.7

    def f(t, x, u):
        # x is shape (1,), keep shape consistent
        return a * x

    t = 0.0
    x0 = jnp.array([1.5])
    u = jnp.array([0.0])  # unused
    dt = 0.1

    x_next = stepfn(f, t, x0, u, dt)
    expected = jnp.exp(a * dt) * x0

    assert jnp.allclose(x_next, jnp.array(expected), atol=tol, rtol=tol)

@pytest.mark.parametrize("stepfn", [euler_step, rk2_step, rk3_step, rk4_step])
def test_integrator_step_depends_on_control(stepfn):
    """
    For dynamics x' = x + u (scalar), integrator should produce different results
    for different control inputs.
    """
    def f(t, x, u):
        return x + u  # scalar

    t = 0.0
    x0 = jnp.array([1.0])
    dt = 0.1

    u1 = jnp.array([0.0])
    u2 = jnp.array([1.0])

    x_next_u1 = stepfn(f, t, x0, u1, dt)
    x_next_u2 = stepfn(f, t, x0, u2, dt)

    # They should differ
    assert not jnp.allclose(x_next_u1, x_next_u2)

    # And the one with larger u should be larger (since derivative is bigger)
    assert x_next_u2[0] > x_next_u1[0]

@pytest.mark.parametrize("stepfn", [euler_step, rk2_step, rk3_step, rk4_step])
def test_integrator_step_zero_dt_is_identity(stepfn):
    """With dt = 0, integrator should return x unchanged."""
    def f(t, x, u):
        return 10.0 * x + 3.0 * u  # arbitrary

    t = 0.0
    x = jnp.array([1.0, -2.0])
    u = jnp.array([0.5, 0.5])
    dt = 0.0

    x_next = stepfn(f, t, x, u, dt)

    assert jnp.allclose(x_next, jnp.array(x), atol=1e-12, rtol=1e-12)
