# Copyright 2026 MIT Lincoln Laboratory
# SPDX-License-Identifier: MIT

import pytest
import jax
import jax.numpy as jnp

from pydgens.ir.timetypes import TimeGrid, compute_ts
from pydgens.ir.trajectorytypes import (
    FixedStepSystemTrajectory,
    FixedStepPrimalDualTrajectory
)
from pydgens.utils.generators import (
    make_random_cost_fn,
    make_random_trajectory
)

# Import module under test
import pydgens.ir.costtypes as costtypes

def test_player_cost_spec_continuous_defaults():
    spec = costtypes.PlayerCostSpecContinuous(running=lambda t, x, u: 0.0)
    assert spec.control_domain == costtypes.ControlDomain.JOINT
    assert spec.control_structure == costtypes.ControlStructure.UNKNOWN
    assert spec.terminal is None


def test_player_cost_spec_continuous_rejects_noncallable_running():
    with pytest.raises(TypeError, match="running must be callable"):
        costtypes.PlayerCostSpecContinuous(running=123)


def test_player_cost_spec_continuous_rejects_noncallable_terminal():
    with pytest.raises(TypeError, match="terminal must be callable or None"):
        costtypes.PlayerCostSpecContinuous(running=lambda t, x, u: 0.0, terminal=object())


def test_make_running_cost_joint_passthrough_for_joint_domain():
    def g(t, x, u):
        return jnp.sum(u**2)

    spec = costtypes.PlayerCostSpecContinuous(running=g, control_domain=costtypes.ControlDomain.JOINT)
    u_splits = jnp.array([2, 1], dtype=jnp.int32)

    g_joint = costtypes.make_running_cost_joint(spec, player_i=0, u_splits=u_splits)
    assert g_joint is g  # passthrough


def test_make_running_cost_joint_wraps_local_domain_correctly():
    # local running cost depends only on player_i's control (here player 1)
    def g_local(t, x, u_i):
        return jnp.sum(x**2) + 3.0 * jnp.sum(u_i**2)

    spec = costtypes.PlayerCostSpecContinuous(running=g_local, control_domain=costtypes.ControlDomain.LOCAL)
    u_splits = jnp.array([2, 3], dtype=jnp.int32)

    t = 0.0
    x = jnp.array([1.0, -2.0])
    u_joint = jnp.array([10.0, 20.0, 1.0, 2.0, 3.0])  # player0=[10,20], player1=[1,2,3]

    g_joint = costtypes.make_running_cost_joint(spec, player_i=1, u_splits=u_splits)

    u_1 = costtypes.get_player_control_vector(u_joint, player_i=1, u_splits=u_splits)
    assert jnp.allclose(g_joint(t, x, u_joint), g_local(t, x, u_1))


def test_validate_player_cost_spec_continuous_accepts_scalar_outputs():
    spec = costtypes.PlayerCostSpecContinuous(
        running=lambda t, x, u: jnp.sum(x) + jnp.sum(u),
        terminal=lambda t, x: jnp.sum(x**2),
        control_domain=costtypes.ControlDomain.JOINT,
    )
    u_splits = jnp.array([2, 1], dtype=jnp.int32)
    costtypes.validate_player_cost_spec_continuous(
        spec,
        t=0.1,
        x=jnp.array([1.0, 2.0]),
        u_joint=jnp.array([0.5, -0.5, 1.0]),
        player_i=0,
        u_splits=u_splits,
    )


def test_validate_player_cost_spec_continuous_rejects_vector_running_output():
    spec = costtypes.PlayerCostSpecContinuous(
        running=lambda t, x, u: x,  # vector output (bad)
        control_domain=costtypes.ControlDomain.JOINT,
    )
    u_splits = jnp.array([1], dtype=jnp.int32)
    with pytest.raises(ValueError, match="running must return a scalar"):
        costtypes.validate_player_cost_spec_continuous(
            spec,
            t=0.0,
            x=jnp.array([1.0, 2.0]),
            u_joint=jnp.array([0.1]),
            player_i=0,
            u_splits=u_splits,
        )


def test_detect_control_structure_reports_block_separable_for_diagonal_hessian():
    # block-separable: sum of squares -> Hessian diagonal -> off-diagonal
    # player blocks are zero
    def g_joint(t, x, u):
        return jnp.sum(u**2)

    u_splits = jnp.array([2, 2], dtype=jnp.int32)
    out = costtypes.detect_control_structure(
        g_joint,
        t=0.0,
        x=jnp.array([0.0]),
        u_joint=jnp.array([1.0, -2.0, 0.5, 3.0]),
        u_splits=u_splits,
        tol=1e-8,
    )
    assert out == costtypes.ControlStructure.BLOCK_SEPARABLE


def test_detect_control_structure_reports_general_for_cross_term():
    # general: cross-player term u0 * u2 couples player0 idx0 with
    # player1 idx0, producing a nonzero off-diagonal player block
    def g_joint(t, x, u):
        return jnp.sum(u**2) + 0.5 * u[0] * u[2]

    u_splits = jnp.array([2, 2], dtype=jnp.int32)
    out = costtypes.detect_control_structure(
        g_joint,
        t=0.0,
        x=jnp.array([0.0]),
        u_joint=jnp.array([1.0, -2.0, 0.5, 3.0]),
        u_splits=u_splits,
        tol=1e-8,
    )
    assert out == costtypes.ControlStructure.GENERAL

def test_quadraticize_cost_playerwise_1():
    # a simple quadraticization that can be computed by hand

    # ~~ ARRANGE ~~
    x = jnp.array([1.0, 2.0])
    n = len(x)  # number of state dimensions
    u = jnp.array([0.5, -1.0, 1.0])  # u_1 = [0.5, -1.0], u_2 = [1.0]
    m = len(u)  # number of joint control dimensions
    u_splits = [2, 1]
    N = len(u_splits)   # number of players
    t = 0.0

    def simple_quadratic_cost(t, x, u):
        return x @ x + u @ u + 2.0 * t

    # ~~ ACT ~~
    # Run the quadraticization
    Q_i, q_i, R_i, r_i = costtypes.quadraticize_cost_joint_ctrl_playerwise(simple_quadratic_cost, t, x, u, u_splits)

    # ~~ ASSERT ~~
    # Expected results:
    # g(x, u) = x₁² + x₂² + u₁² + u₂² + u₃² + 2t
    # ∂g/∂x = [2x₁, 2x₂] = [2, 4]
    # ∂²g/∂x² = 2I
    # ∂g/∂u₁ = [2u₁, 2u₂] = [1.0, -2.0], ∂²g/∂u₁² = 2I
    # ∂g/∂u₂ = [2u₃] = [2.0], ∂²g/∂u₂² = [[2.0]]

    # check shapes
    assert Q_i.shape == (n,n)
    assert q_i.shape == (n,)

    assert R_i.shape == (m,m)
    assert r_i.shape == (m,)

    # check values
    assert jnp.allclose(Q_i, 2.0 * jnp.eye(2))
    assert jnp.allclose(q_i, jnp.array([2.0, 4.0]))

    j = 0
    jstart = 0
    jend = jstart + u_splits[j]
    assert jnp.allclose(R_i[jstart:jend, jstart:jend], 2.0 * jnp.eye(2))
    assert jnp.allclose(r_i[jstart:jend], jnp.array([1.0, -2.0]))

    jstart = jend
    j = 1
    jend = jstart + u_splits[j]
    assert jnp.allclose(R_i[jstart:jend, jstart:jend], jnp.array([[2.0]]))
    assert jnp.allclose(r_i[jstart:jend], jnp.array([2.0]))

def test_quadraticize_cost_playerwise_trajectory_shapes():

    # ~~ ARRANGE ~~
    nt = 5
    x_dim = 3
    u_dims = [2, 2]  # 2 players
    u_dim = sum(u_dims)
    nsteps = nt - 1

    tg = TimeGrid(nt=nt, dt=1.0 / nt)
    xs = jnp.stack([jnp.ones(x_dim) * i for i in range(nt)])
    us = jnp.stack([jnp.ones(sum(u_dims)) * i for i in range(nsteps)])
    op = FixedStepSystemTrajectory(tg=tg, xs=xs, us=us)

    # Define a simple test cost function
    def g_i(t, x, u):
        # Quadratic cost: xᵀQx + u1ᵀR1u1 + u2ᵀR2u2 + t*xᵀx for variety
        u1, u2 = jnp.split(u, 2)
        Q = jnp.eye(x.shape[0])
        R1 = 2.0 * jnp.eye(u1.shape[0])
        R2 = 0.5 * jnp.eye(u2.shape[0])
        return jnp.dot(x, Q @ x) + jnp.dot(u1, R1 @ u1) + jnp.dot(u2, R2 @ u2) + t * jnp.dot(x, x)

    # ~~ ACT ~~
    Q_i, q_i, R_i, r_i = costtypes.quadraticize_cost_joint_ctrl_playerwise_trajectory(
        g_i, op, u_dims
    )

    # ~~ ASSERT ~~
    assert Q_i.shape == (nsteps, x_dim, x_dim)
    assert q_i.shape == (nsteps, x_dim)

    assert R_i.shape == (nsteps, u_dim, u_dim)
    assert r_i.shape == (nsteps, u_dim)

def test_quadraticize_cost_playerwise_trajectory_values():
    # Sanity check that Hessians of known quadratic cost match expected constants

    x_dim = 2
    u_dims = [1, 1]

    tg = TimeGrid(nt=2, dt=0.1)
    xs = jnp.array([[1.0, 2.0], [0.0, 0.0]])
    us = jnp.array([[3.0, 4.0]])
    op = FixedStepSystemTrajectory(tg=tg, xs=xs, us=us)

    def g_i(t, x, u):
        u1, u2 = jnp.split(u, 2)
        return jnp.dot(x, x) + 2.0 * jnp.dot(u1, u1) + 3.0 * jnp.dot(u2, u2)

    Q_i, q_i, R_i, r_i = costtypes.quadraticize_cost_joint_ctrl_playerwise_trajectory(
        g_i, op, u_dims
    )

    assert jnp.allclose(Q_i[0], 2.0 * jnp.eye(x_dim))
    assert jnp.allclose(R_i[0,:1,:1], 4.0 * jnp.eye(1))
    assert jnp.allclose(R_i[0,1:,1:], 6.0 * jnp.eye(1))

def test_quadraticize_cost_playerwise_trajectory_zero_step_case():
    x_dim = 2
    u_dims = [1, 1]
    u_dim = sum(u_dims)

    tg = TimeGrid(nt=1, dt=0.1)
    xs = jnp.array([[1.0, 2.0]])
    us = jnp.zeros((0, u_dim))
    op = FixedStepSystemTrajectory(tg=tg, xs=xs, us=us)

    def g_i(t, x, u):
        return jnp.dot(x, x) + jnp.dot(u, u)

    Q_i, q_i, R_i, r_i = costtypes.quadraticize_cost_joint_ctrl_playerwise_trajectory(
        g_i, op, u_dims
    )

    assert Q_i.shape == (0, x_dim, x_dim)
    assert q_i.shape == (0, x_dim)
    assert R_i.shape == (0, u_dim, u_dim)
    assert r_i.shape == (0, u_dim)

@pytest.mark.regression
@pytest.mark.slow
@pytest.mark.benchmark(group="costtypes-001")
def test_quadraticize_warm_perf(benchmark):
    """benchmark the warm-started performance of quadraticizing arbitrary cost functions"""
    nt, nx, nu, u_splits = 100, 12, 6, [2, 2, 2]
    g_0 = make_random_cost_fn(nx, nu, u_splits, seed=0)[0]
    traj = make_random_trajectory(nt, nx, nu, seed=3)

    def run():
        out = costtypes.quadraticize_cost_joint_ctrl_playerwise_trajectory(g_i=g_0, op=traj, u_splits=u_splits)
        jax.block_until_ready(out[0])
        return out
    
    # warm compile once
    run()

    # benchmark warm-started run
    benchmark(run)

def test_compute_quad_cost_basic_numeric_value():
    # nx=2, nu=1
    Q = jnp.array([[2.0, 0.5],
                   [0.5, 1.0]])
    q = jnp.array([0.1, -0.2])
    R = jnp.array([[4.0]])
    r = jnp.array([-0.3])
    x = jnp.array([1.0, 2.0])
    u = jnp.array([3.0])

    # expected via explicit formula
    expected = 0.5 * (x @ (Q @ x)) + (q @ x) + 0.5 * (u @ (R @ u)) + (r @ u)
    cost = costtypes.compute_quadratic_cost(Q, q, R, r, x, u)
    assert jnp.isclose(cost, expected)


def test_compute_quad_cost_dtype_promotion_from_ints():
    Q = jnp.array([[2, 0],
                   [0, 1]], dtype=jnp.int32)
    q = jnp.array([0, 0], dtype=jnp.int32)
    R = jnp.array([[3]], dtype=jnp.int32)
    r = jnp.array([0], dtype=jnp.int32)
    x = jnp.array([1, 2], dtype=jnp.int32)
    u = jnp.array([3], dtype=jnp.int32)

    cost = costtypes.compute_quadratic_cost(Q, q, R, r, x, u)
    # numeric check
    expected = 0.5 * (x @ (Q @ x)) + (q @ x) + 0.5 * (u @ (R @ u)) + (r @ u)
    assert jnp.isclose(cost, expected)
    # should be floating (due to 0.5 factors)
    assert jnp.issubdtype(cost.dtype, jnp.floating)


def test_compute_quad_cost_zero_terms_yield_zero_cost():
    Q = jnp.zeros((2, 2)); q = jnp.zeros((2,))
    R = jnp.zeros((1, 1)); r = jnp.zeros((1,))
    x = jnp.array([5.0, -3.0]); u = jnp.array([7.0])
    cost = costtypes.compute_quadratic_cost(Q, q, R, r, x, u)
    assert jnp.isclose(cost, 0.0)


def test_compute_quad_cost_nonsymmetric_QR_same_as_symmetric_part():
    # xᵀQx equals xᵀ*sym(Q)*x, similarly for R
    Q = jnp.array([[1.0, 3.0],
                   [0.0, 2.0]])             # nonsymmetric
    R = jnp.array([[5.0, 1.0],
                   [0.0, 4.0]])             # nonsymmetric
    q = jnp.array([0.0, 0.0])
    r = jnp.array([0.0, 0.0])
    x = jnp.array([1.2, -0.7])
    u = jnp.array([0.3, 0.1])

    cost_raw = costtypes.compute_quadratic_cost(Q, q, R, r, x, u)
    Qs = 0.5 * (Q + Q.T)
    Rs = 0.5 * (R + R.T)
    cost_sym = costtypes.compute_quadratic_cost(Qs, q, Rs, r, x, u)
    assert jnp.isclose(cost_raw, cost_sym)


@pytest.mark.parametrize(
    "Q,q,R,r,x,u,err",
    [
        (jnp.eye(2), jnp.zeros(2), jnp.eye(1), jnp.zeros(1), jnp.zeros((2,1)), jnp.zeros(1), "x must be"),
        (jnp.eye(2), jnp.zeros(2), jnp.eye(1), jnp.zeros(1), jnp.zeros(2), jnp.zeros((1,1)), "u must be"),
        (jnp.eye(2), jnp.zeros(2), jnp.eye(1), jnp.zeros(1), jnp.zeros(3), jnp.zeros(1), "Q must be"),
        (jnp.eye(2), jnp.zeros(3), jnp.eye(1), jnp.zeros(1), jnp.zeros(2), jnp.zeros(1), "q must be"),
        (jnp.eye(2), jnp.zeros(2), jnp.eye(2), jnp.zeros(1), jnp.zeros(2), jnp.zeros(1), "R must be"),
        (jnp.eye(2), jnp.zeros(2), jnp.eye(1), jnp.zeros(2), jnp.zeros(2), jnp.zeros(1), "r must be"),
    ],
)
def test_compute_quad_cost_validation_errors(Q, q, R, r, x, u, err):
    with pytest.raises(ValueError, match=err):
        costtypes.compute_quadratic_cost(Q, q, R, r, x, u, validate=True)


def test_grad_cost_shapes_match_inputs():
    """q should have same shape as x, r same shape as u."""
    def g(t, x, u):
        return jnp.dot(x, x) + jnp.dot(u, u) + t

    t = 1.0
    x = jnp.array([1.0, -2.0, 3.0])
    u = jnp.array([0.5, -1.5])

    q, r = costtypes.gradient_cost_local_ctrl_no_checks(g, t, x, u)

    assert q.shape == x.shape
    assert r.shape == u.shape


def test_grad_cost_quadratic_correct():
    """
    For g(t, x, u) = 0.5 * ||x||^2 + ||u||^2 + t,
    ∂g/∂x = x, ∂g/∂u = 2u
    and gradient should not depend on t.
    """
    def g(t, x, u):
        return 0.5 * jnp.dot(x, x) + jnp.dot(u, u) + t

    t = 3.7
    x = jnp.array([1.0, -2.0])
    u = jnp.array([0.5, -1.5, 2.0])

    q, r = costtypes.gradient_cost_local_ctrl_no_checks(g, t, x, u)

    expected_q = x                 # d/dx (0.5 x^T x) = x
    expected_r = 2.0 * u           # d/du (u^T u) = 2u

    assert jnp.allclose(q, expected_q, atol=1e-7, rtol=1e-7)
    assert jnp.allclose(r, expected_r, atol=1e-7, rtol=1e-7)


def test_grad_cost_time_only_affects_constant_term():
    """
    If g has t only as an additive term, gradients w.r.t x and u
    should be independent of t.
    """
    def g(t, x, u):
        return jnp.dot(x, x) + jnp.dot(u, u) + 5.0 * t

    t1 = 0.0
    t2 = 10.0
    x = jnp.array([1.0, 2.0])
    u = jnp.array([-1.0, 0.5])

    q1, r1 = costtypes.gradient_cost_local_ctrl_no_checks(g, t1, x, u)
    q2, r2 = costtypes.gradient_cost_local_ctrl_no_checks(g, t2, x, u)

    assert jnp.allclose(q1, q2, atol=1e-7, rtol=1e-7)
    assert jnp.allclose(r1, r2, atol=1e-7, rtol=1e-7)


def test_grad_cost_cross_coupled():
    """
    Test a cost with x–u coupling to make sure closures behave as expected.

    g(t, x, u) = x^T u_head + t, where u_head is first nx components of u.

    Then:
      ∂g/∂x = u_head
      ∂g/∂u = [x, 0, ..., 0]
    """
    def g(t, x, u):
        nx = x.shape[0]
        u_head = u[:nx]
        return jnp.dot(x, u_head) + t

    t = 2.0
    x = jnp.array([1.0, 2.0, -1.0])        # nx = 3
    u = jnp.array([0.5, -1.0, 3.0, 7.0])   # nu = 4

    q, r = costtypes.gradient_cost_local_ctrl_no_checks(g, t, x, u)

    nx = x.shape[0]
    expected_q = u[:nx]                    # d/dx (x^T u_head) = u_head
    expected_r = jnp.concatenate([x, jnp.zeros((u.shape[0] - nx,))])

    assert jnp.allclose(q, expected_q, atol=1e-7, rtol=1e-7)
    assert jnp.allclose(r, expected_r, atol=1e-7, rtol=1e-7)


def test_grad_cost_function_is_jittable():
    """Ensure gradient_cost_no_checks works correctly inside jax.jit."""
    def g(t, x, u):
        return 0.5 * jnp.dot(x, x) + jnp.dot(u, u) + 3.0 * t

    t = 1.0
    x = jnp.array([1.0, 2.0])
    u = jnp.array([-1.0, 0.5])

    q_eager, r_eager = costtypes.gradient_cost_local_ctrl_no_checks(g, t, x, u)

    jit_grad = jax.jit(costtypes.gradient_cost_local_ctrl_no_checks, static_argnums=(0,))
    q_jit, r_jit = jit_grad(g, t, x, u)

    assert jnp.allclose(q_eager, q_jit, atol=1e-7, rtol=1e-7)
    assert jnp.allclose(r_eager, r_jit, atol=1e-7, rtol=1e-7)

def test_terminal_grad_shape_matches_x():
    """qterm_i should have same shape as x."""
    def gterm(t, x):
        return jnp.dot(x, x) + t  # simple quadratic in x

    t = 1.0
    x = jnp.array([1.0, -2.0, 3.0])

    qterm = costtypes.gradient_terminal_cost_no_checks(gterm, t, x)

    assert qterm.shape == x.shape


def test_terminal_quadratic_grad_correct():
    """
    For g(t, x) = 0.5 * ||x||^2 + 3*t,
    ∂g/∂x = x and does not depend on t.
    """
    def gterm(t, x):
        return 0.5 * jnp.dot(x, x) + 3.0 * t

    t = 5.0
    x = jnp.array([1.0, -2.0])

    qterm = costtypes.gradient_terminal_cost_no_checks(gterm, t, x)
    expected = x  # d/dx(0.5 x^T x) = x

    assert jnp.allclose(qterm, expected, atol=1e-7, rtol=1e-7)


def test_terminal_grad_independent_of_t_when_additive():
    """
    If t appears only additively, gradient wrt x should be independent of t.
    """
    def gterm(t, x):
        # an arbitrary cost function that is non-linear/non-quadratic in x but decoupled from t
        return jnp.sin(jnp.dot(x, x)) + 10.0 * t

    x = jnp.array([1.0, 2.0, -1.0])
    t1 = 0.0
    t2 = 100.0

    q1 = costtypes.gradient_terminal_cost_no_checks(gterm, t1, x)
    q2 = costtypes.gradient_terminal_cost_no_checks(gterm, t2, x)

    assert jnp.allclose(q1, q2, atol=1e-7, rtol=1e-7)


def test_terminal_grad_with_coupled_components():
    """
    Test something slightly nontrivial:

    g(t, x) = x0 * x1 + x2^2

    ∂g/∂x = [x1, x0, 2*x2]
    """
    def gterm(t, x):
        return x[0] * x[1] + x[2] ** 2 + t

    t = 2.0
    x = jnp.array([1.0, 3.0, -2.0])

    qterm = costtypes.gradient_terminal_cost_no_checks(gterm, t, x)
    expected = jnp.array([
        x[1],        # d/dx0 (x0*x1) = x1
        x[0],        # d/dx1 (x0*x1) = x0
        2.0 * x[2],  # d/dx2 (x2^2)   = 2*x2
    ])

    assert jnp.allclose(qterm, expected, atol=1e-7, rtol=1e-7)


# def test_terminal_grad_is_jittable():
#     """Ensure function works correctly under jax.jit."""
#     def gterm(t, x):
#         return 0.5 * jnp.dot(x, x) + 2.0 * t

#     t = 1.0
#     x = jnp.array([1.0, -1.0])

#     eager = gradient_terminal_cost_no_checks(gterm, t, x)
#     jit_fn = jax.jit(gradient_terminal_cost_no_checks)
#     jitted = jit_fn(gterm, t, x)

#     assert jnp.allclose(eager, jitted, atol=1e-7, rtol=1e-7)

def test_gradient_cost_playerwise_trajectory_shapes():

    # ~~ ARRANGE ~~
    num_steps = 5
    x_dim = 3
    u_splits = [2, 3]  # 2 players, p1 has 2-d control, p2 has 3-d controls
    u_dim = sum(u_splits)
    num_players = len(u_splits)
    u_splits = jnp.asarray(u_splits)

    # ts = jnp.linspace(0.0, 1.0, num_steps)
    tg = TimeGrid(nt=num_steps, dt=1.0/num_steps)
    xs = jnp.stack([jnp.ones(x_dim)*i for i in range(num_steps)])
    us = jnp.stack([jnp.ones(u_dim)*i for i in range(num_steps-1)])
    ls = jnp.ones((num_steps-1, num_players, x_dim))
    pdtraj = FixedStepPrimalDualTrajectory(tg=tg, xs=xs, us=us, ls=ls)

    # Define a simple test cost function for player 0
    def g_0(t, x, u):
        # Quadratic cost: xᵀQx + u1ᵀR1u1 + u2ᵀR2u2 + t*xᵀx for variety
        u1, u2 = jnp.split(u, 2)
        Q = jnp.eye(x.shape[0])
        R1 = 2.0 * jnp.eye(u1.shape[0])
        R2 = 0.5 * jnp.eye(u2.shape[0])
        return jnp.dot(x, Q @ x) + jnp.dot(u1, R1 @ u1) + jnp.dot(u2, R2 @ u2) + t * jnp.dot(x, x)
    
    # Define a simple terminal cost function
    def gterm_0(t, x):
        return jnp.dot(x, x) + t  # simple quadratic in x

    # ~~ ACT ~~
    qs_0, rs_0 = costtypes.gradient_cost_local_ctrl_playerwise_trajectory(g_0, gterm_0, pdtraj, 0, u_splits)

    # ~~ ASSERT ~~
    assert qs_0.shape == (num_steps, x_dim)
    assert rs_0.shape == (num_steps-1, u_splits[0])

def test_gradient_cost_playerwise_trajectory_rejects_wrong_type():
    def costfn_i(t, x, u):
        return jnp.dot(x, x) + jnp.dot(u, u)

    def termfn_i(t, x):
        return jnp.dot(x, x)

    # Fake object instead of FixedStepPrimalDualTrajectory
    class DummyTraj:
        pass

    op = DummyTraj()
    u_splits = jnp.array([2])  # anything

    with pytest.raises(ValueError, match="Invalid trajectory type"):
        _ = costtypes.gradient_cost_local_ctrl_playerwise_trajectory(
            costfn_i=costfn_i,
            termfn_i=termfn_i,
            op=op,
            player_i=0,
            u_splits=u_splits,
        )

def test_gradient_cost_playerwise_quadratic_single_player():
    def costfn_i(t, x, u):
        # running cost: 0.5*||x||^2 + ||u||^2
        return 0.5 * jnp.dot(x, x) + jnp.dot(u, u)

    def termfn_i(t, x):
        # terminal cost: 0.5*||x||^2
        return 0.5 * jnp.dot(x, x)

    N = 1   # number of players
    nt = 4  # number of time steps
    nx = 3  # joint state dimension
    nu = 2  # joint control dimension

    tg = TimeGrid(nt=nt, dt=1.0, t0=0.0)

    xs = jnp.array([
        [1.0, 2.0, 2.5],    # x0
        [3.0, 4.0, 4.5],    # x1
        [5.0, 6.0, 6.5],    # x2
        [7.0, 8.0, 8.5],    # x3 (terminal)
    ])

    us = jnp.array([
        [0.1, 0.2],     # u0
        [0.3, 0.4],     # u1
        [0.5, 0.6],     # u2
    ])  # shape (nt-1, nu)

    ls = jnp.ones((nt-1, N, nx))

    # Adjust constructor args if your FixedStepPrimalDualTrajectory differs
    op = FixedStepPrimalDualTrajectory(tg=tg, xs=xs, us=us, ls=ls)

    # One player, controls are all theirs
    u_splits = jnp.array([nu])

    qs_i, rs_i = costtypes.gradient_cost_local_ctrl_playerwise_trajectory(
        costfn_i=costfn_i,
        termfn_i=termfn_i,
        op=op,
        player_i=0,
        u_splits=u_splits,
    )

    # Shape checks
    assert qs_i.shape == (nt, nx)
    assert rs_i.shape == (nt-1, nu)

    # Expected gradients:
    # running: d/dx = x_k, d/du = 2u_k
    # terminal: d/dx_T = x_T
    expected_qs = jnp.array([
        xs[0],      # grad wrt x0 from running cost at k=0
        xs[1],      # grad wrt x1 from running cost at k=1
        xs[2],      # grad wrt x2 from running cost at k=2
        xs[3],      # grad wrt x3 from terminal cost
    ])

    expected_rs = jnp.array([
        2.0 * us[0],
        2.0 * us[1],
        2.0 * us[2],
    ])

    assert jnp.allclose(qs_i, jnp.array(expected_qs), atol=1e-7, rtol=1e-7)
    assert jnp.allclose(rs_i, jnp.array(expected_rs), atol=1e-7, rtol=1e-7)

def test_gradient_cost_playerwise_single_player_terminal_only():
    def costfn_i(t, x, u):
        # No running cost
        return 0.0

    def termfn_i(t, x):
        # terminal cost: 0.5*||x||^2
        return 0.5 * jnp.dot(x, x)

    N = 1
    nt = 5
    nx = 4
    nu = 3

    tg = TimeGrid(nt=nt, dt=10.0, t0=0.1)

    xs = jnp.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 3.0, 0.0],
        [0.0, 0.0, 0.0, 4.0],
        [1.0, 2.0, 3.0, 4.0],   # terminal
    ])

    us = jnp.ones((nt-1, nu))  # controls don't matter for terminal-only cost

    ls = jnp.zeros((nt-1, N, nx))

    op = FixedStepPrimalDualTrajectory(tg=tg, xs=xs, us=us, ls=ls)
    u_splits = jnp.array([nu])    # single player

    qs_i, rs_i = costtypes.gradient_cost_local_ctrl_playerwise_trajectory(
        costfn_i=costfn_i,
        termfn_i=termfn_i,
        op=op,
        player_i=0,
        u_splits=u_splits,
    )

    # Non-terminal state gradients should be ~0
    assert jnp.allclose(qs_i[:-1], 0.0, atol=1e-7, rtol=1e-7)

    # Terminal state gradient should be x_terminal
    expected_q_terminal = xs[-1]
    assert jnp.allclose(qs_i[-1], jnp.array(expected_q_terminal), atol=1e-7, rtol=1e-7)

    # No control dependence => all control gradients zero
    assert jnp.allclose(rs_i, 0.0, atol=1e-7, rtol=1e-7)

@pytest.fixture
def primal_dual_traj_1():
    """
    Hard-coded multi-player FixedStepPrimalDualTrajectory fixture.

    Dimensions:
        N  (players)        = 5
        nt (time steps)     = 17
        nx (state dim)      = 11
        nu (control dim)    = 7

    Shapes:
        xs: (nt,     nx)         = (17, 11)
        us: (nt-1,  nu)         = (16, 7)
        ls: (nt-1,  N,  nx)     = (16, 5, 11)
    """
    N = 5
    nt = 17
    nx = 11
    nu = 7

    # Time grid
    tg = TimeGrid(nt=nt, dt=0.1, t0=0.0)

    # State trajectory: shape (nt, nx)
    # Use a simple ramp so values are deterministic but not trivial.
    xs = jnp.arange(nt * nx, dtype=jnp.float32).reshape(nt, nx)

    # Control trajectory: shape (nt-1, nu)
    us = (jnp.arange((nt - 1) * nu, dtype=jnp.float32)
             .reshape(nt - 1, nu) * 0.1)

    # Dual trajectory (lambdas): shape (nt-1, N, nx)
    ls = (jnp.arange((nt - 1) * N * nx, dtype=jnp.float32)
             .reshape(nt - 1, N, nx) * 0.01)

    op = FixedStepPrimalDualTrajectory(
        tg=tg,
        xs=xs,
        us=us,
        ls=ls,
    )

    return op

def test_gradient_cost_playerwise_multi_player_splitting(primal_dual_traj_1):
    """
    Sanity check: If player i's running cost depends ONLY on their own controls
    (not on state or other players' control dimensions), then:

      - State gradients qs_i should be zero (no x-dependence, no terminal cost).
      - Control gradients rs_i should only have dimensionality of player-i's control space
    """
    op = primal_dual_traj_1

    # Split joint control dims among 5 players: sum must be nu=7
    # Example: [1, 2, 1, 1, 2]  -> total 7
    u_splits = jnp.array([1, 2, 1, 1, 2])
    assert int(u_splits.sum()) == op.nu

    # Running cost for player i: depends only on their control vector u_i
    # g_i(t, x, u_i) = ||u_i||^2
    def make_costfn_for_player():
        def costfn_i(t, x, u_i):
            return 0.5 * jnp.dot(u_i, u_i)
        return costfn_i

    # Terminal cost is zero (no dependence on x to keep qs_i=0)
    def termfn_i(t, x):
        return 0.0
    
    for player_i in range(op.N):
        costfn_i = make_costfn_for_player()

        qs_i, rs_i = costtypes.gradient_cost_local_ctrl_playerwise_trajectory(
            costfn_i=costfn_i,
            termfn_i=termfn_i,
            op=op,
            player_i=player_i,
            u_splits=u_splits,
        )

        # 1) Shape checks
        assert qs_i.shape == (op.nt, op.nx)                 # (nt, nx)
        assert rs_i.shape == (op.nt-1, u_splits[player_i])  # (nt-1, nu_i)

        # 2) No dependence on x or terminal cost -> all state gradients should be ~0
        assert jnp.allclose(qs_i, 0.0, atol=1e-7, rtol=1e-7)

def test_gradient_cost_playerwise_nonquadratic_single_player():
    """
    Compare gradient_cost_playerwise_trajectory against a manual
    per-timestep loop for a nonlinear, time-dependent cost.

    We use:
        running cost:
            g(t, x, u) =
                sin(t) * (x0^2 + 0.5 * exp(x1)) +
                cos(t) * (u0^3 + 0.1 * x0 * u0)

        terminal cost:
            g_T(t, x) =
                sin(2t) * (x0 * x1 + exp(x0 - x1))

    We do NOT derive analytic gradients; instead we build expected
    qs_i, rs_i by explicitly calling gradient_cost_no_checks and
    gradient_terminal_cost_no_checks in a Python loop. This tests that
    gradient_cost_playerwise_trajectory:
      - uses the correct times and trajectory entries,
      - wires nt/nx/nu correctly,
      - handles terminal step separately and only wrt state.
    """
    # ----- Define non-quadratic running and terminal costs -----

    def costfn_i(t, x, u):
        # x: (nx,), u: (nu,)
        # sin(t) * (x0^2 + 0.5 * exp(x1)) + cos(t) * (u0^3 + 0.1 * x0 * u0)
        return (
            jnp.sin(t) * (x[0] ** 2 + 0.5 * jnp.exp(x[1]))
            + jnp.cos(t) * (u[0] ** 3 + 0.1 * x[0] * u[0])
        )

    def termfn_i(t, x):
        # sin(2t) * (x0 * x1 + exp(x0 - x1))
        return jnp.sin(2.0 * t) * (x[0] * x[1] + jnp.exp(x[0] - x[1]))

    # ----- Build a small but nontrivial trajectory -----

    nt = 4      # time steps
    nx = 2      # state dimension
    nu = 1      # control dimension
    N = 1

    tg = TimeGrid(nt=nt, dt=0.3, t0=0.1)

    # xs: (nt, nx)
    xs = jnp.array([
        [0.1, -0.2],   # x0
        [0.3,  0.4],   # x1
        [-0.5, 1.0],   # x2
        [1.2, -0.7],   # x3 (terminal)
    ], dtype=jnp.float32)

    # us: (nt-1, nu)
    us = jnp.array([
        [0.05],        # u0
        [-0.2],        # u1
        [0.7],         # u2
    ], dtype=jnp.float32)

    # duals (ls): (nt-1, N, nx); values irrelevant for cost test
    ls = jnp.zeros((nt - 1, N, nx), dtype=jnp.float32)

    op = FixedStepPrimalDualTrajectory(
        tg=tg,
        xs=xs,
        us=us,
        ls=ls,
    )

    # One player, owns all control dims
    u_splits = jnp.array([nu])

    # ----- Compute result from the function under test -----

    qs_i, rs_i = costtypes.gradient_cost_local_ctrl_playerwise_trajectory(
        costfn_i=costfn_i,
        termfn_i=termfn_i,
        op=op,
        player_i=0,
        u_splits=u_splits,
    )

    # Shape checks
    assert qs_i.shape == (op.nt, op.nx)       # (nt, nx)
    assert rs_i.shape == (op.nt-1, u_splits[0])       # (nt-1, nu)

    # ----- Build expected gradients via explicit per-step loops -----

    # reconstruct time grid
    ts = compute_ts(op.tg)  # (nt,)

    # manual arrays
    expected_qs = jnp.zeros_like(xs)    # (nt, nx)
    expected_rs = jnp.zeros_like(us)    # (nt-1, nu) single player

    # running cost contributions for k = 0..nt-2
    for k in range(nt - 1):
        t_k = ts[k]
        x_k = xs[k]
        u_k = us[k]

        q_k, r_k = costtypes.gradient_cost_local_ctrl_no_checks(
            g_i=costfn_i,
            t=t_k,
            x=x_k,
            u_i=u_k,
        )

        expected_qs = expected_qs.at[k].set(q_k)
        expected_rs = expected_rs.at[k].set(r_k)

    # terminal cost contribution at k = nt-1, only wrt state
    t_T = ts[-1]
    x_T = xs[-1]
    q_T = costtypes.gradient_terminal_cost_no_checks(
        gterm_i=termfn_i,
        t=t_T,
        x=x_T,
    )
    expected_qs = expected_qs.at[-1].set(q_T)

    # ----- Compare function output to manual construction -----

    assert jnp.allclose(qs_i, jnp.array(expected_qs), atol=1e-7, rtol=1e-7)
    assert jnp.allclose(rs_i, jnp.array(expected_rs), atol=1e-7, rtol=1e-7)
