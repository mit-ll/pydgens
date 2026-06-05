# Copyright 2026 MIT Lincoln Laboratory
# SPDX-License-Identifier: MIT

import pytest
import jax.numpy as jnp
import jax
import json
import numpy as np
    
from pathlib import Path

from pydgens.examples.doubleint import DoubleInt_LQLBG_C2
from pydgens.ir.gametypes import LinearQuadraticGameType1
from pydgens.solvers.lqsolver import solve_lqgame_feedback

def eval_quadratic_cost(Q, q, xi):
    return 0.5 * xi.T @ Q @ xi + q.T @ xi

@pytest.fixture
def lbg2():
    return DoubleInt_LQLBG_C2()

@pytest.fixture
def lbg2_Qf():
    # Terminal state cost version of BoublInt_LQLBG_C2
    lbg2 = DoubleInt_LQLBG_C2()



    Qf = lbg2.game.Q[-1]
    qf = lbg2.game.q[-1]

    # modify game object to add terminal state costs 
    # that are continuation of running costs
    lbg2.game = LinearQuadraticGameType1(
        cs = lbg2.game.cs,
        N=lbg2.game.N,
        Q=lbg2.game.Q,
        q=lbg2.game.q,
        R=lbg2.game.R,
        r=lbg2.game.r,
        u_splits=lbg2.game.u_splits,
        Qf=Qf,
        qf=qf
    )

    return lbg2

def test_lbg2_smoketest(lbg2):
    pass

def test_A_B_shapes(lbg2):
    delta = 0.1
    A, B = lbg2.fblin_dynamics(delta, lbg2.PARAMS)
    assert A.shape == (12, 12), "A_game should be 12x12"
    assert B.shape == (12, 6), "B_game should be 12x6"

def test_A_block_diagonal(lbg2):
    delta = 0.1
    A, _ = lbg2.fblin_dynamics(delta, lbg2.PARAMS)
    # Check zero blocks between submatrices
    assert jnp.allclose(A[0:4, 4:12], 0), "A_game upper off-diagonal blocks should be zero"
    assert jnp.allclose(A[4:8, [0,1,2,3,8,9,10,11]], 0), "A_game middle off-diagonal blocks should be zero"
    assert jnp.allclose(A[8:12, 0:8], 0), "A_game lower off-diagonal blocks should be zero"

def test_B_block_diagonal(lbg2):
    delta = 0.1
    _, B = lbg2.fblin_dynamics(delta, lbg2.PARAMS)
    assert jnp.allclose(B[0:4, 2:6], 0), "B_game top row off-diagonal should be zero"
    assert jnp.allclose(B[4:8, [0,1,4,5]], 0), "B_game middle row off-diagonal should be zero"
    assert jnp.allclose(B[8:12, 0:4], 0), "B_game bottom row off-diagonal should be zero"

def test_A_B_known_values(lbg2):
    delta = 0.1
    A, B = lbg2.fblin_dynamics(delta, lbg2.PARAMS)
    A_ref = jnp.array([
        [1.0, 0.0, delta, 0.0],
        [0.0, 1.0, 0.0, delta],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    B_ref = jnp.array([
        [0.5 * delta**2, 0.0],
        [0.0, 0.5 * delta**2],
        [delta, 0.0],
        [0.0, delta]
    ])
    assert jnp.allclose(A[0:4, 0:4], A_ref), "Top-left A block should match A_d"
    assert jnp.allclose(B[0:4, 0:2], B_ref), "Top-left B block should match B_d"
    assert jnp.allclose(A[4:8, 4:8], A_ref), "Middle A block should match A_d"
    assert jnp.allclose(B[4:8, 2:4], B_ref), "Middle B block should match B_d"
    assert jnp.allclose(A[8:12, 8:12], A_ref), "Bottom A block should match A_d"
    assert jnp.allclose(B[8:12, 4:6], B_ref), "Bottom B block should match B_d"

def test_fblin_dynamics_propagation_no_input(lbg2):
    delta = 0.1
    A, B = lbg2.fblin_dynamics(delta, lbg2.PARAMS)

    # Initial state: Bandit x-velocity = 1.0, everything else zero
    xi_0 = jnp.zeros(12)
    xi_0 = xi_0.at[2].set(1.0)  # Bandit vx = 1.0

    mu = jnp.zeros(6)  # No control input

    xi_1 = A @ xi_0 + B @ mu

    # Expected: px should increase by delta * vx
    expected = jnp.zeros(12)
    expected = expected.at[0].set(delta)  # px += delta * vx = 0.1
    expected = expected.at[2].set(1.0)    # vx unchanged

    assert jnp.allclose(xi_1, expected), "State should propagate with constant velocity"

def test_fblin_dynamics_propagation_with_input(lbg2):
    delta = 0.1
    A, B = lbg2.fblin_dynamics(delta, lbg2.PARAMS)

    xi_0 = jnp.zeros(12)
    mu = jnp.zeros(6)
    mu = mu.at[0].set(1.0)  # Bandit ax = 1.0
    mu = mu.at[3].set(1.0)  # Lady ay = 1.0
    mu = mu.at[4].set(-1.0)  # Guard ax = 1.0
    mu = mu.at[5].set(1.0)  # Guard ay = 1.0

    xi_1 = A @ xi_0 + B @ mu

    # Expected change due to acceleration:
    # px += 0.5 * delta^2 * ax
    # vx += delta * ax
    expected = jnp.zeros(12)
    expected = expected.at[0].set(0.5 * delta ** 2)
    expected = expected.at[2].set(delta)
    expected = expected.at[5].set(0.5 * delta ** 2)
    expected = expected.at[7].set(delta)
    expected = expected.at[8].set(-0.5 * delta ** 2)
    expected = expected.at[10].set(-delta)
    expected = expected.at[9].set(0.5 * delta ** 2)
    expected = expected.at[11].set(delta)

    assert jnp.allclose(xi_1, expected), "State should respond correctly to control input"

@pytest.mark.parametrize("c", [0.5, 1.0, 2.0])
def test_cost_bandit_lady_alignment_shapes_and_symmetry(lbg2, c):
    Q, q = lbg2.cost_bandit_lady_alignment_proxy(c, lbg2.PARAMS)
    # Shapes
    assert Q.shape == (12, 12), "Q must be 12×12"
    assert q.shape == (12,), "q must be length 12"
    # Symmetry
    assert jnp.allclose(Q, Q.T), "Q must be symmetric"
    assert jnp.allclose(q, 0)

def test_cost_bandit_lady_aligned(lbg2):
    c = 1.0
    Q, q = lbg2.cost_bandit_lady_alignment_proxy(c, lbg2.PARAMS)

    # Create an auxiliary state xi where:
    # bandit velociity aligns with lady relative position
    xi_v1 = jnp.zeros(12)
    xi_v1 = xi_v1.at[2].set(1.0)  # vx_b
    xi_v1 = xi_v1.at[4].set(1.0)  # p_{L/B,x}
    J_v1 = eval_quadratic_cost(Q, q, xi_v1)
    assert jnp.isclose(J_v1, -1.0)

    # Create an auxiliary state xi where:
    # bandit velociity orthogonal with lady relative position
    xi_v2 = jnp.zeros(12)
    xi_v2 = xi_v2.at[2].set(1.0)  # vx_b
    xi_v2 = xi_v2.at[5].set(1.0)  # p_{L/B,x}
    J_v2 = eval_quadratic_cost(Q, q, xi_v2)
    assert jnp.isclose(J_v2, 0.0)

    # Create an auxiliary state xi where:
    # bandit velociity anti-aligned with lady relative position
    xi_v3 = jnp.zeros(12)
    xi_v3 = xi_v3.at[2].set(1.0)  # vx_b
    xi_v3 = xi_v3.at[4].set(-1.0)  # p_{L/B,x}
    J_v3 = eval_quadratic_cost(Q, q, xi_v3)
    assert jnp.isclose(J_v3, 1.0)

    # Create an auxiliary state xi where:
    # bandit velociity orthogonal with lady relative position
    # thus zeroing this cost
    xi = jnp.zeros(12)
    xi = xi.at[4].set(1.0)  # px_l
    xi = xi.at[5].set(1.0)  # py_l
    xi = xi.at[2].set(-1.0)  # vx_b
    xi = xi.at[3].set(1.0)  # vy_b
    J = eval_quadratic_cost(Q, q, xi)
    assert jnp.isclose(J, 0.0)

    # Create an auxiliary state xi where:
    # guard and bandit it same position, thus zeroing 
    # this cost
    xi = jnp.zeros(12)
    xi = xi.at[0].set(-4.84201112)  # px_b
    xi = xi.at[1].set(-0.45189499)  # py_b
    xi = xi.at[4].set(-4.84201112)  # px_l
    xi = xi.at[5].set(-0.45189499)  # py_l
    xi = xi.at[2].set(-2.96138121)  # vx_b
    xi = xi.at[3].set(-0.13022861)  # vy_b
    J = eval_quadratic_cost(Q, q, xi)
    assert jnp.isclose(J, 0.0, atol=1e-6)

    # Create an auxiliary state xi where:
    # guard and bandit in arbitrary states 
    # with known result
    c = 0.724836858805204
    Q, q = lbg2.cost_bandit_lady_alignment_proxy(c, lbg2.PARAMS)
    xi = jnp.zeros(12)
    xi = xi.at[4].set(2.06935135)  # px_l
    xi = xi.at[5].set(-4.06276323)  # py_l
    xi = xi.at[0].set(-3.42004107)  # px_b
    xi = xi.at[1].set(-4.15076036)  # py_b
    xi = xi.at[2].set(4.59334092)  # vx_b
    xi = xi.at[3].set(1.19424719)  # vy_b
    J_exp = -18.35268164379315
    J = eval_quadratic_cost(Q, q, xi)
    assert jnp.isclose(J, J_exp, atol=1e-6)

def test_cost_bandit_lady_distance_shape_of_symmetry(lbg2):
    Q, q = lbg2.cost_bandit_lady_distance(1.0, lbg2.PARAMS)

    # Shapes
    assert Q.shape == (12, 12)
    assert q.shape == (12,)

    # Symmetry
    assert jnp.allclose(Q, Q.T), "Q must be symmetric"
    assert jnp.allclose(q, 0)


def test_cost_bandit_lady_distance_expected_entries(lbg2):
    c = 1.5
    Q, q = lbg2.cost_bandit_lady_distance(c, lbg2.PARAMS)

    # Diagonal
    assert Q[0, 0] == pytest.approx(2 * c)
    assert Q[1, 1] == pytest.approx(2 * c)
    assert Q[4, 4] == pytest.approx(2 * c)
    assert Q[5, 5] == pytest.approx(2 * c)

    # Cross terms
    assert Q[0, 4] == pytest.approx(-2 * c)
    assert Q[4, 0] == pytest.approx(-2 * c)
    assert Q[1, 5] == pytest.approx(-2 * c)
    assert Q[5, 1] == pytest.approx(-2 * c)

    # Everything else should be zero
    for i in range(12):
        for j in range(12):
            if (i, j) not in [(0, 0), (1, 1), (4, 4), (5, 5), (0, 4), (4, 0), (1, 5), (5, 1)]:
                assert Q[i, j] == pytest.approx(0.0)

def test_cost_bandit_lady_distance_evaluation_zero_dist(lbg2):
    c = 2.0
    Q, q = lbg2.cost_bandit_lady_distance(c, lbg2.PARAMS)

    # xi: Bandit and Lady at same position → distance should be zero
    xi = jnp.zeros(12).at[0].set(5.0).at[1].set(3.0).at[4].set(5.0).at[5].set(3.0)
    cost = 0.5 * xi.T @ Q @ xi + q.T @ xi
    assert jnp.isclose(cost, 0.0)

def test_cost_bandit_lady_distance_evaluation_separated(lbg2):
    c = 1.0
    Q, q = lbg2.cost_bandit_lady_distance(c, lbg2.PARAMS)

    # xi: Bandit at (0, 0), Lady at (3, 4) → distance^2 = 25
    xi = jnp.zeros(12).at[4].set(3.0).at[5].set(4.0)
    cost = 0.5 * xi.T @ Q @ xi + q.T @ xi
    assert jnp.isclose(cost, 25.0)


def test_cost_guard_bandit_alignment_shapes_and_symmetry(lbg2):
    c = 1.5
    Q, q = lbg2.cost_guard_bandit_alignment_proxy(c, lbg2.PARAMS)
    assert Q.shape == (12, 12)
    assert q.shape == (12,)
    assert jnp.allclose(Q, Q.T), "Q matrix must be symmetric"

def test_cost_guard_bandit_alignment_entries(lbg2):
    c = 1.0
    Q, q = lbg2.cost_guard_bandit_alignment_proxy(c, lbg2.PARAMS)

    # Expected values
    assert Q[0, 10] == c
    assert Q[10, 0] == c
    assert Q[8, 10] == -c
    assert Q[10, 8] == -c

    assert Q[1, 11] == c
    assert Q[11, 1] == c
    assert Q[9, 11] == -c
    assert Q[11, 9] == -c

    # Ensure all other entries are zero
    expected_nonzeros = {(0, 10), (10, 0), (8, 10), (10, 8),
                         (1, 11), (11, 1), (9, 11), (11, 9)}
    for i in range(12):
        for j in range(12):
            if (i, j) not in expected_nonzeros:
                assert Q[i, j] == 0.0

    # Check q is zero
    assert jnp.allclose(q, jnp.zeros(12))

def test_cost_guard_bandit_alignment_cost_value(lbg2):
    c = 1.0
    Q, q = lbg2.cost_guard_bandit_alignment_proxy(c, lbg2.PARAMS)

    # Create an auxiliary state xi where:
    # guard velociity aligns with bandit relative position
    # thus "maximizing" this cost
    xi_v1 = jnp.zeros(12)
    xi_v1 = xi_v1.at[0].set(1.0)  # px_b
    xi_v1 = xi_v1.at[10].set(1.0)  # vx_g
    J_v1 = eval_quadratic_cost(Q, q, xi_v1)
    assert jnp.isclose(J_v1, 1.0)

    # Create an auxiliary state xi where:
    # guard velociity orthogonal with bandit relative position
    # thus zeroing this cost
    xi_v2 = jnp.zeros(12)
    xi_v2 = xi_v2.at[0].set(1.0)  # px_b
    xi_v2 = xi_v2.at[11].set(1.0)  # vy_g
    J_v2 = eval_quadratic_cost(Q, q, xi_v2)
    assert jnp.isclose(J_v2, 0.0)

    # Create an auxiliary state xi where:
    # guard velociity anti-aligned with bandit relative position
    # thus "minimizing" this cost
    xi_v3 = jnp.zeros(12)
    xi_v3 = xi_v3.at[0].set(1.0)  # px_b
    xi_v3 = xi_v3.at[10].set(-1.0)  # vx_g
    J_v3 = eval_quadratic_cost(Q, q, xi_v3)
    assert jnp.isclose(J_v3, -1.0)

    # Create an auxiliary state xi where:
    # guard velociity orthogonal with bandit relative position
    # thus zeroing this cost
    xi = jnp.zeros(12)
    xi = xi.at[0].set(1.0)  # px_b
    xi = xi.at[1].set(1.0)  # py_b
    xi = xi.at[10].set(-1.0)  # vx_g
    xi = xi.at[11].set(1.0)  # vy_g
    J = eval_quadratic_cost(Q, q, xi)
    assert jnp.isclose(J, 0.0)

    # Create an auxiliary state xi where:
    # guard and bandit it same position, thus zeroing 
    # this cost
    xi = jnp.zeros(12)
    xi = xi.at[0].set(-4.84201112)  # px_b
    xi = xi.at[1].set(-0.45189499)  # py_b
    xi = xi.at[8].set(-4.84201112)  # px_g
    xi = xi.at[9].set(-0.45189499)  # py_g
    xi = xi.at[10].set(-2.96138121)  # vx_g
    xi = xi.at[11].set(-0.13022861)  # vy_g
    J = eval_quadratic_cost(Q, q, xi)
    assert jnp.isclose(J, 0.0, atol=1e-6)

    # Create an auxiliary state xi where:
    # guard and bandit in arbitrary states 
    # with known result
    c = 0.724836858805204
    Q, q = lbg2.cost_guard_bandit_alignment_proxy(c, lbg2.PARAMS)
    xi = jnp.zeros(12)
    xi = xi.at[0].set(2.06935135)  # px_b
    xi = xi.at[1].set(-4.06276323)  # py_b
    xi = xi.at[8].set(-3.42004107)  # px_g
    xi = xi.at[9].set(-4.15076036)  # py_g
    xi = xi.at[10].set(4.59334092)  # vx_g
    xi = xi.at[11].set(1.19424719)  # vy_g
    J_exp = 18.35268164379315
    J = eval_quadratic_cost(Q, q, xi)
    assert jnp.isclose(J, J_exp, atol=1e-6)

def test_cost_guard_bandit_distance_expected_entries(lbg2):
    c = 1.5
    Q, q = lbg2.cost_guard_bandit_distance(c, lbg2.PARAMS)

    # Diagonal
    assert Q[0, 0] == pytest.approx(-2 * c)
    assert Q[1, 1] == pytest.approx(-2 * c)
    assert Q[8, 8] == pytest.approx(-2 * c)
    assert Q[9, 9] == pytest.approx(-2 * c)

    # Cross terms
    assert Q[0, 8] == pytest.approx(2 * c)
    assert Q[8, 0] == pytest.approx(2 * c)
    assert Q[1, 9] == pytest.approx(2 * c)
    assert Q[9, 1] == pytest.approx(2 * c)

    # Everything else should be zero
    for i in range(12):
        for j in range(12):
            if (i, j) not in [(0, 0), (1, 1), (8, 8), (9, 9), (0, 8), (8, 0), (1, 9), (9, 1)]:
                assert Q[i, j] == pytest.approx(0.0)

def test_cost_guard_bandit_distance_evaluation_zero_dist(lbg2):
    c = 2.0
    Q, q = lbg2.cost_guard_bandit_distance(c, lbg2.PARAMS)

    # xi: Bandit and guard at same position,  distance should be zero
    xi = jnp.zeros(12).at[0].set(5.0).at[1].set(3.0).at[8].set(5.0).at[9].set(3.0)
    cost = 0.5 * xi.T @ Q @ xi + q.T @ xi
    assert jnp.isclose(cost, 0.0)

def test_cost_guard_bandit_distance_evaluation_separated(lbg2):
    c = 1.0
    Q, q = lbg2.cost_guard_bandit_distance(c, lbg2.PARAMS)

    # xi: Bandit at (0, 0), Lady at (3, 4), distance^2 = 25
    xi = jnp.zeros(12).at[8].set(3.0).at[9].set(4.0)
    cost = 0.5 * xi.T @ Q @ xi + q.T @ xi
    assert jnp.isclose(cost, -25.0)

def test_cost_lady_target_distance_evaluation_zero_dist(lbg2):
    c = 2.0
    px_target = 5.0
    py_target = 3.0
    Q, q = lbg2.cost_lady_target_distance(c, px_target, py_target, lbg2.PARAMS)

    # xi: Lady and Target at same position
    # while this should be a minimum of the cost function,
    # cost is not zero because the squared terms of the
    # constant target position are omitted from the cost 
    # function
    xi = jnp.zeros(12).at[4].set(5.0).at[5].set(3.0)
    cost = 0.5 * xi.T @ Q @ xi + q.T @ xi
    assert jnp.isclose(cost + c*(px_target**2 + py_target**2), 0.0)

def test_cost_lady_target_distance_evaluation_separated(lbg2):
    c = 1.0
    px_target = 0
    py_target = 0
    Q, q = lbg2.cost_lady_target_distance(c, px_target, py_target, lbg2.PARAMS)

    # xi: target at (0, 0), Lady at (3, 4), distance^2 = 25
    xi = jnp.zeros(12).at[4].set(3.0).at[5].set(4.0)
    cost = 0.5 * xi.T @ Q @ xi + q.T @ xi
    assert jnp.isclose(cost + c*(px_target**2 + py_target**2), 25.0)

def test_solve_lq_smoketest(lbg2):
    strat = solve_lqgame_feedback(lbg2.game)

@pytest.mark.regression
def test_solve_lq_strat_regress_1(lbg2):
    """regression test to check against approved strategy values of lq solver"""

    # ~~ ARRANGE ~~
    # load approved strategy values
    approved_filepath = Path(__file__).resolve().parent.joinpath(
        "../approvals/example_doubleint/"
    )
    P_app_filepath = approved_filepath.joinpath(
        "test_solve_lq_strat_regress_1_P.json"
    )
    alpha_app_filepath = approved_filepath.joinpath(
        "test_solve_lq_strat_regress_1_alpha.json"
    )
    with open(P_app_filepath) as f:
        P_approved = jnp.array(json.load(f))
    with open(alpha_app_filepath) as f:
        alpha_approved = jnp.array(json.load(f))

    # ~~ ACT ~~
    # solve for lq strategy
    strat = solve_lqgame_feedback(lbg2.game)

    # ~~ ASSERT ~~
    np.testing.assert_allclose(np.asarray(strat.P), np.asarray(P_approved), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(np.asarray(strat.alpha), np.asarray(alpha_approved), rtol=1e-6, atol=1e-7)


@pytest.mark.regression
def test_solve_lq_strat_regress_2_Qf(lbg2_Qf):
    """regression test: approved strategy values of lq solver w/ terminal cost"""

    # ~~ ARRANGE ~~
    # load approved strategy values
    approved_filepath = Path(__file__).resolve().parent.joinpath("../approvals/example_doubleint/")
    P_app_filepath = approved_filepath.joinpath("test_solve_lq_strat_regress_2_Qf_P.json")
    alpha_app_filepath = approved_filepath.joinpath("test_solve_lq_strat_regress_2_Qf_alpha.json")
    with open(P_app_filepath) as f:
        P_approved = jnp.array(json.load(f))
        P_approved = P_approved[1:] # truncate first entry to match update time convention
    with open(alpha_app_filepath) as f:
        alpha_approved = jnp.array(json.load(f))
        alpha_approved = alpha_approved[1:] # truncate first entry to match update time convention

    # ~~ ACT ~~
    # solve for lq strategy
    strat = solve_lqgame_feedback(lbg2_Qf.game)

    # ~~ ASSERT ~~
    np.testing.assert_allclose(np.asarray(strat.P), np.asarray(P_approved), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(np.asarray(strat.alpha), np.asarray(alpha_approved), rtol=1e-6, atol=1e-7)


@pytest.mark.regression
@pytest.mark.slow
@pytest.mark.benchmark(group="doubleintlq-001")
def test_solve_doubleint_lqlbg_warm_perf_1(benchmark):
    """ Benchmark performance of warm-started linear-quadratic solver on 
    2-player integrator system defined in example 37 in 
    https://clearoboticslab.github.io/documents/smooth_game_theory.pdf
    """

    lqg = DoubleInt_LQLBG_C2()

    def run():
        st = solve_lqgame_feedback(lqg.game)
        jax.block_until_ready(st)
    
    # warm compile once
    run()

    # benchmark warm-started run
    benchmark(run)