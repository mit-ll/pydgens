# Copyright 2026 MIT Lincoln Laboratory
# SPDX-License-Identifier: MIT

import pytest
import time
import jax
import numpy as np
import jax.numpy as jnp

from pydgens.examples.ir_unicycle import Unicycle
from pydgens.ir.strategytypes import FixedStepAffineStrategies
from pydgens.ir.trajectorytypes import FixedStepSystemTrajectory
from pydgens.ir.systemtypes import propagate_system_trajectory
from pydgens.ir.gametypes import approx_linear_quadratic_game
from pydgens.solvers.ilqsolver import solve_ilqgame_feedback
from pydgens.utils.utils import is_block_diagonal


@pytest.fixture
def ir_unicycle():
    return Unicycle()

@pytest.mark.parametrize("x,dx",
    [
        ([0, 0, 0, 0], [0, 0, 0, 0]),
        ([0, 0, 0, 1.0], [1.0, 0, 0, 0]),
    ]
)
def test_dynamics_no_input(ir_unicycle, x, dx):
    # check that state derivatives match expectations with no inputs

    # ~~ ARRANGE ~~

    u = jnp.zeros(2)  # No control input
    x = jnp.asarray(x)
    dx_exp = jnp.asarray(dx)

    # ~~ ACT ~~
    dx_act = ir_unicycle.game.dynamics(0, x, u)

    # ~~ ASSERT ~~
    assert jnp.allclose(dx_exp, dx_act, atol=1e-6)

@pytest.mark.parametrize("x,u", 
    [
        ([0, 0, 0, 0], [0, 0]),
        ([0, 0, 0, 0], [0, 1.0]),
    ]
)
def test_dynamics_w_input(ir_unicycle, x, u):
    # check that state derivatives match hard-coded computations

    # ~~ ARRANGE ~~

    x = jnp.asarray(x)
    u = jnp.asarray(u)

    dx_exp = jnp.array([x[3]*jnp.cos(x[2]), x[3]*jnp.sin(x[2]), u[0], u[1]])

    # ~~ ACT ~~
    dx = ir_unicycle.game.dynamics(0, x, u)

    # ~~ ASSERT ~~
    assert jnp.allclose(dx, dx_exp)

def test_approx_lqgame_block_diagonal_cost(ir_unicycle):
    """ensure that lq-approximating produces block-diagonal R matrices, since this check is turned off in ilqsolver"""

    # ~~ ARRANGE ~~
    nlgame = ir_unicycle.game

    # initial state
    x0 = jnp.array([1.0, 1.0, 0.0, 0.5])

    # zero strategy
    strat = FixedStepAffineStrategies(
        tg = nlgame.tg,
        P = jnp.zeros((nlgame.nt-1, nlgame.nu, nlgame.nx)),
        alpha = jnp.zeros((nlgame.nt-1, nlgame.nu)) 
    )

    # trajectory from zero-strategy to use as operating point of lq-approx
    traj = propagate_system_trajectory(
        nlgame.cs,
        x0 = x0,
        strategy = strat
    )

    # ~~ ACT ~~
    lqgame = approx_linear_quadratic_game(nlgame, op=traj)

    # ~~ ASSERT ~~
    for i in range(nlgame.N):
        for k in range(nlgame.nt):
            assert is_block_diagonal(R=lqgame.R[k,i], u_splits=nlgame.u_splits)

@pytest.mark.slow
def test_solve_ir_unicycle_converge(ir_unicycle):
    # Run the iterative linear-quadratic solver on the IR unicycle problem

    # ~~ ARRANGE ~~

    # initial state
    x0 = jnp.array([1.0, 1.0, 0.0, 0.5])

    # ~~ ACT ~~

    # compute nash strategy for nonlinear game
    conv, nl_traj, nl_strat = solve_ilqgame_feedback(ir_unicycle.game, x0)

    # ~~ ASSERT ~~
    
    # Check that solver has converged
    assert conv

def make_unicycle_solver_inputs_1():
    """produce standardized inputs for ilqsolver for unicycle problem"""

    # game wrapper and initial state to be solved
    ir_unicycle = Unicycle(nt=20, dt=0.1)
    x0 = jnp.array([1.0, 1.0, 0.0, 0.5])
    nlgame = ir_unicycle.game

    # optional solver inputs specificied for future consistency
    init_traj = FixedStepSystemTrajectory(
        tg=nlgame.tg,
        xs=jnp.zeros((nlgame.nt, nlgame.nx)),
        us=jnp.zeros((nlgame.nt-1, nlgame.nu)),
    )
    init_strat = FixedStepAffineStrategies(
        tg=nlgame.tg,
        P=jnp.zeros((nlgame.nt-1, nlgame.nu, nlgame.nx)),
        alpha=jnp.zeros((nlgame.nt-1, nlgame.nu)),
    )

    max_iters = 50
    converged_max_diff = 5e-2
    backtrack_max_iters = 20
    backtrack_scale_init = 0.5
    backtrack_scale_step = 0.5
    backtrack_scale_max_diff = 30 * 5e-2
    logger = None

    return (
        ir_unicycle, x0, init_traj, init_strat, 
        max_iters, 
        converged_max_diff, 
        backtrack_max_iters, 
        backtrack_scale_init, 
        backtrack_scale_step, 
        backtrack_scale_max_diff, 
        logger
    )

@pytest.mark.regression
@pytest.mark.slow
def test_unicycle_solve_approved_outputs_1():
    """regression test to ensure unicycle solution continues to match approved results"""
    # ~~ ARRANGE ~~

    (ir_unicycle, x0, init_traj, init_strat, 
        max_iters, 
        converged_max_diff, 
        backtrack_max_iters, 
        backtrack_scale_init, 
        backtrack_scale_step, 
        backtrack_scale_max_diff, 
        logger
    ) = make_unicycle_solver_inputs_1()

    # ---- approved outputs (inline) ----
    # xs_approved = jnp.array(
    #     [[ 1.        ,  1.        ,  0.        ,  0.5       ],
    #     [ 1.052043  ,  0.99557114, -0.16734532,  0.545841  ],
    #     [ 1.106893  ,  0.98165095, -0.32778275,  0.5871501 ],
    #     [ 1.1625289 ,  0.9578413 , -0.4794078 ,  0.6243404 ],
    #     [ 1.2171142 ,  0.92432666, -0.6206441 ,  0.657784  ],
    #     [ 1.2691399 ,  0.8817405 , -0.75030243,  0.68781507],
    #     [ 1.3174993 ,  0.83100986, -0.8675953 ,  0.7147341 ],
    #     [ 1.3615003 ,  0.77320766, -0.97211814,  0.7388102 ],
    #     [ 1.4008312 ,  0.7094319 , -1.0638096 ,  0.7602842 ],
    #     [ 1.4354995 ,  0.6407198 , -1.1429024 ,  0.7793708 ],
    #     [ 1.465758  ,  0.5679967 , -1.2098722 ,  0.79626083],
    #     [ 1.4920317 ,  0.49205428, -1.2653909 ,  0.8111234 ],
    #     [ 1.5148526 ,  0.41354933, -1.310286  ,  0.8241068 ],
    #     [ 1.5348047 ,  0.33301598, -1.3455088 ,  0.8353411 ],
    #     [ 1.5524789 ,  0.25088465, -1.3721068 ,  0.84493864],
    #     [ 1.5684394 ,  0.16750301, -1.3912036 ,  0.8529953 ],
    #     [ 1.5831958 ,  0.08315534, -1.4039812 ,  0.8595916 ],
    #     [ 1.5971848 , -0.0019213 , -1.4116647 ,  0.86479366],
    #     [ 1.6107543 , -0.08752479, -1.4155099 ,  0.8686534 ],
    #     [ 1.6241537 , -0.17347977, -1.4167914 ,  0.8712095 ]]
    # )
    xs_approved = jnp.array([
        [ 1.       ,   1.        ,  0.        ,  0.5       ],
        [ 1.0520906,   0.99647623, -0.13317636,  0.5449668 ],
        [ 1.1074808,   0.9854283 , -0.25906864,  0.5854027 ],
        [ 1.1647724,   0.9665518 , -0.37631497,  0.6217122 ],
        [ 1.222721 ,   0.9399379 , -0.48383918,  0.6542584 ],
        [ 1.2803121,   0.90598416, -0.58088017,  0.68336666],
        [ 1.3367982,   0.8652985 , -0.66699517,  0.7093282 ],
        [ 1.3917003,   0.81861037, -0.7420442 ,  0.73240244],
        [ 1.4447838,   0.76669735, -0.80616236,  0.7528204 ],
        [ 1.4960183,   0.7103317 , -0.85972667,  0.7707859 ],
        [ 1.5455291,   0.650245  , -0.90332174,  0.7864789 ],
        [ 1.5935489,   0.58710855, -0.937706  ,  0.8000562 ],
        [ 1.640372 ,   0.5215246 , -0.96378124,  0.8116536 ],
        [ 1.686314 ,   0.45402503, -0.98256457,  0.82138705],
        [ 1.7316786,   0.3850734 , -0.9951633 ,  0.8293539 ],
        [ 1.776731 ,   0.31506827, -1.0027528 ,  0.8356339 ],
        [ 1.8216774,   0.24434623, -1.0065556 ,  0.8402897 ],
        [ 1.8666512,   0.17318358, -1.0078237 ,  0.84336793],
        [ 1.9117028,   0.10179752, -1.0078237 ,  0.8448994 ],
        [ 1.9567953,   0.0303467 , -1.0078237 ,  0.8448994 ],
    ])
    # us_approved = jnp.array(
    #     [[-1.673453  ,  0.45840925],
    #     [-1.6043743 ,  0.41309094],
    #     [-1.5162501 ,  0.3719036 ],
    #     [-1.4123626 ,  0.33443522],
    #     [-1.2965833 ,  0.3003112 ],
    #     [-1.1729283 ,  0.26919037],
    #     [-1.0452278 ,  0.24076152],
    #     [-0.9169147 ,  0.21474022],
    #     [-0.7909281 ,  0.19086635],
    #     [-0.6696977 ,  0.16890103],
    #     [-0.5551854 ,  0.14862484],
    #     [-0.44895136,  0.12983483],
    #     [-0.35222703,  0.11234319],
    #     [-0.26598057,  0.09597498],
    #     [-0.19096872,  0.08056644],
    #     [-0.12777507,  0.06596369],
    #     [-0.07683513,  0.05202049],
    #     [-0.0384531 ,  0.03859749],
    #     [-0.0128147 ,  0.0255605 ],
    #     [ 0.        ,  0.01277909]]
    # )
    us_approved = jnp.array([
        [-1.3317634 ,  0.44966775],
        [-1.2589228 ,  0.40435964],
        [-1.1724633 ,  0.36309528],
        [-1.0752418 ,  0.32546186],
        [-0.97040963,  0.29108304],
        [-0.8611503 ,  0.259615  ],
        [-0.7504902 ,  0.23074317],
        [-0.64118135,  0.20417875],
        [-0.5356437 ,  0.17965609],
        [-0.4359505 ,  0.15693003],
        [-0.3438426 ,  0.13577324],
        [-0.2607522 ,  0.11597413],
        [-0.18783304,  0.09733486],
        [-0.12598807,  0.07966891],
        [-0.07589467,  0.06279963],
        [-0.03802754,  0.04655835],
        [-0.01268205,  0.03078263],
        [ 0.        ,  0.01531474],
        [ 0.        ,  0.        ],
    ])

    # ~~ ACT & ASSERT ~~

    conv, traj, strat = solve_ilqgame_feedback(
        nlgame=ir_unicycle.game,
        x0=x0,
        init_traj=init_traj,
        init_strat=init_strat,
        max_iters=max_iters,
        converged_max_diff=converged_max_diff,
        backtrack_max_iters=backtrack_max_iters,
        backtrack_scale_init=backtrack_scale_init,
        backtrack_scale_step=backtrack_scale_step,
        backtrack_scale_max_diff=backtrack_scale_max_diff,
        logger=logger
    )
    
    # ---- approved comparisons ----
    assert conv
    np.testing.assert_allclose(np.asarray(traj.xs), np.asarray(xs_approved), rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(np.asarray(traj.us), np.asarray(us_approved), rtol=1e-5, atol=1e-7)


@pytest.mark.regression
@pytest.mark.slow
@pytest.mark.benchmark(group="ir-unicycle-001")
def test_unicycle_solve_cold_perf(benchmark):
    """benchmark the cold-run performance of solving unicycle problem"""
    # ~~ ARRANGE ~~
    (ir_unicycle, x0, init_traj, init_strat, 
        max_iters, 
        converged_max_diff, 
        backtrack_max_iters, 
        backtrack_scale_init, 
        backtrack_scale_step, 
        backtrack_scale_max_diff, 
        logger
    ) = make_unicycle_solver_inputs_1()

    # ~~ ACT & ASSERT ~~

    # Ensure no cached executables so this is truly "cold"
    jax.clear_caches()

    # ---- cold (compile + run) ----
    # t0 = time.perf_counter()
    def cold_run():
        results = solve_ilqgame_feedback(
            nlgame=ir_unicycle.game,
            x0=x0,
            init_traj=init_traj,
            init_strat=init_strat,
            max_iters=max_iters,
            converged_max_diff=converged_max_diff,
            backtrack_max_iters=backtrack_max_iters,
            backtrack_scale_init=backtrack_scale_init,
            backtrack_scale_step=backtrack_scale_step,
            backtrack_scale_max_diff=backtrack_scale_max_diff,
            logger=logger
        )
        # make sure all device work finishes before timing stops
        jax.block_until_ready(results[0])
        return results
    
    # cold = time.perf_counter() - t0
    conv, traj, strat = benchmark.pedantic(
        cold_run,
        iterations=1,       # one timing sample per round
        rounds=1,           # exactly one round → one cold timing
        warmup_rounds=0,    # no warmup
    )


@pytest.mark.regression
@pytest.mark.slow
@pytest.mark.benchmark(group="ir-unicycle-002")
def test_unicycle_solve_warm_perf(benchmark):
    """benchmark the warm-started performance of solving unicycle problem"""
    # ~~ ARRANGE ~~
    (ir_unicycle, x0, init_traj, init_strat, 
        max_iters, 
        converged_max_diff, 
        backtrack_max_iters, 
        backtrack_scale_init, 
        backtrack_scale_step, 
        backtrack_scale_max_diff, 
        logger
    ) = make_unicycle_solver_inputs_1()

    # ~~ ACT & ASSERT ~~
    def run():
        results = solve_ilqgame_feedback(
            nlgame=ir_unicycle.game,
            x0=x0,
            init_traj=init_traj,
            init_strat=init_strat,
            max_iters=max_iters,
            converged_max_diff=converged_max_diff,
            backtrack_max_iters=backtrack_max_iters,
            backtrack_scale_init=backtrack_scale_init,
            backtrack_scale_step=backtrack_scale_step,
            backtrack_scale_max_diff=backtrack_scale_max_diff,
            logger=logger,
        )
        # make sure all device work finishes before timing stops
        jax.block_until_ready(results[0])
        return results
    
    # explicitly invoke solver as warmup round
    run()
    
    # benchmark the warmstarted solver
    benchmark(run)

