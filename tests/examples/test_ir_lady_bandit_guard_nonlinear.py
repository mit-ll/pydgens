# Copyright 2026 MIT Lincoln Laboratory
# SPDX-License-Identifier: MIT

import pytest
import jax.numpy as jnp

from pydgens.examples.ir_lady_bandit_guard_nonlinear import LadyBanditGuardNonlinear
from pydgens.ir.trajectorytypes import FixedStepSystemTrajectory
from pydgens.ir.strategytypes import FixedStepAffineStrategies
from pydgens.ir.costtypes import (
    quadraticize_cost_joint_ctrl_playerwise, 
    quadraticize_cost_joint_ctrl_playerwise_trajectory, 
)
from pydgens.solvers.ilqsolver import solve_ilqgame_feedback


@pytest.fixture
def lbg1():
    return LadyBanditGuardNonlinear()

@pytest.mark.parametrize("x,dx",
    [
        ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        ([0, 0, 0, 1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0], [1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0, 0, 0, 0]), 
        ([0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, -1.0], [0, 0, 0, 0, 1.0, 0, 0, 0, -1.0, 0, 0, 0]),
        ([0, 0, jnp.pi/2, 1.0, 0, 0, jnp.pi, 1.0, 0, 0, 3*jnp.pi/2, 1.0], [0, 1, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0]),
        ([29873, 0.847, jnp.pi/2, 1.0, -2.837, 304.219, jnp.pi, 1.0, -176., 43.98, 3*jnp.pi/2, 1.0], [0, 1, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0]),
    ]
)
def test_dynamics_no_input(lbg1, x, dx):
    # check that state derivatives match expectations with no inputs

    # ~~ ARRANGE ~~

    u = jnp.zeros(6)  # No control input
    x = jnp.asarray(x)
    dx_exp = jnp.asarray(dx)

    # ~~ ACT ~~
    dx_act = lbg1.game.dynamics(0, x, u)

    # ~~ ASSERT ~~
    assert jnp.allclose(dx_exp, dx_act, atol=1e-6)

@pytest.mark.parametrize("x,u", 
    [
        ([0, 0, 0, 1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0], [0, 0, 0, 0, 0, 0]),
        ([0, 0, 0, 0.0, 1, 1, jnp.pi/4.0, 1.0, -1, -1, 5*jnp.pi/4, 1.0], [ 0.208, -1.673,  2.688,  0.16,  -1.576,  1.095]),
        ([0.689, -1.34,   2.385,  0.323, -1.412, -0.698,  1.246, -0.191,  2.071,  1.591, -1.587, -0.302], [ 0.208, -1.673,  2.688,  0.16,  -1.576,  1.095])
    ]
)
def test_dynamics_w_input(lbg1, x, u):
    # check that state derivatives match hard-coded computations

    # ~~ ARRANGE ~~

    x = jnp.asarray(x)
    u = jnp.asarray(u)

    dx_exp = jnp.array([
        x[3] * jnp.cos(x[2]), x[3] * jnp.sin(x[2]), u[0], u[1],
        x[7] * jnp.cos(x[6]), x[7] * jnp.sin(x[6]), u[2], u[3],
        x[11] * jnp.cos(x[10]), x[11] * jnp.sin(x[10]), u[4], u[5],
    ])

    # ~~ ACT ~~
    dx = lbg1.game.dynamics(0, x, u)

    # ~~ ASSERT ~~
    assert jnp.allclose(dx, dx_exp)

@pytest.mark.parametrize("x,c", 
    [
        ([0, 0, 0, 1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0], 0.0),
        ([0, 0, 0, 0.0, 1, 1, jnp.pi/4.0, 1.0, -1, -1, 5*jnp.pi/4, 1.0], 0.0),
        ([0, 0, 0, 1.0, 1, 0, jnp.pi/2.0, 1.0, -1, 0, 0, 0, 0], 1.0),
        ([0.689, -1.34,   jnp.pi,  0.323, 0.689, 38.054,  1.246, -0.191,  2.071,  1.591, -1.587, -0.302], 0.0)
    ]
)
def test_bandit_lady_alignment_cosine(lbg1, x, c):
    # check bandit-lady alignment cosine used in cost computation

    # ~~ ARRANGE ~~
    x = jnp.asarray(x)

    # ~~ ACT ~~
    c_act = lbg1.bandit_lady_alignment_cosine(None, x, None, lbg1.PARAMS)

    # ~~ ASSERT ~~
    assert jnp.isclose(c, c_act, atol=1e-7)

@pytest.mark.parametrize("x,c", 
    [
        ([0, 0, 0, 1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0], 0.0),
        ([0, 0, 0, 0.0, 1, 1, jnp.pi/4.0, 1.0, -1, -1, 5*jnp.pi/4, 1.0], -1.0),
        ([0, 0, 0, 1.0, 1, 0, jnp.pi/2.0, 1.0, -1, 0, 0, 0, 0], 0.0),
        ([2.071,  2.591,   jnp.pi,  0.323, 1.412, -1.34,  1.246, -0.191,  2.071,  1.591, -jnp.pi, -0.302], 0.0)
    ]
)
def test_guard_bandit_alignment_cosine(lbg1, x, c):
    # check guard-bandit alignment cosine used in cost computation

    # ~~ ARRANGE ~~
    x = jnp.asarray(x)

    # ~~ ACT ~~
    c_act = lbg1.guard_bandit_alignment_cosine(None, x, None, lbg1.PARAMS)

    # ~~ ASSERT ~~
    assert jnp.isclose(c, c_act, atol=1e-7)

@pytest.mark.parametrize("x,c", 
    [
        ([0, 0, 0, 1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0], 0.0),
        ([0, 0, 0, 0.0, 1, 1, jnp.pi/4.0, 1.0, -1, -1, 5*jnp.pi/4, 1.0], 2.0),
        ([0, 0, 0, 1.0, 1, 0, jnp.pi/2.0, 1.0, -1, 0, 0, 0, 0], 1.0),
        ([0.689, -1.34,   jnp.pi,  0.323, 1.412, -1.34,  1.246, -0.191,  2.071,  1.591, -1.587, -0.302], (1.412-0.689)**2)
    ]
)
def test_bandit_lady_proximity(lbg1, x, c):
    # check bandit-lady squared distance used in cost computation

    # ~~ ARRANGE ~~
    x = jnp.asarray(x)

    # ~~ ACT ~~
    c_act = lbg1.bandit_lady_proximity(None, x, None, lbg1.PARAMS)

    # ~~ ASSERT ~~
    assert jnp.isclose(c, c_act)

@pytest.mark.parametrize("x,c", 
    [
        ([0, 0, 0, 1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0], 0.0),
        ([0, 0, 0, 0.0, 1, 1, jnp.pi/4.0, 1.0, -1, -1, 5*jnp.pi/4, 1.0], 2.0),
        ([0, 0, 0, 1.0, 1, 0, jnp.pi/2.0, 1.0, -1, 0, 0, 0, 0], 1.0),
        ([0.689, -1.34,   jnp.pi,  0.323, 1.412, -1.34,  1.246, -0.191,  2.071,  1.591, -1.587, -0.302], (0.689-2.071)**2+(-1.34-1.591)**2)
    ]
)
def test_guard_bandit_proximity(lbg1, x, c):
    # check bandit-lady squared distance used in cost computation

    # ~~ ARRANGE ~~
    x = jnp.asarray(x)

    # ~~ ACT ~~
    c_act = lbg1.guard_bandit_proximity(None, x, None, lbg1.PARAMS)

    # ~~ ASSERT ~~
    assert jnp.isclose(c, c_act)

@pytest.mark.parametrize("x,v_tar,c", 
    [
        ([0, 0, 0, 1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0], 1.0, 0.0),
        ([0, 0, 0, 0.0, 1, 1, jnp.pi/4.0, 1.0, -1, -1, 5*jnp.pi/4, 1.0], 1.0, 1.0),
        ([0.689, -1.34, 2.385,  -0.323, -1.412, -0.698,  1.246, -0.191,  2.071,  1.591, -1.587, -0.302], -2.0, (-0.323+2)**2)
    ]
)
def test_bandit_speed_deviation(lbg1, x, v_tar, c):
    # check bandit-lady squared distance used in cost computation

    # ~~ ARRANGE ~~
    x = jnp.asarray(x)

    # ~~ ACT ~~
    c_act = lbg1.bandit_speed_deviation(None, x, None, lbg1.PARAMS, v_tar)

    # ~~ ASSERT ~~
    assert jnp.isclose(c, c_act)

@pytest.mark.parametrize("x,px_tar,py_tar,c", 
    [
        ([0, 0, 0, 1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0], 0.0, 0.0, 0.0),
        ([0, 0, 0, 0.0, 1, 1, jnp.pi/4.0, 1.0, -1, -1, 5*jnp.pi/4, 1.0], 0.0, 0.0, 2.0),
        ([0.689, -1.34, 2.385,  -0.323, -1.412, -0.698,  1.246, -0.191,  2.071,  1.591, -1.587, -0.302], 2.071,  1.591, (-1.412-2.071)**2 + (-0.698-1.591)**2)
    ]
)
def test_lady_target_deviation(lbg1, x, px_tar, py_tar, c):
    # check bandit-lady squared distance used in cost computation

    # ~~ ARRANGE ~~
    x = jnp.asarray(x)

    # ~~ ACT ~~
    c_act = lbg1.lady_target_deviation(None, x, None, lbg1.PARAMS, px_tar, py_tar)

    # ~~ ASSERT ~~
    assert jnp.isclose(c, c_act)

def test_bandit_cost_smoketest(lbg1):
    # check that the bandit composite cost executes without error

    # ~~ ARRANGE ~~
    x = jnp.asarray([0.689, -1.34,   2.385,  0.323, -1.412, -0.698,  1.246, -0.191,  2.071,  1.591, -1.587, -0.302])
    u = jnp.asarray([ 0.208, -1.673,  2.688,  0.16,  -1.576,  1.095])

    # ~~ ACT ~~
    _ = lbg1.bandit_cost(
        0.0, x, u, lbg1.PARAMS, 
        lbg1.px_target,
        lbg1.py_target,
        lbg1.v_b_cruise,
        lbg1.w_b_bl_align, 
        lbg1.w_b_bl_dist, 
        lbg1.w_b_gb_align,
        lbg1.w_b_gb_dist,
        lbg1.w_b_lt_dist,
        lbg1.w_b_spd,
        lbg1.w_b_omg,
        lbg1.w_b_acc
    )

    # ~~ ASSERT ~~
    pass

def test_bandit_cost_components():
    # check bookkeeping of weights by ensuring composite cost repropduces underlying component
    # cost when other weights are zeroed

    # ~~ ARRANGE ~~

    # randomly generated states and game parameters
    t = 0.0
    x = jnp.asarray([ -5.556,  -9.633,  -0.139,  -4.465,   4.552,   4.976,   1.516,   0.227,  17.906, -8.96, -16.503, 9.062])
    u = jnp.asarray([-4.908,  9.381, -6.793,  8.674, -0.627, -1.191])
    lbg1 = LadyBanditGuardNonlinear(
        px_target = -12.087,
        py_target = 8.754,
        v_b_cruise = 65.13,
        w_b_bl_align = 7.613,
        w_b_bl_dist = 6.189,
        w_b_gb_align = 7.395,
        w_b_gb_dist = 1.482,
        w_b_lt_dist = 5.811,
        w_b_spd = 7.635,
        w_b_omg = 8.233,
        w_b_acc = 9.298
    )


    # ~~ ACT & ASSERT ~~
    # bandit-lady alignment cost 
    composite = lbg1.bandit_cost(
        t, x, u, lbg1.PARAMS, 
        lbg1.px_target,
        lbg1.py_target,
        lbg1.v_b_cruise,
        lbg1.w_b_bl_align,
        0.0, 
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    component = -lbg1.w_b_bl_align * lbg1.bandit_lady_alignment_cosine(t, x, u, lbg1.PARAMS)
    assert jnp.isclose(composite, component)

    # bandit-lady proximity cost
    composite = lbg1.bandit_cost(
        t, x, u, lbg1.PARAMS, 
        lbg1.px_target,
        lbg1.py_target,
        lbg1.v_b_cruise,
        0.0,
        lbg1.w_b_bl_dist,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    component = lbg1.w_b_bl_dist * lbg1.bandit_lady_proximity(t, x, u, lbg1.PARAMS)
    assert jnp.isclose(composite, component)

    # guard-bandit alignmnet cost
    composite = lbg1.bandit_cost(
        t, x, u, lbg1.PARAMS, 
        lbg1.px_target,
        lbg1.py_target,
        lbg1.v_b_cruise,
        0.0,
        0.0,
        lbg1.w_b_gb_align,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    component = lbg1.w_b_gb_align * lbg1.guard_bandit_alignment_cosine(t, x, u, lbg1.PARAMS)
    assert jnp.isclose(composite, component)

    # guard-bandit proximity cost
    composite = lbg1.bandit_cost(
        t, x, u, lbg1.PARAMS, 
        lbg1.px_target,
        lbg1.py_target,
        lbg1.v_b_cruise,
        0.0,
        0.0,
        0.0,
        lbg1.w_b_gb_dist,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    component = -lbg1.w_b_gb_dist * lbg1.guard_bandit_proximity(t, x, u, lbg1.PARAMS)
    assert jnp.isclose(composite, component)

    # lady-target deviation cost
    composite = lbg1.bandit_cost(
        t, x, u, lbg1.PARAMS, 
        lbg1.px_target,
        lbg1.py_target,
        lbg1.v_b_cruise,
        0.0,
        0.0,
        0.0,
        0.0,
        lbg1.w_b_lt_dist,
        0.0,
        0.0,
        0.0,
    )
    component = - lbg1.w_b_lt_dist * lbg1.lady_target_deviation(t, x, u, lbg1.PARAMS, lbg1.px_target, lbg1.py_target)
    assert jnp.isclose(composite, component)

    # bandit cruise deviation cost
    composite = lbg1.bandit_cost(
        t, x, u, lbg1.PARAMS, 
        lbg1.px_target,
        lbg1.py_target,
        lbg1.v_b_cruise,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        lbg1.w_b_spd,
        0.0,
        0.0,
    )
    component = lbg1.w_b_spd * lbg1.bandit_speed_deviation(t, x, u, lbg1.PARAMS, lbg1.v_b_cruise)
    assert jnp.isclose(composite, component)

    # bandit turnrate control effort cost
    composite = lbg1.bandit_cost(
        t, x, u, lbg1.PARAMS, 
        lbg1.px_target,
        lbg1.py_target,
        lbg1.v_b_cruise,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        lbg1.w_b_omg,
        0.0,
    )
    component = lbg1.w_b_omg * lbg1.bandit_turnrate_effort(t, x, u, lbg1.PARAMS)
    assert jnp.isclose(composite, component)

    # bandit turnrate control effort cost
    composite = lbg1.bandit_cost(
        t, x, u, lbg1.PARAMS, 
        lbg1.px_target,
        lbg1.py_target,
        lbg1.v_b_cruise,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        lbg1.w_b_acc,
    )
    component = lbg1.w_b_acc * lbg1.bandit_accel_effort(t, x, u, lbg1.PARAMS)
    assert jnp.isclose(composite, component)

def test_lady_cost_components():
    # check bookkeeping of weights by ensuring composite cost repropduces underlying component
    # cost when other weights are zeroed

    # ~~ ARRANGE ~~

    # randomly generated states and game parameters
    t = 0.0
    x = jnp.asarray([ -2.791,  12.611,   3.257, -17.343, -15.026,  16.302,  -5.911,   3.35,    7.827,  2.227,  -2.327,   7.072])
    u = jnp.asarray([-18.685,   1.403,  -3.805,   0.419,   4.469,  -8.042])
    lbg1 = LadyBanditGuardNonlinear(
        px_target = -5.7,
        py_target = 15.369,
        v_l_cruise = 6.482,
        w_l_bl_align = 4.116,
        w_l_bl_dist = 8.768,
        w_l_gb_align = 3.214,
        w_l_gb_dist = 5.144,
        w_l_lt_dist = 6.579,
        w_l_spd = 8.147,
        w_l_omg = 6.836,
        w_l_acc = 3.306
    )

    # ~~ ACT & ASSERT ~~
    # bandit-lady alignment cost 
    composite = lbg1.lady_cost(
        t, x, u, lbg1.PARAMS, 
        lbg1.px_target,
        lbg1.py_target,
        lbg1.v_l_cruise,
        lbg1.w_l_bl_align,
        0.0, 
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    component = lbg1.w_l_bl_align * lbg1.bandit_lady_alignment_cosine(t, x, u, lbg1.PARAMS)
    assert jnp.isclose(composite, component)

    # bandit-lady proximity cost
    composite = lbg1.lady_cost(
        t, x, u, lbg1.PARAMS, 
        lbg1.px_target,
        lbg1.py_target,
        lbg1.v_l_cruise,
        0.0,
        lbg1.w_l_bl_dist,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    component = - lbg1.w_l_bl_dist * lbg1.bandit_lady_proximity(t, x, u, lbg1.PARAMS)
    assert jnp.isclose(composite, component)

    # guard-bandit alignmnet cost
    composite = lbg1.lady_cost(
        t, x, u, lbg1.PARAMS, 
        lbg1.px_target,
        lbg1.py_target,
        lbg1.v_l_cruise,
        0.0,
        0.0,
        lbg1.w_l_gb_align,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    component = - lbg1.w_l_gb_align * lbg1.guard_bandit_alignment_cosine(t, x, u, lbg1.PARAMS)
    assert jnp.isclose(composite, component)

    # guard-bandit proximity cost
    composite = lbg1.lady_cost(
        t, x, u, lbg1.PARAMS, 
        lbg1.px_target,
        lbg1.py_target,
        lbg1.v_l_cruise,
        0.0,
        0.0,
        0.0,
        lbg1.w_l_gb_dist,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    component = lbg1.w_l_gb_dist * lbg1.guard_bandit_proximity(t, x, u, lbg1.PARAMS)
    assert jnp.isclose(composite, component)

    # lady-target deviation cost
    composite = lbg1.lady_cost(
        t, x, u, lbg1.PARAMS, 
        lbg1.px_target,
        lbg1.py_target,
        lbg1.v_l_cruise,
        0.0,
        0.0,
        0.0,
        0.0,
        lbg1.w_l_lt_dist,
        0.0,
        0.0,
        0.0,
    )
    component = lbg1.w_l_lt_dist * lbg1.lady_target_deviation(t, x, u, lbg1.PARAMS, lbg1.px_target, lbg1.py_target)
    assert jnp.isclose(composite, component)

    # lady cruise deviation cost
    composite = lbg1.lady_cost(
        t, x, u, lbg1.PARAMS, 
        lbg1.px_target,
        lbg1.py_target,
        lbg1.v_l_cruise,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        lbg1.w_l_spd,
        0.0,
        0.0,
    )
    component = lbg1.w_l_spd * lbg1.lady_speed_deviation(t, x, u, lbg1.PARAMS, lbg1.v_l_cruise)
    assert jnp.isclose(composite, component)

    # lady turnrate control effort cost
    composite = lbg1.lady_cost(
        t, x, u, lbg1.PARAMS, 
        lbg1.px_target,
        lbg1.py_target,
        lbg1.v_l_cruise,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        lbg1.w_l_omg,
        0.0,
    )
    component = lbg1.w_l_omg * lbg1.lady_turnrate_effort(t, x, u, lbg1.PARAMS)
    assert jnp.isclose(composite, component)

    # bandit turnrate control effort cost
    composite = lbg1.lady_cost(
        t, x, u, lbg1.PARAMS, 
        lbg1.px_target,
        lbg1.py_target,
        lbg1.v_l_cruise,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        lbg1.w_l_acc,
    )
    component = lbg1.w_l_acc * lbg1.lady_accel_effort(t, x, u, lbg1.PARAMS)
    assert jnp.isclose(composite, component)

def test_guard_cost_components():
    # check bookkeeping of weights by ensuring composite cost repropduces underlying component
    # cost when other weights are zeroed

    # ~~ ARRANGE ~~

    # randomly generated states and game parameters
    t = 0.0
    x = jnp.asarray([12.82, -15.095, 1.61, 2.171, 2.798, 3.649, -7.378, 5.781, -8.615, -5.856, 11.777, 7.999])
    u = jnp.asarray([19.188, 15.052, -7.286, -2.191, -6.718, 5.298])
    lbg1 = LadyBanditGuardNonlinear(
        px_target = 6.81,
        py_target = 1.393,
        v_g_cruise = 35.598,
        w_g_bl_align = 1.575,
        w_g_bl_dist = 7.192,
        w_g_gb_align = 7.201,
        w_g_gb_dist = 9.753,
        w_g_lt_dist = 7.95,
        w_g_spd = 0.467,
        w_g_omg = 5.795,
        w_g_acc = 2.178
    )

    # ~~ ACT & ASSERT ~~
    # bandit-lady alignment cost 
    composite = lbg1.guard_cost(
        t, x, u, lbg1.PARAMS, 
        lbg1.px_target,
        lbg1.py_target,
        lbg1.v_g_cruise,
        lbg1.w_g_bl_align,
        0.0, 
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    component = lbg1.w_g_bl_align * lbg1.bandit_lady_alignment_cosine(t, x, u, lbg1.PARAMS)
    assert jnp.isclose(composite, component)

    # bandit-lady proximity cost
    composite = lbg1.guard_cost(
        t, x, u, lbg1.PARAMS, 
        lbg1.px_target,
        lbg1.py_target,
        lbg1.v_g_cruise,
        0.0,
        lbg1.w_g_bl_dist,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    component = - lbg1.w_g_bl_dist * lbg1.bandit_lady_proximity(t, x, u, lbg1.PARAMS)
    assert jnp.isclose(composite, component)

    # guard-bandit alignmnet cost
    composite = lbg1.guard_cost(
        t, x, u, lbg1.PARAMS, 
        lbg1.px_target,
        lbg1.py_target,
        lbg1.v_g_cruise,
        0.0,
        0.0,
        lbg1.w_g_gb_align,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    component = - lbg1.w_g_gb_align * lbg1.guard_bandit_alignment_cosine(t, x, u, lbg1.PARAMS)
    assert jnp.isclose(composite, component)

    # guard-bandit proximity cost
    composite = lbg1.guard_cost(
        t, x, u, lbg1.PARAMS, 
        lbg1.px_target,
        lbg1.py_target,
        lbg1.v_g_cruise,
        0.0,
        0.0,
        0.0,
        lbg1.w_g_gb_dist,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    component = lbg1.w_g_gb_dist * lbg1.guard_bandit_proximity(t, x, u, lbg1.PARAMS)
    assert jnp.isclose(composite, component)

    # lady-target deviation cost
    composite = lbg1.guard_cost(
        t, x, u, lbg1.PARAMS, 
        lbg1.px_target,
        lbg1.py_target,
        lbg1.v_g_cruise,
        0.0,
        0.0,
        0.0,
        0.0,
        lbg1.w_g_lt_dist,
        0.0,
        0.0,
        0.0,
    )
    component = lbg1.w_g_lt_dist * lbg1.lady_target_deviation(t, x, u, lbg1.PARAMS, lbg1.px_target, lbg1.py_target)
    assert jnp.isclose(composite, component)

    # guard cruise deviation cost
    composite = lbg1.guard_cost(
        t, x, u, lbg1.PARAMS, 
        lbg1.px_target,
        lbg1.py_target,
        lbg1.v_g_cruise,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        lbg1.w_g_spd,
        0.0,
        0.0,
    )
    component = lbg1.w_g_spd * lbg1.guard_speed_deviation(t, x, u, lbg1.PARAMS, lbg1.v_g_cruise)
    assert jnp.isclose(composite, component)

    # guard turnrate control effort cost
    composite = lbg1.guard_cost(
        t, x, u, lbg1.PARAMS, 
        lbg1.px_target,
        lbg1.py_target,
        lbg1.v_g_cruise,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        lbg1.w_g_omg,
        0.0,
    )
    component = lbg1.w_g_omg * lbg1.guard_turnrate_effort(t, x, u, lbg1.PARAMS)
    assert jnp.isclose(composite, component)

    # bandit turnrate control effort cost
    composite = lbg1.guard_cost(
        t, x, u, lbg1.PARAMS, 
        lbg1.px_target,
        lbg1.py_target,
        lbg1.v_g_cruise,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        lbg1.w_g_acc,
    )
    component = lbg1.w_g_acc * lbg1.guard_accel_effort(t, x, u, lbg1.PARAMS)
    assert jnp.isclose(composite, component)

@pytest.mark.slow
def test_cost_quadratization(lbg1):
    # Check that the players' costs functions quadratize without error
    
    # ~~ ARRANGE ~~
    # arbitrary state and control for quadratization
    t = 6.041
    x = jnp.asarray([ -0.962, 2.816, -6.2, 10.071, 4.435, 7.258, -18.61, 3.848, 6.957, 5.906, 6.028, 2.297])
    u = jnp.asarray([ -6.259,   3.599,  19.235,   8.971,  -8.279, -11.981])

    # ~~ ACT ~~
    Q, q, R, r = quadraticize_cost_joint_ctrl_playerwise(lbg1.game.costs[0].running, t, x, u, u_splits=lbg1.game.u_splits)
    Q, q, R, r = quadraticize_cost_joint_ctrl_playerwise(lbg1.game.costs[1].running, t, x, u, u_splits=lbg1.game.u_splits)
    Q, q, R, r = quadraticize_cost_joint_ctrl_playerwise(lbg1.game.costs[2].running, t, x, u, u_splits=lbg1.game.u_splits)

    # ~~ ASSERT ~~
    pass

@pytest.mark.slow
def test_cost_quadratization_trajectory(lbg1):
    # Check that the players' costs functions quadratize over a trajectory without error

    # ~~ ARRANGE ~~

    # compose initial operating point trajectory
    traj = FixedStepSystemTrajectory(
        # ts = jnp.linspace(0.0, lbg1.game.nt * lbg1.game.dt, lbg1.game.nt),
        tg = lbg1.game.tg,
        xs = jnp.zeros((lbg1.game.nt, lbg1.game.nx)),
        us = jnp.zeros((lbg1.game.nt-1, lbg1.game.nu))
    )

    # ~~ ACT ~~
    Q, q, R, r = quadraticize_cost_joint_ctrl_playerwise_trajectory(lbg1.game.costs[0].running, op=traj, u_splits=lbg1.game.u_splits)
    Q, q, R, r = quadraticize_cost_joint_ctrl_playerwise_trajectory(lbg1.game.costs[1].running, op=traj, u_splits=lbg1.game.u_splits)
    Q, q, R, r = quadraticize_cost_joint_ctrl_playerwise_trajectory(lbg1.game.costs[2].running, op=traj, u_splits=lbg1.game.u_splits)

    # ~~ ASSERT ~~
    pass

@pytest.mark.slow
def test_solve_ir_lady_bandit_guard_nonlinear_smoketest(lbg1):
    # Run the iterative linear-quadratic solver on the 3player target guarding problem

    # ~~ ARRANGE ~~

    # initial state
    x0 = jnp.zeros((lbg1.game.nx,))

    # compose initial operating point trajectory
    init_traj = FixedStepSystemTrajectory(
        # ts = jnp.linspace(0.0, lbg1.game.nt * lbg1.game.dt, lbg1.game.nt),
        tg = lbg1.game.tg,
        xs = jnp.zeros((lbg1.game.nt, lbg1.game.nx)),
        us = jnp.zeros((lbg1.game.nt-1, lbg1.game.nu))
    )

    # initial strategy 
    P = jnp.broadcast_to(jnp.eye(lbg1.game.nu, lbg1.game.nx), (lbg1.game.nt-1, lbg1.game.nu, lbg1.game.nx))
    alpha = jnp.broadcast_to(jnp.ones(lbg1.game.nu), (lbg1.game.nt-1, lbg1.game.nu))
    init_strat = FixedStepAffineStrategies(tg=lbg1.game.tg, P=P, alpha=alpha)

    # ~~ ACT ~~

    # compute nash strategy for nonlinear game
    conv, nl_traj, nl_strat = solve_ilqgame_feedback(lbg1.game, x0, init_traj, init_strat)

    # ~~ ASSERT ~~
    
    # just checking solver runs without error
    pass