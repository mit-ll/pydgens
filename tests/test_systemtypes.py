# Copyright 2026 MIT Lincoln Laboratory
# SPDX-License-Identifier: MIT

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from types import SimpleNamespace

from pydgens.ir.timetypes import TimeGrid, cont2disc, compute_ts
from pydgens.ir.trajectorytypes import FixedStepSystemTrajectory, FixedStepPrimalDualTrajectory
from pydgens.ir.strategytypes import FixedStepAffineStrategies

from pydgens.utils.generators import (
    make_random_dynamics,
    make_random_strategy
)

# module under test
import pydgens.ir.systemtypes as irsys

# ---- SampledContinuousControlSystem Tests ----

def test_sampled_continuous_system_init_valid():
    nx, nu, nt = 4, 2, 8
    dt = 0.1
    tg = TimeGrid(nt=nt, dt=dt)

    def dummy_dynamics(t, x, u):
        return x + u

    sys = irsys.SampledContinuousSystemType1(tg, nx, nu, dummy_dynamics)
    assert sys.nx == nx
    assert sys.nu == nu
    assert sys.tg.nt == nt
    assert sys.tg.dt == dt
    assert sys.dynamics is dummy_dynamics

def test_sampled_continuous_system_invalid_dynamics():
    nx, nu, nt = 4, 2, 8
    dt = 0.1
    tg = TimeGrid(nt=nt, dt=dt)
    invalid_dynamics = 1.0  # not callable

    with pytest.raises(TypeError, match="dynamics must be callable"):
        irsys.SampledContinuousSystemType1(tg, nx, nu, invalid_dynamics)


def test_sampled_continuous_system_invalid_inputs():
    def dummy_dynamics(t, x, u):
        return x + u
    
    nt, dt = 8, 0.1
    tg = TimeGrid(nt=nt, dt=dt)
    
    nx, nu = (0, 2)
    with pytest.raises(ValueError, match="nx must be positive"):
        irsys.SampledContinuousSystemType1(tg, nx, nu, dummy_dynamics)

    nx, nu = (-10, 2)
    with pytest.raises(ValueError, match="nx must be positive"):
        irsys.SampledContinuousSystemType1(tg, nx, nu, dummy_dynamics)

    nx, nu = (1.1, 2)
    with pytest.raises(TypeError, match="nx must be integer"):
        irsys.SampledContinuousSystemType1(tg, nx, nu, dummy_dynamics)

    nx, nu = (4, 0)
    with pytest.raises(ValueError, match="nu must be positive"):
        irsys.SampledContinuousSystemType1(tg, nx, nu, dummy_dynamics)

    nx, nu = (4, -1)
    with pytest.raises(ValueError, match="nu must be positive"):
        irsys.SampledContinuousSystemType1(tg, nx, nu, dummy_dynamics)

    nx, nu = (4, 2.1)
    with pytest.raises(TypeError, match="nu must be integer"):
        irsys.SampledContinuousSystemType1(tg, nx, nu, dummy_dynamics)

def test_next_x_rk4_constant_velocity():
    nx = 2
    nu = 2
    nt = 8
    dt = 0.1
    tg = TimeGrid(nt=nt, dt=dt)

    # Simple dynamics: dx/dt = u
    def constant_velocity_dynamics(t, x, u):
        return u

    sys = irsys.SampledContinuousSystemType1(tg=tg, nx=nx, nu=nu, dynamics=constant_velocity_dynamics)

    # t0 = 0.0
    k = 0
    x0 = jnp.array([1.0, 2.0])
    u = jnp.array([0.5, -1.0])

    # Analytical solution: x_next = x + dt * u
    expected_x_next = x0 + dt * u
    x_next = irsys.next_x(sys, k, x0, u)

    assert jnp.allclose(x_next, expected_x_next, rtol=1e-6, atol=1e-8)

def test_next_x_rk4_double_integrator():
    nx = 2  # position + velocity
    nu = 1  # scalar input (acceleration)
    nt = 8
    dt = 0.1
    tg = TimeGrid(nt=nt, dt=dt)

    def double_integrator_dynamics(t, x, u):
        # x = [pos, vel], dx/dt = [vel, accel]
        pos = x[0]
        vel = x[1]
        acc = u[0]
        dxdt = jnp.array([vel, acc])
        return dxdt

    sys = irsys.SampledContinuousSystemType1(tg=tg, nx=nx, nu=nu, dynamics=double_integrator_dynamics)

    # t0 = 0.0
    k = 0
    x0 = jnp.array([0.0, 1.0])  # start at pos=0, vel=1
    u = jnp.array([2.0])        # constant acceleration

    # Analytical expected next state using physics
    expected_pos = x0[0] + x0[1] * dt + 0.5 * u[0] * dt**2
    expected_vel = x0[1] + u[0] * dt
    expected_x_next = jnp.array([expected_pos, expected_vel])

    x_next = irsys.next_x(sys, k, x0, u, 10)

    assert jnp.allclose(x_next, expected_x_next, rtol=1e-6, atol=1e-8)

@pytest.mark.parametrize("nx,dt,nt,u_w", 
    [
        (2, 0.1, 11, 1.0),
        (3, 0.1, 11, 1.0),
        (10, 0.5, 3, 1.0),
        (10, 0.5, 3, 10.0),
    ]
)
def test_propagate_system_trajectory_single_integrator_time_varying_control(nx, dt, nt, u_w):
    # n-dim game where each control variable controls the velocity
    # in the respective state dimension, but applies variable control
    # policies over different times

    # ~~ ARRANGE ~~
    nu = nx # each state dimension has a single corresponding control dimension

    x0 = jnp.zeros(nx)

    # define time characteristics
    tg = TimeGrid(nt=nt, dt=dt)

    # control dim 0 controls velocity in state dim 0, ctrl dim 1 for vel in state dim 1, etc.
    # Always apply zero control policy, except at first timestep
    P = jnp.zeros((tg.nsteps, nu, nx))
    alpha = jnp.zeros((tg.nsteps, nu))
    alpha = alpha.at[0].set(-u_w * jnp.ones((nu,)))
    strategy = FixedStepAffineStrategies(tg=tg, P=P, alpha=alpha)

    # define simple dynamics of single integrator on the control input: dx/dt = u
    dyn_single_integrator = lambda t, x, u: u

    # instantiate control system
    cs = irsys.SampledContinuousSystemType1(tg, nx, nu, dyn_single_integrator)

    # ~~ ACT ~~

    # propagate the trajectory of the system
    traj = irsys.propagate_system_trajectory(cs, x0, strategy)

    # ~~ ASSERT ~~

    assert traj.tg == cs.tg
    assert traj.xs.shape == (nt,nx)
    assert traj.us.shape == (tg.nsteps,nu)

    # Final state should just be from non-zero control at final timestep
    x1_expected = u_w * dt * jnp.ones(nx)
    x1_actual = traj.xs[1]
    assert jnp.allclose(x1_actual, x1_expected, atol=1e-5), f"Expected {x1_expected}, got {x1_actual}"

    # control is zero for all time except first
    # therefore state should change from t0 to t1, then remain constant
    assert jnp.allclose(traj.us[0], u_w), f"time {0}: expected all control to be {u_w}, got {traj.us}"
    for t in range(1,nt):
        assert jnp.allclose(traj.xs[t], x1_actual), f"time {t}: expected all states to be  {x1_actual}, got {traj.xs[t]}"
    for t in range(1, tg.nsteps):
        assert jnp.allclose(traj.us[t], 0.0), f"time {t}: expected all control to be 0.0, got {traj.us[t]}"

@pytest.mark.parametrize("nx,dt,nt,u_w", 
    [
        (2, 0.1, 11, 1.0),
        (3, 0.1, 11, 1.0),
        (10, 0.5, 3, 1.0),
        (10, 0.5, 3, 10.0),
    ]
)
def test_propagate_system_trajectory_single_integrator(nx, dt, nt, u_w):
    # n-dim game where each control variable controls the velocity
    # in the respective state dimension, and applies uniform
    # control u_w over all timesteps

    # ~~ ARRANGE ~~
    nu = nx # each state dimension has a single corresponding control dimension

    x0 = jnp.zeros(nx)

    # define time characteristics
    tg = TimeGrid(nt=nt, dt=dt)

    # control dim 0 controls velocity in state dim 0, ctrl dim 1 for vel in state dim 1, etc.
    strategy_P = []
    strategy_a = []
    for _ in range(tg.nsteps):
        P_t = jnp.zeros((nu, nx))  # affine-only control (no state dependence)
        a_t = -u_w * jnp.ones(nu)    # constant control of u_w
        strategy_P.append(P_t)
        strategy_a.append(a_t)

    # convert strategy list to jax arrays to allow indexing with jax tracers
    strategy_P = jnp.asarray(strategy_P)
    strategy_a = jnp.asarray(strategy_a)
    strategy = FixedStepAffineStrategies(tg=tg, P=strategy_P, alpha=strategy_a)

    # define simple dynamics of single integrator on the control input: dx/dt = u
    dyn_single_integrator = lambda t, x, u: u

    # instantiate control system
    cs = irsys.SampledContinuousSystemType1(tg, nx, nu, dyn_single_integrator)

    # ~~ ACT ~~

    # propagate the trajectory of the system
    traj = irsys.propagate_system_trajectory(cs, x0, strategy)

    # ~~ ASSERT ~~

    assert traj.tg == cs.tg
    assert traj.xs.shape == (nt,nx)
    assert traj.us.shape == (tg.nsteps,nu)

    # Final state should be [1.0, 1.0] after 1.0 second
    expected = u_w * dt * (nt-1) * jnp.ones(nx)
    actual = traj.xs[-1]
    assert jnp.allclose(actual, expected, atol=1e-5), f"Expected {expected}, got {actual}"

    # control is constant
    for t in range(tg.nsteps):
        assert jnp.allclose(traj.us[t], u_w), f"time {t}: expected all control to be {u_w}, got {traj.us}"

@pytest.mark.parametrize("dt,nt,u_w,x0", 
    [
        (0.1, 11, 1.0, jnp.zeros(2)),
        (0.1, 101, 1.0, jnp.ones(2)),
        (0.5, 101, 1.0, jnp.array([10., 10.])),
    ]
)
def test_propagate_system_trajectory_single_integrator_origin_tracking(dt, nt, u_w, x0):
    # 2-dim game where each control variable controls velocity of the system within 
    # their respective dimension of the state space space and 
    # applies control to drive system toward origin

    # ~~ ARRANGE ~~

    nx = 2 # 2D state
    nu = 2 # 2D control

    tg = TimeGrid(nt=nt, dt=dt)

    # each control variable controls velocity in the respective state dimension
    strategy_P = []
    strategy_a = []
    for _ in range(tg.nsteps):
        P_t = u_w * jnp.ones((nu, nx))  # linear term: vel control proportionally opposing state
        a_t = jnp.zeros(nu)    # bias term: no bias control term
        strategy_P.append(P_t)
        strategy_a.append(a_t)

    # convert strategy list to jax arrays to allow indexing with jax tracers
    strategy_P = jnp.asarray(strategy_P)
    strategy_a = jnp.asarray(strategy_a)
    strategy = FixedStepAffineStrategies(tg=tg, P=strategy_P, alpha=strategy_a)

    # define simple dynamics of single integrator on the control input: dx/dt = u
    dyn_single_integrator = lambda t, x, u: u

    # instantiate control system
    cs = irsys.SampledContinuousSystemType1(tg, nx, nu, dyn_single_integrator)

    # ~~ ACT ~~

    # propagate the trajectory of the system
    traj = irsys.propagate_system_trajectory(cs, x0, strategy)

    # ~~ ASSERT ~~

    assert traj.tg == cs.tg
    assert traj.xs.shape == (nt,nx)
    assert traj.us.shape == (tg.nsteps,nu)

    # Final state should have driven toward the origin (given sufficient time)
    expected = jnp.zeros(nx)
    actual = traj.xs[-1]

    assert jnp.allclose(actual, expected, atol=1e-5), f"Expected {expected}, got {actual}"


@pytest.mark.parametrize("dt,nt,u_w,x0", 
    [
        (0.1, 11, 1.0, jnp.zeros(2)),
        (0.1, 11, 1.0, jnp.ones(2),),
        (0.1, 166, -1398, jnp.array([-26.3478, 10.2837]),),
    ]
)
def test_propagate_system_trajectory_1d_double_integrator(dt, nt, u_w, x0):
    # 2D game where state is position and velocity and single control 
    # variable is acceleration

    # ~~ ARRANGE ~~

    nx = 2 # 2 state dimensions (pos, vel)
    nu = 1 # player 1 has 1 control dimension
    tg = TimeGrid(nt=nt, dt=dt)
    t_max = dt*(nt-1)

    # control dim 0 controls velocity in state dim 0, ctrl dim 1 for vel in state dim 1, etc.
    strategy_P = []
    strategy_a = []
    for _ in range(tg.nsteps):
        P_t = jnp.zeros((nu, nx))  # affine-only control (no state dependence)
        a_t = -u_w * jnp.ones(nu)    # constant control of u_w
        strategy_P.append(P_t)
        strategy_a.append(a_t)

    # convert strategy list to jax arrays to allow indexing with jax tracers
    strategy_P = jnp.asarray(strategy_P)
    strategy_a = jnp.asarray(strategy_a)
    strategy = FixedStepAffineStrategies(tg=tg, P=strategy_P, alpha=strategy_a)

    # define simple dynamics of double integrator on the control input: dx/dt = [x[1], u]
    dyn_double_integrator = lambda t, x, u: jnp.array([x[1], u[0]])

    # instantiate control system
    cs = irsys.SampledContinuousSystemType1(tg, nx, nu, dyn_double_integrator)

    # ~~ ACT ~~

    # propagate the trajectory of the system
    traj = irsys.propagate_system_trajectory(cs, x0, strategy)

    # ~~ ASSERT ~~

    assert traj.tg == cs.tg
    assert traj.xs.shape == (nt, nx)
    assert traj.us.shape == (tg.nsteps, nu)
    
    # Final state should be double integration of initial state
    expected = jnp.array([
        0.5*u_w*t_max**2 + x0[1]*t_max + x0[0], 
        u_w*t_max + x0[1]
    ])
    actual = traj.xs[-1]

    assert jnp.allclose(actual, expected, atol=1e-5), f"Expected {expected}, got {actual}"

    for t in range(tg.nsteps):
        assert jnp.allclose(traj.us[t], u_w)


def test_prop_scan_is_jittable_for_nonzero_steps():
    """The internal continuous-time rollout helper should remain JIT-friendly.

    This test targets `_prop_scan(...)` directly because it is the jitted
    helper whose zero-step handling was recently changed. Using it directly
    keeps the regression focused on the tracing behavior of the helper itself,
    rather than only on the public wrapper.
    """
    tg = TimeGrid(nt=4, dt=0.1, t0=0.0)
    nx = 2
    nu = 2
    nt_int = 2

    f = lambda t, x, u: u
    P = jnp.zeros((tg.nsteps, nu, nx), dtype=jnp.float32)
    alpha = -jnp.ones((tg.nsteps, nu), dtype=jnp.float32)
    x0 = jnp.array([0.0, 0.0], dtype=jnp.float32)

    xs, us = irsys._prop_scan(f, tg.t0, tg.dt, P, alpha, x0, nt_int)
    jax.block_until_ready(xs)
    jax.block_until_ready(us)

    assert xs.shape == (tg.nt, nx)
    assert us.shape == (tg.nsteps, nu)


def test_prop_scan_is_jittable_for_zero_steps():
    """The jitted helper should also support the degenerate nt=1 case.

    The recent bug fix added a zero-step early return before `lax.scan` is
    traced. This regression test ensures the helper still compiles and returns
    correctly shaped arrays when there are no control intervals.
    """
    tg = TimeGrid(nt=1, dt=0.1, t0=0.0)
    nx = 2
    nu = 3
    nt_int = 2

    f = lambda t, x, u: x + u[:nx]
    P = jnp.zeros((0, nu, nx), dtype=jnp.float32)
    alpha = jnp.zeros((0, nu), dtype=jnp.float32)
    x0 = jnp.array([1.0, -2.0], dtype=jnp.float32)

    xs, us = irsys._prop_scan(f, tg.t0, tg.dt, P, alpha, x0, nt_int)
    jax.block_until_ready(xs)
    jax.block_until_ready(us)

    assert xs.shape == (1, nx)
    assert us.shape == (0, nu)
    np.testing.assert_allclose(np.asarray(xs[0]), np.asarray(x0))

def test_propagate_system_trajectory_4d_unicycle_no_control():
    # propagate 4D unicycle dynamics with no control input

    # ~~ ARRANGE ~~
    nx = 4 # 2 state dimensions (px, py, theta, vt)
    nu = 2 # 2 control dimensions (dtheta, dvt)
    nt = 21  # number of timesteps
    dt = 0.1    # timestep size [s] 
    tg = TimeGrid(nt=nt, dt=dt)
    x0 = jnp.array([1,1,0,0.5])


    # define zero-control strategy
    P = jnp.broadcast_to(jnp.zeros((nu, nx)), (tg.nsteps, nu, nx))
    alpha = jnp.broadcast_to(jnp.zeros(nu), (tg.nsteps, nu))
    strat = FixedStepAffineStrategies(tg=tg, P=P, alpha=alpha)

    # define 4D unicycle dynamics
    dyn = lambda t, x, u: jnp.array([x[3]*jnp.cos(x[2]), x[3]*jnp.sin(x[2]), u[0], u[1]])

    # instantiate control system
    cs = irsys.SampledContinuousSystemType1(tg, nx, nu, dyn)

    # ~~ ACT ~~

    # propagate the trajectory of the system
    traj = irsys.propagate_system_trajectory(cs, x0, strat)

    # ~~ ASSERT ~~

    assert traj.tg == traj.tg
    assert traj.xs.shape == (nt, nx)
    assert traj.us.shape == (tg.nsteps, nu)

    # final position is simple propagation of initial position
    assert jnp.isclose(traj.xs[-1,0], 2.0)
    assert jnp.isclose(traj.xs[-1,1], 1.0)
    assert jnp.isclose(traj.xs[-1,2], 0.0)
    assert jnp.isclose(traj.xs[-1,3], 0.5)

def setup_4d_unicycle_w_strategy():
    # create unicycle control system and strategy generated during ilqsolver debugging
    # ~~ ARRANGE ~~
    nx = 4 # 2 state dimensions (px, py, theta, vt)
    nu = 2 # 2 control dimensions (dtheta, dvt)
    nt = 20  # number of timesteps
    dt = 0.1    # timestep size [s] 
    tg = TimeGrid(nt=nt, dt=dt)
    x0 = jnp.array([1,1,0,0.5])

    # define strategy from output of ilqsolver debugging
    P=jnp.array([[[2.58002132e-01, 6.34731352e-01, 4.42058593e-01, 4.25909422e-02],
        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 9.11455870e-01]],

       [[2.65074879e-01, 5.79918623e-01, 3.93694490e-01, 3.39582823e-02],
        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 9.02862191e-01]],

       [[2.63812840e-01, 5.24151444e-01, 3.45303476e-01, 2.63114553e-02],
        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 8.92468512e-01]],

       [[2.55436629e-01, 4.68481570e-01, 2.97987550e-01, 1.97913144e-02],
        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 8.79923582e-01]],

       [[2.41263613e-01, 4.13851053e-01, 2.52753854e-01, 1.44352736e-02],
        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 8.64820421e-01]],

       [[2.22636938e-01, 3.61072451e-01, 2.10456371e-01, 1.01944301e-02],
        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 8.46692264e-01]],

       [[2.00862557e-01, 3.10818970e-01, 1.71759203e-01, 6.95767393e-03],
        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 8.25012386e-01]],

       [[1.77158177e-01, 2.63622195e-01, 1.37120724e-01, 4.57741832e-03],
        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 7.99197257e-01]],

       [[1.52616620e-01, 2.19879761e-01, 1.06796905e-01, 2.89272191e-03],
        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 7.68616915e-01]],

       [[1.28183976e-01, 1.79869980e-01, 8.08587670e-02, 1.74732716e-03],
        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 7.32612848e-01]],

       [[1.04651041e-01, 1.43770933e-01, 5.92186004e-02, 1.00169203e-03],
        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 6.90527976e-01]],

       [[8.26560706e-02, 1.11682393e-01, 4.16600481e-02, 5.39309578e-04],
        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 6.41747653e-01]],

       [[6.26954883e-02, 8.36471692e-02, 2.78680474e-02, 2.68416334e-04],
        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 5.85755885e-01]],

       [[4.51398529e-02, 5.96706718e-02, 1.74559131e-02, 1.20468911e-04],
        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 5.22201717e-01]],

       [[3.02526820e-02, 3.97373475e-02, 9.98802204e-03, 4.68055550e-05],
        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 4.50973690e-01]],

       [[1.82101950e-02, 2.38234289e-02, 4.99760406e-03, 1.46372140e-05],
        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 3.72271895e-01]],

       [[9.12060309e-03, 1.19059142e-02, 1.99971069e-03, 3.17380932e-06],
        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 2.86666393e-01]],

       [[3.04223737e-03, 3.96781322e-03, 4.99987451e-04, 3.17730127e-07],
        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.95126668e-01]],

       [[0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 9.90099013e-02]],

       [[0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]]
    ])
    alpha = jnp.array([[ 6.8969071e-02, -4.5572788e-01],
       [ 7.1709752e-02, -4.5143110e-01],
       [ 7.0887417e-02, -4.4623423e-01],
       [ 6.7254663e-02, -4.3996179e-01],
       [ 6.1566770e-02, -4.3241018e-01],
       [ 5.4540902e-02, -4.2334610e-01],
       [ 4.6821058e-02, -4.1250616e-01],
       [ 3.8954020e-02, -3.9959860e-01],
       [ 3.1375930e-02, -3.8430843e-01],
       [ 2.4409309e-02, -3.6630642e-01],
       [ 1.8269137e-02, -3.4526399e-01],
       [ 1.3074219e-02, -3.2087383e-01],
       [ 8.8624209e-03, -2.9287794e-01],
       [ 5.6068897e-03, -2.6110086e-01],
       [ 3.2315664e-03, -2.2548684e-01],
       [ 1.6250983e-03, -1.8613595e-01],
       [ 6.5235980e-04, -1.4333320e-01],
       [ 1.6340753e-04, -9.7563334e-02],
       [ 0.0000000e+00, -4.9504951e-02],
       [ 0.0000000e+00,  0.0000000e+00]
    ])
    strat = FixedStepAffineStrategies(tg=tg, P=P[:-1], alpha=alpha[:-1])

    # define 4D unicycle dynamics
    dyn = lambda t, x, u: jnp.array([x[3]*jnp.cos(x[2]), x[3]*jnp.sin(x[2]), u[0], u[1]])

    # instantiate control system
    cs = irsys.SampledContinuousSystemType1(tg, nx, nu, dyn)

    return cs, strat, x0

@pytest.mark.regression
def test_propagate_system_trajectory_4d_unicycle_w_strategy():
    # propagate 4D unicycle dynamics for regression testing of final state

    # ~~ ARRANGE ~~
    cs, strat, x0 = setup_4d_unicycle_w_strategy()
    xf_approved = jnp.array([ 1.800217,    0.52727973, -0.721724,    0.5])

    # ~~ ACT ~~

    # propagate the trajectory of the system
    traj = irsys.propagate_system_trajectory(cs, x0, strat)

    # ~~ ASSERT ~~

    assert traj.tg == traj.tg
    assert traj.xs.shape == (cs.nt, cs.nx)
    assert traj.us.shape == (cs.nsteps, cs.nu)

    np.testing.assert_allclose(np.asarray(traj.xs[-1]), np.asarray(xf_approved))

@pytest.mark.regression
@pytest.mark.slow
@pytest.mark.benchmark(group="systemtypes-001")
def test_prop_4d_unicycle_cold_perf_1(benchmark):
    # test time of cold-started propagatioin of 4D unicycle dynamics
    
    cs, strat, x0 = setup_4d_unicycle_w_strategy()

    # function to be performance tested
    def run():
        traj = irsys.propagate_system_trajectory(cs, x0=x0, strategy=strat)
        jax.block_until_ready(traj)

    # Ensure no cached executables so this is truly "cold"
    jax.clear_caches()

    # benchmark cold-start run
    benchmark.pedantic(
        run,
        iterations=1,       # one timing sample per round
        rounds=1,           # exactly one round → one cold timing
        warmup_rounds=0,    # no warmup
    )

@pytest.mark.regression
@pytest.mark.slow
@pytest.mark.benchmark(group="systemtypes-002")
def test_prop_4d_unicycle_warm_perf_1(benchmark):
    # propagate 4D unicycle dynamics for performance regression testing
    
    cs, strat, x0 = setup_4d_unicycle_w_strategy()
    
    # function to be performance tested
    def run():
        traj = irsys.propagate_system_trajectory(cs, x0=x0, strategy=strat)
        jax.block_until_ready(traj)

    # warm compile once
    run()

    # benchmark warm-started run
    benchmark(run)

def setup_arbitrary_system_and_strat_1():
    # create randomly generated (but deterministic based on fixed seed)
    # dynamics, strategy, and initial state

    # hard-coded params for reproducibility
    seed = 2
    nt, nx, nu, dt = 32, 16, 8, 0.1

    # randomly generated strategy
    strat = make_random_strategy(nt=nt, nx=nx, nu=nu, dt=dt, seed=seed)

    # randomly generated dynamics, and thus, control system
    dyn, _ = make_random_dynamics(nx=nx, nu=nu, seed=seed)
    cs = irsys.SampledContinuousSystemType1(tg=strat.tg, nx=nx, nu=nu, dynamics=dyn)

    # randomly generated initial state
    x0 = jnp.array([
        1.6226422,   
        2.0252647,  
        -0.43359444, 
        -0.07861735,  
        0.1760909,  
        -0.97208923,
        -0.49529874,  
        0.4943786,   
        0.6643493,  
        -0.9501635,   
        2.1795304,  
        -1.9551506,
        0.35857072,  
        0.15779513,  
        1.2770847,   
        1.5104648
    ])

    return cs, strat, x0

@pytest.mark.regression
@pytest.mark.slow
@pytest.mark.benchmark(group="systemtypes-003")
def test_prop_arbitrary_system_cold_perf_1(benchmark):
    # cold-started propagate arbitrary system and strategy for performance regression testing
    cs, strat, x0 = setup_arbitrary_system_and_strat_1()

    # function to be performance tested
    def run():
        traj = irsys.propagate_system_trajectory(cs, x0=x0, strategy=strat)
        jax.block_until_ready(traj)

    # Ensure no cached executables so this is truly "cold"
    jax.clear_caches()
    
    # benchmark cold-start run
    benchmark.pedantic(
        run,
        iterations=1,       # one timing sample per round
        rounds=1,           # exactly one round → one cold timing
        warmup_rounds=0,    # no warmup
    )

@pytest.mark.regression
@pytest.mark.slow
@pytest.mark.benchmark(group="systemtypes-004")
def test_prop_arbitrary_system_warm_perf_1(benchmark):
    # propagate arbitrary system and strategy for performance regression testing
    cs, strat, x0 = setup_arbitrary_system_and_strat_1()

    # function to be performance tested
    def run():
        traj = irsys.propagate_system_trajectory(cs, x0=x0, strategy=strat)
        jax.block_until_ready(traj)

    # warm compile once
    run()

    # benchmark warm-started run
    benchmark(run)


def _make_lindisc_system_and_strategy(nt=5, nx=2, nu=3, dt=0.1, dtype=jnp.float32):
    tg = TimeGrid(nt=nt, dt=dt, t0=0.0)

    # time-varying but simple dynamics
    # A[k] = I + 0.01*(k+1) * I
    # B[k] = 0.1 * ones
    A = []
    B = []
    for k in range(tg.nsteps):
        A.append((1.0 + 0.01 * (k + 1)) * jnp.eye(nx, dtype=dtype))
        B.append(0.1 * jnp.ones((nx, nu), dtype=dtype))
    if tg.nsteps == 0:
        A = jnp.zeros((0, nx, nx), dtype=dtype)
        B = jnp.zeros((0, nx, nu), dtype=dtype)
    else:
        A = jnp.stack(A, axis=0)  # (nsteps,nx,nx)
        B = jnp.stack(B, axis=0)  # (nsteps,nx,nu)

    cs = irsys.LinearDiscreteSystemType1(tg=tg, nx=nx, nu=nu, A=A, B=B)

    # Strategy: P[k] arbitrary, alpha[k] arbitrary
    if tg.nsteps == 0:
        P = jnp.zeros((0, nu, nx), dtype=dtype)
        alpha = jnp.zeros((0, nu), dtype=dtype)
    else:
        P = jnp.stack([0.05 * jnp.ones((nu, nx), dtype=dtype) * (k + 1) for k in range(tg.nsteps)], axis=0)  # (nsteps,nu,nx)
        alpha = jnp.stack([0.01 * jnp.arange(nu, dtype=dtype) for _ in range(tg.nsteps)], axis=0)            # (nsteps,nu)

    strat = FixedStepAffineStrategies(tg=tg, P=P, alpha=alpha)
    return cs, strat


def test_propagate_discrete_strategy_shapes_and_terminal_padding():
    cs, strat = _make_lindisc_system_and_strategy(nt=6, nx=2, nu=3)
    x0 = jnp.array([1.0, -2.0], dtype=jnp.float32)

    traj = irsys.propagate_system_trajectory(cs, x0=x0, strategy=strat)

    assert isinstance(traj, FixedStepSystemTrajectory)
    assert traj.tg == cs.tg
    assert traj.xs.shape == (cs.tg.nt, cs.nx)
    assert traj.us.shape == (cs.tg.nsteps, cs.nu)


def test_propagate_discrete_strategy_zero_step_case():
    cs, strat = _make_lindisc_system_and_strategy(nt=1, nx=2, nu=3)
    x0 = jnp.array([1.0, -2.0], dtype=jnp.float32)

    traj = irsys.propagate_system_trajectory(cs, x0=x0, strategy=strat)

    assert isinstance(traj, FixedStepSystemTrajectory)
    assert traj.xs.shape == (1, cs.nx)
    assert traj.us.shape == (0, cs.nu)
    np.testing.assert_allclose(np.asarray(traj.xs[0]), np.asarray(x0))


def test_propagate_continuous_strategy_zero_step_case():
    tg = TimeGrid(nt=1, dt=0.1, t0=0.0)
    nx = 2
    nu = 3
    x0 = jnp.array([1.0, -2.0], dtype=jnp.float32)

    cs = irsys.SampledContinuousSystemType1(
        tg=tg,
        nx=nx,
        nu=nu,
        dynamics=lambda t, x, u: x,
    )
    strat = FixedStepAffineStrategies(
        tg=tg,
        P=jnp.zeros((0, nu, nx), dtype=jnp.float32),
        alpha=jnp.zeros((0, nu), dtype=jnp.float32),
    )

    traj = irsys.propagate_system_trajectory(cs, x0=x0, strategy=strat)

    assert isinstance(traj, FixedStepSystemTrajectory)
    assert traj.xs.shape == (1, nx)
    assert traj.us.shape == (0, nu)
    np.testing.assert_allclose(np.asarray(traj.xs[0]), np.asarray(x0))


def test_propagate_discrete_strategy_rollout_matches_definition():
    """
    Verifies:
      u_k = -P_k x_k - alpha_k
      x_{k+1} = A_k x_k + B_k u_k
    for all k=0..nt-2.
    """
    cs, strat = _make_lindisc_system_and_strategy(nt=5, nx=2, nu=3)
    x0 = jnp.array([0.2, -0.3], dtype=jnp.float32)

    traj = irsys.propagate_system_trajectory(cs, x0=x0, strategy=strat)

    xs = np.asarray(traj.xs)
    us = np.asarray(traj.us)

    np.testing.assert_allclose(xs[0], x0, atol=1e-6, rtol=1e-6)

    for k in range(cs.tg.nsteps):
        Pk = np.asarray(strat.P[k])
        ak = np.asarray(strat.alpha[k])
        xk = xs[k]

        u_expected = -(Pk @ xk) - ak
        np.testing.assert_allclose(us[k], u_expected, atol=1e-6, rtol=1e-6)

        Ak = np.asarray(cs.A[k])
        Bk = np.asarray(cs.B[k])
        x_next_expected = Ak @ xk + Bk @ u_expected
        np.testing.assert_allclose(xs[k + 1], x_next_expected, atol=1e-6, rtol=1e-6)


def test_propagate_discrete_strategy_rejects_timegrid_mismatch():
    cs, strat = _make_lindisc_system_and_strategy(nt=5, nx=2, nu=3)
    x0 = jnp.array([0.0, 0.0], dtype=jnp.float32)

    tg2 = TimeGrid(nt=5, dt=0.1, t0=1.0)  # different t0 => !=
    strat2 = FixedStepAffineStrategies(tg=tg2, P=strat.P, alpha=strat.alpha)

    with pytest.raises(ValueError, match="Inconsistent time characteristics"):
        irsys.propagate_system_trajectory(cs, x0=x0, strategy=strat2)


def test_propagate_discrete_strategy_rejects_dimension_mismatch_strategy_vs_system():
    cs, strat = _make_lindisc_system_and_strategy(nt=5, nx=2, nu=3)
    x0 = jnp.array([0.0, 0.0], dtype=jnp.float32)

    # Create a strategy with wrong nu (nu=4)
    tg = cs.tg
    P_bad = jnp.zeros((tg.nsteps, cs.nu + 1, cs.nx), dtype=jnp.float32)       # (nsteps,nu+1,nx)
    alpha_bad = jnp.zeros((tg.nsteps, cs.nu + 1), dtype=jnp.float32)          # (nsteps,nu+1)
    strat_bad = FixedStepAffineStrategies(tg=tg, P=P_bad, alpha=alpha_bad)

    # Your code checks strategy.nx and strategy.nu; if those are derived from P/alpha it will trip.
    with pytest.raises(ValueError, match="Inconsistent state/control dimensions"):
        irsys.propagate_system_trajectory(cs, x0=x0, strategy=strat_bad)


def test_propagate_discrete_strategy_rejects_x0_shape_mismatch():
    cs, strat = _make_lindisc_system_and_strategy(nt=5, nx=2, nu=3)
    x0_bad = jnp.zeros((cs.nx, 1), dtype=jnp.float32)

    with pytest.raises(ValueError, match="Inconsistent initial state dimensions"):
        irsys.propagate_system_trajectory(cs, x0=x0_bad, strategy=strat)


def test_propagate_discrete_strategy_alpha_affects_controls():
    """
    If alpha != 0, controls should differ from pure linear feedback -P x.
    """
    cs, strat = _make_lindisc_system_and_strategy(nt=4, nx=2, nu=3)
    x0 = jnp.array([1.0, 1.0], dtype=jnp.float32)

    traj = irsys.propagate_system_trajectory(cs, x0=x0, strategy=strat)

    u0 = np.asarray(traj.us[0])
    P0 = np.asarray(strat.P[0])
    x0_np = np.asarray(traj.xs[0])

    u_linear_only = -(P0 @ x0_np)
    # because alpha is nonzero in fixture, u0 should not equal u_linear_only
    assert not np.allclose(u0, u_linear_only, atol=1e-8, rtol=1e-8)

@pytest.mark.regression
@pytest.mark.slow
@pytest.mark.benchmark(group="systemtypes-005")
def test_next_x_arbitrary_system_cold_perf_1(benchmark):
    # cold-started next_x arbitrary system and strategy for performance regression testing
    cs, strat, x0 = setup_arbitrary_system_and_strat_1()

    # compute control based on strategy and current state
    u0 = -strat.P[0] @ x0 - strat.alpha[0]

    # function to be performance tested
    def run():
        x1 = irsys.next_x(cs, k=0, x=x0, u=u0, nt_int = 2)
        jax.block_until_ready(x1)

    # Ensure no cached executables so this is truly "cold"
    jax.clear_caches()

    # benchmark cold-start run
    benchmark.pedantic(
        run,
        iterations=1,       # one timing sample per round
        rounds=1,           # exactly one round → one cold timing
        warmup_rounds=0,    # no warmup
    )

@pytest.mark.regression
@pytest.mark.slow
@pytest.mark.benchmark(group="systemtypes-006")
def test_next_x_arbitrary_system_warm_perf_1(benchmark):
    # next_x arbitrary system and strategy for performance regression testing
    cs, strat, x0 = setup_arbitrary_system_and_strat_1()

    # compute control based on strategy and current state
    u0 = -strat.P[0] @ x0 - strat.alpha[0]
    
    # function to be performance tested
    def run():
        x1 = irsys.next_x(cs, k=0, x=x0, u=u0, nt_int = 2)
        jax.block_until_ready(x1)

    # warm compile once
    run()

    # benchmark warm-started run
    benchmark(run)

# ---- LinearTimeVaryingDiscreteSystem Tests ----

def test_linear_time_varying_discrete_system_init_valid():
    nx, nu, nt = 3, 2, 5
    dt = 0.2
    tg = TimeGrid(nt=nt, dt=dt)
    A = jnp.tile(jnp.eye(nx), (tg.nsteps, 1, 1))
    B = jnp.ones((tg.nsteps, nx, nu))

    sys = irsys.LinearDiscreteSystemType1(tg, nx, nu, A, B)

    assert sys.nx == nx
    assert sys.nu == nu
    assert sys.dt == dt
    assert sys.nt == nt
    assert sys.A.shape == (tg.nsteps, nx, nx)
    assert sys.B.shape == (tg.nsteps, nx, nu)


def test_linear_time_varying_discrete_system_zero_step_valid():
    nx, nu, nt = 3, 2, 1
    tg = TimeGrid(nt=nt, dt=0.2)
    A = jnp.zeros((0, nx, nx))
    B = jnp.zeros((0, nx, nu))

    sys = irsys.LinearDiscreteSystemType1(tg, nx, nu, A, B)

    assert sys.nt == 1
    assert sys.nsteps == 0
    assert sys.A.shape == (0, nx, nx)
    assert sys.B.shape == (0, nx, nu)

def test_linear_time_varying_discrete_system_invalid_inputs():
    A = jnp.tile(jnp.eye(4), (7, 1, 1))
    B = jnp.ones((7, 4, 2))

    nx, nu, nt, dt = (0, 2, 8, 0.1)
    tg = TimeGrid(nt=nt, dt=dt)
    with pytest.raises(ValueError, match="nx must be positive"):
        irsys.LinearDiscreteSystemType1(tg, nx, nu, A, B)

    nx, nu, nt, dt = (-10, 2, 8, 0.1)
    tg = TimeGrid(nt=nt, dt=dt)
    with pytest.raises(ValueError, match="nx must be positive"):
        irsys.LinearDiscreteSystemType1(tg, nx, nu, A, B)

    nx, nu, nt, dt = (1.1, 2, 8, 0.1)
    tg = TimeGrid(nt=nt, dt=dt)
    with pytest.raises(TypeError, match="nx must be integer"):
        irsys.LinearDiscreteSystemType1(tg, nx, nu, A, B)

    nx, nu, nt, dt = (4, 0, 8, 0.1)
    tg = TimeGrid(nt=nt, dt=dt)
    with pytest.raises(ValueError, match="nu must be positive"):
        irsys.LinearDiscreteSystemType1(tg, nx, nu, A, B)

    nx, nu, nt, dt = (4, -1, 8, 0.1)
    tg = TimeGrid(nt=nt, dt=dt)
    with pytest.raises(ValueError, match="nu must be positive"):
        irsys.LinearDiscreteSystemType1(tg, nx, nu, A, B)

    nx, nu, nt, dt = (4, 2.1, 8, 0.1)
    tg = TimeGrid(nt=nt, dt=dt)
    with pytest.raises(TypeError, match="nu must be integer"):
        irsys.LinearDiscreteSystemType1(tg, nx, nu, A, B)

def test_linear_time_varying_discrete_system_shape_mismatch():
    nx, nu, nt = 2, 1, 5
    dt = 0.1
    tg = TimeGrid(nt=nt, dt=dt)

    # A nstep mismatch
    shape_A, shape_B = ((5, 2, 2), (4, 2, 1))
    A = jnp.zeros(shape_A)
    B = jnp.zeros(shape_B)
    with pytest.raises(ValueError, match="Inconsistent shape for A"):
        irsys.LinearDiscreteSystemType1(tg, nx, nu, A, B)

    # B nstep mismatch
    shape_A, shape_B = ((4, 2, 2), (5, 2, 1)) 
    A = jnp.zeros(shape_A)
    B = jnp.zeros(shape_B)
    with pytest.raises(ValueError, match="Inconsistent shape for B"):
        irsys.LinearDiscreteSystemType1(tg, nx, nu, A, B)

    shape_A, shape_B = ((4, 3, 3), (4, 3, 1))     # A nx mismatch
    A = jnp.zeros(shape_A)
    B = jnp.zeros(shape_B)
    with pytest.raises(ValueError, match="Inconsistent shape for A"):
        irsys.LinearDiscreteSystemType1(tg, nx, nu, A, B)
        
    shape_A, shape_B = ((4, 2, 2), (4, 2, 3))     # B nu mismatch
    A = jnp.zeros(shape_A)
    B = jnp.zeros(shape_B)
    with pytest.raises(ValueError, match="Inconsistent shape for B"):
        irsys.LinearDiscreteSystemType1(tg, nx, nu, A, B)


def test_linearize_dynamics_1():
    # check linearize_dynamics works for a simple non-linear system whose
    # jacobians can be computed by hand
    
    # ~~ ARRANGE ~~
    def dynamics_1(t, x, u):
        return jnp.array([
            x[0]**2 + u[0],
            jnp.sin(x[1]) + u[1]**2
        ])

    # t_traj = jnp.array([1.0, 2.0])
    tg = TimeGrid(nt=2, dt=1.0, t0=1.0)
    x_traj = jnp.array([[1.0, 0.5], [0.8, 0.4]]) 
    u_traj = jnp.array([[0.1, 0.2]])
    op = FixedStepSystemTrajectory(tg=tg, xs=x_traj, us=u_traj)

    # ~~ ACT ~~
    A, B = irsys.linearize_dynamics(dynamics_1, op)

    # ~~ ASSERT ~~
    assert A.shape == (1,2,2)
    assert B.shape == (1,2,2)
    A_exp = jnp.array([
        [[2.0, 0.0], [0.0, jnp.cos(0.5)]],
    ])
    B_exp = jnp.array([
        [[1.0, 0.0], [0.0, 0.4]],
    ])
    for i in range(len(A)):
        assert jnp.allclose(A[i], A_exp[i])
        assert jnp.allclose(B[i], B_exp[i])

def test_linearize_dynamics_piecewise():
    # A toy piecewise-linear dynamics function: f(t, x, u)

    # ~~ ARRANGE ~~
    # time parameters
    tg = TimeGrid(nt=3, dt=1.0)

    def dynamics_pw(t, x, u):
        # Define 3 different dynamics "modes"
        # Define lookup tables as a list of callables
        A_lookup = [
            lambda: jnp.array([[1.0, 0.0], [0.0, 1.0]]),
            lambda: jnp.array([[0.9, 0.1], [0.0, 1.0]]),
            lambda: jnp.array([[1.1, -0.1], [0.0, 1.0]])
        ]
        B_lookup = [
            lambda: jnp.array([[1.0], [0.0]]),
            lambda: jnp.array([[0.5], [0.5]]),
            lambda: jnp.array([[0.0], [1.0]])
        ]

        # appropriate use of indexing in jax
        k = cont2disc(t=t, tg=tg)
        A = jax.lax.switch(k, A_lookup)  
        B = jax.lax.switch(k, B_lookup)
        return A @ x + B @ u

    # Create 3-point trajectory operating point for x and u (each 2D x, 1D u)
    xs = jnp.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0]
    ])
    us = jnp.array([
        [0.0],
        [1.0],
    ])
    op = FixedStepSystemTrajectory(tg, xs, us)

    # ~~ ACT ~~
    # linearize the continuous linear system
    Ac, Bc = irsys.linearize_dynamics(dynamics_pw, op)

def test_discretize_extended_linear_dynamics_euler_static_system_single_timestep():
    # ~~ ARRANGE ~~
    dt = 0.1
    Ac = jnp.array([[[1.0, 0.0],
                     [0.0, 2.0]]])  # shape (1, 2, 2)
    Bc = jnp.array([[[0.0],
                     [1.0]]])       # shape (1, 2, 1)

    # Expected
    A_expected = jnp.array([[[1.1, 0.0],
                             [0.0, 1.2]]])
    B_expected = jnp.array([[[0.0],
                             [0.1]]])

    # ~~ ACT ~~
    A, B = irsys.discretize_extended_linear_dynamics_euler(Ac, Bc, dt)

    # ~~ ASSERT ~~
    assert jnp.allclose(A, A_expected)
    assert jnp.allclose(B, B_expected)

def test_discretize_extended_linear_dynamics_euler_batched_time_varying_system():
    # ~~ ARRANGE ~~
    dt = 0.2
    T, n, m = 3, 2, 1
    Ac = jnp.array([
        [[1.0, 0.0], [0.0, 1.0]],
        [[0.5, 0.0], [0.0, 0.5]],
        [[-1.0, 0.0], [0.0, -1.0]],
    ])
    Bc = jnp.array([
        [[0.0], [1.0]],
        [[1.0], [0.0]],
        [[0.5], [0.5]],
    ])

    A_expected = jnp.array([
        [[1.2, 0.0], [0.0, 1.2]],
        [[1.1, 0.0], [0.0, 1.1]],
        [[0.8, 0.0], [0.0, 0.8]],
    ])
    B_expected = jnp.array([
        [[0.0], [0.2]],
        [[0.2], [0.0]],
        [[0.1], [0.1]],
    ])

    # ~~ ACT ~~
    A, B = irsys.discretize_extended_linear_dynamics_euler(Ac, Bc, dt)

    # ~~ ASSERT ~~
    assert A.shape == (T, n, n)
    assert B.shape == (T, n, m)
    assert jnp.allclose(A, A_expected)
    assert jnp.allclose(B, B_expected)

def test_discretize_extended_linear_dynamics_euler_zero_matrices():
    # ~~ ARRANGE ~~
    dt = 0.05
    T, n, m = 2, 3, 2
    Ac = jnp.zeros((T, n, n))
    Bc = jnp.zeros((T, n, m))

    # Expected
    I = jnp.eye(n)
    A_expected = jnp.stack([I] * T)
    B_expected = jnp.zeros((T, n, m))

    # ~~ ACT ~~
    A, B = irsys.discretize_extended_linear_dynamics_euler(Ac, Bc, dt)

    # ~~ ASSERT ~~
    assert jnp.allclose(A, A_expected)
    assert jnp.allclose(B, B_expected)

def test_approx_linear_discrete_system_2():
    # A toy piecewise-linear dynamics function: f(t, x, u)

    # ~~ ARRANGE ~~
    # time parameters
    tg = TimeGrid(nt=3, dt=1.0)

    def dynamics_pw(t, x, u):
        # Define 3 different dynamics "modes"
        # Define lookup tables as a list of callables
        A_lookup = [
            lambda: jnp.array([[1.0, 0.0], [0.0, 1.0]]),
            lambda: jnp.array([[0.9, 0.1], [0.0, 1.0]]),
            lambda: jnp.array([[1.1, -0.1], [0.0, 1.0]])
        ]
        B_lookup = [
            lambda: jnp.array([[1.0], [0.0]]),
            lambda: jnp.array([[0.5], [0.5]]),
            lambda: jnp.array([[0.0], [1.0]])
        ]

        # Incompatible use of python and jax values causing ConcretizationTypeError
        # A = A_lookup[int(t)]
        # B = B_lookup[int(t)]

        # appropriate use of indexing in jax
        # t must be an int in range(len(A_lookup))
        # NOTE: in more realistic examples, the input t of the dynamics function
        # would likely be a floating point value of time, not an integer index
        # of a lookup table
        k = cont2disc(t=t, tg=tg)
        A = jax.lax.switch(k, A_lookup)  
        B = jax.lax.switch(k, B_lookup)
        return A @ x + B @ u
    
    cont_pw_lin_sys = irsys.SampledContinuousSystemType1(tg=tg, nx=2, nu=1, dynamics=dynamics_pw)

    # Create 3-point trajectory operating point for x and u (each 2D x, 1D u)
    xs = jnp.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0]
    ])
    us = jnp.array([
        [0.0],
        [1.0],
    ])
    # ts = jnp.array([0.0, 1.0, 2.0])
    op = FixedStepSystemTrajectory(tg, xs, us)

    # ~~ ACT ~~
    # linearize the continuous linear system
    Ac, Bc = irsys.linearize_dynamics(dynamics_pw, op)

    # ~~ ASSERT ~~
    
    assert Ac.shape == (2,2,2)
    assert Bc.shape == (2,2,1)

    # Optional: check known values of Jacobians
    A_exp = jnp.array([
        [[1.0, 0.0], [0.0, 1.0]],
        [[0.9, 0.1], [0.0, 1.0]],
    ])
    B_exp = jnp.array([
        [[1.0], [0.0]],
        [[0.5], [0.5]],
    ])

    assert jnp.allclose(Ac, A_exp)
    assert jnp.allclose(Bc, B_exp)

def test_make_discrete_dynamics_step_map_unsupported_type_raises():
    """Calling on an unsupported system type should raise NotImplementedError."""
    class DummySystem:
        pass

    cs = DummySystem()
    t = 0.0
    x = jnp.array([1.0])
    u = jnp.array([0.0])

    with pytest.raises(NotImplementedError):
        _ = irsys.make_discrete_dynamics_step_map(cs, method="rk2")

def test_make_discrete_dynamics_step_map_method_smoketest():
    """
    For a nonlinear system, Euler and RK4 should generally give different
    results for the same dt.
    """

    def f(t, x, u):
        # nonlinear scalar dynamics: x' = x^2 + u
        return jnp.square(x) + jnp.array([0, u[0], 0, u[1]])

    dt = 0.1
    nx = 4
    nu = 2
    tg = TimeGrid(nt=5, dt=dt, t0=0.0)
    cs = irsys.SampledContinuousSystemType1(tg=tg, nx=nx, nu=nu, dynamics=f)

    t = 0.2
    x = jnp.array([0.8, -0.3, 20.0, -11.11])
    u = jnp.array([-0.0004, 3.14, 1e3, -2.23])

    f_euler = irsys.make_discrete_dynamics_step_map(cs, method="euler")
    f_rk2   = irsys.make_discrete_dynamics_step_map(cs, method="rk2")
    f_rk3   = irsys.make_discrete_dynamics_step_map(cs, method="rk3")
    f_rk4   = irsys.make_discrete_dynamics_step_map(cs, method="rk4")

    with pytest.raises(ValueError, match="Unknown integration method"):
        _ = irsys.make_discrete_dynamics_step_map(cs, method="rk5")

    x_next_euler = f_euler(t, x, u)
    x_next_rk2   = f_rk2(t, x, u)
    x_next_rk3   = f_rk3(t, x, u)
    x_next_rk4   = f_rk4(t, x, u)

    # They should not match exactly for a nonlinear f
    assert not np.allclose(x_next_euler, x_next_rk2)
    assert not np.allclose(x_next_euler, x_next_rk3)
    assert not np.allclose(x_next_euler, x_next_rk4)
    assert not np.allclose(x_next_rk2, x_next_rk3)
    assert not np.allclose(x_next_rk2, x_next_rk4)
    assert not np.allclose(x_next_rk3, x_next_rk4)

def test_make_discrete_dynamics_step_map_returns_callable_and_preserves_shape():
    """For a valid system, should return f_d(x,u) with same shape as x."""

    def f(t, x, u):
        # simple vector dynamics: dx/dt = x + 2u
        return x + 2.0 * u

    tg = TimeGrid(nt=10, dt=0.1, t0=0.0)
    nx = 3
    nu = 3
    cs = irsys.SampledContinuousSystemType1(tg=tg, nx=nx, nu=nu, dynamics=f)

    t = 0.3
    x = jnp.array([1.0, -2.0, 3.0])
    u = jnp.array([0.5, 1.0, -0.5])

    f_d = irsys.make_discrete_dynamics_step_map(cs, method="rk2")

    assert callable(f_d)

    x_next = f_d(t, x, u)
    assert x_next.shape == x.shape

def test_make_discrete_dynamics_step_map_euler_matches_definition():
    """
    For x' = a*x + b*u (scalar), Euler discretization gives:
        x_next = x + dt * (a*x + b*u)
    """
    a = -0.7
    b = 2.0

    def f(t, x, u):
        # x, u are shape (1,)
        return a * x + b * u

    dt = 0.05
    tg = TimeGrid(nt=5, dt=dt, t0=0.0)
    nx = 1
    nu = 1
    cs = irsys.SampledContinuousSystemType1(tg=tg, nx=nx, nu=nu, dynamics=f)

    t = 0.1
    x = jnp.array([1.5])
    u = jnp.array([0.3])

    f_d = irsys.make_discrete_dynamics_step_map(cs, method="euler")
    x_next = f_d(t, x, u)

    expected = x + dt * (a * x + b * u)

    assert np.allclose(x_next, np.array(expected), atol=1e-10, rtol=1e-10)

def test_discretized_step_is_jittable():
    """The discrete step function f_d should work under jax.jit."""

    def f(t, x, u):
        return jnp.sin(t) * x + jnp.cos(t) * u

    dt = 0.1
    tg = TimeGrid(nt=5, dt=dt, t0=0.0)
    nx = 2
    nu = 2
    cs = irsys.SampledContinuousSystemType1(tg=tg, nx=nx, nu=nu, dynamics=f)

    t = 0.3
    x = jnp.array([1.0, 2.0])
    u = jnp.array([-0.5, 0.5])

    f_d = irsys.make_discrete_dynamics_step_map(cs, method="rk3")

    # JIT over (x,u); t, dt, method are baked into the closure
    jit_f_d = jax.jit(lambda t_, x_, u_: f_d(t_, x_, u_))

    x_next_eager = f_d(t, x, u)
    x_next_jit   = jit_f_d(t, x, u)

    assert np.allclose(x_next_eager, np.array(x_next_jit), atol=1e-8, rtol=1e-8)

def _make_op_zeros(tg: TimeGrid, nx: int, nu: int, N: int = 1, dtype=jnp.float32):
    nt = tg.nt
    xs = jnp.zeros((nt, nx), dtype=dtype)
    us = jnp.zeros((nt - 1, nu), dtype=dtype)
    ls = jnp.zeros((nt - 1, N, nx), dtype=dtype)
    return FixedStepPrimalDualTrajectory(tg=tg, xs=xs, us=us, ls=ls)

def test_residual_discrete_dynamics_identity_map(monkeypatch):
    """
    If f_d(t,x,u) == x, then residual D_k = x_k - x_{k+1}.
    """
    tg = TimeGrid(nt=5, dt=0.1, t0=0.0)
    nt, nx, nu = tg.nt, 3, 2

    op = _make_op_zeros(tg, nx, nu, N=1)
    xs = jnp.arange(nt * nx, dtype=jnp.float32).reshape(nt, nx) / 10.0
    op = FixedStepPrimalDualTrajectory(tg=tg, xs=xs, us=op.us, ls=op.ls)

    def fake_make_fd(cs, method: str):
        def fd(t, x, u):
            return x
        return fd

    monkeypatch.setattr(irsys, "make_discrete_dynamics_step_map", fake_make_fd)

    cs = SimpleNamespace(tg=tg)
    D = irsys._residual_discrete_dynamics_trajectory(cs, op, method="euler")

    assert D.shape == (nt - 1, nx)
    expected = np.asarray(xs[:-1] - xs[1:])
    np.testing.assert_allclose(np.asarray(D), expected, atol=1e-6, rtol=1e-6)


def test_residual_discrete_dynamics_linear_euler(monkeypatch):
    """
    Continuous linear dynamics xdot = A x + B u.
    Euler discrete step: x_next_pred = x + dt*(A x + B u).
    Residual: D_k = x_next_pred - x_{k+1}.
    """
    tg = TimeGrid(nt=6, dt=0.2, t0=0.0)
    nt, nx, nu = tg.nt, 2, 3
    K = nt - 1

    xs = jnp.arange(nt * nx, dtype=jnp.float32).reshape(nt, nx) * 0.1
    us = jnp.arange(K * nu, dtype=jnp.float32).reshape(K, nu) * 0.05
    ls = jnp.zeros((K, 1, nx), dtype=jnp.float32)
    op = FixedStepPrimalDualTrajectory(tg=tg, xs=xs, us=us, ls=ls)

    A = jnp.array([[0.1, -0.2],
                   [0.3,  0.0]], dtype=jnp.float32)
    B = jnp.array([[1.0, 0.0, 0.5],
                   [0.0, 2.0, -0.1]], dtype=jnp.float32)

    def f_cont(t, x, u):
        return A @ x + B @ u

    def fake_make_fd(cs, method: str):
        assert method == "euler"
        dt = cs.dt
        def fd(t, x, u):
            return x + dt * f_cont(t, x, u)
        return fd

    monkeypatch.setattr(irsys, "make_discrete_dynamics_step_map", fake_make_fd)

    cs = SimpleNamespace(tg=tg, dt=tg.dt)
    D = irsys._residual_discrete_dynamics_trajectory(cs, op, method="euler")

    ts = compute_ts(tg)
    expected = []
    for k in range(K):
        xpred = xs[k] + tg.dt * f_cont(ts[k], xs[k], us[k])
        expected.append(xpred - xs[k + 1])
    expected = jnp.stack(expected, axis=0)

    np.testing.assert_allclose(np.asarray(D), np.asarray(expected), atol=1e-6, rtol=1e-6)

def test_residual_discrete_dynamics_linear_euler_2(monkeypatch):
    """
    Continuous linear dynamics xdot = A x + B u.
    Euler discrete step: x_next_pred = x + dt*(A x + B u).
    Residual: D_k = x_next_pred - x_{k+1}.

    This tests uses actual control system types
    """
    tg = TimeGrid(nt=6, dt=0.2, t0=0.0)
    nt, nx, nu = tg.nt, 2, 3
    K = nt - 1

    xs = jnp.arange(nt * nx, dtype=jnp.float32).reshape(nt, nx) * 0.1
    us = jnp.arange(K * nu, dtype=jnp.float32).reshape(K, nu) * 0.05
    ls = jnp.zeros((K, 1, nx), dtype=jnp.float32)
    op = FixedStepPrimalDualTrajectory(tg=tg, xs=xs, us=us, ls=ls)

    A = jnp.array([[0.1, -0.2],
                   [0.3,  0.0]], dtype=jnp.float32)
    B = jnp.array([[1.0, 0.0, 0.5],
                   [0.0, 2.0, -0.1]], dtype=jnp.float32)

    def f_cont(t, x, u):
        return A @ x + B @ u

    cs = irsys.SampledContinuousSystemType1(tg=tg, nx=nx, nu=nu, dynamics=f_cont)
    D = irsys.residual_discrete_dynamics_trajectory(cs, op, method="euler")

    ts = compute_ts(tg)
    expected = []
    for k in range(K):
        xpred = xs[k] + tg.dt * f_cont(ts[k], xs[k], us[k])
        expected.append(xpred - xs[k + 1])
    expected = jnp.stack(expected, axis=0)

    np.testing.assert_allclose(np.asarray(D), np.asarray(expected), atol=1e-6, rtol=1e-6)


def test_residual_discrete_dynamics_raises_on_timegrid_mismatch(monkeypatch):
    """
    Should fail before calling the discretizer if cs.tg != op.tg.
    """
    tg1 = TimeGrid(nt=5, dt=0.1, t0=0.0)
    tg2 = TimeGrid(nt=5, dt=0.1, t0=1.0)

    op = _make_op_zeros(tg1, nx=2, nu=2, N=1)
    cs = SimpleNamespace(tg=tg2)

    with pytest.raises(ValueError, match="TimeGrid"):
        irsys._residual_discrete_dynamics_trajectory(cs, op, method="euler")

def test_jacobian_discrete_dynamics_step_shapes():
    def f(t, x, u):
        return x + 2.0 * u  # xdot

    dt = 0.1
    tg = TimeGrid(nt=5, dt=dt, t0=0.0)
    nx = 3
    nu = 3
    cs = irsys.SampledContinuousSystemType1(tg=tg, nx=nx, nu=nu, dynamics=f)

    t = 0.2
    x = jnp.array([1.0, -2.0, 3.0])
    u = jnp.array([0.5, 1.0, -0.5])

    A, B = irsys.jacobian_discrete_dynamics_step(cs, t, x, u, method="rk2")

    assert A.shape == (nx, nx)
    assert B.shape == (nx, nu)

def test_jacobian_discrete_dynamics_step_euler_linear_matches_closed_form():
    dt = 0.05
    nt=10
    nx = 2
    nu = 1
    tg = TimeGrid(nt=nt, dt=dt)
    Ac = jnp.array([[1.0, 2.0],
                    [0.0, -1.0]])
    Bc = jnp.array([[3.0],
                    [-2.0]])

    def f(t, x, u):
        return Ac @ x + Bc @ u

    cs = irsys.SampledContinuousSystemType1(tg=tg, nx=nx, nu=nu, dynamics=f)

    t = 0.0
    x = jnp.array([0.2, -0.3])
    u = jnp.array([0.7])

    A, B = irsys.jacobian_discrete_dynamics_step(cs, t, x, u, method="euler")

    A_expected = jnp.eye(2) + dt * Ac
    B_expected = dt * Bc

    assert np.allclose(A, np.array(A_expected), atol=1e-10, rtol=1e-10)
    assert np.allclose(B, np.array(B_expected), atol=1e-10, rtol=1e-10)

def test_jacobian_discrete_dynamics_step_matches_finite_difference_nonlinear():
    """This is a good “smoke test” that your map + jacfwd are wired correctly."""
    dt = 0.1
    nt = 10
    nx = 2
    nu = 1
    tg = TimeGrid(nt=nt, dt=dt)

    def f(t, x, u):
        # nonlinear coupled dynamics
        return jnp.array([
            jnp.sin(t) * x[0] + x[1] ** 2 + u[0],
            x[0] * u[0] + jnp.cos(t) * x[1],
        ])

    cs = irsys.SampledContinuousSystemType1(tg=tg, nx=nx, nu=nu, dynamics=f)
    method = "rk4"

    t = 0.3
    x = jnp.array([0.4, -0.2])
    u = jnp.array([0.1])

    A, B = irsys.jacobian_discrete_dynamics_step(cs, t, x, u, method=method)

    # finite difference against the actual discrete map
    fd = irsys.make_discrete_dynamics_step_map(cs, method=method)
    eps = 1e-3
    tol = 3e-3

    # FD wrt x
    A_fd = np.zeros((2, 2))
    for i in range(2):
        dx = np.zeros((2,))
        dx[i] = eps
        xp = x + jnp.array(dx)
        xm = x - jnp.array(dx)
        fp = fd(t, xp, u)
        fm = fd(t, xm, u)
        A_fd[:, i] = np.array((fp - fm) / (2 * eps))

    # FD wrt u
    B_fd = np.zeros((2, 1))
    du = eps
    up = u + jnp.array([du])
    um = u - jnp.array([du])
    fp = fd(t, x, up)
    fm = fd(t, x, um)
    B_fd[:, 0] = np.array((fp - fm) / (2 * eps))

    assert np.allclose(A, A_fd, atol=tol, rtol=tol)
    assert np.allclose(B, B_fd, atol=tol, rtol=tol)


def _make_primedual_op(tg, xs, us, ls=None):
    # adapt to your actual FixedStepPrimalDualTrajectory ctor
    if ls is None:
        # placeholder duals shape (nt-1, N, nx) if your class requires it;
        # if not required, omit.
        nt, nx = xs.shape
        ls = jnp.zeros((nt - 1, 1, nx), dtype=xs.dtype)
    return FixedStepPrimalDualTrajectory(tg=tg, xs=xs, us=us, ls=ls)

def test_jacobian_discrete_dynamics_trajectory_rejects_cs_type():
    # check single dispatch works as expected and raises error for not-implemented 
    # control system type
    tg = TimeGrid(nt=5, dt=0.1, t0=0.0)
    nx, nu = 11, 7
    xs = jnp.zeros((tg.nt, nx))
    us = jnp.zeros((tg.nt - 1, nu))
    op = _make_primedual_op(tg, xs, us)

    cs = object()

    with pytest.raises(NotImplementedError):
        irsys.jacobian_discrete_dynamics_trajectory(cs, op, method="euler")

def test_jacobian_discrete_dynamics_trajectory_timegrid_mismatch_raises():
    tg1 = TimeGrid(nt=5, dt=0.1, t0=0.0)
    tg2 = TimeGrid(nt=5, dt=0.2, t0=0.0)  # different dt

    nx, nu = 2, 1
    A = jnp.eye(nx)
    B = jnp.ones((nx, nu))

    # cs = _make_linear_system(tg1, A, B)
    # Create control system with linear dynamics
    cs = irsys.SampledContinuousSystemType1(tg1, nx, nu, lambda t, x, u: A @ x + B @ u)

    xs = jnp.zeros((tg2.nt, nx))
    us = jnp.zeros((tg2.nt - 1, nu))
    op = _make_primedual_op(tg2, xs, us)

    with pytest.raises(ValueError, match="time grids"):
        irsys.jacobian_discrete_dynamics_trajectory(cs, op, method="euler")


def test_jacobian_discrete_dynamics_trajectory_shapes():
    tg = TimeGrid(nt=7, dt=0.05, t0=0.0)
    nx, nu = 3, 2
    A = jnp.eye(nx)
    B = jnp.ones((nx, nu))
    
    # Create control system with linear dynamics
    cs = irsys.SampledContinuousSystemType1(tg, nx, nu, lambda t, x, u: A @ x + B @ u)

    xs = jnp.zeros((tg.nt, nx))
    us = jnp.zeros((tg.nt - 1, nu))
    op = _make_primedual_op(tg, xs, us)

    As, Bs = irsys.jacobian_discrete_dynamics_trajectory(cs, op, method="euler")
    assert As.shape == (tg.nt - 1, nx, nx)
    assert Bs.shape == (tg.nt - 1, nx, nu)


def test_jacobian_discrete_dynamics_trajectory_linear_euler_matches_closed_form():
    # For Euler: fd = x + dt*(A x + B u)
    # => dfd/dx = I + dt*A, dfd/du = dt*B (constant over time)
    tg = TimeGrid(nt=6, dt=0.1, t0=0.0)
    nx, nu = 2, 3
    A = jnp.array([[1.0, 2.0],
                   [0.0, -1.0]], dtype=jnp.float32)
    B = jnp.array([[1.0, 0.0, 2.0],
                   [-3.0, 1.0, 0.0]], dtype=jnp.float32)

    # Create control system with linear dynamics
    cs = irsys.SampledContinuousSystemType1(tg, nx, nu, lambda t, x, u: A @ x + B @ u)

    # operating point can be arbitrary for linear system
    key = jax.random.key(0)
    xs = jax.random.normal(key, (tg.nt, nx), dtype=jnp.float32)
    us = jax.random.normal(key, (tg.nt - 1, nu), dtype=jnp.float32)
    op = _make_primedual_op(tg, xs, us)

    As, Bs = irsys.jacobian_discrete_dynamics_trajectory(cs, op, method="euler")

    I = jnp.eye(nx, dtype=jnp.float32)
    A_expected = I + tg.dt * A
    B_expected = tg.dt * B

    # Check all steps equal expected
    assert np.allclose(np.asarray(As), np.asarray(A_expected)[None, :, :], atol=1e-6, rtol=1e-6)
    assert np.allclose(np.asarray(Bs), np.asarray(B_expected)[None, :, :], atol=1e-6, rtol=1e-6)


def test_jacobian_discrete_dynamics_trajectory_time_dependent_varies_with_t():
    # f(t,x,u) = (t)*x + u   (nx==nu for simplicity)
    # Euler: fd = x + dt*(t*x + u)
    # => dfd/dx = I + dt*t*I, dfd/du = dt*I
    tg = TimeGrid(nt=5, dt=0.2, t0=0.0)
    nx = 2
    nu = 2

    # time-dependent dynamics
    def f(t, x, u):
        return t * x + u

    cs = irsys.SampledContinuousSystemType1(tg , nx, nu, dynamics=f)

    xs = jnp.zeros((tg.nt, nx), dtype=jnp.float32)
    us = jnp.zeros((tg.nt - 1, nu), dtype=jnp.float32)
    op = _make_primedual_op(tg, xs, us)

    As, Bs = irsys.jacobian_discrete_dynamics_trajectory(cs, op, method="euler")

    ts = compute_ts(tg)[:-1]
    I = jnp.eye(nx, dtype=jnp.float32)
    # Check two different steps differ
    A0 = I + tg.dt * ts[0] * I
    A_last = I + tg.dt * ts[-1] * I

    assert np.allclose(np.asarray(As[0]), np.asarray(A0), atol=1e-6, rtol=1e-6)
    assert np.allclose(np.asarray(As[-1]), np.asarray(A_last), atol=1e-6, rtol=1e-6)

    B_expected = tg.dt * jnp.eye(nu, dtype=jnp.float32)
    assert np.allclose(np.asarray(Bs[0]), np.asarray(B_expected), atol=1e-6, rtol=1e-6)
    assert np.allclose(np.asarray(Bs[-1]), np.asarray(B_expected), atol=1e-6, rtol=1e-6)
