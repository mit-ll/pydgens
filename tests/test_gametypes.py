# Copyright 2026 MIT Lincoln Laboratory
# SPDX-License-Identifier: MIT

import pytest
import jax
import jax.numpy as jnp
from random import randint
from dataclasses import FrozenInstanceError

# import helpers from pydgens
from pydgens.ir.timetypes import TimeGrid
from pydgens.ir.systemtypes import (
    LinearDiscreteSystemType1,
    SampledContinuousSystemType1,
    propagate_system_trajectory
)
from pydgens.ir.costtypes import PlayerCostSpecContinuous
from pydgens.ir.costtypes import ControlDomain as CostControlDomain
from pydgens.ir.costtypes import ControlStructure as CostControlStructure
from pydgens.ir.constrainttypes import (
    GameConstraintGridMap, 
    ConstraintBlockGridMap, 
)
from pydgens.ir.strategytypes import FixedStepAffineStrategies
from pydgens.utils.generators import (
    make_random_cost_fn,
    make_random_dynamics
)

# import the module under test
import pydgens.ir.gametypes as pdg_gt

@pytest.fixture(params=[
    (2, 5, 3, 4),
    (4, 128, 64, 8)
])
def make_afhgame_valid_inputs(request):
    N, nt, nx, nu = request.param
    u_splits = [nu // N] * N
    return N, nt, nx, nu, u_splits

@pytest.fixture
def make_lqgame1_valid_inputs(make_afhgame_valid_inputs):
    N, nt, nx, nu, u_splits = make_afhgame_valid_inputs
    dt = 0.1
    tg = TimeGrid(nt=nt, dt=dt)
    A = jnp.zeros((tg.nsteps, nx, nx))
    B = jnp.zeros((tg.nsteps, nx, nu))
    Q = jnp.zeros((tg.nsteps, N, nx, nx))
    q = jnp.zeros((tg.nsteps, N, nx))
    R = jnp.zeros((tg.nsteps, N, nu, nu))
    r = jnp.zeros((tg.nsteps, N, nu))
    u_splits = jnp.asarray(u_splits)
    cs = LinearDiscreteSystemType1(tg=tg, nx=nx, nu=nu, A=A, B=B)
    return cs, N, Q, q, R, r, u_splits

def test_lqgame1_properties(make_lqgame1_valid_inputs):
    cs, N, Q, q, R, r, u_splits = make_lqgame1_valid_inputs
    game = pdg_gt.LinearQuadraticGameType1(cs, N, Q, q, R, r, u_splits)

    assert game.tg == cs.tg
    assert game.nt == game.tg.nt == cs.nt == cs.tg.nt
    assert game.nsteps == game.tg.nsteps == cs.nsteps == cs.tg.nsteps
    assert game.dt == game.tg.dt == cs.dt == cs.tg.dt
    assert game.t0 == game.tg.t0 == cs.t0 == cs.tg.t0
    assert game.N == N
    assert game.nx == cs.nx
    assert game.nu == cs.nu
    assert game.cs.A.shape == (cs.nsteps, cs.nx, cs.nx)
    assert game.cs.B.shape == (cs.nsteps, cs.nx, cs.nu)
    assert game.Q.shape == (cs.nsteps, N, cs.nx, cs.nx)
    assert game.q.shape == (cs.nsteps, N, cs.nx)
    assert game.R.shape == (cs.nsteps, N, cs.nu, cs.nu)
    assert game.r.shape == (cs.nsteps, N, cs.nu)
    assert game.Qf.shape == (N, cs.nx, cs.nx)
    assert game.qf.shape == (N, cs.nx)


def test_lqgame1_accepts_explicit_terminal_state_cost():
    N, nt, nx, nu = 2, 5, 3, 4
    tg = TimeGrid(nt=nt, dt=0.1)
    A = jnp.zeros((tg.nsteps, nx, nx))
    B = jnp.zeros((tg.nsteps, nx, nu))
    Q = jnp.zeros((tg.nsteps, N, nx, nx))
    q = jnp.zeros((tg.nsteps, N, nx))
    R = jnp.zeros((tg.nsteps, N, nu, nu))
    r = jnp.zeros((tg.nsteps, N, nu))
    Qf = jnp.ones((N, nx, nx))
    qf = jnp.ones((N, nx))
    u_splits = jnp.asarray([2, 2])
    cs = LinearDiscreteSystemType1(tg=tg, nx=nx, nu=nu, A=A, B=B)

    game = pdg_gt.LinearQuadraticGameType1(cs, N, Q, q, R, r, u_splits, Qf=Qf, qf=qf)

    assert jnp.array_equal(game.Qf, Qf)
    assert jnp.array_equal(game.qf, qf)

def test_lqgame1_invalid_shape_R(make_lqgame1_valid_inputs):
    cs, N, Q, q, R, r, u_splits = make_lqgame1_valid_inputs
    R_bad = jnp.zeros((cs.nsteps, N, cs.nu + 1, cs.nu))  # Wrong m dimension
    with pytest.raises(ValueError):
        pdg_gt.LinearQuadraticGameType1(cs, N, Q, q, R_bad, r, u_splits)

def test_lqgame1_invalid_u_splits_length(make_lqgame1_valid_inputs):
    cs, N, Q, q, R, r, u_splits = make_lqgame1_valid_inputs
    u_splits_bad = jnp.asarray([1] * (N + 1))  # Too long
    with pytest.raises(ValueError):
        pdg_gt.LinearQuadraticGameType1(cs, N, Q, q, R, r, u_splits_bad)

def test_lqgame1_invalid_u_splits_total(make_lqgame1_valid_inputs):
    cs, N, Q, q, R, r, u_splits = make_lqgame1_valid_inputs
    u_splits_bad = jnp.asarray([1] * N)  # Does not sum to nu
    with pytest.raises(ValueError):
        pdg_gt.LinearQuadraticGameType1(cs, N, Q, q, R, r, u_splits_bad)

def test_lqgame1_dataclass_is_frozen(make_lqgame1_valid_inputs):
    cs, N, Q, q, R, r, u_splits = make_lqgame1_valid_inputs
    # construct game
    g = pdg_gt.LinearQuadraticGameType1(cs, N, Q, q, R, r, u_splits)
    with pytest.raises(FrozenInstanceError):
        g.N = g.N + 1

def test_lqgame1_invalid_u_splits_dtype(make_lqgame1_valid_inputs):
    cs, N, Q, q, R, r, u_splits = make_lqgame1_valid_inputs
    u_splits_float = u_splits.astype('float32')
    with pytest.raises(TypeError, match="u_splits must be an integer"):
        pdg_gt.LinearQuadraticGameType1(cs, N, Q, q, R, r, u_splits_float)

@pytest.fixture
def make_nlgame1_valid_inputs(make_afhgame_valid_inputs):
    N, T, nx, nu, u_splits = make_afhgame_valid_inputs
    dynamics = lambda t, x, u: x
    costs = [PlayerCostSpecContinuous(
        running=lambda t, x, u: t,
        # terminal, domain, coupling left as default values
        # which should be valid for NonlinearGameType1
    ) for _ in range(N)]
    nt = 8
    dt = 0.1
    tg = TimeGrid(nt=nt, dt=dt)
    cs = SampledContinuousSystemType1(tg=tg, nx=nx, nu=nu, dynamics=dynamics)
    u_splits = jnp.asarray(u_splits)
    return cs, N, T, costs, u_splits

def test_nlgame1_properties(make_nlgame1_valid_inputs):
    cs, N, T, costs, u_splits = make_nlgame1_valid_inputs
    game = pdg_gt.NonlinearGameType1(cs, N, costs, u_splits)

    assert game.tg == cs.tg
    assert game.nt == game.tg.nt == cs.nt == cs.tg.nt
    assert game.dt == game.tg.dt == cs.dt == cs.tg.dt
    assert game.t0 == game.tg.t0 == cs.t0 == cs.tg.t0
    assert game.N == N
    assert game.nx == cs.nx
    assert game.nu == cs.nu

def test_nlgame1_invalid_u_splits_length(make_nlgame1_valid_inputs):
    cs, N, T, costs, u_splits = make_nlgame1_valid_inputs
    u_splits_bad = jnp.asarray([1] * (N + 1))  # Too long
    with pytest.raises(ValueError):
        pdg_gt.NonlinearGameType1(cs, N, costs, u_splits_bad)

def test_nlgame1_invalid_costs_length(make_nlgame1_valid_inputs):
    cs, N, T, costs, u_splits = make_nlgame1_valid_inputs
    costs_bad = [PlayerCostSpecContinuous(running=lambda t, x, u: t) for _ in range(N+1)] # too long
    with pytest.raises(ValueError, match="costs must have length N"):
        pdg_gt.NonlinearGameType1(cs, N, costs_bad, u_splits)

def test_nlgame1_invalid_costs_not_spec(make_nlgame1_valid_inputs):
    cs, N, T, costs, u_splits = make_nlgame1_valid_inputs
    costs[randint(0,N-1)] = 1.0   # float, not a callable
    with pytest.raises(TypeError, match="costs must be PlayerCostSpecContinuous"):
        pdg_gt.NonlinearGameType1(cs, N, costs, u_splits)

def test_nlgame1_invalid_costs_terminal(make_nlgame1_valid_inputs):
    cs, N, T, costs, u_splits = make_nlgame1_valid_inputs
    costs_bad = [PlayerCostSpecContinuous(
        running = costs[i].running,
        terminal = lambda t, x: 1.0,
    ) for i in range(N)]
    with pytest.raises(ValueError, match="terminal costs not supported"):
        pdg_gt.NonlinearGameType1(cs, N, costs_bad, u_splits)

def test_nlgame1_invalid_costs_local_domain(make_nlgame1_valid_inputs):
    cs, N, T, costs, u_splits = make_nlgame1_valid_inputs
    costs_bad = [PlayerCostSpecContinuous(
        running = costs[i].running,
        control_domain = CostControlDomain.LOCAL
    ) for i in range(N)]
    with pytest.raises(ValueError, match="cost functions take joint control vectors"):
        pdg_gt.NonlinearGameType1(cs, N, costs_bad, u_splits)

def test_nlgame1_invalid_costs_coupled(make_nlgame1_valid_inputs):
    cs, N, T, costs, u_splits = make_nlgame1_valid_inputs
    costs_bad = [PlayerCostSpecContinuous(
        running = costs[i].running,
        control_structure = CostControlStructure.GENERAL
    ) for i in range(N)]
    with pytest.raises(ValueError, match="GENERAL control structure"):
        pdg_gt.NonlinearGameType1(cs, N, costs_bad, u_splits)

# def test_nlgame1_invalid_costs_not_callable(make_nlgame1_valid_inputs):
#     cs, N, T, costs, u_splits = make_nlgame1_valid_inputs
#     costs[randint(0,N-1)] = 1.0   # float, not a callable
#     with pytest.raises(TypeError, match="costs must be PlayerCostSpecContinuous"):
#         NonlinearGameType1(cs, N, costs, u_splits)

def test_nlgame1_dataclass_is_frozen(make_nlgame1_valid_inputs):
    cs, N, T, costs, u_splits = make_nlgame1_valid_inputs
    # construct game
    g = pdg_gt.NonlinearGameType1(cs=cs, N=N, costs=costs, u_splits=u_splits)
    with pytest.raises(FrozenInstanceError):
        g.N = g.N + 1

def test_nlgame1_rejects_u_splits_non_integer_dtype(make_nlgame1_valid_inputs):
    cs, N, T, costs, u_splits = make_nlgame1_valid_inputs
    u_splits_float = u_splits.astype('float32')
    with pytest.raises(TypeError, match="u_splits must be an integer"):
        pdg_gt.NonlinearGameType1(cs=cs, N=N, costs=costs, u_splits=u_splits_float)

@pytest.mark.regression
@pytest.mark.slow
@pytest.mark.benchmark(group="gametypes-001")
def test_approx_lqgame_warm_perf(benchmark):
    """benchmark the warm-started performance of approx_linear_quadratic_game
    with arbitrary (i.e. randomized, but with fixed seed) dynamics and cost
    functions
    """

    # hardcoded params of test to be similar to Unicycle1 system
    # but note that the dynamics and costs won't be the same
    seed = 1    # trying to vary seed from other tests
    # nt, nx, nu, u_splits = 100, 12, 6, [2, 2, 2]
    nt, nx, nu, u_splits = 20, 4, 2, [1, 1]

    # generator randomized functions from params
    N = len(u_splits)
    tg = TimeGrid(nt=nt, dt=0.1)
    dyn, _ = make_random_dynamics(nx=nx, nu=nu, seed=seed)
    costfns = make_random_cost_fn(nx=nx, nu=nu, u_splits=u_splits, seed=seed)
    costs = [PlayerCostSpecContinuous(running=cfn) for cfn in costfns] 

    # create control system and game 
    cs = SampledContinuousSystemType1(tg=tg, nx=nx, nu=nu, dynamics=dyn)
    nlgame = pdg_gt.NonlinearGameType1(cs=cs, N=N, costs=costs, u_splits=jnp.asarray(u_splits))

    # define a zero-strategy to be used to generate operating point trajectory
    strat = FixedStepAffineStrategies(
        tg=nlgame.tg,
        P=jnp.zeros((nlgame.nsteps, nlgame.nu, nlgame.nx)),
        alpha=jnp.zeros((nlgame.nsteps, nlgame.nu)),
    )

    # propagate operating point trajectory from origin initial state
    x0 = jnp.zeros((nx,))
    traj = propagate_system_trajectory(cs, x0=x0, strategy=strat)

    def run():
        lqgame = pdg_gt.approx_linear_quadratic_game(nlgame, op=traj)
        jax.block_until_ready(lqgame)
        return lqgame
    
    # warm compile once
    run()

    # benchmark warm-started run
    benchmark(run)



def _make_dummy_system(nx=4, nu=7, nt=10, dt=0.1):
    """
    Helper to build a minimal SampledContinuousSystemType1 instance for tests.
    Adjust field names to match your actual system class.
    """
    tg = TimeGrid(nt=nt, dt=dt, t0=0.0)

    def dynamics(t, x, u):
        # simple stable-ish dynamics: xdot = -x + Bu
        B = jnp.ones((nx, nu)) * 0.01
        return -x + B @ u

    # NOTE: adjust constructor args to your SampledContinuousSystemType1
    cs = SampledContinuousSystemType1(tg=tg, nx=nx, nu=nu, dynamics=dynamics)
    return cs

def _make_empty_constraints():
    # empty constraints are valid
    return GameConstraintGridMap(ineq_blocks=(), eq_blocks=())


def _make_player_specific_costs(N):
    # cost per player: scalar
    return [PlayerCostSpecContinuous(
        running=lambda t, x, u, i=i: jnp.sum(x**2) + (i + 1) * jnp.sum(u**2),
        terminal=lambda t, x: i*10.0,
        control_domain=CostControlDomain.LOCAL,
        control_structure=CostControlStructure.LOCAL_ONLY
    ) for i in range(N)]


def test_nlgame2_construct_valid_ok():
    # parameterize game
    N = 3
    nt = 11
    dt = 0.1
    nx = 7
    nu = 4
    u_splits = jnp.array([1, 2, 1], dtype=jnp.int32)
    seed = 1234
    dyn, _ = make_random_dynamics(nx=nx, nu=nu, seed=seed)
    runcosts = make_random_cost_fn(nx=nx, nu=nu, u_splits=u_splits, seed=seed)
    termcosts = [lambda t, x: i*10.0 for i in range(N)]
    costs = [PlayerCostSpecContinuous(
        running=runcosts[i],
        terminal=termcosts[i],
        control_domain=CostControlDomain.LOCAL,
        control_structure=CostControlStructure.LOCAL_ONLY
    ) for i in range(N)]
    constraints = _make_empty_constraints()
    tg = TimeGrid(nt=nt, dt=dt)
    cs = SampledContinuousSystemType1(tg=tg, nx=nx, nu=nu, dynamics=dyn)

    # construct game
    g = pdg_gt.NonlinearGameType2(cs=cs, N=N, costs=costs, constraints=constraints, u_splits=u_splits)

    assert g.N == N
    assert g.nx == 7
    assert g.nu == 4
    assert g.nt == 11
    assert g.dt == pytest.approx(0.1)
    assert callable(g.dynamics)


def test_nlgame2_rejects_wrong_system_type():
    N = 2
    constraints = _make_empty_constraints()
    costs = _make_player_specific_costs(N)
    u_splits = jnp.array([1, 1], dtype=jnp.int32)

    with pytest.raises(TypeError, match="SampledContinuousSystemType1"):
        pdg_gt.NonlinearGameType2(cs=object(), N=N, costs=costs, constraints=constraints, u_splits=u_splits)


def test_nlgame2_rejects_costs_wrong_length():
    N = 3
    cs = _make_dummy_system(nu=7)
    constraints = _make_empty_constraints()
    costs = _make_player_specific_costs(N - 1)  # wrong
    u_splits = jnp.array([2, 3, 2], dtype=jnp.int32)

    with pytest.raises(ValueError, match="costs must have length"):
        pdg_gt.NonlinearGameType2(cs=cs, N=N, costs=costs, constraints=constraints, u_splits=u_splits)


def test_nlgame2_rejects_non_spec_cost():
    N = 2
    cs = _make_dummy_system(nu=4)
    constraints = _make_empty_constraints()
    costs = [_make_player_specific_costs(1)[0], lambda t, x, u: 0.0]  # second callable but not PlayerCostSpecContinuous
    u_splits = jnp.array([2, 2], dtype=jnp.int32)

    with pytest.raises(TypeError, match="costs must be PlayerCostSpecContinuous"):
        pdg_gt.NonlinearGameType2(cs=cs, N=N, costs=costs, constraints=constraints, u_splits=u_splits)

def test_nlgame2_rejects_costs_no_terminal():
    N = 2
    cs = _make_dummy_system(nu=4)
    constraints = _make_empty_constraints()
    costs = _make_player_specific_costs(N)
    costs_bad = [PlayerCostSpecContinuous(
        running = costs[i].running,
        terminal = None,
        control_domain=costs[i].control_domain,
        control_structure=costs[i].control_structure
    ) for i in range(N)]
    u_splits = jnp.array([2, 2], dtype=jnp.int32)
    with pytest.raises(ValueError, match="terminal costs functions must be defined"):
        pdg_gt.NonlinearGameType2(cs, N, costs_bad, constraints, u_splits)

def test_nlgame2_rejects_costs_joint_domain():
    N = 2
    cs = _make_dummy_system(nu=4)
    constraints = _make_empty_constraints()
    costs = _make_player_specific_costs(N)
    costs_bad = [PlayerCostSpecContinuous(
        running = costs[i].running,
        terminal = costs[i].terminal,
        control_domain=CostControlDomain.JOINT,
        control_structure=costs[i].control_structure
    ) for i in range(N)]
    u_splits = jnp.array([2, 2], dtype=jnp.int32)
    with pytest.raises(ValueError, match="cost functions take local control vectors"):
        pdg_gt.NonlinearGameType2(cs, N, costs_bad, constraints, u_splits)

def test_nlgame2_rejects_costs_not_local_only():
    N = 2
    cs = _make_dummy_system(nu=4)
    constraints = _make_empty_constraints()
    costs = _make_player_specific_costs(N)
    costs_bad = [PlayerCostSpecContinuous(
        running = costs[i].running,
        terminal = costs[i].terminal,
        control_domain=costs[i].control_domain,
        control_structure=CostControlStructure.UNKNOWN
    ) for i in range(N)]
    u_splits = jnp.array([2, 2], dtype=jnp.int32)
    with pytest.raises(ValueError, match="must be explicitly declared LOCAL_ONLY"):
        pdg_gt.NonlinearGameType2(cs, N, costs_bad, constraints, u_splits)

def test_nlgame2_rejects_wrong_constraints_type():
    N = 2
    cs = _make_dummy_system(nu=4)
    costs = _make_player_specific_costs(N)
    u_splits = jnp.array([2, 2], dtype=jnp.int32)

    with pytest.raises(TypeError, match="GameConstraintGridMap"):
        pdg_gt.NonlinearGameType2(cs=cs, N=N, costs=costs, constraints=object(), u_splits=u_splits)


def test_nlgame2_rejects_u_splits_wrong_shape():
    N = 3
    cs = _make_dummy_system(nu=7)
    constraints = _make_empty_constraints()
    costs = _make_player_specific_costs(N)

    bad_u_splits = jnp.array([[2, 3, 2]], dtype=jnp.int32)  # ndim=2
    with pytest.raises(ValueError, match=r"u_splits must be shape"):
        pdg_gt.NonlinearGameType2(cs=cs, N=N, costs=costs, constraints=constraints, u_splits=bad_u_splits)

    bad_u_splits2 = jnp.array([2, 5], dtype=jnp.int32)  # length != N
    with pytest.raises(ValueError, match=r"u_splits must be shape"):
        pdg_gt.NonlinearGameType2(cs=cs, N=N, costs=costs, constraints=constraints, u_splits=bad_u_splits2)


def test_nlgame2_rejects_u_splits_sum_mismatch():
    N = 3
    cs = _make_dummy_system(nu=7)
    constraints = _make_empty_constraints()
    costs = _make_player_specific_costs(N)

    bad_u_splits = jnp.array([2, 2, 2], dtype=jnp.int32)  # sums to 6, should be 7
    with pytest.raises(ValueError, match="u_splits must sum"):
        pdg_gt.NonlinearGameType2(cs=cs, N=N, costs=costs, constraints=constraints, u_splits=bad_u_splits)

def test_nlgame2_accepts_game_constraint_map_with_basic_constraints():
    N = 2
    cs = _make_dummy_system(nx=3, nu=5, nt=9, dt=0.1)

    # one inequality and one equality constraint
    # c_ineq = BasicConstraint(func=lambda t, x, u: u[0] - 1.0, iseq=False)
    # c_eq   = BasicConstraint(func=lambda t, x, u: x[0],       iseq=True)
    c_ineq = ConstraintBlockGridMap(
        tg=cs.tg, 
        func=lambda t, x, u: u[0] - 1.0,
        cdim_out_step=1,
        iseq=False
    )
    c_eq   = ConstraintBlockGridMap(
        tg=cs.tg, 
        func=lambda t, x, u: x[0],
        cdim_out_step=1,
        iseq=True
    )

    constraints = GameConstraintGridMap(ineq_blocks=(c_ineq,), eq_blocks=(c_eq,))
    costs = _make_player_specific_costs(N)
    u_splits = jnp.array([2, 3], dtype=jnp.int32)  # sums to 5

    g = pdg_gt.NonlinearGameType2(cs=cs, N=N, costs=costs, constraints=constraints, u_splits=u_splits)

    assert isinstance(g.constraints, GameConstraintGridMap)
    assert g.constraints.nc_blocks_ineq == 1
    assert g.constraints.nc_blocks_eq == 1
    assert g.constraints.ineq_blocks[0].iseq is False
    assert g.constraints.eq_blocks[0].iseq is True

def test_nlgame2_rejects_u_splits_non_integer_dtype():
    N = 3
    cs = _make_dummy_system(nu=7)
    constraints = _make_empty_constraints()
    costs = _make_player_specific_costs(N)

    u_splits_float = jnp.array([2.0, 3.0, 2.0], dtype=jnp.float32)
    with pytest.raises(TypeError, match="u_splits must be an integer"):
        pdg_gt.NonlinearGameType2(cs=cs, N=N, costs=costs, constraints=constraints, u_splits=u_splits_float)

def test_nlgame2_dataclass_is_frozen():
    # parameterize game
    N = 3
    nt = 11
    dt = 0.1
    nx = 7
    nu = 4
    u_splits = jnp.array([1, 2, 1], dtype=jnp.int32)
    seed = 1234
    dyn, _ = make_random_dynamics(nx=nx, nu=nu, seed=seed)
    costs = _make_player_specific_costs(N)
    constraints = _make_empty_constraints()
    tg = TimeGrid(nt=nt, dt=dt)
    cs = SampledContinuousSystemType1(tg=tg, nx=nx, nu=nu, dynamics=dyn)

    # construct game
    g = pdg_gt.NonlinearGameType2(cs=cs, N=N, costs=costs, constraints=constraints, u_splits=u_splits)

    with pytest.raises(FrozenInstanceError):
        g.N = g.N + 1
