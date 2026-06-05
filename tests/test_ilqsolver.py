# Copyright 2026 MIT Lincoln Laboratory
# SPDX-License-Identifier: MIT

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from pydgens.ir.timetypes import TimeGrid
from pydgens.ir.trajectorytypes import FixedStepSystemTrajectory
from pydgens.ir.strategytypes import FixedStepAffineStrategies
from pydgens.ir.systemtypes import (
    SampledContinuousSystemType1,
    LinearDiscreteSystemType1,
    propagate_system_trajectory,
    linearize_dynamics,
    discretize_extended_linear_dynamics_euler
)
from pydgens.ir.gametypes import (
    NonlinearGameType1, 
    LinearQuadraticGameType1,
    approx_linear_quadratic_game
)
from pydgens.ir.costtypes import (
    PlayerCostSpecContinuous,
    quadraticize_cost_joint_ctrl_playerwise_trajectory
)
from pydgens.solvers.lqsolver import solve_lqgame_feedback
from pydgens.solvers.ilqsolver import (
    scale_strategy,
    backtrack_scale_strategy,
    solve_ilqgame_feedback
)

def test_scale_strategy_identity():
    nt, nu, nx = 3, 2, 2
    tg = TimeGrid(nt=nt, dt=0.1)
    P = jnp.stack([jnp.eye(nx) for _ in range(nt-1)])
    alpha = jnp.ones((nt-1, nu))
    strategy = FixedStepAffineStrategies(tg=tg, P=P, alpha=alpha)

    scaled = scale_strategy(strategy, 1.0)
    assert strategy.tg is scaled.tg
    assert jnp.allclose(scaled.P, P)
    assert jnp.allclose(scaled.alpha, alpha)

def test_scale_strategy_half():
    nt, nu, nx = 2, 2, 2
    tg = TimeGrid(nt=nt, dt=0.1)
    P = jnp.ones((nt-1, nu, nx))
    alpha = jnp.array([[2.0, 4.0]])
    strategy = FixedStepAffineStrategies(tg=tg, P=P, alpha=alpha)

    scaled = scale_strategy(strategy, 0.5)
    expected_alpha = alpha * 0.5

    assert strategy.tg is scaled.tg
    assert jnp.allclose(scaled.P, P)
    assert jnp.allclose(scaled.alpha, expected_alpha)

def test_scale_strategy_zero():
    nt, nu, nx = 2, 2, 2
    tg = TimeGrid(nt=nt, dt=0.1)
    P = jnp.array([[[1.0, 2.0], [3.0, 4.0]],])
    alpha = jnp.array([[5.0, -5.0],])
    strategy = FixedStepAffineStrategies(tg=tg, P=P, alpha=alpha)

    scaled = scale_strategy(strategy, 0.0)
    assert strategy.tg is scaled.tg
    assert jnp.allclose(scaled.P, P)
    assert jnp.allclose(scaled.alpha, jnp.zeros_like(alpha))

def test_scale_strategy_does_not_mutate_original():
    nt, nu, nx = 2, 2, 2
    tg = TimeGrid(nt=nt, dt=0.1)
    P = jnp.stack([jnp.eye(nx) for _ in range(nt-1)])
    alpha = jnp.array([[1.0, 2.0], ])
    strategy = FixedStepAffineStrategies(tg=tg, P=P, alpha=alpha)

    scaled = scale_strategy(strategy, 0.5)

    assert strategy.tg is tg
    assert jnp.allclose(strategy.alpha, alpha)
    assert jnp.allclose(strategy.P, P)

def test_scale_strategy_invalid_alpha_scale():
    nt, nu, nx = 2, 2, 2
    tg = TimeGrid(nt=nt, dt=0.1)
    P = jnp.stack([jnp.eye(nx) for _ in range(nt-1)])
    alpha = jnp.ones((nt-1, nu))
    strategy = FixedStepAffineStrategies(tg=tg, P=P, alpha=alpha)

    with pytest.raises(ValueError, match=r"alpha_scale.*\[0, 1\]"):
        scale_strategy(strategy, -0.1)

    with pytest.raises(ValueError, match=r"alpha_scale.*\[0, 1\]"):
        scale_strategy(strategy, 1.5)

@pytest.fixture(params=[
    (2, 16, 8, 4, 0.1),
    (4, 128, 32, 8, 0.01),
    (1, 8, 4, 2, 1.0)
])
def nlgame_stationary(request):
    """Nonlinear game with stationary dynamics and zero costs"""
    N, nt, nx, nu, dt = request.param
    tg = TimeGrid(nt=nt, dt=dt)
    dyn = lambda t, x, u: jnp.zeros_like(x)
    cs = SampledContinuousSystemType1(tg=tg, nx=nx, nu=nu, dynamics=dyn)
    costs = [PlayerCostSpecContinuous(running=lambda t, x, u: 0) for _ in range(N)]
    return NonlinearGameType1(
        cs=cs,
        N=N,
        costs = costs,
        u_splits=jnp.asarray([nu//N] * N)
    )

def make_zero_trajectory(game):
    """Trajectory of all zeros"""
    # ts = jnp.linspace(0.0, game.dt*game.T, game.T)
    tg = TimeGrid(nt=game.nt, dt=game.dt)
    xs = jnp.zeros((game.nt, game.nx))
    us = jnp.zeros((game.nt-1, game.nu))
    return FixedStepSystemTrajectory(tg=tg, xs=xs, us=us)

def make_zero_strategy(game):
    """Strategy that maps to all-zero control, regardless of state"""
    tg = TimeGrid(nt=game.nt, dt=game.dt)
    P = jnp.zeros((game.nt-1, game.nu, game.nx))
    alpha = jnp.zeros((game.nt-1, game.nu))
    return FixedStepAffineStrategies(tg=tg, P=P, alpha=alpha)

def make_uniform_feedforward_strategy(game, alpha_val):
    """Strategy where control is independent of state (only feedforward term with all the same entries)"""
    tg = TimeGrid(nt=game.nt, dt=game.dt)
    P = jnp.zeros((game.nt-1, game.nu, game.nx))
    alpha = jnp.full((game.nt-1, game.nu), alpha_val)
    return FixedStepAffineStrategies(tg=tg, P=P, alpha=alpha)

def test_backtrack_scale_strategy_shapes(nlgame_stationary):
    # Check backtrack_scale_strategy executes without error and provides correctly shaped outputs
    game = nlgame_stationary
    scaled_strategy, new_traj, success = backtrack_scale_strategy(
        strat_del=make_zero_strategy(game),
        op=make_zero_trajectory(game),
        nlgame=game,
        max_iters=3,
        alpha_scale_init=1.0,
        alpha_scale_step=0.5,
        max_elwise_diff=1e-4
    )

    assert success
    assert scaled_strategy.P.shape == (game.nt-1, game.nu, game.nx)
    assert scaled_strategy.alpha.shape == (game.nt-1, game.nu)
    # assert new_traj.ts.shape == (game.T,)
    assert new_traj.tg is game.tg
    assert new_traj.xs.shape == (game.nt, game.nx)
    assert new_traj.us.shape == (game.nt-1, game.nu)

@pytest.fixture(params=[
    (2, 16, 8, 8, 0.1),
])
def nlgame_direct(request):
    """Nonlinear game with direct-control (i.e. dx/dt=u) linear dynamics, and zero costs"""
    N, nt, nx, nu, dt = request.param
    assert nx == nu, "For direct-control dynamics, state dims (n) must equal control dims (m)"
    tg = TimeGrid(nt=nt, dt=dt)
    dyn = lambda t, x, u: u
    cs = SampledContinuousSystemType1(tg=tg, nx=nx, nu=nu, dynamics=dyn)
    costs = [PlayerCostSpecContinuous(running=lambda t, x, u: 0) for _ in range(N)]
    return NonlinearGameType1(
        cs=cs,
        N=N,
        costs=costs,
        u_splits=jnp.asarray([nu//N] * N)
    )

def test_backtrack_scale_strategy_linear_dynamics_feedforward_strategy(nlgame_direct):
    game = nlgame_direct
    max_elwise_diff = 0.2
    alpha_del_val = 1.0
    
    # Construct previous zero trajectory and initial strategy
    last_op = make_zero_trajectory(game)
    strat_del = make_uniform_feedforward_strategy(game, alpha_val=alpha_del_val)

    # Run backtrack_scale
    scaled_strategy, new_op, success = backtrack_scale_strategy(
        strat_del,
        last_op,
        game,
        max_elwise_diff=max_elwise_diff,
        alpha_scale_init=1.0,       # Deliberately large
        alpha_scale_step=0.5,
        max_iters=10,
    )

    assert success, "backtrack_scale should find a feasible scaled strategy"

    # Confirm that the new trajectory is close enough
    diff = jnp.abs(new_op.xs - last_op.xs)
    max_diff = jnp.max(diff)
    assert max_diff > 0 # there should be some difference in operating point trajectories to 
                        # ensure backtrack scaling has not just returned the original op
    assert max_diff < max_elwise_diff, f"Max deviation {max_diff} exceeds threshold"

    # Confirm strategy was scaled down but original candidate not modified
    # NOTE: cannot directly check this downscaling of the absolute-valued (x, u) strategy
    # since downscaling occurs in del-valued (delx, delu) space
    # assert jnp.all(jnp.abs(scaled_strategy.alpha) < alpha_del_val), "Strategy should be scaled down"
    assert jnp.allclose(strat_del.alpha, alpha_del_val), "Candidate strategy should not be modified in place"

    # Confirm that resulting trajectory is nonzero (shows it took a real step) but that original op not modified
    assert jnp.any(new_op.xs != 0), "New trajectory should not be all zeros"
    assert jnp.all(last_op.xs == 0.0), "Old trajectory should not be modified in place"


def test_backtrack_scale_strategy_zero_step_game():
    """Backtracking should gracefully handle a game with no control intervals.

    This is a high-value regression test for the new time convention:
    ``nt = 1`` means there is exactly one state node and zero control stages.
    The iLQ backtracking path should therefore accept empty ``P``, ``alpha``,
    and ``u`` arrays without trying to pair them against the terminal state.
    """
    tg = TimeGrid(nt=1, dt=0.1)
    nx = 2
    nu = 2
    N = 2

    cs = SampledContinuousSystemType1(
        tg=tg,
        nx=nx,
        nu=nu,
        dynamics=lambda t, x, u: jnp.zeros_like(x),
    )
    nlgame = NonlinearGameType1(
        cs=cs,
        N=N,
        costs=[PlayerCostSpecContinuous(running=lambda t, x, u: 0) for _ in range(N)],
        u_splits=jnp.array([1, 1]),
    )

    op = FixedStepSystemTrajectory(
        tg=tg,
        xs=jnp.zeros((1, nx)),
        us=jnp.zeros((0, nu)),
    )
    strat_del = FixedStepAffineStrategies(
        tg=tg,
        P=jnp.zeros((0, nu, nx)),
        alpha=jnp.zeros((0, nu)),
    )

    scaled_strategy, new_traj, success = backtrack_scale_strategy(
        strat_del=strat_del,
        op=op,
        nlgame=nlgame,
        max_iters=3,
        alpha_scale_init=1.0,
        alpha_scale_step=0.5,
        max_elwise_diff=1e-6,
    )

    assert success
    assert scaled_strategy.P.shape == (0, nu, nx)
    assert scaled_strategy.alpha.shape == (0, nu)
    assert new_traj.xs.shape == (1, nx)
    assert new_traj.us.shape == (0, nu)
    assert jnp.allclose(new_traj.xs, op.xs)
    assert jnp.allclose(new_traj.us, op.us)


@pytest.fixture
def simple_2N_4T_lqgame():
    # very basic 2-player, one-timestep, linear-quadratic game
    # which is being encoded as a nonlinear game
    N = 2   # 2 players
    nt = 5   # 5 time nodes -> 4 time steps
    nx = 2   # 2D joint state space
    nu = 2   # 2D joint control space
    dt = 0.1    # [sec]
    tg = TimeGrid(nt=nt, dt=dt)
    u_splits = [1,1] # each player controls one dimension of joint control space

    # Formulate very simple time-invariant dynamics
    # NOTE: there is no time index because the nonlinear game definition allows for 
    # continuous dynamics and cost functions, in contrast to the linear-quadratic game definition
    # that is encoded as discrete time steps. The linearization process in the 
    # nonlinear solver creates the time-indexed discretization 
    A = jnp.eye(nx)
    B = jnp.eye(nx,nu)
    dynamics = lambda t, x, u: A @ x + B @ u

    # define control system
    cs = SampledContinuousSystemType1(tg=tg, nx=nx, nu=nu, dynamics=dynamics)

    # Formulate very simple, time-invariant cost function with only quadratic terms
    # NOTE: see note above about lack of time index in nonlinear game definition
    Q = jnp.zeros((N, nx, nx))
    Q = Q.at[0,:,:].set(jnp.eye(nx)) # player 1
    Q = Q.at[1,:,:].set(2.0*jnp.eye(nx)) # player 2

    q = jnp.zeros((N,nx))
    q = q.at[0,0].set(1.1)
    q = q.at[1,1].set(2.2)

    R = jnp.zeros((N,nu,nu))
    R = R.at[0,0,0].set(3.0)    # player 1
    R = R.at[1,1,1].set(4.0)    # player 2

    r = jnp.zeros((N,nu))
    r = r.at[0,0].set(3.3)
    r = r.at[1,1].set(4.4)

    costfns = N*[None]
    for i in range(N):
        # NOTE: need to bind the index i such that lambda in a loop doesn't use the final value of i
        # for all instances of the lambda function in the loop
        costfns[i] = lambda t, x, u, i=i: 0.5 * x.T @ Q[i] @ x + q[i].T @ x + 0.5 * u.T @ R[i] @ u + r[i].T @ u
    costs = [PlayerCostSpecContinuous(running=costfns[i]) for i in range(N)]

    # compose linear quadratic game as NonlinearGame object and equivalent linear quadratic game
    nlgame = NonlinearGameType1(
        cs=cs,
        N=N,
        costs=costs,
        u_splits=jnp.asarray(u_splits)
    )

    return nlgame, A, B, Q, q, R, r

def test_linearize_dynamics_linear(simple_2N_4T_lqgame):
    # check that linearization of a linear function returns the same function
    
    # ~~ ARRANGE ~~
    nlgame, A_exp, B_exp, _, _, _, _ = simple_2N_4T_lqgame

    # define a simple strategy to provide non-zero control inputs 
    # during trajectory generation
    P = jnp.broadcast_to(jnp.eye(nlgame.nu, nlgame.nx), (nlgame.nt-1, nlgame.nu, nlgame.nx))
    alpha = jnp.broadcast_to(jnp.ones(nlgame.nu), (nlgame.nt-1, nlgame.nu))
    strat = FixedStepAffineStrategies(tg=nlgame.tg, P=P, alpha=alpha)

    # compute trajectory of linear system following a strategy
    traj = propagate_system_trajectory(nlgame.cs,
        x0 = jnp.zeros(nlgame.nx),
        strategy = strat
    )

    # ~~ ACT ~~
    A, B = linearize_dynamics(f=nlgame.cs.dynamics, op=traj)

    # ~~ ASSERT ~~
    assert A.shape == (nlgame.nt-1, nlgame.nx, nlgame.nx)
    assert B.shape == (nlgame.nt-1, nlgame.nx, nlgame.nu)
    for t in range(nlgame.nt-1):
        # check that linearized dynamics in nonlinear game match
        # the underlying linear dynamics used to define the nonlinear game
        # object
        assert jnp.allclose(A[t], A_exp)
        assert jnp.allclose(B[t], B_exp)

def test_quadraticize_cost_playerwise_trajectory_quadratic(simple_2N_4T_lqgame):
    # check that quadratication of a quadratic cost returns the same function
    
    # ~~ ARRANGE ~~
    # Unpack the underlying quadratic function terms from the
    # nonlinear/nonquadratic game object that elides them
    nlgame, _, _, Qa, qa, Ra, ra = simple_2N_4T_lqgame

    # Unpack quadratic cost terms for each player, which are time invariant
    Qa_p1 = Qa[0]
    qa_p1 = qa[0]
    Ra_p1 = Ra[0]
    ra_p1 = ra[0]
    Qa_p2 = Qa[1]
    qa_p2 = qa[1]
    Ra_p2 = Ra[1]
    ra_p2 = ra[1]

    # define a simple strategy to provide non-zero control inputs 
    # during trajectory generation
    P = jnp.broadcast_to(jnp.eye(nlgame.nu, nlgame.nx), (nlgame.nt-1, nlgame.nu, nlgame.nx))
    alpha = jnp.broadcast_to(jnp.ones(nlgame.nu), (nlgame.nt-1, nlgame.nu))
    strat = FixedStepAffineStrategies(tg=nlgame.tg, P=P, alpha=alpha)

    # compute trajectory of linear system following a strategy
    traj = propagate_system_trajectory(nlgame.cs,
        x0 = jnp.zeros(nlgame.nx),
        strategy = strat
    )

    # ~~ ACT ~~

    # compute the gradient and hessian of each player's cost function 
    # with respect to x and u
    # NOTE: this is where the subtle distinction between the underlying
    # quadratic functions, defined with Qa, qa, Ra, ra, and the 
    # quadratic approximation that uses the second-order Taylor series 
    # about point (xt, ut) with terms Q, q, R, r which correspond to the
    # hessian wrt x at xt, jacobian wrt x at xt, hessian wrt u at ut, and
    # jacobian wrt u at ut, respectively. Even though the second-order
    # Taylor series is an exact approximation of the underlying quadratic
    # function, they have different form such that q_act != q and r_act != r.
    # This can be seen by equating the underlying quadratic with it's exact
    # second order Taylor series "approximation":
    # g_i(x,u) = 0.5 * x.T @ Qa @ x + qa.T @ x + 0.5 * u.T @ Ra @ u + ra.T @ u = 
    #   g_i(x0,u0) + 0.5*(x-x0).T @ Q @ (x-x0) + q.T @ (x-x0) + 0.5*(u-u0).T @ R @ (u-u0) + r.T @ (u-u0)
    # if you rearrange this expression and match terms based on constants, terms-of-x,
    # terms-of-x^2, terms-of-u, and terms-of-u^2, you will find that
    # Q = Qa, R = Ra, but q = qa + Qa @ xt and r = ra + Ra @ ut

    # Player 1: quadratic approximation terms at each time step t of trajectory
    Qt_p1, qt_p1, Rt_p1, rt_p1 = quadraticize_cost_joint_ctrl_playerwise_trajectory(
        nlgame.costs[0].running,
        traj, 
        nlgame.u_splits
    )

    # Player 2: quadratic approximation terms at each time step t of trajectory
    Qt_p2, qt_p2, Rt_p2, rt_p2 = quadraticize_cost_joint_ctrl_playerwise_trajectory(
        nlgame.costs[1].running,
        traj, 
        nlgame.u_splits
    )

    # ~~ ASSERT ~~
    assert Qt_p1.shape == (nlgame.nt-1, nlgame.nx, nlgame.nx)
    assert qt_p1.shape == (nlgame.nt-1, nlgame.nx)
    assert Rt_p1.shape == (nlgame.nt-1, nlgame.nu, nlgame.nu)
    assert rt_p1.shape == (nlgame.nt-1, nlgame.nu)
    assert Qt_p2.shape == (nlgame.nt-1, nlgame.nx, nlgame.nx)
    assert qt_p2.shape == (nlgame.nt-1, nlgame.nx)
    assert Rt_p2.shape == (nlgame.nt-1, nlgame.nu, nlgame.nu)
    assert rt_p2.shape == (nlgame.nt-1, nlgame.nu)

    for tidx in range(nlgame.nt-1):
        # check that quadraticized cost in nonlinear game match
        # the underlying quadratic costs used to define the nonlinear game
        # object
        xt = traj.xs[tidx]
        ut = traj.us[tidx]
        assert jnp.allclose(Qt_p1[tidx], Qa_p1)
        assert jnp.allclose(Rt_p1[tidx], Ra_p1)
        assert jnp.allclose(qt_p1[tidx], qa_p1 + Qa_p1 @ xt)
        assert jnp.allclose(rt_p1[tidx], ra_p1 + Ra_p1 @ ut)
        assert jnp.allclose(Qt_p2[tidx], Qa_p2)
        assert jnp.allclose(Rt_p2[tidx], Ra_p2)
        assert jnp.allclose(qt_p2[tidx], qa_p2 + Qa_p2 @ xt)
        assert jnp.allclose(rt_p2[tidx], ra_p2 + Ra_p2 @ ut)

@pytest.fixture(params=[
    (2, 4, 2, 2, 0.1, [1, 1]),
    (2, 20, 4, 2, 0.1, [1, 1]),
    (4, 8, 32, 16, 1.0, [2, 2, 4, 8]),
    # (5, 10, 87, 34, 3.45, [10, 2, 12, 6, 4]),
])
def arbitrary_time_varying_lqgame(request):
    # Multiplayer linear-quadratic, time-varying, game with arbitrarily generated parameters
    N, nt, nx, nu, dt, u_splits = request.param
    tg = TimeGrid(nt=nt, dt=dt)

    # Formulate arbitrary time-varying linear dynamics, but do so with a fixed
    # random number key generator so that it is the same set of dynamics
    # everytime the test is run
    # Note that the multiplication by scaling factor (e.g. 1e-3) is to try to
    # prevent system trajectories from "exploding" to large numbers that can
    # casue numerical errors that cause test failures
    key = jax.random.PRNGKey(42)  # fixed seed
    R_keys = jax.random.split(key, len(u_splits))
    A = jax.random.normal(key, shape=(nt-1,nx,nx)) * 1e-4
    B = jax.random.normal(key, shape=(nt-1,nx,nu)) * 1e-4
    def dynamics(t, x, u):
        t_idx = jnp.floor(t / dt).astype(int)
        t_idx = jnp.clip(t_idx, 0, nt - 2)  # ensure index is valid
        A_t = A[t_idx]
        B_t = B[t_idx]
        return A_t @ x + B_t @ u

    # Formulate arbitrary time-varying quadratic cost terms for each player, 
    # ensuring they are the same everytime the test is run
    Q = jnp.zeros((nt-1, N, nx, nx))
    q = jnp.zeros((nt-1, N, nx))
    R = jnp.zeros((nt-1, N, nu, nu))
    r = jnp.zeros((nt-1, N, nu))

    for t_idx in range(nt-1):
        for p_idx in range(N):
            Q = Q.at[t_idx, p_idx].set(jax.random.normal(key, shape=(nx,nx)))
            Q = Q.at[t_idx, p_idx].set(Q[t_idx,p_idx] + Q[t_idx,p_idx].T)   # makes random matrix symmetric
            q = q.at[t_idx, p_idx].set(jax.random.normal(key, shape=(nx,)))
            r = r.at[t_idx, p_idx].set(jax.random.normal(key, shape=(nu,)))

            # need to make R symmetric and block diagonal
            start = 0
            for size, subkey in zip(u_splits, R_keys):
                # Generate symmetric block
                R_block = jax.random.normal(subkey, (size, size))
                R_block = R_block + R_block.T

                # Insert into R
                R = R.at[t_idx, p_idx, start:start+size, start:start+size].set(R_block)
                start += size

    # encode the quadratic terms as quadratic cost functions
    costs = N*[None]
    for i in range(N):
        
        def cost_func(t, x, u, p_idx):
            t_idx = jnp.floor(t / dt).astype(int)
            t_idx = jnp.clip(t_idx, 0, nt - 2)  # ensure index is valid
            Q_t = Q[t_idx,p_idx]
            q_t = q[t_idx,p_idx]
            R_t = R[t_idx,p_idx]
            r_t = r[t_idx,p_idx]
            return 0.5 * x.T @ Q_t @ x + q_t.T @ x + 0.5 * u.T @ R_t @ u + r_t.T @ u

        # NOTE: need to bind the index i such that lambda in a loop doesn't use the final value of i
        # for all instances of the lambda function in the loop
        costs[i] = PlayerCostSpecContinuous(running=lambda t, x, u, i=i: cost_func(t, x, u, i))

    # compose linear quadratic game as NonlinearGame object and equivalent linear quadratic game
    cs = SampledContinuousSystemType1(tg=tg, nx=nx, nu=nu, dynamics=dynamics)
    nlgame = NonlinearGameType1(cs=cs, N=N, costs=costs, u_splits=jnp.asarray(u_splits))

    return nlgame, A, B, Q, q, R, r

@pytest.mark.slow
def test_arbitrary_time_varying_lqgame_lin_and_quad(arbitrary_time_varying_lqgame):
    # check that the linearization and quadraticization of an arbitrary, time-vaying 
    # linear-quadratic game returns the known, underlying linear and quadratic functions
    # NOTE: this is a bit of a misleading test because we are simply
    # linearizing the dynamics (which are, in fact, already linear), 
    # not discretizing them, which is the second
    # step required for converting a sampled continuous system into the
    # linearized approximation. However, this does check the 
    # underlying linearization functions, so we've kept it as-is
    
    # ~~ ARRANGE ~~
    # Unpack the underlying linear and quadratic function terms from the
    # nonlinear/nonquadratic game object that elides them
    nlgame, Aa, Ba, Qa, qa, Ra, ra = arbitrary_time_varying_lqgame

    # define a simple strategy to provide non-zero control inputs 
    # during trajectory generation
    key = jax.random.PRNGKey(42)  # fixed seed
    P = jax.random.normal(key, shape=(nlgame.nt-1, nlgame.nu, nlgame.nx))
    alpha = jax.random.normal(key, shape=(nlgame.nt-1, nlgame.nu))
    strat = FixedStepAffineStrategies(tg=nlgame.tg, P=P, alpha=alpha)

    # compute trajectory of linear system following a strategy
    traj = propagate_system_trajectory(nlgame.cs,
        x0 = jnp.zeros(nlgame.nx),
        strategy = strat
    )

    # ~~ ACT ~~

    # Compute time-varying linearization of dynamics
    A, B = linearize_dynamics(f=nlgame.cs.dynamics, op=traj)

    # Compute time-varying quadraticization of costs
    Q = jnp.zeros((nlgame.nt-1, nlgame.N, nlgame.nx, nlgame.nx))
    q = jnp.zeros((nlgame.nt-1, nlgame.N, nlgame.nx))
    R = jnp.zeros((nlgame.nt-1, nlgame.N, nlgame.nu, nlgame.nu))
    r = jnp.zeros((nlgame.nt-1, nlgame.N, nlgame.nu))
    for pidx in range(nlgame.N):
        Qp, qp, Rp, rp = quadraticize_cost_joint_ctrl_playerwise_trajectory(
            nlgame.costs[pidx].running, 
            traj, 
            nlgame.u_splits
        )
        Q = Q.at[:, pidx, :, :].set(Qp)
        q = q.at[:, pidx, :].set(qp)
        R = R.at[:, pidx, :, :].set(Rp)
        r = r.at[:, pidx, :].set(rp)

    # ~~ ASSERT ~~
    assert A.shape == (nlgame.nt-1, nlgame.nx, nlgame.nx)
    assert B.shape == (nlgame.nt-1, nlgame.nx, nlgame.nu)
    assert Q.shape == (nlgame.nt-1, nlgame.N, nlgame.nx, nlgame.nx)
    assert q.shape == (nlgame.nt-1, nlgame.N, nlgame.nx)
    assert R.shape == (nlgame.nt-1, nlgame.N, nlgame.nu, nlgame.nu)
    assert r.shape == (nlgame.nt-1, nlgame.N, nlgame.nu)
    for tidx in range(nlgame.nt-1):
        # check that linearized dynamics in nonlinear game match
        # the underlying linear dynamics used to define the nonlinear game
        # object
        # NOTE: checking just the linearization of dynamics, not the full
        # conversion to a linearized approximation, which also requires 
        # discretization
        np.testing.assert_allclose(np.asarray(A[tidx]), np.asarray(Aa[tidx]))
        np.testing.assert_allclose(np.asarray(B[tidx]), np.asarray(Ba[tidx]))

        # 
        xt = traj.xs[tidx]
        ut = traj.us[tidx]
        for pidx in range(nlgame.N):
            np.testing.assert_allclose(np.asarray(Q[tidx,pidx]), np.asarray(Qa[tidx,pidx]))
            np.testing.assert_allclose(np.asarray(R[tidx,pidx]), np.asarray(Ra[tidx,pidx]))
            np.testing.assert_allclose(np.asarray(q[tidx,pidx]), np.asarray(qa[tidx,pidx] + Qa[tidx,pidx] @ xt), rtol=1e-3)
            np.testing.assert_allclose(np.asarray(r[tidx,pidx]), np.asarray(ra[tidx,pidx] + Ra[tidx,pidx] @ ut), rtol=1e-3)

@pytest.mark.slow
def test_arbitrary_time_varying_lqgame_trajectory(arbitrary_time_varying_lqgame):
    # check that trajectory of underlying linear system matches trajectory
    # of linearized system
    
    # ~~ ARRANGE ~~
    # Unpack the underlying linear and quadratic function terms from the
    # nonlinear/nonquadratic game object that elides them
    nlgame, Ac, Bc, Q, q, R, r = arbitrary_time_varying_lqgame

    # discretize the underlying linear-quadratic game object
    A, B = discretize_extended_linear_dynamics_euler(Ac, Bc, nlgame.dt)

    # compose the underlyuing discretized linear-quadratic game object
    cs =LinearDiscreteSystemType1(
        tg = nlgame.tg,
        nx = nlgame.nx,
        nu = nlgame.nu,
        A = A, 
        B = B
    )
    lqgame = LinearQuadraticGameType1(
        cs = cs,
        N = nlgame.N,
        Q = Q,
        q = q,
        R = R,
        r = r,
        u_splits=nlgame.u_splits
    )

    # compute the nash strategy for the equivalent linear-quadratic game
    lq_strategy = solve_lqgame_feedback(lqgame)

    # compute the open-loop trajectory of the LQ nash strategy to be 
    # used as comparison with final nonlinear nash trajectory
    lq_trajectory = propagate_system_trajectory(nlgame.cs,
        x0 = jnp.zeros(nlgame.nx),
        strategy = lq_strategy
    )

    # ~~ ACT ~~ 
    # approximate the linear-quadratic game of nonlinear game (which is in fact
    # linear under the hood). 
    # NOTE: this approximation implies a change of variable space to delx, delu
    # that must be accounted for
    lq_approx_game_del = approx_linear_quadratic_game(nlgame, op=lq_trajectory)

    # solve for the nash strategy of the approximated lq game
    # in (delx, delu) state and control space
    # Note: enforcing check of block diagnonal R matrix that is output from 
    # approx_linear_quadratic_game. In solve_ilqgame_feedback, this check is 
    # turned off to accelerate computation
    lq_approx_strategy_del = solve_lqgame_feedback(lq_approx_game_del, check_block_diag=True)
    # lq_approx_strategy = solve_approx_lqgame_feedback(nlgame, lq_trajectory)

    # map the lq_approx strategy into absolute (x, u) variable space
    lq_approx_strategy = FixedStepAffineStrategies(
        tg = nlgame.tg,
        P=lq_approx_strategy_del.P, 
        alpha=lq_approx_strategy_del.alpha - lq_trajectory.us - jax.vmap(lambda P_t, x_t: P_t @ x_t)(lq_approx_strategy_del.P, lq_trajectory.xs[:-1])
    )

    # propagate trajectory of nonlinear system (which is actually linear under the hood)
    # using nash strategy of linear-quadratic approximate game mapped into 
    # absolut state and control space (x, u)
    lq_approx_trajectory = propagate_system_trajectory(nlgame.cs,
        x0 = jnp.zeros(nlgame.nx),
        strategy = lq_approx_strategy
    )

    # ~~ ASSERT ~~
    np.testing.assert_allclose(np.asarray(lq_trajectory.xs), np.asarray(lq_approx_trajectory.xs), atol=1e-4, rtol=1e-4)

@pytest.mark.slow
def test_solve_ilqgame_feedback_lq_converge(arbitrary_time_varying_lqgame):
    # Run the iterative linear-quadratic solver on a game that is actually linear-quadratic
    # to check that solver converges

    # ~~ ARRANGE ~~

    # Unpack the underlying linear and quadratic function terms from the
    # nonlinear/nonquadratic game object that elides them 
    nlgame, Ac, Bc, Q, q, R, r = arbitrary_time_varying_lqgame

    # initial state
    x0 = jnp.zeros((nlgame.nx,))

    # initial randomized strategy 
    # (should not affect end solution since underlying system is linear-quadratic)
    key = jax.random.PRNGKey(42)  # fixed seed
    P = jax.random.normal(key, shape=(nlgame.nt-1, nlgame.nu, nlgame.nx))
    alpha = jax.random.normal(key, shape=(nlgame.nt-1, nlgame.nu))
    init_strat = FixedStepAffineStrategies(tg=nlgame.tg, P=P, alpha=alpha)

    # ~~ ACT ~~

    # compute nash strategy for nonlinear game
    conv, nl_traj, nl_strat = solve_ilqgame_feedback(nlgame, x0, init_strat=init_strat, backtrack_max_iters=20)

    # ~~ ASSERT ~~

    assert conv


def test_solve_ilqgame_feedback_zero_step_game():
    """The full iLQ solve path should support ``nt = 1``.

    This checks the solver layer itself, rather than relying only on lower-level
    IR tests, because the default initialization inside ``solve_ilqgame_feedback``
    was one of the places that recently changed from ``nt``- to
    ``nsteps``-indexed control arrays.
    """
    tg = TimeGrid(nt=1, dt=0.1)
    nx = 3
    nu = 2
    N = 2
    x0 = jnp.array([1.0, -2.0, 0.5])

    cs = SampledContinuousSystemType1(
        tg=tg,
        nx=nx,
        nu=nu,
        dynamics=lambda t, x, u: jnp.zeros_like(x),
    )
    nlgame = NonlinearGameType1(
        cs=cs,
        N=N,
        costs=[PlayerCostSpecContinuous(running=lambda t, x, u: 0.0) for _ in range(N)],
        u_splits=jnp.array([1, 1]),
    )

    converged, traj, strat = solve_ilqgame_feedback(
        nlgame=nlgame,
        x0=x0,
        max_iters=3,
        backtrack_max_iters=3,
    )

    assert converged
    assert traj.xs.shape == (1, nx)
    assert traj.us.shape == (0, nu)
    assert strat.P.shape == (0, nu, nx)
    assert strat.alpha.shape == (0, nu)
    assert jnp.allclose(traj.xs[0], x0)

# def test_solve_ilqgame_feedback_lq(arbitrary_time_varying_lqgame):
#     # Run the iterative linear-quadratic solver on a game that is actually linear-quadratic
#     # to check that the iterative-LQ algorithm returns the same nash trajectory as the lq-solver

#     # ~~ ARRANGE ~~

#     # Unpack the underlying linear and quadratic function terms from the
#     # nonlinear/nonquadratic game object that elides them 
#     # nlgame, A, B, Q, q, R, r = request.getfixturevalue(fixture_name)
#     nlgame, A, B, Q, q, R, r = arbitrary_time_varying_lqgame

#     # compose the underlyuing linear-quadratic game object
#     # NOTE: must discretize the dynamics, first
#     Ad, Bd = discretize_extended_linear_dynamics_euler(Ac=A, Bc=B, dt=nlgame.dt)
#     lq_cs = LinearDiscreteSystemType1(
#         tg = nlgame.tg,
#         nx = nlgame.nx,
#         nu = nlgame.nu,
#         A = Ad,
#         B = Bd
#     )
#     lqgame = LinearQuadraticGameType1(
#         cs = lq_cs,
#         N = nlgame.N,
#         Q = Q,
#         q = q,
#         R = R,
#         r = r,
#         u_splits = nlgame.u_splits
#     )

#     # compute the nash strategy for the equivalent linear-quadratic game
#     lq_strategy = solve_lqgame_feedback(lqgame)

#     # compute the open-loop trajectory of the LQ nash strategy to be 
#     # used as comparison with final nonlinear nash trajectory
#     lq_trajectory = propagate_system_trajectory(
#         nlgame.cs,
#         x0 = jnp.zeros((nlgame.nx,)),
#         strategy = lq_strategy
#     )
#     # lq_trajectory = get_game_trajectory(
#     #     x0=jnp.zeros((nlgame.n,)), 
#     #     strategy=lq_strategy, 
#     #     dynamics_eom=nlgame.dynamics, 
#     #     dt=nlgame.dt
#     # )

#     # initial zero trajectory
#     init_traj = SystemTrajectory(
#         ts = jnp.linspace(0.0, nlgame.T * nlgame.dt, nlgame.T),
#         xs = jnp.zeros((nlgame.T, nlgame.n)),
#         us = jnp.zeros((nlgame.T, nlgame.m))
#     )

#     # initial randomized strategy 
#     # (should not affect end solution since underlying system is linear-quadratic)
#     key = jax.random.PRNGKey(42)  # fixed seed
#     P = jax.random.normal(key, shape=(nlgame.T, nlgame.m, nlgame.n))
#     alpha = jax.random.normal(key, shape=(nlgame.T, nlgame.m))
#     init_strat = AffineStrategy(P=P, alpha=alpha)

#     # ~~ ACT ~~

#     # compute nash strategy for nonlinear game
#     conv, nl_traj, nl_strategy = solve_ilqgame_feedback(nlgame, x0, init_traj, init_strat, backtrack_max_iters=20)

#     # ~~ ASSERT ~~

#     # check that strategies are equivalent
#     # assert jnp.allclose(nl_strategy.P, lq_strategy.P)
#     # assert jnp.allclose(nl_strategy.alpha, lq_strategy.alpha)

#     # compute the open-loop trajectory of the nonlinear nash strategy to be 
#     # used as the initial trajectory of the iLQ algorithm
#     nl_trajectory = get_game_trajectory(
#         x0=jnp.zeros((nlgame.n,)), 
#         strategy=nl_strategy, 
#         dynamics_eom=nlgame.dynamics, 
#         dt=nlgame.dt
#     )

#     # print(f"DEBUG: nl_traj.xs={nl_trajectory.xs}")
#     # print(f"DEBUG: lq_traj.xs={lq_trajectory.xs}")
#     xs_diff = jnp.abs(nl_trajectory.xs-lq_trajectory.xs)
#     tmax, imax = jnp.unravel_index(jnp.argmax(xs_diff), xs_diff.shape)
#     print(f"DEBUG: max xs diff={jnp.max(xs_diff)} at (t,i)={tmax,imax}")
#     print(f"DEBUG: nl_traj.x(t,i)={nl_trajectory.xs[tmax,imax]}, lq_traj.x(t,i)={lq_trajectory.xs[tmax,imax]}")
#     assert jnp.allclose(nl_trajectory.xs, lq_trajectory.xs, atol=1e-3, rtol=1e-2)
#     assert nl_trajectory.are_xs_close(other=lq_trajectory, max_elwise_diff=1e-3)


# def test_solve_ilqgame_feedback_lq(arbitrary_time_varying_lqgame):
#     # Run the iterative linear-quadratic solver on a game that is actually linear-quadratic
#     # to check that the iterative-LQ algorithm returns the same nash trajectory as the lq-solver

#     # ~~ ARRANGE ~~

#     # Unpack the underlying linear and quadratic function terms from the
#     # nonlinear/nonquadratic game object that elides them 
#     # nlgame, A, B, Q, q, R, r = request.getfixturevalue(fixture_name)
#     nlgame, A, B, Q, q, R, r = arbitrary_time_varying_lqgame

#     # define trajectory-generator strategy
#     # define a simple strategy to provide non-zero control inputs 
#     # during trajectory generation
#     key = jax.random.PRNGKey(42)  # fixed seed
#     P = jax.random.normal(key, shape=(nlgame.nt, nlgame.nu, nlgame.nx))
#     alpha = jax.random.normal(key, shape=(nlgame.nt, nlgame.nu))
#     gen_strat = FixedStepAffineStrategies(tg=nlgame.tg, P=P, alpha=alpha)

#     # propagate operating point trajectory for linearization
#     op = propagate_system_trajectory(nlgame.cs,
#         x0 = jnp.zeros(nlgame.nx),
#         strategy = gen_strat
#     )

#     # compose the underlyuing linear-quadratic game object
#     # in (delx, delu) state and control space
#     lq_game_del = approx_linear_quadratic_game(nlgame, op=op        )

#     # compute the nash strategy for the equivalent linear-quadratic game
#     lq_strat_del = solve_lqgame_feedback(lq_game_del)

#     # map the (delx, delu) strategy of the lq approximation (which should be
#     # exact, once accounting for change of vars and discretization, since
#     # the underlying system is linear-quadratic) into the absolute
#     # (x, u) state and control space
#     lq_strat = FixedStepAffineStrategies(
#         tg = nlgame.tg,
#         P=lq_strat_del.P, 
#         alpha=lq_strat_del.alpha - op.us - jax.vmap(lambda P_t, x_t: P_t @ x_t)(lq_strat_del.P, op.xs)
#     )

#     # compute the open-loop trajectory of the LQ nash strategy to be 
#     # used as comparison with final nonlinear nash trajectory
#     lq_traj = propagate_system_trajectory(nlgame.cs, x0=op.xs[0], strategy=lq_strat)

#     # # initial zero trajectory
#     # init_traj = SystemTrajectory(
#     #     ts = jnp.linspace(0.0, nlgame.T * nlgame.dt, nlgame.T),
#     #     xs = jnp.zeros((nlgame.T, nlgame.n)),
#     #     us = jnp.zeros((nlgame.T, nlgame.m))
#     # )

#     # # initial randomized strategy 
#     # # (should not affect end solution since underlying system is linear-quadratic)
#     # key = jax.random.PRNGKey(42)  # fixed seed
#     # P = jax.random.normal(key, shape=(nlgame.T, nlgame.m, nlgame.n))
#     # alpha = jax.random.normal(key, shape=(nlgame.T, nlgame.m))
#     # init_strat = AffineStrategy(P=P, alpha=alpha)

#     # ~~ ACT ~~

#     # compute nash strategy for nonlinear game
#     conv, nl_traj, nl_strategy = solve_ilqgame_feedback(nlgame, x0, init_traj, init_strat, backtrack_max_iters=20)

#     # ~~ ASSERT ~~

#     # check that strategies are equivalent
#     # assert jnp.allclose(nl_strategy.P, lq_strategy.P)
#     # assert jnp.allclose(nl_strategy.alpha, lq_strategy.alpha)

#     # compute the open-loop trajectory of the nonlinear nash strategy to be 
#     # used as the initial trajectory of the iLQ algorithm
#     nl_trajectory = get_game_trajectory(
#         x0=jnp.zeros((nlgame.n,)), 
#         strategy=nl_strategy, 
#         dynamics_eom=nlgame.dynamics, 
#         dt=nlgame.dt
#     )

#     # print(f"DEBUG: nl_traj.xs={nl_trajectory.xs}")
#     # print(f"DEBUG: lq_traj.xs={lq_trajectory.xs}")
#     xs_diff = jnp.abs(nl_trajectory.xs-lq_trajectory.xs)
#     tmax, imax = jnp.unravel_index(jnp.argmax(xs_diff), xs_diff.shape)
#     print(f"DEBUG: max xs diff={jnp.max(xs_diff)} at (t,i)={tmax,imax}")
#     print(f"DEBUG: nl_traj.x(t,i)={nl_trajectory.xs[tmax,imax]}, lq_traj.x(t,i)={lq_trajectory.xs[tmax,imax]}")
#     assert jnp.allclose(nl_trajectory.xs, lq_trajectory.xs, atol=1e-3, rtol=1e-2)
#     assert nl_trajectory.are_xs_close(other=lq_trajectory, max_elwise_diff=1e-3)
