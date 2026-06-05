# Copyright 2026 MIT Lincoln Laboratory
# SPDX-License-Identifier: MIT

import pytest
import jax
import jax.numpy as jnp
import numpy as np
import json

from copy import deepcopy
from pathlib import Path

from pydgens.ir.timetypes import TimeGrid
from pydgens.ir.systemtypes import LinearDiscreteSystemType1 as LinSys
from pydgens.ir.gametypes import LinearQuadraticGameType1 as LQGame
from pydgens.ir.costtypes import goal_cost_quadratic

# module under test
# from pydgens.solvers.lqsolver import solve_lqgame_feedback, solve_lqgame_feedback_old
import pydgens.solvers.lqsolver as lqsolver


def setup_simple_2N_1T_lqgame():
    # very basic 2-player, one-stage, linear-quadratic game with
    # only a quadratic control cost
    N = 2   # 2 players
    nt = 2   # two nodes => one control stage
    nx = 2   # 2D joint state space
    nu = 2   # 2D joint control space
    dt = 1.0    # time step of 1 second
    tg = TimeGrid(nt=nt, dt=dt)
    u_splits = jnp.asarray([1,1]) # each player controls one dimension of joint control space

    # One stage of simple identity dynamics.
    A = jnp.expand_dims(jnp.eye(nx), axis=0)
    B = jnp.expand_dims(jnp.eye(nx,nu), axis=0)

    # Compose control system
    cs = LinSys(tg=tg, nx=nx, nu=nu, A=A, B=B)

    # Formulate very simple cost function with only quadratic terms
    Q = jnp.zeros((tg.nsteps, N, nx, nx))
    # Q = Q.at[0,0,:,:].set(jnp.eye(nx))
    # Q = Q.at[0,1,:,:].set(jnp.eye(nx))

    q = jnp.zeros((tg.nsteps,N,nx))
    # q = q.at[0,0,:].set(jnp.ones(n))
    # q = q.at[0,1,:].set(jnp.ones(n))

    R = jnp.zeros((tg.nsteps,N,nu,nu))
    R = R.at[0,0,0,0].set(1.0)
    R = R.at[0,1,1,1].set(1.0)

    r = jnp.zeros((tg.nsteps,N,nu))
    # r = r.at[0,0,0].set(1.0)
    # r = r.at[0,1,1].set(1.0)

    # compose linear quadratic game object
    lqgame = LQGame(cs=cs, N=N, Q=Q, q=q, R=R, r=r, u_splits=u_splits)

    return lqgame

@pytest.fixture
def simple_2N_1T_lqgame():
    # define fixture wrapper so the underlying function can
    # also be called directly
    return setup_simple_2N_1T_lqgame()

@pytest.mark.slow
def test_solve_lqg_fb_2N_1T(simple_2N_1T_lqgame):
    # Check Nash of simple 2-player, one-step game is zero control input

    # ~~ ARRANGE ~~
    lqg = simple_2N_1T_lqgame

    # ~~ ACT ~~ 
    st = lqsolver.solve_lqgame_feedback(lqg)

    # ~~ ASSERT ~~
    assert st.P.shape == (lqg.nsteps, lqg.nu, lqg.nx)
    assert st.alpha.shape == (lqg.nsteps, lqg.nu)
    assert jnp.allclose(st.P, jnp.zeros((lqg.nsteps, lqg.nu, lqg.nx)))
    assert jnp.allclose(st.alpha, jnp.zeros((lqg.nsteps, lqg.nu)))

def test_solve_lqg_fb_2N_1T_affine_control(simple_2N_1T_lqgame):
    # If control cost has affine term, Nash control should shift of origin

    # ~~ ARRANGE ~~
    # shorten for brevity
    lqg = simple_2N_1T_lqgame

    # add affine components to control cost
    r_new = deepcopy(lqg.r)
    r_new = r_new.at[0,0,0].set(1.0)
    r_new = r_new.at[0,1,1].set(1.0)

    # create new game instance with affine control components
    lqg = LQGame(cs=lqg.cs, N=lqg.N, Q=lqg.Q, q=lqg.q, R=lqg.R, r=r_new, u_splits=lqg.u_splits)

    # ~~ ACT ~~ 
    st = lqsolver.solve_lqgame_feedback(lqg)

    # ~~ ASSERT ~~
    # now Nash control should be off the origin to minimize (u.T @ R_i + 2r_i.T) @ u
    assert st.P.shape == (lqg.nsteps, lqg.nu, lqg.nx)
    assert st.alpha.shape == (lqg.nsteps, lqg.nu)
    assert jnp.allclose(st.P, jnp.zeros((lqg.nsteps, lqg.nu, lqg.nx)))
    assert jnp.allclose(st.alpha, jnp.ones((lqg.nsteps, lqg.nu)))

def test_solve_lqg_fb_non_block_diagnonal_R(simple_2N_1T_lqgame):
    # most basic, 2-player, one-step game that should resolve to zero control input

    # ~~ ARRANGE ~~
    # shorten for brevity
    lqg = simple_2N_1T_lqgame

    # update R to be non-block diagonal
    R = deepcopy(lqg.R)
    R = R.at[0,0].set(jnp.ones((lqg.nu, lqg.nu)))

    # create new game instance with affine control components
    lqg = LQGame(lqg.cs, lqg.N, lqg.Q, lqg.q, R, lqg.r, lqg.u_splits)

    # ~~ ACT ~~ 
    with pytest.raises(ValueError, match="Non-block-diagonal"):
        st = lqsolver.solve_lqgame_feedback(lqg)
    

def setup_simple_2N_2T_lqgame_v1():
    # very basic 2-player, 2-stage, linear-quadratic game
    # where controls are very lightly penalized and each agent is just trying 
    # to move to the origin with "delta" dynamics where the next state (position)
    # is equal to player i's current state plus it's action (i.e. control is
    # change in position)
    N = 2   # 2 players
    nt = 3   # 3 nodes => 2 control stages
    nx = 2   # 2D joint state space (i.e position of each player)
    nu = 2   # 2D joint control space (i.e. change in position of each player)
    dt = 0.1    # time step of 0.1 sec
    tg = TimeGrid(nt=nt, dt=dt)
    u_splits = jnp.asarray([1,1]) # each player controls one dimension of joint control space

    # Formulate very simple dynamics for single timestep (i.e. need for expand dims)
    A = jnp.zeros((tg.nsteps,nx,nx))
    B = jnp.zeros((tg.nsteps,nx,nu))
    Q = jnp.zeros((tg.nsteps, N, nx, nx))
    q = jnp.zeros((tg.nsteps,N,nx))
    R = jnp.zeros((tg.nsteps,N,nu,nu))
    r = jnp.zeros((tg.nsteps,N,nu))

    # dynamics and costs are same at all time steps
    for t in range(tg.nsteps):
        # formulate simple "delta" dynamics
        A = A.at[t].set(jnp.eye(nx))
        B = B.at[t].set(jnp.eye(nx,nu))

        # Formulate very simple cost function with only quadratic costs
        # (no affine terms)
        Q = Q.at[t,0,0,0].set(1.0)
        Q = Q.at[t,1,1,1].set(1.0)

        R = R.at[t,0,0,0].set(1e-9)
        R = R.at[t,1,1,1].set(1e-9)

    # compose control system
    cs = LinSys(tg=tg, nx=nx, nu=nu, A=A, B=B)

    # compose linear quadratic game object
    lqgame = LQGame(cs=cs, N=N, Q=Q, q=q, R=R, r=r, u_splits=u_splits)

    return lqgame

def setup_simple_2N_2T_lqgame_v2():
    # very basic 2-player, 2-stage, terminal cost linear-quadratic game
    # where controls are very lightly penalized and each agent is just trying 
    # to move to the origin with "delta" dynamics where the next state (position)
    # is equal to player i's current state plus it's action (i.e. control is
    # change in position)
    N = 2   # 2 players
    nt = 3   # 3 time nodes => 2 control stages (+ terminal state cost)
    nx = 2   # 2D joint state space (i.e position of each player)
    nu = 2   # 2D joint control space (i.e. change in position of each player)
    dt = 0.1    # time step of 0.1 sec
    tg = TimeGrid(nt=nt, dt=dt)
    u_splits = jnp.asarray([1,1]) # each player controls one dimension of joint control space

    # Formulate very simple dynamics for single timestep (i.e. need for expand dims)
    A = jnp.zeros((tg.nsteps,nx,nx))
    B = jnp.zeros((tg.nsteps,nx,nu))
    Q = jnp.zeros((tg.nsteps,N,nx,nx))
    q = jnp.zeros((tg.nsteps,N,nx))
    R = jnp.zeros((tg.nsteps,N,nu,nu))
    r = jnp.zeros((tg.nsteps,N,nu))

    # dynamics and costs are same at all time steps
    for t in range(tg.nsteps):
        # formulate simple "delta" dynamics
        A = A.at[t].set(jnp.eye(nx))
        B = B.at[t].set(jnp.eye(nx,nu))

        # Formulate very simple cost function with only quadratic costs
        # (no affine terms)
        Q = Q.at[t,0,0,0].set(1.0)
        Q = Q.at[t,1,1,1].set(1.0)

        R = R.at[t,0,0,0].set(1e-9)
        R = R.at[t,1,1,1].set(1e-9)

    # terminal state costs: continuation of running costs
    Qf = Q[-1]
    qf = q[-1]

    # compose control system
    cs = LinSys(tg=tg, nx=nx, nu=nu, A=A, B=B)

    # compose linear quadratic game object
    lqgame = LQGame(cs=cs, N=N, Q=Q, q=q, R=R, r=r, Qf=Qf, qf=qf, u_splits=u_splits)

    return lqgame

@pytest.fixture
def simple_2N_2T_lqgame_v1():
    return setup_simple_2N_2T_lqgame_v1()

@pytest.fixture
def simple_2N_2T_lqgame_v2():
    return setup_simple_2N_2T_lqgame_v2()

def test_solve_lqg_fb_2N_2T_1(simple_2N_2T_lqgame_v1):
    # Check Nash of simple 2-player, two-step game is zero control input

    # ~~ ARRANGE ~~
    # shorten for brevity
    lqg = simple_2N_2T_lqgame_v1

    # ~~ ACT ~~ 
    st = lqsolver.solve_lqgame_feedback(lqg)

    # ~~ ASSERT ~~
    assert st.P.shape == (lqg.nsteps, lqg.nu, lqg.nx)
    assert st.alpha.shape == (lqg.nsteps, lqg.nu)

    # at time 0, both agents should just trying to "leap" to 
    # the origin, i.e. 0 position for their respective states, 
    # mostly ignoring the cost of control which has small penalty
    # in the equation u = -P @ x - alpha, this should just appear
    # as simply the identy matrix
    assert jnp.allclose(st.P[0], jnp.eye(lqg.nu, lqg.nx))
    assert jnp.allclose(st.alpha[0], jnp.zeros(lqg.nu,))
    assert jnp.allclose(st.P[1], jnp.zeros((lqg.nu, lqg.nx)))
    assert jnp.allclose(st.alpha[1], jnp.zeros((lqg.nu,)))

@pytest.mark.regression
def test_solve_2N_2T_1_strat_regress(simple_2N_2T_lqgame_v1):
    """regression test to check against approved strategy values of lq solver"""
    # ~~ ARRANGE ~~
    # shorten for brevity
    lqg = simple_2N_2T_lqgame_v1

    # load approved strategy values
    approved_filepath = Path(__file__).resolve().parent.joinpath("approvals/lqsolver/")
    P_app_filepath = approved_filepath.joinpath("test_solve_2N_2T_1_strat_regress_P.json")
    alpha_app_filepath = approved_filepath.joinpath("test_solve_2N_2T_1_strat_regress_alpha.json")
    with open(P_app_filepath) as f:
        P_approved = jnp.array(json.load(f))
    with open(alpha_app_filepath) as f:
        alpha_approved = jnp.array(json.load(f))

    # ~~ ACT ~~ 
    st = lqsolver.solve_lqgame_feedback(lqg)

    # ~~ ASSERT ~~
    np.testing.assert_allclose(np.asarray(st.P), np.asarray(P_approved))
    np.testing.assert_allclose(np.asarray(st.alpha), np.asarray(alpha_approved))

@pytest.mark.regression
def test_solve_2N_2T_2_Qf_strat_regress(simple_2N_2T_lqgame_v2):
    """
    regression test to check against approved strategy values of lq solver 
    with 2-stage + terminal state cost
    """
    # ~~ ARRANGE ~~
    # shorten for brevity
    lqg = simple_2N_2T_lqgame_v2

    # load approved strategy values
    approved_filepath = Path(__file__).resolve().parent.joinpath("approvals/lqsolver/")
    P_app_filepath = approved_filepath.joinpath(
        "test_solve_2N_2T_2_Qf_strat_regress_P.json"
    )
    alpha_app_filepath = approved_filepath.joinpath(
        "test_solve_2N_2T_2_Qf_strat_regress_alpha.json"
    )
    with open(P_app_filepath) as f:
        P_approved = jnp.array(json.load(f))
    with open(alpha_app_filepath) as f:
        alpha_approved = jnp.array(json.load(f))

    # ~~ ACT ~~ 
    st = lqsolver.solve_lqgame_feedback(lqg)

    # ~~ ASSERT ~~
    np.testing.assert_allclose(np.asarray(st.P), np.asarray(P_approved))
    np.testing.assert_allclose(np.asarray(st.alpha), np.asarray(alpha_approved))

def test_compare_solve_lqgame_feedback_implementations(simple_2N_2T_lqgame_v1):

    game = simple_2N_2T_lqgame_v1

    # run both
    strat_old = lqsolver.solve_lqgame_feedback_old(game, check_block_diag=False)
    strat_new = lqsolver.solve_lqgame_feedback(game, check_block_diag=False)

    P_old, a_old = strat_old.P, strat_old.alpha
    P_new, a_new = strat_new.P, strat_new.alpha

    assert np.allclose(P_old, P_new)
    assert np.allclose(a_old, a_new)

    # overall diffs
    P_abs = jnp.max(jnp.abs(P_old - P_new))
    a_abs = jnp.max(jnp.abs(a_old - a_new))
    print("max |ΔP|:", float(P_abs), "  max |Δα|:", float(a_abs))

    # per-step diffs to find the first time index with nontrivial difference
    diffs = jnp.max(jnp.abs(P_old - P_new), axis=(1,2))
    t_bad = int(jnp.argmax(diffs))
    print("first biggest Δ at t =", t_bad, "  |ΔP_t|:", float(diffs[t_bad]))
    # return t_bad

def test_bad_terminal_init_is_not_nash_for_one_stage_game():
    # This test is meant to investigate the claim that iLQGames.jl's 
    # initialization of Z0 and zeta0 is in error (as of v0.2.7). 
    # We solve a simple, single-step LQ game using iLQGames.jl's 
    # initializations of Z=Q[-1] and zeta=q[-1] and show that
    # each player can in fact unilaterily improve their performance
    # by deviating from this "bad terminal init" strategy, thus
    # showing it is not a Nash strategy

    # One control stage: nt=2, nsteps=1
    N = 2
    nx = 2
    nu = 2
    tg = TimeGrid(nt=2, dt=1.0)
    u_splits = jnp.array([1, 1])

    A = jnp.expand_dims(jnp.eye(nx), axis=0)
    B = jnp.expand_dims(jnp.eye(nx, nu), axis=0)

    # Nonzero running state cost, no affine terms, PD local control penalties
    Q = jnp.zeros((1, N, nx, nx))
    Q = Q.at[0, 0, 0, 0].set(1.0)
    Q = Q.at[0, 1, 1, 1].set(1.0)

    q = jnp.zeros((1, N, nx))

    R = jnp.zeros((1, N, nu, nu))
    R = R.at[0, 0, 0, 0].set(1.0)
    R = R.at[0, 1, 1, 1].set(1.0)

    r = jnp.zeros((1, N, nu))

    cs = LinSys(tg=tg, nx=nx, nu=nu, A=A, B=B)
    lqg = LQGame(cs=cs, N=N, Q=Q, q=q, R=R, r=r, u_splits=u_splits)

    # Correct solver (Z0, zeta0 -> zeros)
    st_z0 = lqsolver.solve_lqgame_feedback_old(lqg)
    assert jnp.allclose(st_z0.P[0], jnp.zeros((nu, nx)))
    assert jnp.allclose(st_z0.alpha[0], jnp.zeros((nu,)))

    # Wrong solver boundary condition (Z0, zeta0 -> Q[-1], q[-1])
    st_zq = lqsolver.solve_lqgame_feedback_old(
        lqgame = lqg,
        Z0 = Q[-1],
        zeta0 = q[-1]
    )

    x0 = jnp.array([2.0, -3.0])
    # Single-stage game: take the only joint control vector u[0].
    u_zq = -st_zq.P[0] @ x0 - st_zq.alpha[0]

    # For a one-stage game with no terminal cost and r=0, best response is u_i = 0
    # since x0 is fixed and only the local quadratic control term depends on u_i.
    for i, (s, e) in enumerate([(0, 1), (1, 2)]):
        ui_zq = u_zq[s:e]

        # Joint control under bad strategy
        u_joint_zq = u_zq

        # Joint control after unilateral deviation to the true one-stage best response u_i = 0
        u_joint_dev = u_zq.at[s:e].set(jnp.zeros_like(ui_zq))

        Qi = lqg.Q[0, i]
        qi = lqg.q[0, i]
        Ri = lqg.R[0, i]
        ri = lqg.r[0, i]

        J_bad = 0.5 * (x0 @ (Qi @ x0)) + (qi @ x0) + 0.5 * (u_joint_zq @ (Ri @ u_joint_zq)) + (ri @ u_joint_zq)
        J_dev = 0.5 * (x0 @ (Qi @ x0)) + (qi @ x0) + 0.5 * (u_joint_dev @ (Ri @ u_joint_dev)) + (ri @ u_joint_dev)

        assert J_dev < J_bad


def _lq_feedback_residuals_given_terminal_boundary(lqg, strat, Z_terminal, zeta_terminal):
    """
    Replay the backward LQ recursion using a fixed terminal boundary condition
    and measure how well a provided strategy satisfies the stage-wise linear
    system S * [P_k | alpha_k] = [YP_k | Ya_k].

    This is useful for distinguishing a true indexing/boundary-condition issue
    from a mere implementation difference:
    - if a strategy is the correct solution under the zero terminal boundary,
      these residuals should be ~0 at every stage
    - if a strategy was computed using a different boundary condition, the
      residuals should generally be nonzero when evaluated under the zero
      terminal boundary

    Returns
    -------
    P_residuals : jnp.ndarray of shape (nsteps,)
        Infinity-norm residual for each P[k].
    alpha_residuals : jnp.ndarray of shape (nsteps,)
        Infinity-norm residual for each alpha[k].
    """
    g = lqg
    K = g.nsteps

    u_sizes = [int(s) for s in g.u_splits]
    u_starts = [0]
    for s in u_sizes[:-1]:
        u_starts.append(u_starts[-1] + s)
    u_ends = [a + b for a, b in zip(u_starts, u_sizes)]

    Z = Z_terminal
    zeta = zeta_terminal
    P_residuals = []
    alpha_residuals = []

    for k in range(K - 1, -1, -1):
        A_k = g.A[k]
        B_k = g.B[k]
        Q_k = g.Q[k]
        q_k = g.q[k]
        R_k = g.R[k]
        r_k = g.r[k]

        S = jnp.zeros((g.nu, g.nu), dtype=A_k.dtype)
        YP = jnp.zeros((g.nu, g.nx), dtype=A_k.dtype)
        Ya = jnp.zeros((g.nu,), dtype=A_k.dtype)

        for i, (s, e) in enumerate(zip(u_starts, u_ends)):
            B_i = B_k[:, s:e]
            B_i_Z_i = B_i.T @ Z[i]
            S = S.at[s:e, :].set(B_i_Z_i @ B_k)
            S = S.at[s:e, s:e].add(R_k[i, s:e, s:e])
            YP = YP.at[s:e, :].set(B_i_Z_i @ A_k)
            Ya = Ya.at[s:e].set(B_i.T @ zeta[i] + r_k[i, s:e])

        Y = jnp.concatenate([YP, Ya[:, None]], axis=1)
        X = jax.scipy.linalg.solve(S, Y)
        P_expected = X[:, :-1]
        alpha_expected = X[:, -1]

        P_residuals.append(jnp.max(jnp.abs(strat.P[k] - P_expected)))
        alpha_residuals.append(jnp.max(jnp.abs(strat.alpha[k] - alpha_expected)))

        # Advance the value recursion backward using the provided strategy so we
        # test whether *that* strategy is self-consistent with the boundary.
        P_k = strat.P[k]
        alpha_k = strat.alpha[k]
        F_k = A_k - B_k @ P_k
        beta_k = -B_k @ alpha_k

        Z_next = Z
        zeta_next = zeta
        for i in range(g.N):
            R_ki = R_k[i]
            Z_i = Q_k[i] + P_k.T @ R_ki @ P_k + F_k.T @ Z[i] @ F_k
            zeta_i = (
                q_k[i]
                + P_k.T @ (R_ki @ alpha_k - r_k[i])
                + F_k.T @ (zeta[i] + Z[i] @ beta_k)
            )
            Z_next = Z_next.at[i].set(Z_i)
            zeta_next = zeta_next.at[i].set(zeta_i)

        Z = Z_next
        zeta = zeta_next

    return jnp.flip(jnp.asarray(P_residuals)), jnp.flip(jnp.asarray(alpha_residuals))


def test_final_stage_stationarity_depends_only_on_running_control_cost():
    """
    Validate the final-stage first-order condition directly.

    For a running-cost-only LQ game, the final control stage k = K-1 has no
    future cost-to-go beyond it. Therefore the final-stage stationarity
    condition depends only on R[K-1] and r[K-1], not on Q[K-1] or q[K-1].

    This test uses a two-stage game whose final-stage running state cost is
    deliberately nonzero. The correct zero-boundary recursion should still
    produce P[K-1] = 0 and alpha[K-1] = 0 because r[K-1] = 0. The "bad init"
    recursion, by contrast, incorrectly feeds Q[K-1] into the final-stage
    control law and therefore produces a nonzero feedback gain.
    """
    lqg = setup_simple_2N_2T_lqgame_v1()

    # Emphasize the final-stage state cost so a bad terminal initialization has
    # a strong, easily detectable effect.
    Q_new = deepcopy(lqg.Q)
    Q_new = Q_new.at[-1, 0, 0, 0].set(7.0)
    Q_new = Q_new.at[-1, 1, 1, 1].set(11.0)
    lqg = LQGame(cs=lqg.cs, N=lqg.N, Q=Q_new, q=lqg.q, R=lqg.R, r=lqg.r, u_splits=lqg.u_splits)

    # Correct boundary condition: no terminal cost beyond the last running-cost stage.
    st_z0 = lqsolver.solve_lqgame_feedback_old(lqg)

    # Julia-style initialization that treats the last running-cost slice like a terminal cost.
    st_zq = lqsolver.solve_lqgame_feedback_old(
        lqgame=lqg,
        Z0=lqg.Q[-1],
        zeta0=lqg.q[-1],
    )

    # The true final-stage Nash condition depends only on R_last and r_last.
    # Here r_last = 0 and R_last is block diagonal positive definite on each
    # player's own control, so the unique final-stage equilibrium is u = 0,
    # which implies zero feedback gain and zero offset.
    assert jnp.allclose(st_z0.P[-1], jnp.zeros((lqg.nu, lqg.nx)))
    assert jnp.allclose(st_z0.alpha[-1], jnp.zeros((lqg.nu,)))

    # If the last running state cost is incorrectly injected into the boundary
    # condition, the final-stage gain becomes spuriously nonzero.
    assert not jnp.allclose(st_zq.P[-1], jnp.zeros((lqg.nu, lqg.nx)))


def test_terminal_state_cost_drives_nonzero_final_feedback_gain():
    """
    A one-stage game with only terminal state cost should induce a nonzero
    final-stage feedback gain.

    This exercises the new explicit Qf/qf pathway rather than relying on the
    old "inject Q[-1] into Z0" convention.
    """
    N = 2
    nx = 2
    nu = 2
    tg = TimeGrid(nt=2, dt=1.0)
    u_splits = jnp.array([1, 1])

    A = jnp.expand_dims(jnp.eye(nx), axis=0)
    B = jnp.expand_dims(jnp.eye(nx, nu), axis=0)
    Q = jnp.zeros((1, N, nx, nx))
    q = jnp.zeros((1, N, nx))
    R = jnp.zeros((1, N, nu, nu))
    R = R.at[0, 0, 0, 0].set(1.0)
    R = R.at[0, 1, 1, 1].set(1.0)
    r = jnp.zeros((1, N, nu))
    Qf = jnp.zeros((N, nx, nx))
    Qf = Qf.at[0, 0, 0].set(4.0)
    Qf = Qf.at[1, 1, 1].set(9.0)
    qf = jnp.zeros((N, nx))

    cs = LinSys(tg=tg, nx=nx, nu=nu, A=A, B=B)
    lqg = LQGame(cs=cs, N=N, Q=Q, q=q, R=R, r=r, u_splits=u_splits, Qf=Qf, qf=qf)

    st_new = lqsolver.solve_lqgame_feedback(lqg)
    st_old = lqsolver.solve_lqgame_feedback_old(lqg)

    assert not jnp.allclose(st_new.P[0], jnp.zeros((nu, nx)))
    assert jnp.allclose(st_new.P, st_old.P)
    assert jnp.allclose(st_new.alpha, st_old.alpha)


def test_zero_terminal_boundary_recursion_residuals_distinguish_correct_vs_bad_init():
    """
    Check the actual backward-recursion equations, not just the resulting
    controls on one example.

    We replay the stage-wise linear system under the intended zero terminal
    boundary condition Z_K = 0, zeta_K = 0. A correct strategy should satisfy
    those equations at every stage up to numerical tolerance. A strategy
    generated from the nonzero "bad init" boundary should fail this test,
    because it solves a different dynamic program.
    """
    lqg = setup_integrator_2N_20T_lqgame_v1()

    st_z0 = lqsolver.solve_lqgame_feedback_old(lqg)
    st_zq = lqsolver.solve_lqgame_feedback_old(
        lqgame=lqg,
        Z0=lqg.Q[-1],
        zeta0=lqg.q[-1],
    )

    Z_terminal = jnp.zeros((lqg.N, lqg.nx, lqg.nx), dtype=lqg.A.dtype)
    zeta_terminal = jnp.zeros((lqg.N, lqg.nx), dtype=lqg.A.dtype)

    P_res_z0, a_res_z0 = _lq_feedback_residuals_given_terminal_boundary(
        lqg, st_z0, Z_terminal, zeta_terminal
    )
    P_res_zq, a_res_zq = _lq_feedback_residuals_given_terminal_boundary(
        lqg, st_zq, Z_terminal, zeta_terminal
    )

    # The correct strategy should satisfy the zero-terminal-boundary recursion
    # essentially exactly up to floating-point error.
    assert jnp.max(P_res_z0) < 1e-8
    assert jnp.max(a_res_z0) < 1e-8

    # The bad-init strategy should violate that same recursion. We do not need
    # to prescribe where the violation appears; it is enough that at least one
    # stage fails by a meaningful margin.
    assert (jnp.max(P_res_zq) > 1e-6) or (jnp.max(a_res_zq) > 1e-6)

def setup_integrator_2N_20T_lqgame_v1():
    # integrator dynamics with 2-players, over 20 control stages, linear-quadratic game
    # Ref: https://clearoboticslab.github.io/documents/smooth_game_theory.pdf example 37
    N = 2   # 2 players
    nt = 21   # 21 nodes => 20 control stages
    nx = 8   # joint state space (i.e position of each player)
    nx1 = 4
    nx2 = 4
    nu = 4   # joint control space (i.e. change in position of each player)
    u_splits = jnp.asarray([2,2]) # each player controls one dimension of joint control space
    nu1 = u_splits[0]
    nu2 = u_splits[1]

    # Formulate very simple dynamics for single timestep (i.e. need for expand dims)
    tg = TimeGrid(nt=nt, dt=1.0)
    A = jnp.zeros((tg.nsteps,nx,nx))
    B = jnp.zeros((tg.nsteps,nx,nu))
    Q = jnp.zeros((tg.nsteps, N, nx, nx))
    q = jnp.zeros((tg.nsteps,N,nx))
    R = jnp.zeros((tg.nsteps,N,nu,nu))
    r = jnp.zeros((tg.nsteps,N,nu))

    # dynamics and costs are same at all time steps
    dt = 1.0    # [s] time step size of system
    for t in range(tg.nsteps):
        # formulate integrator dynamics
        A = A.at[t,0:nx1,0:nx1].set(jnp.eye(nx1))
        A = A.at[t,nx1:nx,nx1:nx].set(jnp.eye(nx2))
        A = A.at[t,0,1].set(dt)
        A = A.at[t,2,3].set(dt)
        A = A.at[t,4,5].set(dt)
        A = A.at[t,6,7].set(dt)
        
        B = B.at[t,1,0].set(dt)
        B = B.at[t,3,1].set(dt)
        B = B.at[t,5,2].set(dt)
        B = B.at[t,7,3].set(dt)

        # Formulate cost function where 
        # player 1 minimizes state of player two (drives it to origin and zero vel)
        # player 2 minimizes difference in state to player 1 (matches pos and vel)
        # both players minimize ther control effort
        # (no affine terms)
        Q = Q.at[t,0,nx1:nx,nx1:nx].set(jnp.eye(nx2))
        Q = Q.at[t,1].set(jnp.eye(nx))
        Q = Q.at[t,1,0:nx1,nx1:nx].set(-jnp.eye(nx2))
        Q = Q.at[t,1,nx1:nx,0:nx1].set(-jnp.eye(nx2))

        R = R.at[t,0,0:nu1,0:nu1].set(jnp.eye(nu1))
        R = R.at[t,1,nu1:nu,nu1:nu].set(jnp.eye(nu2))

    # compose control system
    cs = LinSys(tg=tg, nx=nx, nu=nu, A=A, B=B)

    # compose linear quadratic game object
    lqgame = LQGame(cs=cs, N=N, Q=Q, q=q, R=R, r=r, u_splits=u_splits)

    return lqgame

@pytest.fixture
def integrator_2N_20T_lqgame_v1():
    # define fixture wrapper so the underlying function can
    # also be called directly (i.e. in main when running this as a script)
    return setup_integrator_2N_20T_lqgame_v1()

def setup_integrator_2N_20T_lqgame_v2():
    # integrator dynamics with 2-players, over 20 control stages, 
    # terminal state cost, linear-quadratic game
    
    # copy version 1 with only running costs
    lqg1 = setup_integrator_2N_20T_lqgame_v1()

    # create terminal costs that are continuation of running costs
    Qf = lqg1.Q[-1]
    qf = lqg1.q[-1]

    lqg2 = LQGame(
        cs=lqg1.cs,
        N=lqg1.N,
        Q=lqg1.Q,
        q=lqg1.q,
        R=lqg1.R,
        r=lqg1.r,
        u_splits=lqg1.u_splits,
        Qf=Qf,
        qf=qf
    )

    return lqg2


@pytest.mark.regression
def test_solve_integrator_2N_20T_v1_strat_regress(integrator_2N_20T_lqgame_v1):
    """regression test to check against approved strategy values of lq solver"""
    # ~~ ARRANGE ~~
    # shorten for brevity
    lqg = integrator_2N_20T_lqgame_v1

    # load approved strategy values
    approved_filepath = Path(__file__).resolve().parent.joinpath("approvals/lqsolver/")
    P_app_filepath = approved_filepath.joinpath("test_solve_integrator_2N_20T_v1_strat_regress_P.json")
    alpha_app_filepath = approved_filepath.joinpath("test_solve_integrator_2N_20T_v1_strat_regress_alpha.json")
    with open(P_app_filepath) as f:
        P_approved = jnp.array(json.load(f))
    with open(alpha_app_filepath) as f:
        alpha_approved = jnp.array(json.load(f))
    

    # ~~ ACT ~~ 
    st = lqsolver.solve_lqgame_feedback(lqg)

    # ~~ ASSERT ~~
    print("")
    print(f"P_approved.shape={P_approved.shape}")
    print(f"P_computed.shape={st.P.shape}")
    print("")
    print(f"P_approved={P_approved[-1]}")
    print(f"P_computed={st.P[-1]}")

    print("")
    print(f"alpha_approved.shape={alpha_approved.shape}")
    print(f"alpha_computed.shape={st.alpha.shape}")
    print("")
    print(f"alpha_approved={alpha_approved[-1]}")
    print(f"alpha_computed={st.alpha[-1]}")


    np.testing.assert_allclose(np.asarray(st.P), np.asarray(P_approved), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(st.alpha), np.asarray(alpha_approved), rtol=1e-5, atol=1e-6)

@pytest.mark.regression
def test_solve_integrator_2N_20T_v2_Qf_strat_regress():
    """
    regression test to check against approved strategy values
      of terminal cost lq solver
    """
    # ~~ ARRANGE ~~
    # shorten for brevity
    lqg = setup_integrator_2N_20T_lqgame_v2()

    # load approved strategy values
    approved_filepath = Path(__file__).resolve().parent.joinpath("approvals/lqsolver/")
    P_app_filepath = approved_filepath.joinpath(
        "test_solve_integrator_2N_20T_v2_Qf_strat_regress_P.json"
    )
    alpha_app_filepath = approved_filepath.joinpath(
        "test_solve_integrator_2N_20T_v2_Qf_strat_regress_alpha.json"
    )
    with open(P_app_filepath) as f:
        P_approved = jnp.array(json.load(f))
    with open(alpha_app_filepath) as f:
        alpha_approved = jnp.array(json.load(f))
    

    # ~~ ACT ~~ 
    st = lqsolver.solve_lqgame_feedback(lqg)

    # ~~ ASSERT ~~

    np.testing.assert_allclose(np.asarray(st.P), np.asarray(P_approved), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(st.alpha), np.asarray(alpha_approved), rtol=1e-5, atol=1e-6)


@pytest.mark.slow
def test_solve_lqg_fb_integrator_2N_20T_v1(integrator_2N_20T_lqgame_v1):
    # For example 37 in https://clearoboticslab.github.io/documents/smooth_game_theory.pdf
    # check that the feedback nash solution converges toward the origin
    # (zero pos and vel for both players)

    # ~~ ARRANGE ~~
    # shorten for brevity
    lqg = integrator_2N_20T_lqgame_v1

    # ~~ ACT ~~ 
    st = lqsolver.solve_lqgame_feedback(lqg)

    # define initial joint state
    x0 = jnp.array([-.15, 0, -0.6, 0, -0.45, 0, -0.05, 0])

    # propagate system
    x = x0
    for t in range(lqg.cs.nsteps):
        u = - st.P[t] @ x - st.alpha[t]
        x = lqg.cs.A[t] @ x + lqg.cs.B[t] @ u

    # ~~ ASSERT ~~
    assert st.P.shape == (lqg.nsteps, lqg.nu, lqg.nx)
    assert st.alpha.shape == (lqg.nsteps, lqg.nu)
    
    # final x should have converged toward origin
    assert jnp.allclose(x, jnp.zeros(lqg.cs.nx,), atol=1e-4)

@pytest.mark.regression
@pytest.mark.slow
@pytest.mark.benchmark(group="lqsolver-001")
def test_solve_lqg_fb_warm_perf_1(benchmark):
    """ Benchmark performance of warm-started linear-quadratic solver on 
    2-player integrator system defined in example 37 in 
    https://clearoboticslab.github.io/documents/smooth_game_theory.pdf
    """

    lqg = setup_integrator_2N_20T_lqgame_v1()

    def run():
        st = lqsolver.solve_lqgame_feedback(lqg)
        jax.block_until_ready(st)
    
    # warm compile once
    run()

    # benchmark warm-started run
    benchmark(run)

@pytest.mark.regression
@pytest.mark.slow
@pytest.mark.benchmark(group="lqsolver-002")
def test_solve_lqg_fb_warm_perf_2(benchmark):
    # Check Nash of simple 2-player, one-step game is zero control input

    # ~~ ARRANGE ~~
    lqg = setup_simple_2N_1T_lqgame()

    # ~~ ACT ~~ 
    def run():
        st = lqsolver.solve_lqgame_feedback(lqg)
        jax.block_until_ready(st)
    
    # warm compile once
    run()

    # benchmark warm-started run
    benchmark(run)

@pytest.mark.regression
@pytest.mark.slow
@pytest.mark.benchmark(group="lqsolver-003")
def test_solve_lqg_fb_warm_perf_3(benchmark):
    # Check Nash of simple 2-player, one-step game is zero control input

    # ~~ ARRANGE ~~
    lqg = setup_simple_2N_2T_lqgame_v1()

    # ~~ ACT ~~ 
    def run():
        st = lqsolver.solve_lqgame_feedback(lqg)
        jax.block_until_ready(st)
    
    # warm compile once
    run()

    # benchmark warm-started run
    benchmark(run)

@pytest.mark.slow
def test_solve_lqg_fb_integrator_2N_20T_v1_target(integrator_2N_20T_lqgame_v1):
    # For example 37 in https://clearoboticslab.github.io/documents/smooth_game_theory.pdf
    # check that the feedback nash solution converges toward non-zero target state
    # i.e. player 1 wants to drive player 2 to a target state, and player 2 wants
    # to minimize distance to player 1

    # ~~ ARRANGE ~~
    # unpack variables from fixture
    # shorten for brevity
    lqg = integrator_2N_20T_lqgame_v1
    nx1 = 4
    nx2 = 4
    
    # define initial joint state
    x0 = jnp.array([0, 0, 0, 0, 0, 0, 0, 0])

    # define target state of player 1
    # Note: that player 1 is "driving" player 2
    # toward the target, but "doesn't care" about 
    # its own position because Q for player 1 only
    # has non-zero entries for player two's state
    # thus we can arbitrarily assign player 1's 
    # state in the target and it should not affect
    # the resulting system
    xg1 = jnp.array([
        23489., -12.873, 19.20, -176.,    # player 1's target state for itself (irrelevant)
        10.0, 0, 10.0, 0    # player 1's target state for player 2
    ])

    # compute goal cost quadratic function for player 1
    q = deepcopy(lqg.q)
    for t in range(lqg.cs.nsteps):
        _, qt = goal_cost_quadratic(xg1, lqg.Q[t,0])
        q = q.at[t,0].set(qt)

    # create new game object
    lqg = LQGame(cs=lqg.cs, N=lqg.N, Q=lqg.Q, q=q, R=lqg.R, r=lqg.r, u_splits=lqg.u_splits)

    # ~~ ACT ~~ 
    st = lqsolver.solve_lqgame_feedback(lqg)

    # propagate system
    x = x0
    for t in range(lqg.cs.nsteps):
        u = - st.P[t] @ x - st.alpha[t]
        x = lqg.cs.A[t] @ x + lqg.cs.B[t] @ u

    # ~~ ASSERT ~~
    assert st.P.shape == (lqg.nsteps, lqg.nu, lqg.nx)
    assert st.alpha.shape == (lqg.nsteps, lqg.nu)
    
    # final x should have converged toward origin
    assert jnp.allclose(x[nx1:lqg.nx], xg1[nx1:lqg.nx], atol=1e-3)  # player 2 should have been driven to target
    assert jnp.allclose(x[0:nx1], xg1[nx1:lqg.nx], atol=1e-3)  # player 1 should have followed to target
