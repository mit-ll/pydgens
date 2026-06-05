# Copyright 2026 MIT Lincoln Laboratory
# SPDX-License-Identifier: MIT

import pytest
import jax.numpy as jnp

from copy import deepcopy

from pydgens.ir.timetypes import TimeGrid

# module under test
import pydgens.ir.trajectorytypes as irtraj

def test_valid_creation():
    nt = 3
    dt = 0.1
    t0 = -1.0
    tg = TimeGrid(nt=nt, dt=dt, t0=t0)

    xs = jnp.array([[1.0, 2.0], [1.1, 2.1], [1.2, 2.2]])
    us = jnp.array([[0.0, 0.1], [0.1, 0.2]])

    traj = irtraj.FixedStepSystemTrajectory(tg, xs, us)

    assert traj.nt == 3
    assert traj.nsteps == 2
    assert traj.nx == 2
    assert traj.nu == 2
    assert jnp.isclose(traj.dt, dt)
    assert jnp.isclose(traj.t0, t0)


def test_valid_zero_step_creation():
    nt = 1
    dt = 0.1
    tg = TimeGrid(nt=nt, dt=dt)

    xs = jnp.array([[1.0, 2.0]])
    us = jnp.empty((0, 2))

    traj = irtraj.FixedStepSystemTrajectory(tg, xs, us)

    assert traj.nt == 1
    assert traj.nsteps == 0
    assert traj.nx == 2
    assert traj.nu == 2
    assert jnp.isclose(traj.dt, dt)


def test_make_system_trajectory_accepts_zero_step_controls():
    tg = TimeGrid(nt=1, dt=0.1)

    traj = irtraj.make_system_trajectory(
        tg=tg,
        xs=[[1.0, 2.0]],
        us=jnp.zeros((0, 2)),
    )

    assert traj.nt == 1
    assert traj.nsteps == 0
    assert traj.nu == 2


def test_make_system_trajectory_rejects_rank1_empty_controls_for_zero_step_case():
    tg = TimeGrid(nt=1, dt=0.1)

    with pytest.raises(ValueError, match="empty 2D array"):
        irtraj.make_system_trajectory(
            tg=tg,
            xs=[[1.0, 2.0]],
            us=[],
        )


def test_invalid_lengths_raise():
    tg = TimeGrid(nt=3, dt=0.1)
    xs = jnp.array([[1.0, 2.0], [1.1, 2.1], [1.2, 2.2]])
    us = jnp.array([[0.0, 0.1], [0.1, 0.2]])

    xs_wrong = jnp.array([[1.0, 2.0], [1.1, 2.1]])
    with pytest.raises(ValueError, match="Inconsistent number of state steps"):
        irtraj.FixedStepSystemTrajectory(tg, xs_wrong, us)

    us_wrong = jnp.array([[0.0, 0.1], [0.1, 0.2], [0.2, 0.3]])
    with pytest.raises(ValueError, match="Inconsistent number of control steps"):
        irtraj.FixedStepSystemTrajectory(tg, xs, us_wrong)

    tg_zero = TimeGrid(nt=1, dt=0.1)
    xs_zero = jnp.array([[1.0, 2.0]])
    us_zero_wrong = jnp.array([[0.0, 0.1]])
    with pytest.raises(ValueError, match="Inconsistent number of control steps"):
        irtraj.FixedStepSystemTrajectory(tg_zero, xs_zero, us_zero_wrong)


def test_invalid_dimensions_raise():
    tg = TimeGrid(nt=3, dt=0.1)
    xs_invalid = jnp.array([1.0, 2.0, 3.0])  # Should be 2D
    us = jnp.array([[0.0, 0.1], [0.1, 0.2]])

    with pytest.raises(ValueError, match="xs must be 2-dimensional"):
        irtraj.FixedStepSystemTrajectory(tg, xs_invalid, us)

    xs = jnp.array([[1.0, 2.0], [1.1, 2.1], [1.2, 2.2]])
    us_invalid = jnp.array([0.0, 0.1, 0.2])  # Should be 2D

    with pytest.raises(ValueError, match="us must be 2-dimensional"):
        irtraj.FixedStepSystemTrajectory(tg, xs, us_invalid)

def test_are_xs_close_exact_match():
    traj = irtraj.make_system_trajectory(
        TimeGrid(nt=3, dt=0.1, t0=0.0),
        xs=[[1.0, 2.0], [1.1, 2.1], [1.2, 2.2]],
        us=[[0.0, 0.0], [0.1, 0.1]],
    )
    assert irtraj.are_xs_close(traj, traj, max_elwise_diff=1e-6)


def test_are_xs_close_exact_match_zero_step_case():
    traj = irtraj.make_system_trajectory(
        TimeGrid(nt=1, dt=0.1, t0=0.0),
        xs=[[1.0, 2.0]],
        us=jnp.zeros((0, 2)),
    )

    assert irtraj.are_xs_close(traj, traj, max_elwise_diff=1e-6)

def test_are_xs_close_within_threshold():
    traj1 = irtraj.make_system_trajectory(
        TimeGrid(nt=2, dt=0.1),
        xs=[[1.0, 2.0], [1.1, 2.1]],
        us=[[0.0, 0.0]]
    )
    traj2 = irtraj.make_system_trajectory(
        TimeGrid(nt=2, dt=0.1),
        xs=[[1.001, 2.001], [1.099, 2.101]],
        us=[[0.0, 0.0]]
    )
    assert irtraj.are_xs_close(traj1, traj2, max_elwise_diff=0.01)

def test_are_xs_close_within_threshold_1():
    traj1 = irtraj.make_system_trajectory(
        TimeGrid(nt=2, dt=0.1),
        xs=[[1.0, 2.0], [1.1, 2.1]],
        us=[[0.0, 0.0]]
    )
    traj2 = irtraj.make_system_trajectory(
        TimeGrid(nt=2, dt=0.1),
        xs=[[1.001, 2.001], [1.099, 2.101]],
        us=[[0.0, 0.0]]
    )
    assert irtraj.are_xs_close(traj1, traj2, max_elwise_diff=0.01)

def test_are_xs_close_exceeds_threshold():
    traj1 = irtraj.make_system_trajectory(
        TimeGrid(nt=2, dt=0.1),
        xs=[[1.0, 2.0], [1.1, 2.1]],
        us=[[0.0, 0.0]]
    )
    traj2 = irtraj.make_system_trajectory(
        TimeGrid(nt=2, dt=0.1),
        xs=[[1.0, 2.5], [1.1, 2.1]],  # large change in one element
        us=[[0.0, 0.0]]
    )
    assert not irtraj.are_xs_close(traj1, traj2, max_elwise_diff=0.2)

def test_are_xs_close_mismatched_xs_shape():
    traj1 = irtraj.make_system_trajectory(
        TimeGrid(nt=2, dt=0.1),
        xs=[[1.0, 2.0], [1.1, 2.1]],
        us=[[0.0, 0.0]]
    )
    traj2 = irtraj.make_system_trajectory(
        TimeGrid(nt=1, dt=0.1),
        xs=[[1.0, 2.0]],  # only one step
        us=jnp.empty((0,2))
    )
    with pytest.raises(ValueError, match="State trajectories"):
        irtraj.are_xs_close(traj1, traj2, max_elwise_diff=0.01)

def test_are_xs_close_inconsistent_times():
    traj1 = irtraj.make_system_trajectory(
        tg=TimeGrid(nt=2, dt=0.1, t0=-1),
        xs=[[1.0, 2.0], [1.1, 2.1]],
        us=[[0.0, 0.0]]
    )
    traj2 = irtraj.make_system_trajectory(
        tg=TimeGrid(nt=2, dt=0.11, t0=-1),
        xs=[[1.0, 2.0], [1.1, 2.1]],
        us=[[0.0, 0.0]],
    )
    with pytest.raises(ValueError, match="Inconsistent time"):
        irtraj.are_xs_close(traj1, traj2, max_elwise_diff=0.01)

    traj2 = irtraj.make_system_trajectory(
        tg=TimeGrid(nt=2, dt=0.1, t0=1.0),
        xs=[[1.0, 2.0], [1.1, 2.1]],
        us=[[0.0, 0.0]],
    )
    with pytest.raises(ValueError, match="Inconsistent time"):
        irtraj.are_xs_close(traj1, traj2, max_elwise_diff=0.01)

def test_property_access():
    tg = TimeGrid(nt=3, dt=0.1)
    xs = jnp.array([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])
    us = jnp.array([[1.0, 1.0], [2.0, 2.0]])

    traj = irtraj.FixedStepSystemTrajectory(tg, xs, us)

    assert traj.nt == 3
    assert traj.nsteps == 2
    assert traj.nx == 2
    assert traj.nu == 2

@pytest.fixture
def example_traj():
    tg = TimeGrid(nt=5, dt=1.0, t0=0.0)
    xs = jnp.ones((5, 3))
    us = jnp.zeros((4, 2))
    return irtraj.FixedStepSystemTrajectory(tg, xs, us)


def test_cannot_set_nt(example_traj):
    with pytest.raises(AttributeError, match="cannot assign to field 'nt'"):
        example_traj.nt = 10

def test_cannot_set_nsteps(example_traj):
    with pytest.raises(AttributeError, match="cannot assign to field 'nsteps'"):
        example_traj.nsteps = 10

def test_cannot_set_nx(example_traj):
    with pytest.raises(AttributeError, match="cannot assign to field 'nx'"):
        example_traj.nx = 4

def test_cannot_set_nu(example_traj):
    with pytest.raises(AttributeError, match="cannot assign to field 'nu'"):
        example_traj.nu = 1

@pytest.fixture
def example_traj_16_8_4_v1():
    tg = TimeGrid(nt=16, dt=10.0)
    xs = [
        [6.62080750e+01, -1.70932868e+02, -2.17225560e-01,  2.33421393e+01,
        -6.22468383e+01,  1.62632844e+02,  1.77398604e+02, -8.67130339e+01],
        [-1.35707241e+02, -1.18578085e+02, -2.64879690e+02,  9.12417588e+01,
        -8.47448659e+00,  5.49756746e+01,  6.11770699e+01,  1.31229025e+02],
        [ 1.39258884e+02, -4.20648004e+01, -6.79083811e+01, -1.24434910e+02,
        1.29942260e+02,  6.90858259e+01, -1.53810227e+02,  4.94275148e+01],
        [ 1.48300099e+02,  1.71515169e+02,  1.36301294e+02, -2.20067391e+00,
        6.12930378e+01,  1.13367166e+02, -2.34947809e+02, -1.43833525e+02],
        [-1.67653829e+02,  1.07031848e+02, -6.31894920e+01,  8.64615602e+01,
        -4.15764698e+01, -6.31320134e+00, -5.00966377e+01,  6.70791683e+01],
        [ 5.76862882e+01, -1.41745373e+01, -1.90854409e+02,  6.73548335e+01,
        1.36437350e+02, -1.27445562e+00,  2.15541908e+01, -6.46275503e+00],
        [-4.78258620e+01,  2.00045463e+01, -9.27779388e+01,  8.04341133e+01,
        -6.89844465e+01, -1.05570505e+02,  2.29083099e+01, -6.93607218e+01],
        [-1.55865307e+01,  1.47854634e+02, -1.19297280e+02, -5.92236983e+01,
        1.82981645e+02,  2.79904819e+01, -5.76457711e+01,  6.44514028e+01],
        [ 2.07361676e+01,  1.11199350e+02,  1.82605346e+01,  1.44148251e+02,
        -1.89884933e+02,  4.82942209e+01,  5.35467331e+01,  6.81415728e+01],
        [-9.08601842e+00,  1.77039354e+02, -1.37406885e+02,  1.35601250e+02,
        5.21844068e+01, -1.78643214e+02, -4.31357264e+01,  4.88401591e+01],
        [-2.02652696e+02, -1.48211119e+02, -1.76164755e+01, -5.46760883e+01,
        -1.68809335e+01,  1.02615992e+02, -3.64377859e+02, -7.46574906e+01],
        [-5.62137307e+00,  7.92933310e+01,  2.30632813e+01, -2.81036294e+01,
        1.12618618e+02,  4.80647453e+00,  4.05281421e+00,  6.16143038e+00],
        [-8.71509546e+01,  2.69258007e+01, -2.61969535e+01, -3.55332289e+01,
        1.39009758e+02,  3.87591420e+01, -1.27689913e+02, -1.02931922e+02],
        [ 1.19940436e+02, -1.47024355e+02,  9.52549015e+01, -3.49535043e+01,
        4.82098854e+00,  3.03655755e+01,  1.48398257e+02, -3.59212871e+01],
        [-1.77690338e+02, -5.00991265e+00,  8.72212112e+01,  5.51062768e+01,
        1.26448410e+02,  1.36453915e+02,  1.05704371e+02,  1.06799868e+02],
        [ 1.02078883e+01,  2.32475441e+01,  3.45640407e+01,  7.96913251e+01,
        5.33373960e+01, -9.26331528e+01, -7.59000554e+01, -2.03161578e+01]
    ]
    us = [
        [ 2.25868239e+02, -6.73011168e+01, -1.11282653e+02,  5.06237605e+01],
        [-3.56449898e+01, -4.39285551e+01, -4.12300655e+01, -4.72211445e+01],
        [ 1.21921053e+01, -3.63263652e+01,  1.03968861e+01, -1.27242110e+02],
        [-2.57329292e+02, -9.94375144e+01,  1.22002779e+02,  1.46613740e+01],
        [ 3.39086391e+01,  1.21506407e+02, -1.46234024e+02,  1.04727138e+02],
        [-1.04679253e+02,  2.68620087e+01,  7.09798108e+01,  3.26951782e+01],
        [-8.82838667e+00,  1.07062573e+01,  1.54561419e+01, -6.26397130e+01],
        [-1.08352197e+02,  3.50135315e+01,  1.19429137e+02,  1.44877535e+02],
        [ 2.11143846e+01,  2.54323945e+01, -1.36042426e+01, -1.68787695e+01],
        [-2.40711739e+01,  4.77363432e+01, -4.54889787e+00, -1.11031861e+01],
        [-1.60793609e+02,  1.30599514e-01, -3.09398876e+01, -1.65138159e+00],
        [-9.84827641e+01, -1.15530538e+02,  2.65859056e+01,  6.15605253e+01],
        [ 1.18829309e+02,  3.53003590e+01, -1.13078245e+02,  1.53365538e+02],
        [-5.47598139e+01,  1.70237856e+01, -1.13335233e+02, -1.96307862e+01],
        [ 2.39362988e+02,  2.04289747e+02,  1.54625493e+02, -5.95631623e+01],
    ]
    return irtraj.make_system_trajectory(tg, xs, us)

def test_are_xs_close_long1_success(example_traj_16_8_4_v1):
    traj1 = example_traj_16_8_4_v1
    xs2 = deepcopy(traj1.xs)
    xs2 = xs2.at[:].add(1e-8)
    traj2 = irtraj.make_system_trajectory(traj1.tg, xs2, traj1.us)

    assert irtraj.are_xs_close(traj1, traj2, max_elwise_diff=1e-6)

def test_are_xs_close_long1_failure(example_traj_16_8_4_v1):
    traj1 = example_traj_16_8_4_v1
    xs2 = deepcopy(traj1.xs)
    xs2 = xs2.at[:].add(1e-5)
    traj2 = irtraj.make_system_trajectory(traj1.tg, xs2, traj1.us)

    assert not irtraj.are_xs_close(traj1, traj2, max_elwise_diff=1e-6)

def test_pdtraj_valid_creation():
    nt = 3
    dt = 0.1
    t0 = -1.0
    tg = TimeGrid(nt=nt, dt=dt, t0=t0)

    xs = jnp.array([[1.0, 2.0, 3.0, 4.0], [1.1, 2.1, 3.1, 4.1], [1.2, 2.2, 3.2, 4.2]])
    us = jnp.array([[0.0], [0.1]])
    ls = jnp.array([
        [[-1.0, -2.0, -3.0, -4.0], [-1.1, -2.1, -3.1, -4.1]],   # t=0
        [[-10.0, -20.0, -30.0, -40.0], [-10.1, -20.1, -30.1, -40.1]]    # t=1
    ])

    pdtraj = irtraj.FixedStepPrimalDualTrajectory(tg, xs, us, ls)

    assert pdtraj.nt == 3
    assert pdtraj.nx == 4
    assert pdtraj.nu == 1
    assert pdtraj.N == 2
    assert jnp.isclose(pdtraj.dt, dt)
    assert jnp.isclose(pdtraj.t0, t0)

def test_pdtraj_invalid_raises():
    tg = TimeGrid(nt=3, dt=0.1)
    xs = jnp.array([[1.0, 2.0, 3.0, 4.0], [1.1, 2.1, 3.1, 4.1], [1.2, 2.2, 3.2, 4.2]])
    us = jnp.array([[0.0], [0.1]])
    ls = jnp.array([
        [[-1.0, -2.0, -3.0, -4.0], [-1.1, -2.1, -3.1, -4.1]],   # t=0
        [[-10.0, -20.0, -30.0, -40.0], [-10.1, -20.1, -30.1, -40.1]]    # t=1
    ])

    # Too many dimensions in state
    xs_wrong = jnp.array([[[1.0]], [[2.0]]])
    with pytest.raises(ValueError, match="xs must be 2-dimensional"):
        irtraj.FixedStepPrimalDualTrajectory(tg, xs_wrong, us, ls)

    # Too few state steps
    xs_wrong = jnp.array([[1.0, 2.0, 3.0, 4.0], [1.1, 2.1, 3.1, 4.1]])
    with pytest.raises(ValueError, match="Inconsistent number of state steps"):
        irtraj.FixedStepPrimalDualTrajectory(tg, xs_wrong, us, ls)

    # Too many dimensions in control trajectory
    us_wrong = jnp.array([[[1.0]], [[2.0]]])
    with pytest.raises(ValueError, match="us must be 2-dimensional"):
        irtraj.FixedStepPrimalDualTrajectory(tg, xs, us_wrong, ls)

    # Too many control steps
    us_wrong = jnp.array([[0.1, 0.2, 0.3]])
    with pytest.raises(ValueError, match="Inconsistent number of control steps"):
        irtraj.FixedStepPrimalDualTrajectory(tg, xs, us_wrong, ls)

    # Too few dimensions in dual trajectory (Lagrange multipliers)
    ls_wrong = jnp.array([
        [-1.0, -2.0, -3.0, -4.0],   # t=0
        [-10.0, -20.0, -30.0, -40.0]    # t=1
    ])
    with pytest.raises(ValueError, match="ls must be 3-dimensional"):
        irtraj.FixedStepPrimalDualTrajectory(tg, xs, us, ls_wrong)

    # Too many dual trajectory steps
    ls_wrong = jnp.array([
        [[-1.0, -2.0, -3.0, -4.0], [-1.1, -2.1, -3.1, -4.1]],   # t=0
        [[-10.0, -20.0, -30.0, -40.0], [-10.1, -20.1, -30.1, -40.1]],    # t=1
        [[-100.0, -200.0, -300.0, -400.0], [-100.1, -200.1, -300.1, -400.1]]    # t=1
    ])
    with pytest.raises(ValueError, match="Inconsistent number of Lagrange"):
        irtraj.FixedStepPrimalDualTrajectory(tg, xs, us, ls_wrong)

    # Langrange multipliers don't match state dimensions
    ls_wrong = jnp.array([
        [[-1.0, -2.0, -3.0], [-1.1, -2.1, -3.1]],   # t=0
        [[-10.0, -20.0, -30.0], [-10.1, -20.1, -30.1]],    # t=1
    ])
    with pytest.raises(ValueError, match="Inconsistent Lagrange multiplier shape"):
        irtraj.FixedStepPrimalDualTrajectory(tg, xs, us, ls_wrong)


@pytest.mark.parametrize('trajfix', ['example_traj', 'example_traj_16_8_4_v1'])
def test_get_ctrl_traj_single_player_all_ctrls(trajfix, request):
    """If there is only one player, they should get the full control trajectory."""
    traj = request.getfixturevalue(trajfix) # FixedStepSystemTrajector, us of length nt, not nt-1
    u_splits = jnp.array([traj.nu])  # one player with all control dims

    us_i = irtraj.get_player_control_trajectory(traj, 0, u_splits)

    assert us_i.shape == (traj.nsteps, traj.nu)
    assert jnp.allclose(us_i, traj.us)

def test_get_ctrl_traj_three_player_correct_slices(example_traj_16_8_4_v1):
    """If there is only one player, they should get the full control trajectory."""
    traj = example_traj_16_8_4_v1 # FixedStepSystemTrajector, us of length nt, not nt-1
    nu_0 = 1
    nu_1 = 2
    nu_2 = 1
    u_splits = jnp.array([nu_0, nu_1, nu_2])  # one player with all control dims

    us_0 = irtraj.get_player_control_trajectory(traj, 0, u_splits)
    us_1 = irtraj.get_player_control_trajectory(traj, 1, u_splits)
    us_2 = irtraj.get_player_control_trajectory(traj, 2, u_splits)

    nu_cum = 0 

    assert us_0.shape == (traj.nsteps, nu_0)
    assert jnp.allclose(us_0, traj.us[:, 0:nu_0])
    nu_cum += nu_0

    assert us_1.shape == (traj.nsteps, nu_1)
    assert jnp.allclose(us_1, traj.us[:, nu_cum:nu_cum+nu_1])
    nu_cum += nu_1

    assert us_2.shape == (traj.nsteps, nu_2)
    assert jnp.allclose(us_2, traj.us[:, nu_cum:nu_cum+nu_2])
