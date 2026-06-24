# Copyright 2026 MIT Lincoln Laboratory
# SPDX-License-Identifier: MIT

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from types import SimpleNamespace

# helper types and functions
from pydgens.ir.timetypes import TimeGrid, compute_ts
from pydgens.ir.trajectorytypes import FixedStepPrimalDualTrajectory
from pydgens.ir.systemtypes import (
    SampledContinuousSystemType1, 
    LinearDiscreteSystemType1, 
    propagate_system_trajectory, 
    residual_discrete_dynamics_trajectory
)
from pydgens.ir.costtypes import PlayerCostSpecContinuous
from pydgens.ir.costtypes import ControlDomain as CostControlDomain
from pydgens.ir.costtypes import ControlStructure as CostControlStructure
from pydgens.ir.constrainttypes import (
    ConstraintBlockGridMap,
    GameConstraintGridMap,
    build_constraint_step_linearizations
)
from pydgens.ir.gametypes import NonlinearGameType2, LinearQuadraticGameType1
from pydgens.solvers.lqsolver import solve_lqgame_feedback

# module under test
import pydgens.solvers.alsolver as pdg_alsolver
import pydgens.ir.altypes as altypes

def _make_game_and_op(nt=6, nx=3, nu=7, N=4):
    tg = TimeGrid(nt=nt, dt=0.1, t0=0.0)
    K = nt - 1

    xs = jnp.zeros((nt, nx), dtype=jnp.float32)
    us = jnp.zeros((K, nu), dtype=jnp.float32)
    ls = jnp.zeros((K, N, nx), dtype=jnp.float32)
    op = SimpleNamespace(tg=tg, xs=xs, us=us, ls=ls, nt=nt, nsteps=K, nx=nx, nu=nu)

    # minimal nlgame mock: compute_al_residual_struct needs constraints for shared ingredients
    cs = SimpleNamespace(tg=tg)
    constraints = GameConstraintGridMap(ineq_blocks=(), eq_blocks=())
    nlgame = SimpleNamespace(tg=tg, cs=cs, N=N, nt=nt, nx=nx, nu=nu, constraints=constraints)

    return nlgame, op


# -------------------------
# ALResidualStruct + validate
# -------------------------

def test_validate_al_residual_struct_accepts_good_shapes():
    N, K, nx, nu = 3, 5, 2, 4
    r = altypes.ALResidualStruct(
        dLdX=jnp.zeros((N, K, nx), dtype=jnp.float32),
        dLdU=jnp.zeros((N, K, nu), dtype=jnp.float32),
        dyn_res=jnp.zeros((K, nx), dtype=jnp.float32),
    )
    # should not raise
    altypes.validate_al_residual_struct(r)


@pytest.mark.parametrize(
    "dLdX_shape,dLdU_shape,dyn_shape,err_match",
    [
        ((3, 5), (3, 5, 4), (5, 2), "dLdX"),           # wrong rank
        ((3, 5, 2), (3, 5), (5, 2), "dLdU"),           # wrong rank
        ((3, 5, 2), (3, 5, 4), (5, 2, 1), "dyn_res"),  # wrong rank
        ((3, 5, 2), (4, 5, 4), (5, 2), "share"),       # N mismatch
        ((3, 5, 2), (3, 6, 4), (5, 2), "share"),       # K mismatch between dLdX and dLdU
        ((3, 5, 2), (3, 5, 4), (6, 2), "dyn_res"),     # K mismatch with dyn
        ((3, 5, 2), (3, 5, 4), (5, 3), "dyn_res"),     # nx mismatch with dyn
    ],
)
def test_validate_al_residual_struct_rejects_bad_shapes(dLdX_shape, dLdU_shape, dyn_shape, err_match):
    r = altypes.ALResidualStruct(
        dLdX=jnp.zeros(dLdX_shape, dtype=jnp.float32),
        dLdU=jnp.zeros(dLdU_shape, dtype=jnp.float32),
        dyn_res=jnp.zeros(dyn_shape, dtype=jnp.float32),
    )
    with pytest.raises(ValueError, match=err_match):
        altypes.validate_al_residual_struct(r)

def test_alstate_post_init_rejects_non_jax_array_type():
    # Passing numpy arrays or python lists should fail with current strictness
    with pytest.raises(TypeError, match="must be a jax.Array"):
        altypes.JointAugmentedLagrangianState(
            lam_ineq=[0.0, 0.0],          # not a jax.Array
            rho_ineq=jnp.zeros((2,)),
            lam_eq=jnp.zeros((1,)),
            rho_eq=jnp.zeros((1,)),
        )

def test_alstate_post_init_rejects_non_1d_arrays():
    with pytest.raises(ValueError, match="must be 1D"):
        altypes.JointAugmentedLagrangianState(
            lam_ineq=jnp.zeros((2, 1)),
            rho_ineq=jnp.zeros((2,)),
            lam_eq=jnp.zeros((1,)),
            rho_eq=jnp.zeros((1,)),
        )

def test_alstate_post_init_rejects_shape_mismatch_ineq():
    with pytest.raises(ValueError, match="lam_ineq and rho_ineq must have same shape"):
        altypes.JointAugmentedLagrangianState(
            lam_ineq=jnp.zeros((2,)),
            rho_ineq=jnp.zeros((3,)),
            lam_eq=jnp.zeros((1,)),
            rho_eq=jnp.zeros((1,)),
        )

def test_alstate_post_init_rejects_shape_mismatch_eq():
    with pytest.raises(ValueError, match="lam_eq and rho_eq must have same shape"):
        altypes.JointAugmentedLagrangianState(
            lam_ineq=jnp.zeros((2,)),
            rho_ineq=jnp.zeros((2,)),
            lam_eq=jnp.zeros((1,)),
            rho_eq=jnp.zeros((2,)),
        )

def test_alstate_post_init_rejects_dtype_mismatch():
    with pytest.raises(TypeError, match="All AL arrays must share the same dtype"):
        altypes.JointAugmentedLagrangianState(
            lam_ineq=jnp.zeros((2,), dtype=jnp.float32),
            rho_ineq=jnp.zeros((2,), dtype=jnp.float32),
            lam_eq=jnp.zeros((1,), dtype=jnp.int32),     # mismatch that always exists
            rho_eq=jnp.zeros((1,), dtype=jnp.int32),
        )

def test_alstate_flax_struct_immutable_and_replace_works():
    st = altypes.init_joint_augmented_lagrangian_state(nc_ineq=3, nc_eq=2)
    with pytest.raises(Exception):
        st.lam_ineq = jnp.ones((3,))

    st2 = st.replace(lam_ineq=jnp.array([1.0, 2.0, 3.0], dtype=st.lam_ineq.dtype))
    assert jnp.all(st.lam_ineq == 0.0)
    assert jnp.all(st2.lam_ineq == jnp.array([1.0, 2.0, 3.0], dtype=st.lam_ineq.dtype))

def test_alstate_jit_sanity_replace():
    st = altypes.init_joint_augmented_lagrangian_state(nc_ineq=2, nc_eq=1)

    @jax.jit
    def f(s: altypes.JointAugmentedLagrangianState):
        return s.replace(lam_ineq=s.lam_ineq + 1.0)

    st2 = f(st)
    assert jnp.all(st2.lam_ineq == st.lam_ineq + 1.0)

def test_init_joint_augmented_lagrangian_state_shapes_values_dtype():
    st = altypes.init_joint_augmented_lagrangian_state(
        nc_ineq=5, nc_eq=3, lam0=0.0, rho0=2.5, dtype=jnp.float32
    )
    assert st.lam_ineq.shape == (5,)
    assert st.rho_ineq.shape == (5,)
    assert st.lam_eq.shape == (3,)
    assert st.rho_eq.shape == (3,)
    assert st.lam_ineq.dtype == jnp.float32
    assert jnp.all(st.lam_ineq == 0.0)
    assert jnp.all(st.rho_ineq == 2.5)
    assert st.nc_all == 8


def test_alstate_zero_dim_ok():
    st = altypes.init_joint_augmented_lagrangian_state(nc_ineq=4, nc_eq=0)
    assert st.lam_eq.shape == (0,)
    assert st.rho_eq.shape == (0,)
    assert st.nc_all == 4

    st2 = altypes.init_joint_augmented_lagrangian_state(nc_ineq=0, nc_eq=6)
    assert st2.lam_ineq.shape == (0,)
    assert st2.rho_ineq.shape == (0,)
    assert st2.nc_all == 6

def _make_op(nt=6, nx=3, nu=4, N=2, dtype=jnp.float32):
    tg = TimeGrid(nt=nt, dt=0.1, t0=0.0)

    # deterministic values for easy debugging
    xs = (jnp.arange(nt * nx, dtype=dtype).reshape(nt, nx) + 1) / 10.0
    us = (jnp.arange((nt - 1) * nu, dtype=dtype).reshape(nt - 1, nu) + 1) / 10.0
    ls = (jnp.arange((nt - 1) * N * nx, dtype=dtype).reshape(nt - 1, N, nx) + 1) / 10.0

    return FixedStepPrimalDualTrajectory(tg=tg, xs=xs, us=us, ls=ls)


def test_pack_decision_vars_length_formula():
    op = _make_op(nt=7, nx=2, nu=5, N=3)
    z = pdg_alsolver.pack_decision_vars_1d(op)

    nt, nx = op.xs.shape
    K = nt - 1
    nu = op.us.shape[1]
    N = op.ls.shape[1]
    expected = K * nx + K * nu + K * N * nx

    assert z.ndim == 1
    assert int(z.shape[0]) == expected


def test_pack_unpacked_roundtrip_preserves_x0_and_restores_rest():
    op = _make_op(nt=5, nx=3, nu=2, N=4)

    z = pdg_alsolver.pack_decision_vars_1d(op)
    op2 = pdg_alsolver.unpack_decision_vars(z, op)

    # x0 preserved exactly
    np.testing.assert_allclose(np.asarray(op2.xs[0]), np.asarray(op.xs[0]))

    # everything else matches exactly
    np.testing.assert_allclose(np.asarray(op2.xs[1:]), np.asarray(op.xs[1:]))
    np.testing.assert_allclose(np.asarray(op2.us), np.asarray(op.us))
    np.testing.assert_allclose(np.asarray(op2.ls), np.asarray(op.ls))

    # tg preserved
    assert op2.tg == op.tg


def test_pack_unpacked_roundtrip_changes_x0_if_template_changes():
    """
    Since x0 is not packed, unpacking takes x0 from the template.
    This test ensures that behavior is explicit/verified.
    """
    op = _make_op(nt=5, nx=2, nu=3, N=2)
    z = pdg_alsolver.pack_decision_vars_1d(op)

    # Make a different template with different x0 but identical shapes
    xs2 = op.xs.at[0].set(op.xs[0] + 100.0)
    template2 = FixedStepPrimalDualTrajectory(tg=op.tg, xs=xs2, us=op.us, ls=op.ls)

    op2 = pdg_alsolver.unpack_decision_vars(z, template2)

    np.testing.assert_allclose(np.asarray(op2.xs[0]), np.asarray(template2.xs[0]))
    np.testing.assert_allclose(np.asarray(op2.xs[1:]), np.asarray(op.xs[1:]))


def test_unpack_rejects_wrong_length():
    op = _make_op()
    z = pdg_alsolver.pack_decision_vars_1d(op)

    z_bad = z[:-1]
    with pytest.raises(ValueError, match="wrong length"):
        pdg_alsolver.unpack_decision_vars(z_bad, op)


def test_unpack_rejects_non_1d_z():
    op = _make_op()
    z = pdg_alsolver.pack_decision_vars_1d(op)

    z2 = z.reshape(1, -1)
    with pytest.raises(ValueError, match="must be 1D"):
        pdg_alsolver.unpack_decision_vars(z2, op)


def test_pack_rejects_bad_us_shape_when_checks_enabled():
    op = _make_op(nt=5, nx=2, nu=3, N=2)

    # Corrupt us to have wrong length
    bad_us = jnp.zeros((op.tg.nt, op.us.shape[1]), dtype=op.us.dtype)  # should be nsteps
    op_bad = SimpleNamespace(tg=op.tg, nsteps=op.nsteps, xs=op.xs, us=bad_us, ls=op.ls)

    with pytest.raises(ValueError, match="us first dim must be nsteps"):
        pdg_alsolver.pack_decision_vars_1d(op_bad, check_shapes=True)


def test_pack_rejects_bad_ls_shape_when_checks_enabled():
    op = _make_op(nt=5, nx=2, nu=3, N=2)

    # Corrupt ls: wrong last dim
    bad_ls = jnp.zeros((op.tg.nt - 1, op.ls.shape[1], op.nx + 1), dtype=op.ls.dtype)
    op_bad = SimpleNamespace(tg=op.tg, nsteps=op.nsteps, xs=op.xs, us=op.us, ls=bad_ls)

    with pytest.raises(ValueError, match="ls must have shape"):
        pdg_alsolver.pack_decision_vars_1d(op_bad, check_shapes=True)


def test_no_checks_variants_run_and_roundtrip_on_valid_input():
    """
    Sanity: no-check functions should work on correct shapes.
    """
    op = _make_op(nt=6, nx=3, nu=4, N=2)

    z = pdg_alsolver.pack_decision_vars_no_checks(op)
    op2 = pdg_alsolver.unpack_decision_vars_no_checks(z, op)

    np.testing.assert_allclose(np.asarray(op2.xs[0]), np.asarray(op.xs[0]))
    np.testing.assert_allclose(np.asarray(op2.xs[1:]), np.asarray(op.xs[1:]))
    np.testing.assert_allclose(np.asarray(op2.us), np.asarray(op.us))
    np.testing.assert_allclose(np.asarray(op2.ls), np.asarray(op.ls))

def test_pack_unpack_nt_equals_1_edge_case():
    nt, nx, nu, N = 1, 3, 2, 4
    tg = TimeGrid(nt=nt, dt=0.1, t0=0.0)

    xs = jnp.array([[1.0, 2.0, 3.0]], dtype=jnp.float32)          # (1, nx)
    us = jnp.zeros((0, nu), dtype=jnp.float32)                    # (0, nu)
    ls = jnp.zeros((0, N, nx), dtype=jnp.float32)                 # (0, N, nx)

    op = FixedStepPrimalDualTrajectory(tg=tg, xs=xs, us=us, ls=ls)

    z = pdg_alsolver.pack_decision_vars_1d(op)
    assert z.ndim == 1
    assert int(z.shape[0]) == 0

    op2 = pdg_alsolver.unpack_decision_vars(z, op)

    np.testing.assert_allclose(np.asarray(op2.xs), np.asarray(op.xs))
    assert op2.us.shape == (0, nu)
    assert op2.ls.shape == (0, N, nx)
    assert op2.tg == op.tg

def _make_primedual_op(nt, dt, nx, nu, N, player_i=0, mu_i=None):
    """
    Create a minimal prime-dual operating point with 
    ability to specify one players lagrange multipliers:
    - op.ls holds multipliers (nt-1, N, nx)
    - op.xs holds states (nt, nx) only used for nt sizing
    - op.us holds controls (nt-1, nu) not used here, but realistic
    """
    K = nt - 1
    tg = TimeGrid(nt=nt, dt=dt)
    if mu_i is None:
        mu_i = jnp.zeros((K, nx), dtype=jnp.float32)

    ls = jnp.zeros((K, N, nx), dtype=mu_i.dtype)
    ls = ls.at[:, player_i, :].set(mu_i)

    xs = jnp.zeros((nt, nx), dtype=mu_i.dtype)
    us = jnp.zeros((K, nu), dtype=mu_i.dtype)
    return FixedStepPrimalDualTrajectory(tg=tg, xs=xs, us=us, ls=ls)

def _make_primedual_op_v2(tg: TimeGrid, nx: int, nu: int, N: int, dtype=jnp.float32):
    nt = tg.nt
    xs = jnp.arange(nt * nx, dtype=dtype).reshape(nt, nx) / 10.0
    us = jnp.arange((nt - 1) * nu, dtype=dtype).reshape(nt - 1, nu) / 10.0
    ls = jnp.ones((nt-1, N, nx))
    return FixedStepPrimalDualTrajectory(tg=tg, xs=xs, us=us, ls=ls)


def test_grad_aug_lagrange_traj_dyn_term_shapes_state_and_control_1(monkeypatch):
    """
    Basic shape sanity check:
    - dL/dX should be (nt, nx)
    - dL/dU (joint) should be (nt-1, nu)
    """
    nt, dt, nx, nu, N = 6, 1.0, 3, 4, 2
    K = nt - 1

    A = jnp.zeros((K, nx, nx), dtype=jnp.float32)
    B = jnp.zeros((K, nx, nu), dtype=jnp.float32)

    def fake_jacobian(cs, op, method):
        return A, B

    # monkeypatch the underlysing jacobian_discrete_dynamics_trajectory function
    # since it is not the funtion under test, but the funciton under test
    # depends upon it
    monkeypatch.setattr(pdg_alsolver.systypes, "jacobian_discrete_dynamics_trajectory", fake_jacobian)

    cs = SimpleNamespace()  # unused by fake
    op = _make_primedual_op(nt, dt, nx, nu, N, player_i=0)

    dX, dU = pdg_alsolver.gradient_aug_lagrangian_playerwise_trajectory_dynamics(
        cs=cs, player_i=0, op=op, discretize_method="rk2"
    )

    assert dX.shape == (nt, nx)
    assert dU.shape == (K, nu)

def test_grad_aug_lagrange_traj_dyn_term_shapes_state_and_control_2():
    """
    Basic shape sanity check without monkey patching subroutines:
    - dL/dX should be (nt, nx)
    - dL/dU (joint) should be (nt-1, nu)
    """
    nt, dt, nx, nu, N = 16, 1e-3, 8, 4, 3
    tg = TimeGrid(nt=nt, dt=dt)

    # define control system
    cs = SampledContinuousSystemType1(
        tg=tg,
        nx=nx,
        nu=nu,
        dynamics=lambda t, x, u: x
    )

    # define operating point at which gradient computed
    op = FixedStepPrimalDualTrajectory(
        tg=tg,
        xs=jnp.zeros((nt, nx)),
        us=jnp.zeros((nt-1, nu)),
        ls=jnp.zeros((nt-1, N, nx)),
    )

    dX, dU = pdg_alsolver.gradient_aug_lagrangian_playerwise_trajectory_dynamics(
        cs=cs, player_i=0, op=op, discretize_method="rk2"
    )

    assert dX.shape == (nt, nx)
    assert dU.shape == (nt-1, nu)

def test_grad_aug_lagrange_traj_dyn_term_zero_when_multipliers_zero(monkeypatch):
    """
    If μ_k = 0 for all k, then:
      A_k^T μ_k = 0 and B_k^T μ_k = 0 and -μ_k = 0
    so both gradients are identically zero regardless of A,B.
    """
    nt, dt, nx, nu, N = 5, 0.1, 2, 3, 3
    K = nt - 1

    rng = np.random.default_rng(0)
    A = jnp.array(rng.standard_normal((K, nx, nx)), dtype=jnp.float32)
    B = jnp.array(rng.standard_normal((K, nx, nu)), dtype=jnp.float32)

    def fake_jacobian(cs, op, method):
        return A, B

    # monkeypatch the underlysing jacobian_discrete_dynamics_trajectory function
    # since it is not the funtion under test, but the funciton under test
    # depends upon it
    monkeypatch.setattr(pdg_alsolver.systypes, "jacobian_discrete_dynamics_trajectory", fake_jacobian)

    mu = jnp.zeros((K, nx), dtype=jnp.float32)
    cs = SimpleNamespace()
    op = _make_primedual_op(nt, dt, nx, nu, N, player_i=1, mu_i=mu)

    dX, dU = pdg_alsolver.gradient_aug_lagrangian_playerwise_trajectory_dynamics(
        cs=cs, player_i=1, op=op, discretize_method="rk2"
    )

    assert np.allclose(np.asarray(dX), 0.0)
    assert np.allclose(np.asarray(dU), 0.0)


def test_grad_aug_lagrange_traj_dyn_state_term_identity_A_expected_pattern(monkeypatch):
    """
    If A_k = I for all k, then term_x[k] = A_k^T μ_k = μ_k.

    Assembly rule:
      dX[0]      +=  μ_0
      dX[1]      +=  μ_1  - μ_0
      ...
      dX[K-1]    +=  μ_{K-1} - μ_{K-2}
      dX[K]      += -μ_{K-1}

    This is a great test because it checks the "shift-by-one" logic of D_k = f_d - x_{k+1}.
    """
    nt, dt, nx, nu, N = 5, 10.0, 3, 2, 2
    K = nt - 1

    A = jnp.tile(jnp.eye(nx, dtype=jnp.float32)[None, :, :], (K, 1, 1))
    B = jnp.zeros((K, nx, nu), dtype=jnp.float32)  # control irrelevant here

    def fake_jacobian(cs, op, method):
        return A, B

    # monkeypatch the underlysing jacobian_discrete_dynamics_trajectory function
    # since it is not the funtion under test, but the funciton under test
    # depends upon it
    monkeypatch.setattr(pdg_alsolver.systypes, "jacobian_discrete_dynamics_trajectory", fake_jacobian)

    mu = jnp.arange(K * nx, dtype=jnp.float32).reshape(K, nx)
    cs = SimpleNamespace()
    op = _make_primedual_op(nt, dt, nx, nu, N, player_i=0, mu_i=mu)

    dX, dU = pdg_alsolver.gradient_aug_lagrangian_playerwise_trajectory_dynamics(
        cs=cs, player_i=0, op=op, discretize_method="rk2"
    )

    dX_np = np.asarray(dX)
    mu_np = np.asarray(mu)

    assert np.allclose(dX_np[0], mu_np[0])
    for t in range(1, K):
        assert np.allclose(dX_np[t], mu_np[t] - mu_np[t - 1])
    assert np.allclose(dX_np[K], -mu_np[K - 1])

    # Since B=0, control gradient must be zero
    assert np.allclose(np.asarray(dU), 0.0)


def test_grad_aug_lagrange_traj_dyn_control_term_matches_manual_loop(monkeypatch):
    """
    Verify control contribution:
      dU[k] = B_k^T μ_k

    We'll randomize B and μ and compare against a manual NumPy loop.
    """
    nt, dt, nx, nu, N = 6, 0.2, 2, 4, 3
    K = nt - 1
    rng = np.random.default_rng(123)

    A = jnp.zeros((K, nx, nx), dtype=jnp.float32)  # state irrelevant for this test
    B = jnp.array(rng.standard_normal((K, nx, nu)), dtype=jnp.float32)

    def fake_jacobian(cs, op, method):
        return A, B

    monkeypatch.setattr(pdg_alsolver.systypes, "jacobian_discrete_dynamics_trajectory", fake_jacobian)

    mu = jnp.array(rng.standard_normal((K, nx)), dtype=jnp.float32)
    cs = SimpleNamespace()
    op = _make_primedual_op(nt, dt, nx, nu, N, player_i=2, mu_i=mu)

    dX, dU = pdg_alsolver.gradient_aug_lagrangian_playerwise_trajectory_dynamics(
        cs=cs, player_i=2, op=op, discretize_method="rk2"
    )

    # Manual reference for dU
    B_np = np.asarray(B)
    mu_np = np.asarray(mu)
    refU = np.zeros((K, nu), dtype=np.float32)
    for k in range(K):
        refU[k] = B_np[k].T @ mu_np[k]

    assert np.allclose(np.asarray(dU), refU, atol=1e-6, rtol=1e-6)


def _make_alstate_v1(lam_ineq, lam_eq, dtype=jnp.float32):
    # rho is unused by this function, but your dataclass requires it
    lam_ineq = jnp.asarray(lam_ineq, dtype=dtype)
    lam_eq   = jnp.asarray(lam_eq, dtype=dtype)
    rho_ineq = jnp.ones_like(lam_ineq)
    rho_eq   = jnp.ones_like(lam_eq)
    return altypes.JointAugmentedLagrangianState(
        lam_ineq=lam_ineq, rho_ineq=rho_ineq,
        lam_eq=lam_eq, rho_eq=rho_eq,
    )


def test_grad_aug_lag_traj_con_empty_constraints_returns_zeros():
    tg = TimeGrid(nt=5, dt=0.1, t0=0.0)
    N = 1
    op = _make_primedual_op_v2(tg, nx=3, nu=2, N=N)

    constraints = GameConstraintGridMap(ineq_blocks=(), eq_blocks=())
    alstate = _make_alstate_v1(lam_ineq=[], lam_eq=[])

    dX, dU = pdg_alsolver.gradient_aug_lagrangian_trajectory_constraints(constraints, alstate, op)

    assert dX.shape == (tg.nt, 3)
    assert dU.shape == (tg.nt - 1, 2)
    assert np.allclose(np.array(dX), 0.0)
    assert np.allclose(np.array(dU), 0.0)


def test_grad_aug_lag_traj_con_single_scalar_block_affine_adds_at_active_steps():
    tg = TimeGrid(nt=6, dt=0.2, t0=0.0)
    N, nx, nu = 3, 3, 4
    op = _make_primedual_op_v2(tg=tg, nx=nx, nu=nu, N=N)

    a = jnp.array([1.0, -2.0, 0.5], dtype=jnp.float32)
    b = jnp.array([0.25, -1.5, 2.0, 0.0], dtype=jnp.float32)

    def c_step(t, x, u):
        return jnp.dot(a, x) + jnp.dot(b, u)  # scalar, cdim_out_step=1

    block = ConstraintBlockGridMap(
        tg=tg,
        func=c_step,
        cdim_out_step=1,
        active_steps=(1, 3),  # only steps 1 and 3
        iseq=False,
        terminal=False,
    )
    constraints = GameConstraintGridMap(ineq_blocks=(block,), eq_blocks=())

    # two active steps * cdim=1 => nc_ineq = 2
    lam = jnp.array([2.0, -1.0], dtype=jnp.float32)
    alstate = _make_alstate_v1(lam_ineq=lam, lam_eq=[])

    dX, dU = pdg_alsolver.gradient_aug_lagrangian_trajectory_constraints(constraints, alstate, op)

    # Expect dX[k] += a * lam_step, dU[k] += b * lam_step at k=1,3
    expected_dX = np.zeros((tg.nt, 3), dtype=np.float32)
    expected_dU = np.zeros((tg.nt - 1, 4), dtype=np.float32)

    expected_dX[1, :] += np.array(a) * 2.0
    expected_dU[1, :] += np.array(b) * 2.0

    expected_dX[3, :] += np.array(a) * (-1.0)
    expected_dU[3, :] += np.array(b) * (-1.0)

    assert np.allclose(np.array(dX), expected_dX, atol=1e-6, rtol=1e-6)
    assert np.allclose(np.array(dU), expected_dU, atol=1e-6, rtol=1e-6)


def test_grad_aug_lag_traj_con_vector_valued_block_slices_lambda_correctly():
    tg = TimeGrid(nt=5, dt=0.1, t0=0.0)
    N, nx, nu = 2, 2, 2
    op = _make_primedual_op_v2(tg=tg, nx=nx, nu=nu, N=N)

    # cdim=2 output
    def c_step(t, x, u):
        return jnp.array([x[0] + u[0], 2.0 * x[1] - u[1]])

    block = ConstraintBlockGridMap(
        tg=tg,
        func=c_step,
        cdim_out_step=2,
        active_steps=(0, 2),
        iseq=False,
        terminal=False,
    )
    constraints = GameConstraintGridMap(ineq_blocks=(block,), eq_blocks=())

    # nc_ineq = 2 active steps * cdim 2 = 4
    lam = jnp.array([1.0, 10.0,   -2.0, 3.0], dtype=jnp.float32)
    alstate = _make_alstate_v1(lam_ineq=lam, lam_eq=[])

    dX, dU = pdg_alsolver.gradient_aug_lagrangian_trajectory_constraints(constraints, alstate, op)

    # At k=0: lam_step=[1,10]
    # Jx = [[1,0],[0,2]] ; Ju=[[1,0],[0,-1]]
    expected0_dx = np.array([1.0, 2.0 * 10.0], dtype=np.float32)
    expected0_du = np.array([1.0, -10.0], dtype=np.float32)

    # At k=2: lam_step=[-2,3]
    expected2_dx = np.array([-2.0, 2.0 * 3.0], dtype=np.float32)
    expected2_du = np.array([-2.0, -3.0], dtype=np.float32)

    assert np.allclose(np.array(dX[0]), expected0_dx)
    assert np.allclose(np.array(dU[0]), expected0_du)

    assert np.allclose(np.array(dX[2]), expected2_dx)
    assert np.allclose(np.array(dU[2]), expected2_du)


def test_grad_aug_lag_traj_con_terminal_block_only_hits_final_state():
    tg = TimeGrid(nt=6, dt=0.1, t0=0.0)
    N, nx, nu = 2, 3, 2
    op = _make_primedual_op_v2(tg=tg, nx=nx, nu=nu, N=N)

    # terminal scalar constraint: c(t,x) = x0^2
    def c_term(t, x):
        return x[0] ** 2

    block = ConstraintBlockGridMap(
        tg=tg,
        func=c_term,
        cdim_out_step=1,
        active_steps=None,  # should default to (nt-1,)
        iseq=True,
        terminal=True,
    )
    constraints = GameConstraintGridMap(ineq_blocks=(), eq_blocks=(block,))

    lam_eq = jnp.array([3.0], dtype=jnp.float32)
    alstate = _make_alstate_v1(lam_ineq=[], lam_eq=lam_eq)

    dX, dU = pdg_alsolver.gradient_aug_lagrangian_trajectory_constraints(constraints, alstate, op)

    # d/dx0 of x0^2 is 2*x0, multiplied by lambda
    x_final = np.array(op.xs[-1])
    expected = np.zeros((tg.nt, 3), dtype=np.float32)
    expected[-1, 0] = 3.0 * (2.0 * x_final[0])

    assert np.allclose(np.array(dX), expected, atol=1e-6, rtol=1e-6)
    assert np.allclose(np.array(dU), 0.0)


def test_grad_aug_lag_traj_con_raises_on_lambda_shape_mismatch():
    tg = TimeGrid(nt=5, dt=0.1, t0=0.0)
    N, nx, nu = 2, 2, 2
    op = _make_primedual_op_v2(tg=tg, nx=nx, nu=nu, N=N)

    def c_step(t, x, u):
        return x[0]

    block = ConstraintBlockGridMap(
        tg=tg, func=c_step, cdim_out_step=1, active_steps=(0, 1), iseq=False, terminal=False
    )
    constraints = GameConstraintGridMap(ineq_blocks=(block,), eq_blocks=())

    # nc_ineq should be 2, but give length 1
    alstate = _make_alstate_v1(lam_ineq=[1.0], lam_eq=[])

    with pytest.raises(ValueError, match="lam_ineq must have shape"):
        pdg_alsolver.gradient_aug_lagrangian_trajectory_constraints(constraints, alstate, op)


def test_grad_aug_lag_traj_con_raises_on_timegrid_mismatch():
    tg_op = TimeGrid(nt=5, dt=0.1, t0=0.0)
    tg_c  = TimeGrid(nt=5, dt=0.1, t0=0.5)

    N, nx, nu = 1, 2, 2
    op = _make_primedual_op_v2(tg=tg_op, nx=nx, nu=nu, N=N)

    def c_step(t, x, u):
        return x[0]

    block = ConstraintBlockGridMap(
        tg=tg_c, func=c_step, cdim_out_step=1, active_steps=(0,), iseq=False, terminal=False
    )
    constraints = GameConstraintGridMap(ineq_blocks=(block,), eq_blocks=())
    alstate = _make_alstate_v1(lam_ineq=[1.0], lam_eq=[])

    with pytest.raises(ValueError, match="TimeGrid"):
        pdg_alsolver.gradient_aug_lagrangian_trajectory_constraints(constraints, alstate, op)


def test_grad_aug_lag_traj_con_raises_if_nonterminal_block_uses_final_step():
    tg = TimeGrid(nt=5, dt=0.1, t0=0.0)

    def c_step(t, x, u):
        return x[0]

    # illegal: active at terminal node nt-1 but terminal=False
    with pytest.raises(ValueError, match="stage grid"):
        ConstraintBlockGridMap(
            tg=tg,
            func=c_step,
            cdim_out_step=1,
            active_steps=(tg.nt - 1,),
            iseq=False,
            terminal=False,
        )


def _as_1d(c_val: jnp.ndarray, expected: int) -> jnp.ndarray:
    c_val = jnp.asarray(c_val)
    if c_val.ndim == 0:
        c_val = c_val[None]
    if c_val.ndim != 1:
        raise ValueError(f"Constraint kernel must return scalar or 1D array, got {c_val.shape}")
    if int(c_val.shape[0]) != expected:
        raise ValueError(f"Constraint output dim mismatch: expected {expected}, got {c_val.shape[0]}")
    return c_val


def _L_C_scalar(constraints, # GameConstraintGridMap
                alstate,  # JointAugmentedLagrangianState
                tg: TimeGrid,
                xs: jnp.ndarray,
                us: jnp.ndarray) -> jnp.ndarray:
    """
    Scalar reference implementation of L_C = lam^T C(X,U) with the same stacking order
    as gradient_aug_lagrangian_playerwise_trajectory_constraints.
    """
    ts = compute_ts(tg)

    # Inequalities
    total = jnp.array(0.0, dtype=xs.dtype)
    lam_ptr = 0
    for b in constraints.ineq_blocks:
        lam_block = alstate.lam_ineq[lam_ptr: lam_ptr + b.nc_block]
        lam_ptr += b.nc_block

        for j_step, k in enumerate(b.active_steps):
            lam_step = lam_block[j_step * b.cdim_out_step : (j_step + 1) * b.cdim_out_step]
            if b.terminal:
                c = _as_1d(b.func(ts[k], xs[k]), b.cdim_out_step)
            else:
                if k == tg.nt - 1:
                    raise ValueError("Non-terminal block active at nt-1 in reference; should match main code.")
                c = _as_1d(b.func(ts[k], xs[k], us[k]), b.cdim_out_step)
            total = total + jnp.dot(lam_step, c)

    # Equalities
    lam_ptr = 0
    for b in constraints.eq_blocks:
        lam_block = alstate.lam_eq[lam_ptr: lam_ptr + b.nc_block]
        lam_ptr += b.nc_block

        for j_step, k in enumerate(b.active_steps):
            lam_step = lam_block[j_step * b.cdim_out_step : (j_step + 1) * b.cdim_out_step]
            if b.terminal:
                c = _as_1d(b.func(ts[k], xs[k]), b.cdim_out_step)
            else:
                if k == tg.nt - 1:
                    raise ValueError("Non-terminal block active at nt-1 in reference; should match main code.")
                c = _as_1d(b.func(ts[k], xs[k], us[k]), b.cdim_out_step)
            total = total + jnp.dot(lam_step, c)

    return total


def test_grad_aug_lag_traj_con_golden_matches_autodiff_mixed_blocks():
    tg = TimeGrid(nt=6, dt=0.2, t0=0.0)
    nt, nx, nu = tg.nt, 3, 2
    K = nt - 1

    # Use nontrivial values so gradients aren't accidentally zero
    xs0 = jnp.linspace(0.1, 1.0, nt * nx, dtype=jnp.float32).reshape(nt, nx)
    us0 = jnp.linspace(-0.3, 0.7, K * nu, dtype=jnp.float32).reshape(K, nu)

    # ineq block: vector-valued (cdim=2), active at steps 0 and 2
    def c_ineq_vec(t, x, u):
        return jnp.array([
            x[0] + 2.0 * u[0] + 0.1 * t,
            x[1] - u[1] + 0.2 * t,
        ])

    b1 = ConstraintBlockGridMap(
        tg=tg, func=c_ineq_vec, cdim_out_step=2, active_steps=(0, 2),
        iseq=False, terminal=False
    )

    # eq block: scalar (cdim=1), active at step 1
    def c_eq_scalar(t, x, u):
        return (x[2] * u[0]) + 0.3 * t

    b2 = ConstraintBlockGridMap(
        tg=tg, func=c_eq_scalar, cdim_out_step=1, active_steps=(1,),
        iseq=True, terminal=False
    )

    # terminal eq block: scalar (cdim=1), enforced at nt-1
    def c_term(t, x):
        return x[0] ** 2 + 0.5 * t

    b3 = ConstraintBlockGridMap(
        tg=tg, func=c_term, cdim_out_step=1, active_steps=None,
        iseq=True, terminal=True
    )

    constraints = GameConstraintGridMap(ineq_blocks=(b1,), eq_blocks=(b2, b3))

    # Dimensions:
    # nc_ineq = 2 active steps * cdim 2 = 4
    # nc_eq = (1 step * 1) + (terminal 1 step * 1) = 2
    alstate = _make_alstate_v1(
        lam_ineq=jnp.array([1.0, -2.0, 0.5, 3.0], dtype=jnp.float32),
        lam_eq=jnp.array([-1.5, 2.0], dtype=jnp.float32),
    )

    op = FixedStepPrimalDualTrajectory(tg=tg, xs=xs0, us=us0, ls=jnp.ones((K,2,nx)))
    dX_impl, dU_impl = pdg_alsolver.gradient_aug_lagrangian_trajectory_constraints(constraints, alstate, op)

    # Autodiff reference
    def L_of_xu(xs, us):
        return _L_C_scalar(constraints, alstate, tg, xs, us)

    dX_ref, dU_ref = jax.grad(L_of_xu, argnums=(0, 1))(xs0, us0)

    assert dX_impl.shape == dX_ref.shape == (nt, nx)
    assert dU_impl.shape == dU_ref.shape == (K, nu)

    np.testing.assert_allclose(np.array(dX_impl), np.array(dX_ref), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(np.array(dU_impl), np.array(dU_ref), atol=1e-5, rtol=1e-5)

def test_grad_aug_lag_traj_con_time_dependence_uses_ts():
    tg = TimeGrid(nt=4, dt=0.5, t0=1.0)  # ts = [1.0,1.5,2.0,2.5]
    nt, nx, nu = tg.nt, 2, 1
    K = nt - 1

    xs = jnp.ones((nt, nx), dtype=jnp.float32)
    us = jnp.zeros((K, nu), dtype=jnp.float32)

    # c(t,x,u) = t * x0 -> dc/dx = [t, 0]
    def c_step(t, x, u):
        return t * x[0]

    b = ConstraintBlockGridMap(
        tg=tg, func=c_step, cdim_out_step=1, active_steps=(0, 1), iseq=False, terminal=False
    )
    constraints = GameConstraintGridMap(ineq_blocks=(b,), eq_blocks=())

    lam = jnp.array([1.0, 1.0], dtype=jnp.float32)
    alstate = _make_alstate_v1(lam_ineq=lam, lam_eq=[])

    op = FixedStepPrimalDualTrajectory(tg=tg, xs=xs, us=us, ls=jnp.ones((K,1,nx)))
    dX, dU = pdg_alsolver.gradient_aug_lagrangian_trajectory_constraints(constraints, alstate, op)

    ts = np.array(compute_ts(tg))
    expected_dX = np.zeros((nt, nx), dtype=np.float32)
    expected_dX[0, 0] = ts[0]
    expected_dX[1, 0] = ts[1]

    np.testing.assert_allclose(np.array(dX), expected_dX, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(np.array(dU), 0.0)

def test_grad_aug_lag_traj_con_mixed_ineq_eq_independent():
    tg = TimeGrid(nt=5, dt=0.1, t0=0.0)
    nt, nx, nu = tg.nt, 2, 2
    K = nt - 1

    xs = jnp.zeros((nt, nx), dtype=jnp.float32)
    us = jnp.zeros((K, nu), dtype=jnp.float32)

    def cI(t, x, u):  # affects x0 only
        return x[0]

    def cE(t, x, u):  # affects u1 only
        return u[1]

    bI = ConstraintBlockGridMap(tg=tg, func=cI, cdim_out_step=1, active_steps=(0,), iseq=False, terminal=False)
    bE = ConstraintBlockGridMap(tg=tg, func=cE, cdim_out_step=1, active_steps=(0,), iseq=True,  terminal=False)

    constraints = GameConstraintGridMap(ineq_blocks=(bI,), eq_blocks=(bE,))
    alstate = _make_alstate_v1(lam_ineq=[2.0], lam_eq=[3.0])

    op = FixedStepPrimalDualTrajectory(tg=tg, xs=xs, us=us, ls=jnp.ones((K,2,nx)))
    dX, dU = pdg_alsolver.gradient_aug_lagrangian_trajectory_constraints(constraints, alstate, op)

    expected_dX = np.zeros((nt, nx), dtype=np.float32)
    expected_dU = np.zeros((K, nu), dtype=np.float32)

    expected_dX[0, 0] = 2.0  # from ineq
    expected_dU[0, 1] = 3.0  # from eq

    np.testing.assert_allclose(np.array(dX), expected_dX, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(np.array(dU), expected_dU, atol=1e-6, rtol=1e-6)

def _make_alstate_from_lists(lam_ineq, rho_ineq, lam_eq, rho_eq, dtype=jnp.float32):
    lam_ineq = jnp.asarray(lam_ineq, dtype=dtype)
    rho_ineq = jnp.asarray(rho_ineq, dtype=dtype)
    lam_eq   = jnp.asarray(lam_eq, dtype=dtype)
    rho_eq   = jnp.asarray(rho_eq, dtype=dtype)
    return altypes.JointAugmentedLagrangianState(
        lam_ineq=lam_ineq, rho_ineq=rho_ineq,
        lam_eq=lam_eq, rho_eq=rho_eq,
    )

def test_grad_aug_lag_traj_pen_empty_constraints_returns_zeros():
    nt, dt, nx, nu, N = 5, 0.1, 3, 2, 1
    tg = TimeGrid(nt=nt, dt=dt, t0=0.0)
    op = _make_primedual_op_v2(tg=tg, nx=nx, nu=nu, N=N)

    constraints = GameConstraintGridMap(ineq_blocks=(), eq_blocks=())
    alstate = altypes.JointAugmentedLagrangianState(
        lam_ineq=jnp.array([], dtype=jnp.float32), 
        rho_ineq=jnp.array([], dtype=jnp.float32), 
        lam_eq=jnp.array([], dtype=jnp.float32), 
        rho_eq=jnp.array([], dtype=jnp.float32))

    dX, dU = pdg_alsolver.gradient_aug_lagrangian_trajectory_penalty(constraints, alstate, op, ineq_activation="none")

    assert dX.shape == (tg.nt, 3)
    assert dU.shape == (tg.nt - 1, 2)
    assert np.allclose(np.asarray(dX), 0.0)
    assert np.allclose(np.asarray(dU), 0.0)

def test_grad_aug_lag_traj_pen_single_affine_scalar_block_matches_closed_form_none_activation():
    """
    Constraint: c = a^T x + b^T u (scalar), active at k in {1,3}
    Penalty:    1/2 * rho * c^2
    Gradient:   d/dx = rho * c * a,   d/du = rho * c * b  (at active steps)
    """
    nt, dt, nx, nu, N = 6, 0.2, 3, 4, 3
    tg = TimeGrid(nt=nt, dt=dt, t0=0.0)
    op = _make_primedual_op_v2(tg=tg, nx=nx, nu=nu, N=N)
    xs, us = op.xs, op.us

    a = jnp.array([1.0, -2.0, 0.5], dtype=jnp.float32)
    b = jnp.array([0.25, -1.5, 2.0, 0.0], dtype=jnp.float32)

    def c_step(t, x, u):
        return jnp.dot(a, x) + jnp.dot(b, u)  # scalar

    block = ConstraintBlockGridMap(
        tg=tg, func=c_step, cdim_out_step=1, active_steps=(1, 3), iseq=False, terminal=False
    )
    constraints = GameConstraintGridMap(ineq_blocks=(block,), eq_blocks=())

    # nc_ineq = 2 steps * 1 dim = 2
    rho = jnp.array([2.0, 5.0], dtype=jnp.float32)
    alstate = _make_alstate_from_lists(lam_ineq=[0.0, 0.0], rho_ineq=rho, lam_eq=[], rho_eq=[])

    dX, dU = pdg_alsolver.gradient_aug_lagrangian_trajectory_penalty(constraints, alstate, op, ineq_activation="none")

    expected_dX = np.zeros((tg.nt, op.nx), dtype=np.float32)
    expected_dU = np.zeros((tg.nt - 1, op.nu), dtype=np.float32)

    # step 1
    c1 = float(jnp.dot(a, xs[1]) + jnp.dot(b, us[1]))
    expected_dX[1] += np.array(a) * float(rho[0]) * c1
    expected_dU[1] += np.array(b) * float(rho[0]) * c1

    # step 3
    c3 = float(jnp.dot(a, xs[3]) + jnp.dot(b, us[3]))
    expected_dX[3] += np.array(a) * float(rho[1]) * c3
    expected_dU[3] += np.array(b) * float(rho[1]) * c3

    np.testing.assert_allclose(np.asarray(dX), expected_dX, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(dU), expected_dU, atol=1e-5, rtol=1e-5)

def test_grad_aug_lag_traj_pen_terminal_eq_block_only_hits_final_state():
    """
    Terminal constraint: c = x0^2 (scalar), penalty 1/2*rho*c^2
    d/dx0 = rho * c * (dc/dx0) = rho * (x0^2) * (2*x0) = 2*rho*x0^3
    """
    nt, dt, nx, nu, N = 6, 0.1, 3, 2, 2
    tg = TimeGrid(nt=nt, dt=dt, t0=0.0)
    op = _make_primedual_op_v2(tg=tg, nx=nx, nu=nu, N=N)
    xs = op.xs

    def c_term(t, x):
        return x[0] ** 2

    block = ConstraintBlockGridMap(
        tg=tg, func=c_term, cdim_out_step=1, active_steps=None, iseq=True, terminal=True
    )
    constraints = GameConstraintGridMap(ineq_blocks=(), eq_blocks=(block,))

    # nc_eq = 1
    rho_eq = jnp.array([4.0], dtype=jnp.float32)
    alstate = _make_alstate_from_lists(lam_ineq=[], rho_ineq=[], lam_eq=[0.0], rho_eq=rho_eq)

    dX, dU = pdg_alsolver.gradient_aug_lagrangian_trajectory_penalty(constraints, alstate, op, ineq_activation="none")

    expected_dX = np.zeros((tg.nt, op.nx), dtype=np.float32)
    expected_dU = np.zeros((tg.nt - 1, op.nu), dtype=np.float32)

    x0 = float(xs[-1, 0])
    expected_dX[-1, 0] = 2.0 * float(rho_eq[0]) * (x0 ** 3)

    np.testing.assert_allclose(np.asarray(dX), expected_dX, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(dU), expected_dU, atol=1e-5, rtol=1e-5)

def test_grad_aug_lag_traj_pen_raises_on_rho_shape_mismatch():
    nt, dt, nx, nu, N = 5, 0.1, 2, 2, 1
    tg = TimeGrid(nt=nt, dt=dt, t0=0.0)
    op = _make_primedual_op_v2(tg=tg, nx=nx, nu=nu, N=N)

    def c_step(t, x, u):
        return x[0]

    block = ConstraintBlockGridMap(
        tg=tg, func=c_step, cdim_out_step=1, active_steps=(0, 1), iseq=False, terminal=False
    )
    constraints = GameConstraintGridMap(ineq_blocks=(block,), eq_blocks=())

    # nc_ineq = 2, but provide rho_ineq length 1
    alstate = _make_alstate_from_lists(lam_ineq=[0.0], rho_ineq=[1.0], lam_eq=[], rho_eq=[])

    with pytest.raises(ValueError, match="rho_ineq"):
        pdg_alsolver.gradient_aug_lagrangian_trajectory_penalty(constraints, alstate, op, ineq_activation="none")

def test_grad_aug_lag_traj_pen_altro_activation_zeroes_inactive_constraints_when_c_negative_and_lambda_nonpositive():
    """
    With altro activation: a = (c >= 0) OR (lambda > 0)
    If c < 0 and lambda <= 0 => a=0 => c_eff=0 => no penalty gradient contribution.
    """
    nt, dt, nx, nu, N = 4, 0.1, 2, 1, 1
    tg = TimeGrid(nt=nt, dt=dt, t0=0.0)
    op = _make_primedual_op_v2(tg=tg, nx=nx, nu=nu, N=N)

    # c = x0 - 100 => definitely negative for our xs in [0.1..1.0]
    def c_step(t, x, u):
        return x[0] - 100.0

    block = ConstraintBlockGridMap(
        tg=tg, func=c_step, cdim_out_step=1, active_steps=(0, 1, 2), iseq=False, terminal=False
    )
    constraints = GameConstraintGridMap(ineq_blocks=(block,), eq_blocks=())

    # nc_ineq = 3
    lam = jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32)  # not activating
    rho = jnp.array([5.0, 5.0, 5.0], dtype=jnp.float32)
    alstate = _make_alstate_from_lists(lam_ineq=lam, rho_ineq=rho, lam_eq=[], rho_eq=[])

    dX, dU = pdg_alsolver.gradient_aug_lagrangian_trajectory_penalty(constraints, alstate, op, ineq_activation="altro")

    assert np.allclose(np.asarray(dX), 0.0)
    assert np.allclose(np.asarray(dU), 0.0)

def _make_dummy_game_and_op():
    tg = TimeGrid(nt=5, dt=0.1, t0=0.0)
    nt = tg.nt
    nx = 3
    nu = 6
    N = 2
    u_splits = jnp.array([2, 4], dtype=jnp.int32)

    # minimal op fixture
    xs = jnp.zeros((nt, nx), dtype=jnp.float32)
    us = jnp.zeros((nt - 1, nu), dtype=jnp.float32)
    ls = jnp.zeros((nt - 1, N, nx), dtype=jnp.float32)
    op = SimpleNamespace(tg=tg, xs=xs, us=us, ls=ls, nt=nt, nx=nx, nu=nu)

    # minimal nlgame fixture
    # costs is indexed but not called directly (we patch the cost gradient function)
    dummy_cost = SimpleNamespace(running=lambda t, x, u: 0.0, terminal=lambda t, x: 0.0)
    nlgame = SimpleNamespace(
        tg=tg, nt=nt, nx=nx, nu=nu, N=N,
        u_splits=u_splits,
        cs=SimpleNamespace(tg=tg, nx=nx, nu=nu, nt=nt),  # placeholder
        constraints=SimpleNamespace(tg=tg, nc_ineq=0, nc_eq=0, nc_all=0),
        costs=[dummy_cost for _ in range(N)],
    )

    # AL state placeholder (shapes irrelevant since we patch constraint gradients)
    alstate = SimpleNamespace(
        lam_ineq=jnp.zeros((0,), dtype=jnp.float32),
        rho_ineq=jnp.zeros((0,), dtype=jnp.float32),
        lam_eq=jnp.zeros((0,), dtype=jnp.float32),
        rho_eq=jnp.zeros((0,), dtype=jnp.float32),
    )

    return nlgame, op, alstate

def test_gradient_aug_lagrangian_trajectory_shapes_and_shared_broadcast(monkeypatch):
    nlgame, op, alstate = _make_dummy_game_and_op()
    N, nt, nx, nu = nlgame.N, nlgame.nt, nlgame.nx, nlgame.nu
    K = nt - 1

    # Shared constraint gradients (linear + penalty) -> total shared = ones
    shared_X = jnp.ones((nt, nx), dtype=jnp.float32)
    shared_U = jnp.ones((K, nu), dtype=jnp.float32) * 2.0

    monkeypatch.setattr(pdg_alsolver.contypes, "build_constraint_step_linearizations",
                        lambda constraints, op: (tuple(), tuple()))
    monkeypatch.setattr(pdg_alsolver, "_gradient_aug_lagrangian_trajectory_constraints_from_linearizations",
                        lambda **kwargs: (shared_X, shared_U))
    monkeypatch.setattr(pdg_alsolver, "_gradient_aug_lagrangian_trajectory_penalty_from_linearizations",
                        lambda **kwargs: (shared_X, shared_U))

    # Player-specific cost gradients: zero
    monkeypatch.setattr(pdg_alsolver.costtypes, "gradient_cost_local_ctrl_playerwise_trajectory",
                        lambda **kwargs: (jnp.zeros((nt, nx), jnp.float32),
                                          jnp.zeros((K, int(nlgame.u_splits[int(kwargs["player_i"])])), jnp.float32)))

    # Player-specific dynamics gradients: zero
    monkeypatch.setattr(pdg_alsolver, "gradient_aug_lagrangian_playerwise_trajectory_dynamics",
                        lambda **kwargs: (jnp.zeros((nt, nx), jnp.float32),
                                          jnp.zeros((K, nu), jnp.float32)))

    dX_all, dU_all = pdg_alsolver._gradient_aug_lagrangian_trajectory(nlgame, op, alstate, 
        discretize_method = "rk2",
        ineq_activation = "altro",
    )

    assert dX_all.shape == (N, nt, nx)
    assert dU_all.shape == (N, K, nu)

    # shared total = shared_X + shared_X, shared_U + shared_U
    expected_X = np.array(2.0 * shared_X)
    expected_U = np.array(2.0 * shared_U)

    for i in range(N):
        np.testing.assert_allclose(np.array(dX_all[i]), expected_X)
        np.testing.assert_allclose(np.array(dU_all[i]), expected_U)


def test_gradient_aug_lagrangian_trajectory_reuses_constraint_linearizations_for_linear_and_penalty(monkeypatch):
    nlgame, op, alstate = _make_dummy_game_and_op()
    N, nt, nx, nu = nlgame.N, nlgame.nt, nlgame.nx, nlgame.nu
    K = nt - 1

    calls = {"count": 0}

    def fake_build_constraint_step_linearizations(constraints, op):
        calls["count"] += 1
        return tuple(), tuple()

    monkeypatch.setattr(
        pdg_alsolver.contypes,
        "build_constraint_step_linearizations",
        fake_build_constraint_step_linearizations,
    )
    monkeypatch.setattr(
        pdg_alsolver,
        "_gradient_aug_lagrangian_trajectory_constraints_from_linearizations",
        lambda **kwargs: (
            jnp.ones((nt, nx), dtype=jnp.float32),
            jnp.ones((K, nu), dtype=jnp.float32),
        ),
    )
    monkeypatch.setattr(
        pdg_alsolver,
        "_gradient_aug_lagrangian_trajectory_penalty_from_linearizations",
        lambda **kwargs: (
            jnp.ones((nt, nx), dtype=jnp.float32),
            jnp.ones((K, nu), dtype=jnp.float32),
        ),
    )
    monkeypatch.setattr(
        pdg_alsolver.costtypes,
        "gradient_cost_local_ctrl_playerwise_trajectory",
        lambda **kwargs: (
            jnp.zeros((nt, nx), jnp.float32),
            jnp.zeros((K, int(nlgame.u_splits[int(kwargs["player_i"])])), jnp.float32),
        ),
    )
    monkeypatch.setattr(
        pdg_alsolver,
        "gradient_aug_lagrangian_playerwise_trajectory_dynamics",
        lambda **kwargs: (
            jnp.zeros((nt, nx), jnp.float32),
            jnp.zeros((K, nu), jnp.float32),
        ),
    )

    dX_all, dU_all = pdg_alsolver._gradient_aug_lagrangian_trajectory(
        nlgame,
        op,
        alstate,
        discretize_method="rk2",
        ineq_activation="altro",
    )

    assert calls["count"] == 1
    assert dX_all.shape == (N, nt, nx)
    assert dU_all.shape == (N, K, nu)


def test_gradient_aug_lagrangian_trajectory_inserts_local_control_slices(monkeypatch):
    nlgame, op, alstate = _make_dummy_game_and_op()
    N, nt, nx, nu = nlgame.N, nlgame.nt, nlgame.nx, nlgame.nu
    K = nt - 1
    u_splits = np.array(nlgame.u_splits)

    # No shared constraints / no dynamics
    monkeypatch.setattr(pdg_alsolver.contypes, "build_constraint_step_linearizations",
                        lambda constraints, op: (tuple(), tuple()))
    monkeypatch.setattr(pdg_alsolver, "_gradient_aug_lagrangian_trajectory_constraints_from_linearizations",
                        lambda **kwargs: (jnp.zeros((nt, nx), jnp.float32),
                                          jnp.zeros((K, nu), jnp.float32)))
    monkeypatch.setattr(pdg_alsolver, "_gradient_aug_lagrangian_trajectory_penalty_from_linearizations",
                        lambda **kwargs: (jnp.zeros((nt, nx), jnp.float32),
                                          jnp.zeros((K, nu), jnp.float32)))
    monkeypatch.setattr(pdg_alsolver, "gradient_aug_lagrangian_playerwise_trajectory_dynamics",
                        lambda **kwargs: (jnp.zeros((nt, nx), jnp.float32),
                                          jnp.zeros((K, nu), jnp.float32)))

    # Cost gradients: state zero, local control = constant different per player
    def fake_cost_grad(**kwargs):
        i = int(kwargs["player_i"])
        nu_i = int(u_splits[i])
        dX = jnp.zeros((nt, nx), dtype=jnp.float32)
        dUi = jnp.ones((K, nu_i), dtype=jnp.float32) * (10.0 + i)
        return dX, dUi

    monkeypatch.setattr(pdg_alsolver.costtypes, "gradient_cost_local_ctrl_playerwise_trajectory", fake_cost_grad)

    dX_all, dU_all = pdg_alsolver._gradient_aug_lagrangian_trajectory(nlgame, op, alstate,
        discretize_method = "rk2",
        ineq_activation = "altro",
    )

    # Check each player's joint control gradient has nonzero entries only in their slice
    starts = np.cumsum([0] + list(u_splits[:-1]))
    for i in range(N):
        sl = slice(int(starts[i]), int(starts[i] + u_splits[i]))
        dU_i = np.array(dU_all[i])

        # in-slice should be (10+i), out-of-slice should be 0
        assert np.allclose(dU_i[:, sl], 10.0 + i)
        out = np.ones((K, nu), dtype=np.float32)
        out[:, sl] = 0.0
        assert np.allclose(dU_i[out.astype(bool)], 0.0)

def test_gradient_aug_lagrangian_trajectory_adds_joint_dynamics_control_to_each_player(monkeypatch):
    nlgame, op, alstate = _make_dummy_game_and_op()
    N, nt, nx, nu = nlgame.N, nlgame.nt, nlgame.nx, nlgame.nu
    K = nt - 1

    # No shared constraints / no cost
    monkeypatch.setattr(pdg_alsolver.contypes, "build_constraint_step_linearizations",
                        lambda constraints, op: (tuple(), tuple()))
    monkeypatch.setattr(pdg_alsolver, "_gradient_aug_lagrangian_trajectory_constraints_from_linearizations",
                        lambda **kwargs: (jnp.zeros((nt, nx), jnp.float32),
                                          jnp.zeros((K, nu), jnp.float32)))
    monkeypatch.setattr(pdg_alsolver, "_gradient_aug_lagrangian_trajectory_penalty_from_linearizations",
                        lambda **kwargs: (jnp.zeros((nt, nx), jnp.float32),
                                          jnp.zeros((K, nu), jnp.float32)))
    monkeypatch.setattr(pdg_alsolver.costtypes, "gradient_cost_local_ctrl_playerwise_trajectory",
                        lambda **kwargs: (jnp.zeros((nt, nx), jnp.float32),
                                          jnp.zeros((K, int(nlgame.u_splits[int(kwargs["player_i"])])), jnp.float32)))

    # Dynamics gradient per player: return a constant joint control gradient that depends on player index
    def fake_dyn_grad(**kwargs):
        i = int(kwargs["player_i"])
        dX = jnp.zeros((nt, nx), dtype=jnp.float32)
        dU = jnp.ones((K, nu), dtype=jnp.float32) * (3.0 + i)
        return dX, dU

    monkeypatch.setattr(pdg_alsolver, "gradient_aug_lagrangian_playerwise_trajectory_dynamics", fake_dyn_grad)

    dX_all, dU_all = pdg_alsolver._gradient_aug_lagrangian_trajectory(nlgame, op, alstate,
        discretize_method = "rk2",
        ineq_activation = "altro",
    )

    for i in range(N):
        np.testing.assert_allclose(np.array(dU_all[i]), (3.0 + i) * np.ones((K, nu), dtype=np.float32))
        np.testing.assert_allclose(np.array(dX_all[i]), 0.0)


def test_gradient_aug_lagrangian_trajectory_reuses_dynamics_jacobians_across_players(monkeypatch):
    N = 3
    nt = 5
    K = nt - 1
    nx = 2
    nu = 3
    tg = TimeGrid(nt=nt, dt=0.1, t0=0.0)
    u_splits = jnp.array([1, 1, 1], dtype=jnp.int32)

    def f_cont(t, x, u):
        return jnp.zeros_like(x)

    cs = SampledContinuousSystemType1(tg=tg, dynamics=f_cont, nx=nx, nu=nu)
    xs = jnp.zeros((nt, nx), dtype=jnp.float32)
    us = jnp.zeros((K, nu), dtype=jnp.float32)
    ls = jnp.ones((K, N, nx), dtype=jnp.float32)
    op = FixedStepPrimalDualTrajectory(tg=tg, xs=xs, us=us, ls=ls)

    dummy_cost = SimpleNamespace(running=lambda t, x, u: 0.0, terminal=lambda t, x: 0.0)
    nlgame = SimpleNamespace(
        tg=tg,
        nt=nt,
        nx=nx,
        nu=nu,
        N=N,
        u_splits=u_splits,
        cs=cs,
        constraints=SimpleNamespace(tg=tg, nc_ineq=0, nc_eq=0, nc_all=0),
        costs=[dummy_cost for _ in range(N)],
    )
    alstate = SimpleNamespace(
        lam_ineq=jnp.zeros((0,), dtype=jnp.float32),
        rho_ineq=jnp.zeros((0,), dtype=jnp.float32),
        lam_eq=jnp.zeros((0,), dtype=jnp.float32),
        rho_eq=jnp.zeros((0,), dtype=jnp.float32),
    )

    monkeypatch.setattr(
        pdg_alsolver.contypes,
        "build_constraint_step_linearizations",
        lambda constraints, op: (tuple(), tuple()),
    )
    monkeypatch.setattr(
        pdg_alsolver,
        "_gradient_aug_lagrangian_trajectory_constraints_from_linearizations",
        lambda **kwargs: (
            jnp.zeros((nt, nx), dtype=jnp.float32),
            jnp.zeros((K, nu), dtype=jnp.float32),
        ),
    )
    monkeypatch.setattr(
        pdg_alsolver,
        "_gradient_aug_lagrangian_trajectory_penalty_from_linearizations",
        lambda **kwargs: (
            jnp.zeros((nt, nx), dtype=jnp.float32),
            jnp.zeros((K, nu), dtype=jnp.float32),
        ),
    )
    monkeypatch.setattr(
        pdg_alsolver.costtypes,
        "gradient_cost_local_ctrl_playerwise_trajectory",
        lambda **kwargs: (
            jnp.zeros((nt, nx), dtype=jnp.float32),
            jnp.zeros((K, 1), dtype=jnp.float32),
        ),
    )

    calls = {"count": 0}

    def fake_dynamics_jacobian(cs, op, method):
        calls["count"] += 1
        As = jnp.tile(jnp.eye(nx, dtype=jnp.float32)[None, :, :], (K, 1, 1))
        Bs = jnp.ones((K, nx, nu), dtype=jnp.float32)
        return As, Bs

    monkeypatch.setattr(
        pdg_alsolver.systypes,
        "jacobian_discrete_dynamics_trajectory",
        fake_dynamics_jacobian,
    )

    dX_all, dU_all = pdg_alsolver._gradient_aug_lagrangian_trajectory(
        nlgame,
        op,
        alstate,
        discretize_method="rk2",
        ineq_activation="altro",
    )

    assert calls["count"] == 1
    assert dX_all.shape == (N, nt, nx)
    assert dU_all.shape == (N, K, nu)


def test_build_al_residual_ingredients_collects_shared_linearizations_and_dynamics_jacobians(monkeypatch):
    tg = TimeGrid(nt=4, dt=0.1, t0=0.0)
    K = tg.nt - 1
    nx = 2
    nu = 1

    def f_cont(t, x, u):
        return jnp.zeros_like(x)

    cs = SampledContinuousSystemType1(tg=tg, dynamics=f_cont, nx=nx, nu=nu)
    constraints = GameConstraintGridMap(ineq_blocks=(), eq_blocks=())
    nlgame = SimpleNamespace(tg=tg, cs=cs, constraints=constraints)
    op = FixedStepPrimalDualTrajectory(
        tg=tg,
        xs=jnp.zeros((tg.nt, nx), dtype=jnp.float32),
        us=jnp.zeros((K, nu), dtype=jnp.float32),
        ls=jnp.zeros((K, 1, nx), dtype=jnp.float32),
    )

    ineq_lins = ("ineq-linearization",)
    eq_lins = ("eq-linearization",)
    dfd_dx = jnp.ones((K, nx, nx), dtype=jnp.float32)
    dfd_du = jnp.ones((K, nx, nu), dtype=jnp.float32) * 2.0
    calls = {"constraints": 0, "dynamics": 0}

    def fake_build_constraint_step_linearizations(constraints, op):
        calls["constraints"] += 1
        return ineq_lins, eq_lins

    def fake_jacobian_discrete_dynamics_trajectory(cs, op, method):
        calls["dynamics"] += 1
        return dfd_dx, dfd_du

    monkeypatch.setattr(
        pdg_alsolver.contypes,
        "build_constraint_step_linearizations",
        fake_build_constraint_step_linearizations,
    )
    monkeypatch.setattr(
        pdg_alsolver.systypes,
        "jacobian_discrete_dynamics_trajectory",
        fake_jacobian_discrete_dynamics_trajectory,
    )

    ingredients = pdg_alsolver.build_al_residual_ingredients(
        nlgame,
        op,
        discretize_method="rk2",
    )

    assert calls == {"constraints": 1, "dynamics": 1}
    assert ingredients.ineq_lins is ineq_lins
    assert ingredients.eq_lins is eq_lins
    assert ingredients.dfd_dx is dfd_dx
    assert ingredients.dfd_du is dfd_du


def test_gradient_aug_lagrangian_trajectory_raises_on_timegrid_mismatch(monkeypatch):
    nlgame, op, alstate = _make_dummy_game_and_op()
    # change op.tg
    op_bad = SimpleNamespace(**{**op.__dict__, "tg": TimeGrid(nt=op.tg.nt, dt=op.tg.dt, t0=op.tg.t0 + 1.0)})

    with pytest.raises(ValueError, match="TimeGrid"):
        pdg_alsolver._gradient_aug_lagrangian_trajectory(nlgame, op_bad, alstate,
            discretize_method = "rk2",
            ineq_activation = "altro",
        )

def _u_slice_from_splits(u_splits: jnp.ndarray, i: int) -> slice:
    # Use numpy for deterministic Python ints
    splits = np.asarray(u_splits, dtype=int)
    start = int(np.sum(splits[:i]))
    stop = start + int(splits[i])
    return slice(start, stop)


def test_gradient_aug_lagrangian_trajectory_golden_tiny_game():
    # ---------- tiny game dimensions ----------
    N = 2
    tg = TimeGrid(nt=4, dt=0.3, t0=0.0)  # nt=4 => K=3 control steps
    nt = tg.nt
    K = nt - 1
    nx = 2
    u_splits = jnp.array([1, 2], dtype=jnp.int32)
    nu = int(np.sum(np.asarray(u_splits)))

    # ---------- operating point (xs, us, ls) ----------
    key = jax.random.PRNGKey(0)
    xs = jax.random.normal(key, (nt, nx), dtype=jnp.float32) * 0.2
    us = jax.random.normal(key, (K, nu), dtype=jnp.float32) * 0.3
    # dynamics multipliers: (K, N, nx)
    ls = jax.random.normal(key, (K, N, nx), dtype=jnp.float32) * 0.1

    op = FixedStepPrimalDualTrajectory(tg=tg, xs=xs, us=us, ls=ls)

    # ---------- simple linear continuous dynamics ----------
    # xdot = A x + B u
    A = jnp.array([[0.2, -0.1],
                   [0.05, 0.1]], dtype=jnp.float32)
    B = jnp.array([[1.0, 0.2, -0.1],
                   [0.0, 0.7,  0.3]], dtype=jnp.float32)

    def f_cont(t, x, u):
        return A @ x + B @ u

    # Minimal cs compatible with your code (your dynamics jacobian routine dispatches on type;
    cs = SampledContinuousSystemType1(tg=tg, nx=nx, nu=nu, dynamics=f_cont)

    # ---------- one inequality constraint block applied at every stage step ----------
    # c_k = alpha^T x_k + beta^T u_k  (scalar)
    alpha = jnp.array([0.3, -0.4], dtype=jnp.float32)
    beta  = jnp.array([0.2, -0.1, 0.05], dtype=jnp.float32)

    def c_ineq(t, x, u):
        return alpha @ x + beta @ u

    ineq_block = ConstraintBlockGridMap(
        tg=tg,
        func=c_ineq,
        cdim_out_step=1,
        active_steps=tuple(range(K)),  # steps 0..K-1
        iseq=False,
        terminal=False,
    )
    constraints = GameConstraintGridMap(ineq_blocks=(ineq_block,), eq_blocks=())

    # ---------- AL state (lam,rho) for ineq only ----------
    # stacking: one scalar per active step => nc_ineq = K
    lam_ineq = jnp.array([0.5, -0.2, 0.1], dtype=jnp.float32)  # length K
    rho_ineq = jnp.array([1.5,  2.0, 0.7], dtype=jnp.float32)  # length K

    alstate = altypes.JointAugmentedLagrangianState(
        lam_ineq=lam_ineq,
        rho_ineq=rho_ineq,
        lam_eq=jnp.zeros((0,), dtype=jnp.float32),
        rho_eq=jnp.zeros((0,), dtype=jnp.float32),
    )

    # ---------- player costs (local control dependence) ----------
    # Player 0 uses u slice [0:1], Player 1 uses u slice [1:3]
    def running_cost_0(t, x, u0):
        # scalar
        return 0.7 * (x @ x) + 1.2 * (u0 @ u0)

    def terminal_cost_0(t, x):
        return 0.9 * (x @ x)

    def running_cost_1(t, x, u1):
        return 0.5 * (x @ x) + 0.8 * (u1 @ u1)

    def terminal_cost_1(t, x):
        return 0.4 * (x @ x)

    costs = [
        PlayerCostSpecContinuous(running=running_cost_0, terminal=terminal_cost_0, control_domain=CostControlDomain.LOCAL, control_coupling=CostControlStructure.LOCAL_ONLY),
        PlayerCostSpecContinuous(running=running_cost_1, terminal=terminal_cost_1, control_domain=CostControlDomain.LOCAL, control_coupling=CostControlStructure.LOCAL_ONLY),
    ]

    # ---------- construct game ----------
    nlgame = NonlinearGameType2(
        cs=cs,
        N=N,
        costs=costs,
        constraints=constraints,
        u_splits=u_splits
    )

    # ---------- compute assembled gradients ----------
    dX_all, dU_all = pdg_alsolver.gradient_aug_lagrangian_trajectory(
        nlgame,
        op,
        alstate,
        discretize_method="euler",
        ineq_activation="none",
    )
    assert dX_all.shape == (N, nt, nx)
    assert dU_all.shape == (N, K, nu)

    # ---------- reference scalar L_i and its gradients via autodiff ----------
    ts = compute_ts(tg)

    def euler_fd(x, u):
        return x + tg.dt * f_cont(0.0, x, u)  # t treated as constant here

    def L_i_scalar(i: int, X: jnp.ndarray, U: jnp.ndarray) -> jnp.ndarray:
        """
        Scalar augmented Lagrangian for player i with:
          - stage + terminal cost
          - dynamics multipliers mu_i · D_k
          - shared linear constraint term lam^T C
          - shared quadratic penalty 1/2 C^T diag(rho) C
        """
        # cost
        sl = _u_slice_from_splits(u_splits, i)
        J = 0.0
        for k in range(K):
            ui = U[k, sl]
            J = J + costs[i].running(ts[k], X[k], ui)
        J = J + costs[i].terminal(ts[-1], X[-1])

        # dynamics multiplier term: sum_k mu_k^T (f_d(x_k,u_k) - x_{k+1})
        mu = op.ls[:, i, :]  # (K,nx) fixed for this reference
        Dterm = 0.0
        for k in range(K):
            x_next = euler_fd(X[k], U[k])
            Dk = x_next - X[k + 1]
            Dterm = Dterm + jnp.dot(mu[k], Dk)

        # shared constraints (only ineq, scalar at each step)
        C_lin = 0.0
        C_pen = 0.0
        for k in range(K):
            ck = c_ineq(ts[k], X[k], U[k])  # scalar
            C_lin = C_lin + lam_ineq[k] * ck
            C_pen = C_pen + 0.5 * rho_ineq[k] * (ck ** 2)

        return J + Dterm + C_lin + C_pen

    # Compare per-player grads (w.r.t. joint X and joint U)
    for i in range(N):
        def Li_wrt_XU(X, U):
            return L_i_scalar(i, X, U)

        dX_ref, dU_ref = jax.grad(Li_wrt_XU, argnums=(0, 1))(xs, us)

        np.testing.assert_allclose(np.asarray(dX_all[i]), np.asarray(dX_ref), atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(np.asarray(dU_all[i]), np.asarray(dU_ref), atol=1e-5, rtol=1e-5)


def test_gradient_aug_lagrangian_trajectory_golden_vector_constraints_and_eq_blocks():
    # --- dimensions ---
    N = 2
    tg = TimeGrid(nt=5, dt=0.2, t0=0.0)   # nt=5 => K=4
    nt, K = tg.nt, tg.nt - 1
    nx = 2
    u_splits = jnp.array([2, 1], dtype=jnp.int32)  # nontrivial slices
    nu = int(np.sum(np.asarray(u_splits)))

    # --- operating point ---
    key = jax.random.PRNGKey(1)
    xs = jax.random.normal(key, (nt, nx), dtype=jnp.float32) * 0.3
    us = jax.random.normal(key, (K, nu), dtype=jnp.float32) * 0.2
    ls = jax.random.normal(key, (K, N, nx), dtype=jnp.float32) * 0.1

    op = FixedStepPrimalDualTrajectory(tg=tg, xs=xs, us=us, ls=ls)

    # --- simple continuous dynamics + Euler discretization in reference ---
    A = jnp.array([[0.1, 0.0],
                   [0.0, 0.2]], dtype=jnp.float32)
    B = jnp.array([[1.0, 0.0, 0.3],
                   [0.0, 1.0, -0.1]], dtype=jnp.float32)

    def f_cont(t, x, u):
        return A @ x + B @ u
    
    cs = SampledContinuousSystemType1(tg=tg, nx=nx, nu=nu, dynamics=f_cont)

    # --- costs: local control only ---
    def run0(t, x, u0):
        return 0.6 * (x @ x) + 0.4 * (u0 @ u0)
    def term0(t, x):
        return 0.2 * (x @ x)

    def run1(t, x, u1):
        return 0.5 * (x @ x) + 0.9 * (u1 @ u1)
    def term1(t, x):
        return 0.1 * (x @ x)

    costs = [
        PlayerCostSpecContinuous(running=run0, terminal=term0, control_domain=CostControlDomain.LOCAL, control_coupling=CostControlStructure.LOCAL_ONLY),
        PlayerCostSpecContinuous(running=run1, terminal=term1, control_domain=CostControlDomain.LOCAL, control_coupling=CostControlStructure.LOCAL_ONLY),
    ]

    # --- constraints: multiple blocks, vector-valued, sparse active_steps, eq + ineq ---
    ts = compute_ts(tg)

    # Inequality block A: cdim=2, active at steps (0,2)
    # cA = [ x0 + u0 ; x1 - u2 ]
    def cA(t, x, u):
        return jnp.array([x[0] + u[0], x[1] - u[2]], dtype=jnp.float32)

    bA = ConstraintBlockGridMap(
        tg=tg, func=cA, cdim_out_step=2, active_steps=(0, 2), iseq=False, terminal=False
    )

    # Inequality block B: cdim=1, active at step (1,)
    def cB(t, x, u):
        return x[0] - 0.1 * u[1]
    bB = ConstraintBlockGridMap(
        tg=tg, func=cB, cdim_out_step=1, active_steps=(1,), iseq=False, terminal=False
    )

    # Equality block C: cdim=1, active at steps (0,3)
    def cC(t, x, u):
        return u[0] + u[1]
    bC = ConstraintBlockGridMap(
        tg=tg, func=cC, cdim_out_step=1, active_steps=(0, 3), iseq=True, terminal=False
    )

    constraints = GameConstraintGridMap(ineq_blocks=(bA, bB), eq_blocks=(bC,))

    # Dimensions:
    # ineq: bA => 2 dims *2 steps = 4, bB => 1*1 = 1 => nc_ineq = 5
    # eq:   bC => 1*2 = 2 => nc_eq = 2
    lam_ineq = jnp.array([0.5, -1.0,  0.2, 0.3,  -0.7], dtype=jnp.float32)
    rho_ineq = jnp.array([1.2,  0.7,  1.5, 2.0,   0.9], dtype=jnp.float32)
    lam_eq   = jnp.array([0.25, -0.4], dtype=jnp.float32)
    rho_eq   = jnp.array([1.1,  0.6], dtype=jnp.float32)

    alstate = altypes.JointAugmentedLagrangianState(
        lam_ineq=lam_ineq, rho_ineq=rho_ineq,
        lam_eq=lam_eq,     rho_eq=rho_eq,
    )

    nlgame = NonlinearGameType2(cs=cs, N=N, costs=costs, constraints=constraints, u_splits=u_splits)

    # --- assembled grads from your implementation ---
    dX_all, dU_all = pdg_alsolver.gradient_aug_lagrangian_trajectory(
        nlgame, op, alstate, discretize_method="euler", ineq_activation="none"
    )

    # --- scalar reference L_i and autodiff ---
    def euler_fd(x, u):
        return x + tg.dt * f_cont(0.0, x, u)

    def L_i_scalar(i: int, X: jnp.ndarray, U: jnp.ndarray) -> jnp.ndarray:
        # cost
        sl = _u_slice_from_splits(u_splits, i)
        J = 0.0
        for k in range(K):
            ui = U[k, sl]
            J = J + costs[i].running(ts[k], X[k], ui)
        J = J + costs[i].terminal(ts[-1], X[-1])

        # dynamics multiplier
        mu = op.ls[:, i, :]
        Dterm = 0.0
        for k in range(K):
            Dk = euler_fd(X[k], U[k]) - X[k + 1]
            Dterm = Dterm + jnp.dot(mu[k], Dk)

        # constraints stack in the same block order + active_steps order
        # linear + penalty
        C_lin = 0.0
        C_pen = 0.0

        # ineq: bA (k=0,2), cdim=2
        # slices: [0:2] for k=0, [2:4] for k=2
        c = cA(ts[0], X[0], U[0]); C_lin += jnp.dot(lam_ineq[0:2], c); C_pen += 0.5*jnp.sum(rho_ineq[0:2]*c*c)
        c = cA(ts[2], X[2], U[2]); C_lin += jnp.dot(lam_ineq[2:4], c); C_pen += 0.5*jnp.sum(rho_ineq[2:4]*c*c)
        # ineq: bB (k=1), cdim=1 => slice [4:5]
        c = jnp.asarray(cB(ts[1], X[1], U[1]))[None]
        C_lin += lam_ineq[4] * c[0]
        C_pen += 0.5 * rho_ineq[4] * (c[0]**2)

        # eq: bC (k=0,3), cdim=1 => slices [0:1], [1:2] within eq vectors
        c = jnp.asarray(cC(ts[0], X[0], U[0]))[None]
        C_lin += lam_eq[0] * c[0]
        C_pen += 0.5 * rho_eq[0] * (c[0]**2)

        c = jnp.asarray(cC(ts[3], X[3], U[3]))[None]
        C_lin += lam_eq[1] * c[0]
        C_pen += 0.5 * rho_eq[1] * (c[0]**2)

        return J + Dterm + C_lin + C_pen

    for i in range(N):
        dX_ref, dU_ref = jax.grad(lambda X, U: L_i_scalar(i, X, U), argnums=(0, 1))(xs, us)
        np.testing.assert_allclose(np.asarray(dX_all[i]), np.asarray(dX_ref), atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(np.asarray(dU_all[i]), np.asarray(dU_ref), atol=1e-5, rtol=1e-5)


def test_gradient_aug_lagrangian_trajectory_golden_altro_activation_stopgrad():
    N = 1  # keep it minimal; we're testing activation, not multi-player coupling
    tg = TimeGrid(nt=4, dt=0.1, t0=0.0)
    nt, K = tg.nt, tg.nt - 1
    nx, nu = 1, 1
    u_splits = jnp.array([1], dtype=jnp.int32)

    # operating point
    xs = jnp.array([[0.0], [0.0], [0.0], [0.0]], dtype=jnp.float32)
    us = jnp.array([[0.0], [0.0], [0.0]], dtype=jnp.float32)
    ls = jnp.zeros((K, N, nx), dtype=jnp.float32)
    op = FixedStepPrimalDualTrajectory(tg=tg, xs=xs, us=us, ls=ls)

    # no cost, no dynamics
    cs = SampledContinuousSystemType1(tg=tg, nx=nx, nu=nu, dynamics=lambda t, x, u: jnp.zeros_like(x))
    costs = [PlayerCostSpecContinuous(running=lambda t, x, ui: 0.0, terminal=lambda t, x: 0.0, control_domain=CostControlDomain.LOCAL, control_coupling=CostControlStructure.LOCAL_ONLY)]

    ts = compute_ts(tg)

    # inequality constraint: c = x - 1.0  (always negative here since x=0)
    def c_ineq(t, x, u):
        return x[0] - 1.0

    block = ConstraintBlockGridMap(
        tg=tg, func=c_ineq, cdim_out_step=1, active_steps=(0, 1, 2), iseq=False, terminal=False
    )
    constraints = GameConstraintGridMap(ineq_blocks=(block,), eq_blocks=())

    # choose lambda so step 1 is "active by lambda>0" even though c<0; others inactive
    lam_ineq = jnp.array([0.0, 1.0, 0.0], dtype=jnp.float32)
    rho_ineq = jnp.array([2.0, 2.0, 2.0], dtype=jnp.float32)
    alstate = altypes.JointAugmentedLagrangianState(
        lam_ineq=lam_ineq, rho_ineq=rho_ineq,
        lam_eq=jnp.zeros((0,), dtype=jnp.float32),
        rho_eq=jnp.zeros((0,), dtype=jnp.float32),
    )

    nlgame = NonlinearGameType2(cs=cs, N=N, costs=costs, constraints=constraints, u_splits=u_splits)

    # assembled grads
    dX_all, dU_all = pdg_alsolver.gradient_aug_lagrangian_trajectory(
        nlgame, op, alstate, discretize_method="euler", ineq_activation="altro"
    )

    # reference scalar L with stop-grad activation
    def L_scalar(X, U):
        total = 0.0
        for k in range(K):
            c = jnp.asarray(c_ineq(ts[k], X[k], U[k]))[None]  # (1,)
            lam = lam_ineq[k:k+1]
            rho = rho_ineq[k:k+1]
            a = (c >= 0) | (lam > 0)
            a = jax.lax.stop_gradient(a.astype(c.dtype))
            c_eff = a * c
            total = total + lam[0] * c[0] + 0.5 * rho[0] * (c_eff[0] ** 2)
        return total

    dX_ref, dU_ref = jax.grad(L_scalar, argnums=(0, 1))(xs, us)

    np.testing.assert_allclose(np.asarray(dX_all[0]), np.asarray(dX_ref), atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(dU_all[0]), np.asarray(dU_ref), atol=1e-6, rtol=1e-6)

def test_gradient_aug_lagrangian_trajectory_golden_terminal_activation_multi_player():
    # -----------------
    # Problem dimensions
    # -----------------
    N = 3
    tg = TimeGrid(nt=5, dt=0.2, t0=0.0)  # nt=5 => K=4
    nt, K = tg.nt, tg.nt - 1
    nx = 2
    u_splits = jnp.array([1, 1, 2], dtype=jnp.int32)
    nu = int(np.sum(np.asarray(u_splits)))

    # -----------------
    # Operating point
    # -----------------
    key = jax.random.PRNGKey(7)
    xs = jax.random.normal(key, (nt, nx), dtype=jnp.float32) * 0.2
    us = jax.random.normal(key, (K, nu), dtype=jnp.float32) * 0.3

    # set dynamics multipliers to zero so dynamics term is exactly zero in both impl and reference
    ls = jnp.zeros((K, N, nx), dtype=jnp.float32)

    op = FixedStepPrimalDualTrajectory(tg=tg, xs=xs, us=us, ls=ls)

    # -----------------
    # Dynamics (irrelevant since mu=0, but provide a valid cs)
    # -----------------
    def f_cont(t, x, u):
        return jnp.zeros_like(x)

    cs = SampledContinuousSystemType1(tg=tg, nx=nx, nu=nu, dynamics=f_cont)

    # -----------------
    # Costs (player-local control dependence)
    # -----------------
    ts = compute_ts(tg)

    # Make running costs depend on u_i only (so dU differs across players),
    # and small terminal costs so dX differs across players too.
    def make_cost(i: int):
        def running(t, x, ui):
            return (0.1 + 0.05 * i) * (x @ x) + (1.0 + 0.2 * i) * (ui @ ui)
        def terminal(t, x):
            return (0.2 + 0.1 * i) * (x @ x)
        return PlayerCostSpecContinuous(running=running, terminal=terminal, control_domain=CostControlDomain.LOCAL, control_coupling=CostControlStructure.LOCAL_ONLY)

    costs = [make_cost(i) for i in range(N)]

    # -----------------
    # Constraints: one terminal inequality block + activation
    # -----------------
    # Terminal inequality kernel: c_T(t, x) = x0 - 10  (very negative) so activation depends on lam>0
    def c_term_ineq(t, x):
        return x[0] - 10.0

    bT = ConstraintBlockGridMap(
        tg=tg,
        func=c_term_ineq,
        cdim_out_step=1,
        active_steps=None,   # defaults to (nt-1,)
        iseq=False,
        terminal=True,
    )

    constraints = GameConstraintGridMap(ineq_blocks=(bT,), eq_blocks=())

    # AL params: nc_ineq = 1 (terminal only)
    lam_ineq = jnp.array([0.7], dtype=jnp.float32)  # >0 so Altro activation should turn penalty ON even though c<0
    rho_ineq = jnp.array([2.0], dtype=jnp.float32)

    alstate = altypes.JointAugmentedLagrangianState(
        lam_ineq=lam_ineq,
        rho_ineq=rho_ineq,
        lam_eq=jnp.zeros((0,), dtype=jnp.float32),
        rho_eq=jnp.zeros((0,), dtype=jnp.float32),
    )

    # -----------------
    # Game object (minimal fields needed by your implementation)
    # -----------------
    nlgame = NonlinearGameType2(cs=cs, N=N, costs=costs, constraints=constraints, u_splits=u_splits)

    # -----------------
    # Implementation gradients (assembled)
    # -----------------
    dX_all, dU_all = pdg_alsolver.gradient_aug_lagrangian_trajectory(
        nlgame, op, alstate, discretize_method="euler", ineq_activation="altro"
    )
    assert dX_all.shape == (N, nt, nx)
    assert dU_all.shape == (N, K, nu)

    # -----------------
    # Scalar reference L_i and autodiff gradients
    # -----------------
    def L_i_scalar(i: int, X: jnp.ndarray, U: jnp.ndarray) -> jnp.ndarray:
        sl = _u_slice_from_splits(u_splits, i)

        # cost (running on steps 0..K-1, terminal on nt-1)
        J = 0.0
        for k in range(K):
            ui = U[k, sl]
            J = J + costs[i].running(ts[k], X[k], ui)
        J = J + costs[i].terminal(ts[-1], X[-1])

        # dynamics term: mu=0 => exactly zero (kept for conceptual completeness)
        Dterm = 0.0

        # terminal inequality constraint: linear + activated quadratic penalty
        c = jnp.asarray(c_term_ineq(ts[-1], X[-1]))[None]  # (1,)
        lam = lam_ineq
        rho = rho_ineq

        # linear term always applies
        C_lin = lam[0] * c[0]

        # Altro activation for penalty term, with stop_gradient on mask
        a = (c >= 0) | (lam > 0)
        a = jax.lax.stop_gradient(a.astype(c.dtype))
        c_eff = a * c
        C_pen = 0.5 * rho[0] * (c_eff[0] ** 2)

        return J + Dterm + C_lin + C_pen

    # Compare player-by-player to autodiff reference
    for i in range(N):
        dX_ref, dU_ref = jax.grad(lambda X, U: L_i_scalar(i, X, U), argnums=(0, 1))(xs, us)
        np.testing.assert_allclose(np.asarray(dX_all[i]), np.asarray(dX_ref), atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(np.asarray(dU_all[i]), np.asarray(dU_ref), atol=1e-5, rtol=1e-5)

    # Extra sanity: terminal constraint contribution should be identical across players
    # (it is shared), so differences in dX at terminal index should come only from costs.
    # We don't assert exact decomposition here, but we at least check the *constraint part*
    # is nonzero at terminal state for all players (since lam>0 triggers activation).
    assert np.any(np.asarray(dX_all[:, -1, :]) != 0.0)

def test_compute_al_residual_struct_shapes_and_x0_exclusion(monkeypatch):
    nt, nx, nu, N = 6, 3, 7, 4
    K = nt - 1
    nlgame, op = _make_game_and_op(nt=nt, nx=nx, nu=nu, N=N)

    # fake gradient_aug_lagrangian_trajectory outputs:
    dL_dX_all = jnp.ones((N, nt, nx), dtype=jnp.float32) * 10.0
    dL_dU_all = jnp.ones((N, K, nu), dtype=jnp.float32) * 20.0

    def fake_grad(nlgame_, op_, alstate_, **kwargs):
        return dL_dX_all, dL_dU_all

    def fake_dyn_res(cs_, op_, method):
        return jnp.ones((K, nx), dtype=jnp.float32) * 30.0

    monkeypatch.setattr(pdg_alsolver, "gradient_aug_lagrangian_trajectory", fake_grad)
    monkeypatch.setattr(pdg_alsolver.systypes, "residual_discrete_dynamics_trajectory", fake_dyn_res)

    alstate = SimpleNamespace()  # unused by fake_grad

    r = pdg_alsolver.compute_al_residual_struct_from_traj(
        nlgame, op, alstate, discretize_method="rk2", ineq_activation="altro"
    )

    assert r.dLdX.shape == (N, K, nx)
    assert r.dLdU.shape == (N, K, nu)
    assert r.dyn_res.shape == (K, nx)

    # x0 excluded: should match dL_dX_all[:,1:,:]
    np.testing.assert_allclose(np.asarray(r.dLdX), np.asarray(dL_dX_all[:, 1:, :]))
    np.testing.assert_allclose(np.asarray(r.dLdU), np.asarray(dL_dU_all))
    np.testing.assert_allclose(np.asarray(r.dyn_res), np.asarray(jnp.ones((K, nx)) * 30.0))


def test_compute_al_residual_struct_builds_and_passes_al_residual_ingredients_once(monkeypatch):
    nt, nx, nu, N = 5, 2, 3, 2
    K = nt - 1
    nlgame, op = _make_game_and_op(nt=nt, nx=nx, nu=nu, N=N)

    ingredients = pdg_alsolver._ALResidualIngredients(
        ineq_lins=("ineq-linearization",),
        eq_lins=("eq-linearization",),
        dfd_dx=jnp.ones((K, nx, nx), dtype=jnp.float32),
        dfd_du=jnp.ones((K, nx, nu), dtype=jnp.float32),
    )
    calls = {"ingredients": 0, "gradient": 0}

    def fake_build_ingredients(nlgame_, op_, *, discretize_method):
        calls["ingredients"] += 1
        assert nlgame_ is nlgame
        assert op_ is op
        assert discretize_method == "rk2"
        return ingredients

    def fake_grad(nlgame_, op_, alstate_, **kwargs):
        calls["gradient"] += 1
        assert kwargs["ingredients"] is ingredients
        return (
            jnp.ones((N, nt, nx), dtype=jnp.float32),
            jnp.ones((N, K, nu), dtype=jnp.float32),
        )

    def fake_dyn_res(cs_, op_, method):
        return jnp.zeros((K, nx), dtype=jnp.float32)

    monkeypatch.setattr(pdg_alsolver, "build_al_residual_ingredients", fake_build_ingredients)
    monkeypatch.setattr(pdg_alsolver, "gradient_aug_lagrangian_trajectory", fake_grad)
    monkeypatch.setattr(pdg_alsolver.systypes, "residual_discrete_dynamics_trajectory", fake_dyn_res)

    r = pdg_alsolver.compute_al_residual_struct_from_traj(
        nlgame,
        op,
        SimpleNamespace(),
        discretize_method="rk2",
        ineq_activation="altro",
    )

    assert calls == {"ingredients": 1, "gradient": 1}
    assert r.dLdX.shape == (N, K, nx)
    assert r.dLdU.shape == (N, K, nu)
    assert r.dyn_res.shape == (K, nx)


def test_compute_al_residual_struct_raises_on_timegrid_mismatch(monkeypatch):
    nt, nx, nu, N = 5, 2, 3, 2
    nlgame, op = _make_game_and_op(nt=nt, nx=nx, nu=nu, N=N)

    # mismatch
    nlgame = SimpleNamespace(**{**nlgame.__dict__, "tg": TimeGrid(nt=nt, dt=0.1, t0=1.0)})

    with pytest.raises(ValueError, match="TimeGrid"):
        pdg_alsolver.compute_al_residual_struct_from_traj(nlgame, op, SimpleNamespace(), discretize_method="euler", ineq_activation="altro")


@pytest.mark.benchmark(group="alsolver-residual-001")
def test_compute_al_residual_flat_from_decision_vars_warm_perf(benchmark):
    tg = TimeGrid(nt=8, dt=0.1, t0=0.0)
    nt = tg.nt
    K = nt - 1
    nx = 2
    N = 2
    u_splits = jnp.array([1, 1], dtype=jnp.int32)
    nu = int(np.sum(np.asarray(u_splits)))

    A = jnp.array([[0.0, 1.0], [-0.2, -0.1]], dtype=jnp.float32)
    B = jnp.array([[0.0, 0.0], [1.0, 0.5]], dtype=jnp.float32)

    def f_cont(t, x, u):
        return A @ x + B @ u

    cs = SampledContinuousSystemType1(tg=tg, dynamics=f_cont, nx=nx, nu=nu)

    def make_cost(i):
        target = jnp.array([1.0 + i, 0.0], dtype=jnp.float32)

        def running(t, x, u_i):
            return 0.5 * (x - target) @ (x - target) + 0.1 * (u_i @ u_i)

        def terminal(t, x):
            return 2.0 * ((x - target) @ (x - target))

        return PlayerCostSpecContinuous(
            running=running,
            terminal=terminal,
            control_domain=CostControlDomain.LOCAL,
            control_coupling=CostControlStructure.LOCAL_ONLY,
        )

    constraints = GameConstraintGridMap(ineq_blocks=(), eq_blocks=())
    nlgame = NonlinearGameType2(
        cs=cs,
        N=N,
        costs=[make_cost(0), make_cost(1)],
        constraints=constraints,
        u_splits=u_splits,
    )

    xs = jnp.stack(
        [
            jnp.linspace(0.0, 1.0, nt, dtype=jnp.float32),
            jnp.linspace(0.5, -0.5, nt, dtype=jnp.float32),
        ],
        axis=1,
    )
    us = jnp.ones((K, nu), dtype=jnp.float32) * 0.1
    ls = jnp.zeros((K, N, nx), dtype=jnp.float32)
    op = FixedStepPrimalDualTrajectory(tg=tg, xs=xs, us=us, ls=ls)
    z = pdg_alsolver.pack_decision_vars_1d(op)
    alstate = altypes.JointAugmentedLagrangianState(
        lam_ineq=jnp.zeros((0,), dtype=jnp.float32),
        rho_ineq=jnp.zeros((0,), dtype=jnp.float32),
        lam_eq=jnp.zeros((0,), dtype=jnp.float32),
        rho_eq=jnp.zeros((0,), dtype=jnp.float32),
    )

    def run():
        g = pdg_alsolver.compute_al_residual_flat_from_decision_vars(
            nlgame,
            z,
            op,
            alstate,
            discretize_method="euler",
            ineq_activation="altro",
        )
        return g.block_until_ready()

    benchmark(run)


def _make_constraint_heavy_residual_problem(*, nt=24):
    tg = TimeGrid(nt=nt, dt=0.05, t0=0.0)
    K = nt - 1
    nx = 4
    N = 2
    u_splits = jnp.array([1, 1], dtype=jnp.int32)
    nu = int(np.sum(np.asarray(u_splits)))

    def f_cont(t, x, u):
        return jnp.array(
            [
                x[1],
                u[0] - 0.05 * x[1],
                x[3],
                u[1] - 0.05 * x[3],
            ],
            dtype=x.dtype,
        )

    cs = SampledContinuousSystemType1(tg=tg, dynamics=f_cont, nx=nx, nu=nu)

    def make_cost(i):
        target = jnp.array([1.0 + i, 0.2, 1.5 + i, -0.1], dtype=jnp.float32)

        def running(t, x, u_i):
            dx = x - target
            return 0.1 * (dx @ dx) + 0.05 * (u_i @ u_i)

        def terminal(t, x):
            dx = x - target
            return 2.0 * (dx @ dx)

        return PlayerCostSpecContinuous(
            running=running,
            terminal=terminal,
            control_domain=CostControlDomain.LOCAL,
            control_coupling=CostControlStructure.LOCAL_ONLY,
        )

    active_all = tuple(range(K))

    def control_bounds(t, x, u):
        return jnp.array(
            [
                u[0] - 1.5,
                -u[0] - 1.5,
                u[1] - 1.5,
                -u[1] - 1.5,
            ],
            dtype=u.dtype,
        )

    def state_bounds(t, x, u):
        return jnp.array(
            [
                x[1] - 3.0,
                -x[1] - 0.2,
                x[3] - 3.0,
                -x[3] - 0.2,
            ],
            dtype=x.dtype,
        )

    def separation(t, x, u):
        return jnp.array([0.75 - (x[2] - x[0])], dtype=x.dtype)

    constraints = GameConstraintGridMap(
        ineq_blocks=(
            ConstraintBlockGridMap(
                tg=tg,
                func=control_bounds,
                cdim_out_step=4,
                active_steps=active_all,
                iseq=False,
                terminal=False,
            ),
            ConstraintBlockGridMap(
                tg=tg,
                func=state_bounds,
                cdim_out_step=4,
                active_steps=active_all,
                iseq=False,
                terminal=False,
            ),
            ConstraintBlockGridMap(
                tg=tg,
                func=separation,
                cdim_out_step=1,
                active_steps=active_all,
                iseq=False,
                terminal=False,
            ),
        ),
        eq_blocks=(),
    )

    nlgame = NonlinearGameType2(
        cs=cs,
        N=N,
        costs=[make_cost(0), make_cost(1)],
        constraints=constraints,
        u_splits=u_splits,
    )

    ts = jnp.linspace(0.0, 1.0, nt, dtype=jnp.float32)
    xs = jnp.stack(
        [
            0.5 * ts,
            0.2 + 0.1 * ts,
            1.2 + 0.6 * ts,
            0.3 - 0.1 * ts,
        ],
        axis=1,
    )
    us = jnp.stack(
        [
            0.1 * jnp.ones((K,), dtype=jnp.float32),
            -0.05 * jnp.ones((K,), dtype=jnp.float32),
        ],
        axis=1,
    )
    ls = jnp.zeros((K, N, nx), dtype=jnp.float32)
    op = FixedStepPrimalDualTrajectory(tg=tg, xs=xs, us=us, ls=ls)
    alstate = altypes.JointAugmentedLagrangianState(
        lam_ineq=jnp.zeros((constraints.nc_ineq,), dtype=jnp.float32),
        rho_ineq=jnp.ones((constraints.nc_ineq,), dtype=jnp.float32),
        lam_eq=jnp.zeros((constraints.nc_eq,), dtype=jnp.float32),
        rho_eq=jnp.ones((constraints.nc_eq,), dtype=jnp.float32),
    )

    return nlgame, op, alstate


@pytest.mark.benchmark(group="alsolver-residual-constraints-001")
def test_compute_al_residual_flat_from_decision_vars_constraint_heavy_warm_perf(benchmark):
    nlgame, op, alstate = _make_constraint_heavy_residual_problem(nt=24)
    z = pdg_alsolver.pack_decision_vars_1d(op)

    def run():
        g = pdg_alsolver.compute_al_residual_flat_from_decision_vars(
            nlgame,
            z,
            op,
            alstate,
            discretize_method="euler",
            ineq_activation="altro",
        )
        return g.block_until_ready()

    benchmark(run)


@pytest.mark.benchmark(group="alsolver-stationarity-metrics-constraints-001")
def test_newton_solve_stationarity_start_metrics_constraint_heavy_warm_perf(benchmark):
    nlgame, op, alstate = _make_constraint_heavy_residual_problem(nt=24)

    def run():
        op_out, diag = pdg_alsolver.newton_solve_stationarity_autodiff(
            nlgame,
            op,
            alstate,
            discretize_method="euler",
            ineq_activation="altro",
            opt_tol=1e9,
            dyn_tol=1e9,
            max_iters=0,
            max_rejects=0,
            step_rtol=1e-7,
            step_atol=1e-8,
            reg0=0.0,
            reg1_min=1e-12,
            reg_increase=10.0,
            reg_max=1e8,
            reg_max_iters=64,
            ls_alpha0=1.0,
            ls_tau=0.5,
            ls_beta=0.25,
            ls_max_iters=20,
            normkind="l1_mean",
        )
        assert op_out is op
        assert diag.reason == "opt_dyn_tol_at_start"
        return jnp.asarray(diag.merit_norms[-1]).block_until_ready()

    benchmark(run)


@pytest.mark.benchmark(group="alsolver-jacobian-constraints-001")
def test_jacobian_al_residual_flat_autodiff_constraint_heavy_warm_perf(benchmark):
    tg = TimeGrid(nt=8, dt=0.05, t0=0.0)
    nt = tg.nt
    K = nt - 1
    nx = 4
    N = 2
    u_splits = jnp.array([1, 1], dtype=jnp.int32)
    nu = int(np.sum(np.asarray(u_splits)))

    def f_cont(t, x, u):
        return jnp.array(
            [
                x[1],
                u[0] - 0.05 * x[1],
                x[3],
                u[1] - 0.05 * x[3],
            ],
            dtype=x.dtype,
        )

    cs = SampledContinuousSystemType1(tg=tg, dynamics=f_cont, nx=nx, nu=nu)

    def make_cost(i):
        target = jnp.array([1.0 + i, 0.2, 1.5 + i, -0.1], dtype=jnp.float32)

        def running(t, x, u_i):
            dx = x - target
            return 0.1 * (dx @ dx) + 0.05 * (u_i @ u_i)

        def terminal(t, x):
            dx = x - target
            return 2.0 * (dx @ dx)

        return PlayerCostSpecContinuous(
            running=running,
            terminal=terminal,
            control_domain=CostControlDomain.LOCAL,
            control_coupling=CostControlStructure.LOCAL_ONLY,
        )

    active_all = tuple(range(K))

    def control_bounds(t, x, u):
        return jnp.array(
            [
                u[0] - 1.5,
                -u[0] - 1.5,
                u[1] - 1.5,
                -u[1] - 1.5,
            ],
            dtype=u.dtype,
        )

    def state_bounds(t, x, u):
        return jnp.array(
            [
                x[1] - 3.0,
                -x[1] - 0.2,
                x[3] - 3.0,
                -x[3] - 0.2,
            ],
            dtype=x.dtype,
        )

    def separation(t, x, u):
        return jnp.array([0.75 - (x[2] - x[0])], dtype=x.dtype)

    constraints = GameConstraintGridMap(
        ineq_blocks=(
            ConstraintBlockGridMap(
                tg=tg,
                func=control_bounds,
                cdim_out_step=4,
                active_steps=active_all,
                iseq=False,
                terminal=False,
            ),
            ConstraintBlockGridMap(
                tg=tg,
                func=state_bounds,
                cdim_out_step=4,
                active_steps=active_all,
                iseq=False,
                terminal=False,
            ),
            ConstraintBlockGridMap(
                tg=tg,
                func=separation,
                cdim_out_step=1,
                active_steps=active_all,
                iseq=False,
                terminal=False,
            ),
        ),
        eq_blocks=(),
    )

    nlgame = NonlinearGameType2(
        cs=cs,
        N=N,
        costs=[make_cost(0), make_cost(1)],
        constraints=constraints,
        u_splits=u_splits,
    )

    ts = jnp.linspace(0.0, 1.0, nt, dtype=jnp.float32)
    xs = jnp.stack(
        [
            0.5 * ts,
            0.2 + 0.1 * ts,
            1.2 + 0.6 * ts,
            0.3 - 0.1 * ts,
        ],
        axis=1,
    )
    us = jnp.stack(
        [
            0.1 * jnp.ones((K,), dtype=jnp.float32),
            -0.05 * jnp.ones((K,), dtype=jnp.float32),
        ],
        axis=1,
    )
    ls = jnp.zeros((K, N, nx), dtype=jnp.float32)
    op = FixedStepPrimalDualTrajectory(tg=tg, xs=xs, us=us, ls=ls)
    z = pdg_alsolver.pack_decision_vars_1d(op)
    alstate = altypes.JointAugmentedLagrangianState(
        lam_ineq=jnp.zeros((constraints.nc_ineq,), dtype=jnp.float32),
        rho_ineq=jnp.ones((constraints.nc_ineq,), dtype=jnp.float32),
        lam_eq=jnp.zeros((constraints.nc_eq,), dtype=jnp.float32),
        rho_eq=jnp.ones((constraints.nc_eq,), dtype=jnp.float32),
    )

    def run():
        H = pdg_alsolver.jacobian_al_residual_flat_autodiff(
            nlgame,
            z,
            op,
            alstate,
            discretize_method="euler",
            ineq_activation="altro",
            mode="jacfwd",
        )
        return H.block_until_ready()

    benchmark(run)


# -------------------------
# pack_al_residual_1d
# -------------------------

def test_pack_al_residual_1d_length_and_order(monkeypatch):
    """
    Create a tiny residual struct with known values and verify:
    - packed length is correct
    - order is [dLdX0, dLdU0_local, dLdX1, dLdU1_local, dyn_res]
    """
    N, K, nx = 2, 3, 2
    u_splits = jnp.array([1, 2], dtype=jnp.int32)
    nu = int(u_splits.sum())

    # Construct distinct values so ordering mistakes show up
    # dLdX[0] = 1.., dLdX[1] = 100..
    dLdX0 = jnp.arange(K * nx, dtype=jnp.float32).reshape(K, nx) + 1.0
    dLdX1 = jnp.arange(K * nx, dtype=jnp.float32).reshape(K, nx) + 101.0
    dLdX = jnp.stack([dLdX0, dLdX1], axis=0)

    # dLdU joint: fill with player-identifiable columns
    # player0 slice col0, player1 slice col1-2
    dLdU = jnp.zeros((N, K, nu), dtype=jnp.float32)
    dLdU = dLdU.at[0, :, 0].set(10.0)          # p0 local
    dLdU = dLdU.at[1, :, 1].set(20.0)          # p1 local col1
    dLdU = dLdU.at[1, :, 2].set(30.0)          # p1 local col2

    dyn_res = jnp.arange(K * nx, dtype=jnp.float32).reshape(K, nx) + 1000.0

    r = altypes.ALResidualStruct(dLdX=dLdX, dLdU=dLdU, dyn_res=dyn_res)
    altypes.validate_al_residual_struct(r)

    g = pdg_alsolver.pack_al_residual_1d(r, u_splits=u_splits)
    assert g.ndim == 1

    expected_len = (N * K * nx) + (K * nu) + (K * nx)  # local controls across players sum to nu
    assert int(g.shape[0]) == expected_len

    # Build expected packed vector manually
    expected_parts = []
    expected_parts.append(jnp.ravel(dLdX0))              # p0 X
    expected_parts.append(jnp.ravel(dLdU[0, :, 0:1]))    # p0 U0 local
    expected_parts.append(jnp.ravel(dLdX1))              # p1 X
    expected_parts.append(jnp.ravel(dLdU[1, :, 1:3]))    # p1 U1 local
    expected_parts.append(jnp.ravel(dyn_res))            # D

    expected = jnp.concatenate(expected_parts)

    np.testing.assert_allclose(np.asarray(g), np.asarray(expected), atol=1e-6, rtol=1e-6)


def test_pack_al_residual_1d_rejects_bad_u_splits_shape():
    N, K, nx, nu = 2, 3, 2, 3
    r = altypes.ALResidualStruct(
        dLdX=jnp.zeros((N, K, nx), dtype=jnp.float32),
        dLdU=jnp.zeros((N, K, nu), dtype=jnp.float32),
        dyn_res=jnp.zeros((K, nx), dtype=jnp.float32),
    )

    with pytest.raises(ValueError):
        pdg_alsolver.pack_al_residual_1d(r, u_splits=jnp.array([1], dtype=jnp.int32))


def test_pack_al_residual_1d_rejects_u_splits_sum_mismatch():
    N, K, nx, nu = 2, 3, 2, 3
    r = altypes.ALResidualStruct(
        dLdX=jnp.zeros((N, K, nx), dtype=jnp.float32),
        dLdU=jnp.zeros((N, K, nu), dtype=jnp.float32),
        dyn_res=jnp.zeros((K, nx), dtype=jnp.float32),
    )

    # sums to 4 != nu=3
    with pytest.raises(ValueError, match="sum"):
        pdg_alsolver.pack_al_residual_1d(r, u_splits=jnp.array([2, 2], dtype=jnp.int32))


def _finite_difference_jacobian(f, z, eps=1e-6):
    z = np.asarray(z, dtype=np.float64)
    f0 = np.asarray(f(z), dtype=np.float64)
    m = f0.size
    n = z.size
    J = np.zeros((m, n), dtype=np.float64)
    for j in range(n):
        zp = z.copy(); zp[j] += eps
        zm = z.copy(); zm[j] -= eps
        fp = np.asarray(f(zp), dtype=np.float64)
        fm = np.asarray(f(zm), dtype=np.float64)
        J[:, j] = (fp - fm) / (2.0 * eps)
    return J


def _make_linear_dynamics_only_al_problem(*, nt=5, dtype=jnp.float32):
    tg = TimeGrid(nt=nt, dt=0.1, t0=0.0)
    K = tg.nsteps
    nx = 2
    u_splits = jnp.array([1, 2], dtype=jnp.int32)
    N = int(u_splits.shape[0])
    nu = int(np.sum(np.asarray(u_splits)))

    A = jnp.array([[0.0, 1.0], [-0.3, -0.2]], dtype=dtype)
    B = jnp.array([[0.0, 0.4, -0.1], [1.0, 0.2, 0.5]], dtype=dtype)

    def f_cont(t, x, u):
        return A @ x + B @ u

    cs = SampledContinuousSystemType1(tg=tg, dynamics=f_cont, nx=nx, nu=nu)

    def make_zero_cost():
        def running(t, x, u_i):
            return jnp.array(0.0, dtype=x.dtype)

        def terminal(t, x):
            return jnp.array(0.0, dtype=x.dtype)

        return PlayerCostSpecContinuous(
            running=running,
            terminal=terminal,
            control_domain=CostControlDomain.LOCAL,
            control_coupling=CostControlStructure.LOCAL_ONLY,
        )

    constraints = GameConstraintGridMap(ineq_blocks=(), eq_blocks=())
    nlgame = NonlinearGameType2(
        cs=cs,
        N=N,
        costs=[make_zero_cost() for _ in range(N)],
        constraints=constraints,
        u_splits=u_splits,
    )

    xs = (jnp.arange(nt * nx, dtype=dtype).reshape(nt, nx) * 0.1).at[0].set(
        jnp.array([0.2, -0.1], dtype=dtype)
    )
    us = jnp.linspace(-0.2, 0.3, K * nu, dtype=dtype).reshape(K, nu)
    ls = jnp.linspace(-0.4, 0.5, K * N * nx, dtype=dtype).reshape(K, N, nx)
    op = FixedStepPrimalDualTrajectory(tg=tg, xs=xs, us=us, ls=ls)
    alstate = altypes.JointAugmentedLagrangianState(
        lam_ineq=jnp.zeros((0,), dtype=dtype),
        rho_ineq=jnp.zeros((0,), dtype=dtype),
        lam_eq=jnp.zeros((0,), dtype=dtype),
        rho_eq=jnp.zeros((0,), dtype=dtype),
    )

    return nlgame, op, alstate


def test_jacobian_al_residual_flat_autodiff_matches_finite_difference(monkeypatch):
    """
    Golden test: compare autodiff Jacobian of G(z) to finite-difference Jacobian
    on a tiny, deliberately linear residual construction.
    """

    # enable float64 jax for better numerical precision
    jax.config.update("jax_enable_x64", True)

    # tiny dimensions
    N = 2
    tg = TimeGrid(nt=3, dt=0.1, t0=0.0)  # nt=3 => K=2
    nt = tg.nt
    K = nt - 1
    nx = 1
    nu = 3
    u_splits = jnp.array([1, 2], dtype=jnp.int32)

    # template operating point (x0 fixed)
    xs = jnp.array([[0.5], [0.0], [0.0]], dtype=jnp.float64)   # x0=0.5, rest overwritten by z
    us = jnp.zeros((K, nu), dtype=jnp.float64)
    ls = jnp.zeros((K, N, nx), dtype=jnp.float64)
    template_op = FixedStepPrimalDualTrajectory(tg=tg, xs=xs, us=us, ls=ls)

    # minimal nlgame mock: must provide fields used by compute_al_residual_struct
    cs = SimpleNamespace(tg=tg)
    constraints = GameConstraintGridMap(ineq_blocks=(), eq_blocks=())
    nlgame = SimpleNamespace(tg=tg, cs=cs, constraints=constraints, u_splits=u_splits, N=N)

    # minimal AL state (not used by fake residual pieces)
    alstate = SimpleNamespace()

    # ---- monkeypatched stationarity gradients (linear in xs/us) ----
    def fake_grad_aug_lagrangian_trajectory(nlgame_, op_, alstate_, **kwargs):
        # dL/dX shape (N, nt, nx), linear in xs
        # player 0: 1*xs, player 1: 2*xs
        dX0 = op_.xs[None, :, :] * 1.0
        dX1 = op_.xs[None, :, :] * 2.0
        dL_dX_all = jnp.concatenate([dX0, dX1], axis=0)

        # dL/dU shape (N, K, nu), linear in us
        # player 0: 3*us, player 1: 4*us
        dU0 = op_.us[None, :, :] * 3.0
        dU1 = op_.us[None, :, :] * 4.0
        dL_dU_all = jnp.concatenate([dU0, dU1], axis=0)
        return dL_dX_all, dL_dU_all

    monkeypatch.setattr(pdg_alsolver, "gradient_aug_lagrangian_trajectory", fake_grad_aug_lagrangian_trajectory)

    # ---- monkeypatched dynamics residual (linear in xs/us) ----
    def fake_dyn_res(cs_, op_, method):
        # D_k = x_k - x_{k+1} + 0.5*u_k0
        return op_.xs[:-1, :] - op_.xs[1:, :] + 0.5 * op_.us[:, 0:1]

    monkeypatch.setattr(pdg_alsolver.systypes, "residual_discrete_dynamics_trajectory", fake_dyn_res)

    # build some nontrivial z by packing from a nontrivial op
    xs2 = jnp.array([[0.5], [1.0], [-2.0]], dtype=jnp.float64)
    us2 = jnp.array([[0.2, -0.1, 0.3],
                     [0.4,  0.0, -0.2]], dtype=jnp.float64)
    ls2 = jnp.arange(K * N * nx, dtype=jnp.float64).reshape(K, N, nx) * 0.1  # arbitrary
    op2 = FixedStepPrimalDualTrajectory(tg=tg, xs=xs2, us=us2, ls=ls2)
    z0 = pdg_alsolver.pack_decision_vars_1d(op2)  # uses your existing packer (xs[1:], us, ls)

    # define G(z) for finite-diff (numpy expects numpy arrays)
    def G_np(z_np):
        z_jax = jnp.asarray(z_np)
        g = pdg_alsolver.compute_al_residual_flat_from_decision_vars(
            nlgame,
            z_jax,
            template_op,
            alstate,
            discretize_method="rk2",
            ineq_activation="altro",
        )
        return np.asarray(g)

    # autodiff Jacobian
    H_ad = pdg_alsolver.jacobian_al_residual_flat_autodiff(
        nlgame,
        z0,
        template_op,
        alstate,
        discretize_method="rk2",
        ineq_activation="altro",
        mode="jacfwd",
    )
    H_ad = np.asarray(H_ad)

    # finite-difference Jacobian
    H_fd = _finite_difference_jacobian(G_np, np.asarray(z0), eps=1e-5)

    np.testing.assert_allclose(H_ad, H_fd, atol=5e-4, rtol=5e-4)

    # disable float64 so other tests don't run under these conditions 
    # which can fail regression tests
    jax.config.update("jax_enable_x64", False)


def test_jacobian_al_residual_flat_structured_dynamics_matches_autodiff_on_linear_problem():
    """
    First structured-Jacobian slice should be exact when omitted terms are zero:
    linear dynamics, zero costs, and no auxiliary constraints.
    """
    nlgame, op, alstate = _make_linear_dynamics_only_al_problem(nt=5, dtype=jnp.float32)
    z = pdg_alsolver.pack_decision_vars_1d(op)

    H_ad = pdg_alsolver.jacobian_al_residual_flat_autodiff(
        nlgame,
        z,
        op,
        alstate,
        discretize_method="euler",
        ineq_activation="none",
        mode="jacfwd",
    )
    H_struct = pdg_alsolver.jacobian_al_residual_flat_structured(
        nlgame,
        z,
        op,
        alstate,
        discretize_method="euler",
        ineq_activation="none",
        include_second_order=False,
    )

    assert H_struct.shape == H_ad.shape
    np.testing.assert_allclose(np.asarray(H_struct), np.asarray(H_ad), rtol=2e-5, atol=2e-5)


def test_jacobian_al_residual_flat_structured_rejects_second_order_request():
    nlgame, op, alstate = _make_linear_dynamics_only_al_problem(nt=4, dtype=jnp.float32)
    z = pdg_alsolver.pack_decision_vars_1d(op)

    with pytest.raises(NotImplementedError, match="first-order dynamics"):
        pdg_alsolver.jacobian_al_residual_flat_structured(
            nlgame,
            z,
            op,
            alstate,
            discretize_method="euler",
            ineq_activation="none",
            include_second_order=True,
        )


@pytest.mark.benchmark(group="alsolver-jacobian-dynamics-001")
def test_jacobian_al_residual_flat_autodiff_dynamics_warm_perf(benchmark):
    nlgame, op, alstate = _make_linear_dynamics_only_al_problem(nt=12, dtype=jnp.float32)
    z = pdg_alsolver.pack_decision_vars_1d(op)

    def run():
        H = pdg_alsolver.jacobian_al_residual_flat_autodiff(
            nlgame,
            z,
            op,
            alstate,
            discretize_method="euler",
            ineq_activation="none",
            mode="jacfwd",
        )
        return H.block_until_ready()

    benchmark(run)


@pytest.mark.benchmark(group="alsolver-jacobian-dynamics-001")
def test_jacobian_al_residual_flat_structured_dynamics_warm_perf(benchmark):
    nlgame, op, alstate = _make_linear_dynamics_only_al_problem(nt=12, dtype=jnp.float32)
    z = pdg_alsolver.pack_decision_vars_1d(op)

    def run():
        H = pdg_alsolver.jacobian_al_residual_flat_structured(
            nlgame,
            z,
            op,
            alstate,
            discretize_method="euler",
            ineq_activation="none",
            include_second_order=False,
        )
        return H.block_until_ready()

    benchmark(run)


# ---------- FD helper (x64 + Richardson) ----------
def _fd_jacobian_richardson(f, z, eps=1e-5):
    z = np.asarray(z, dtype=np.float64)
    f0 = np.asarray(f(z), dtype=np.float64)
    m, n = f0.size, z.size

    def fd(e):
        J = np.zeros((m, n), dtype=np.float64)
        for j in range(n):
            zp = z.copy(); zp[j] += e
            zm = z.copy(); zm[j] -= e
            fp = np.asarray(f(zp), dtype=np.float64)
            fm = np.asarray(f(zm), dtype=np.float64)
            J[:, j] = (fp - fm) / (2.0 * e)
        return J

    J1 = fd(eps)
    J2 = fd(eps / 2.0)
    return (4.0 * J2 - J1) / 3.0


def test_jacobian_al_residual_flat_autodiff_matches_fd_integration():
    # Enable x64 for stable FD comparisons
    jax.config.update("jax_enable_x64", True)

    # ---- tiny problem dims ----
    tg = TimeGrid(nt=4, dt=0.2, t0=0.0)    # nt=4 -> K=3
    nt = tg.nt
    K = nt - 1
    nx = 2
    N = 2
    u_splits = jnp.array([1, 1], dtype=jnp.int32)
    nu = int(np.sum(np.asarray(u_splits)))

    # ---- linear continuous dynamics xdot = A x + B u ----
    A = jnp.array([[0.1, -0.2],
                   [0.3,  0.0]], dtype=jnp.float64)
    B = jnp.array([[1.0, 0.0],
                   [0.0, 2.0]], dtype=jnp.float64)

    def f_cont(t, x, u):
        return A @ x + B @ u

    # instantiate system (adjust if your constructor differs)
    cs = SampledContinuousSystemType1(tg=tg, nx=nx, nu=nu, dynamics=f_cont)

    # ---- quadratic costs, player-local control ----
    # player i uses u_i scalar (since u_splits=[1,1])
    Q0 = jnp.diag(jnp.array([1.0, 2.0], dtype=jnp.float64))
    R0 = jnp.array([[0.7]], dtype=jnp.float64)
    Qf0 = jnp.diag(jnp.array([1.5, 0.5], dtype=jnp.float64))

    Q1 = jnp.diag(jnp.array([0.5, 1.0], dtype=jnp.float64))
    R1 = jnp.array([[1.2]], dtype=jnp.float64)
    Qf1 = jnp.diag(jnp.array([0.8, 1.1], dtype=jnp.float64))

    def running0(t, x, u0):
        return 0.5 * (x @ (Q0 @ x)) + 0.5 * (u0 @ (R0 @ u0))
    def terminal0(t, x):
        return 0.5 * (x @ (Qf0 @ x))

    def running1(t, x, u1):
        return 0.5 * (x @ (Q1 @ x)) + 0.5 * (u1 @ (R1 @ u1))
    def terminal1(t, x):
        return 0.5 * (x @ (Qf1 @ x))

    # costs = [
    #     SimpleNamespace(running=running0, terminal=terminal0),
    #     SimpleNamespace(running=running1, terminal=terminal1),
    # ]
    costs = [
        PlayerCostSpecContinuous(running=running0, terminal=terminal0, control_domain=CostControlDomain.LOCAL, control_coupling=CostControlStructure.LOCAL_ONLY),
        PlayerCostSpecContinuous(running=running1, terminal=terminal1, control_domain=CostControlDomain.LOCAL, control_coupling=CostControlStructure.LOCAL_ONLY)
    ]

    # ---- one linear inequality constraint block active at all stage steps ----
    # c(t,x,u) = a^T x + b^T u - 0.1  <= 0
    a = jnp.array([0.2, -0.1], dtype=jnp.float64)
    b = jnp.array([0.05, 0.03], dtype=jnp.float64)

    def c_ineq(t, x, u):
        return a @ x + b @ u - 0.1  # scalar

    ineq_block = ConstraintBlockGridMap(
        tg=tg,
        func=c_ineq,
        cdim_out_step=1,
        active_steps=tuple(range(K)),
        iseq=False,
        terminal=False,
    )
    constraints = GameConstraintGridMap(ineq_blocks=(ineq_block,), eq_blocks=())

    # ---- AL state: nc_ineq = K * 1 ----
    lam_ineq = jnp.array([0.2, -0.1, 0.05], dtype=jnp.float64)  # length K
    rho_ineq = jnp.array([1.0,  1.5, 0.7], dtype=jnp.float64)
    alstate = altypes.JointAugmentedLagrangianState(
        lam_ineq=lam_ineq,
        rho_ineq=rho_ineq,
        lam_eq=jnp.zeros((0,), dtype=jnp.float64),
        rho_eq=jnp.zeros((0,), dtype=jnp.float64),
    )

    # ---- game ----
    nlgame = NonlinearGameType2(
        cs=cs,
        N=N,
        costs=costs,
        constraints=constraints,
        u_splits=u_splits,
    )

    # ---- initial primal-dual trajectory (x0 fixed) ----
    key = jax.random.PRNGKey(0)
    xs = jax.random.normal(key, (nt, nx), dtype=jnp.float64) * 0.2
    us = jax.random.normal(key, (K, nu), dtype=jnp.float64) * 0.1
    ls = jnp.zeros((K, N, nx), dtype=jnp.float64)   # start mu at zero
    op = FixedStepPrimalDualTrajectory(tg=tg, xs=xs, us=us, ls=ls)

    # ---- decision vector z and autodiff Jacobian ----
    z0 = pdg_alsolver.pack_decision_vars_1d(op).astype(jnp.float64)

    H_ad = pdg_alsolver.jacobian_al_residual_flat_autodiff(
        nlgame,
        z0,
        op,          # template_op
        alstate,
        discretize_method="euler",
        ineq_activation="none",   # keep smooth for FD agreement
        mode="jacfwd",
    )
    H_ad = np.asarray(H_ad, dtype=np.float64)

    # ---- FD Jacobian of the same packed residual ----
    def G_np(z_np):
        z_jax = jnp.asarray(z_np, dtype=jnp.float64)
        g = pdg_alsolver.compute_al_residual_flat_from_decision_vars(
            nlgame,
            z_jax,
            op,        # template_op
            alstate,
            discretize_method="euler",
            ineq_activation="none",
        )
        return np.asarray(g, dtype=np.float64)

    H_fd = _fd_jacobian_richardson(G_np, np.asarray(z0, dtype=np.float64), eps=1e-5)

    np.testing.assert_allclose(H_ad, H_fd, atol=2e-5, rtol=2e-5)

    # disable float64 so other tests don't run under these conditions 
    # which can fail regression tests
    jax.config.update("jax_enable_x64", False)

def test_residual_norm_empty_returns_zero():
    g = jnp.zeros((0,), dtype=jnp.float32)
    assert pdg_alsolver.residual_norm(g, "l1") == 0.0
    assert pdg_alsolver.residual_norm(g, "l2") == 0.0
    assert pdg_alsolver.residual_norm(g, "l1_mean") == 0.0
    assert pdg_alsolver.residual_norm(g, "l2_rms") == 0.0


def test_residual_norm_vanilla_l1_l2():
    g = jnp.array([1.0, -2.0, 3.0], dtype=jnp.float32)
    assert pdg_alsolver.residual_norm(g, "l1") == pytest.approx(6.0)
    assert pdg_alsolver.residual_norm(g, "l2") == pytest.approx(np.sqrt(14.0))


def test_residual_norm_l1_mean_and_l2_rms():
    g = jnp.array([1.0, -2.0, 3.0], dtype=jnp.float32)
    n = 3
    l1 = 6.0
    l2 = np.sqrt(14.0)

    assert pdg_alsolver.residual_norm(g, "l1_mean") == pytest.approx(l1 / n)
    assert pdg_alsolver.residual_norm(g, "l2_rms") == pytest.approx(l2 / np.sqrt(n))


def test_residual_norm_handles_non_vector_shape():
    g = jnp.array([[1.0, -2.0], [3.0, 4.0]], dtype=jnp.float32)  # size=4
    # l1 = 1+2+3+4 = 10
    # l2 = sqrt(1+4+9+16)=sqrt(30)
    assert pdg_alsolver.residual_norm(g, "l1") == pytest.approx(10.0)
    assert pdg_alsolver.residual_norm(g, "l2") == pytest.approx(np.sqrt(30.0))
    assert pdg_alsolver.residual_norm(g, "l1_mean") == pytest.approx(10.0 / 4.0)
    assert pdg_alsolver.residual_norm(g, "l2_rms") == pytest.approx(np.sqrt(30.0) / np.sqrt(4.0))


def test_residual_norm_rejects_unknown_kind():
    g = jnp.array([1.0], dtype=jnp.float32)
    with pytest.raises(ValueError):
        pdg_alsolver.residual_norm(g, "banana")  # type: ignore[arg-type]


def test_residual_norm_linf_vector():
    g = jnp.array([1.0, -3.5, 2.0], dtype=jnp.float32)
    assert pdg_alsolver.residual_norm(g, "linf") == pytest.approx(3.5)


def test_residual_norm_linf_matrix_flattens():
    g = jnp.array([[1.0, -2.0], [3.0, 4.0]], dtype=jnp.float32)
    # max abs entry is 4.0
    assert pdg_alsolver.residual_norm(g, "linf") == pytest.approx(4.0)


def _fake_nlgame_for_opt_test(*, nt: int, nx: int, u_splits) -> SimpleNamespace:
    u_splits = jnp.asarray(u_splits, dtype=jnp.int32)
    nu = int(jnp.sum(u_splits))
    N = int(u_splits.shape[0])
    return SimpleNamespace(nt=nt, nx=nx, nu=nu, N=N, u_splits=u_splits)


def test_optimality_violation_inf_ignores_cross_player_joint_control_entries(monkeypatch):
    """
    Regression test for the joint-control bug:

    Build a residual struct where each player's dLdU has:
      - small entries on the player's own control slice
      - huge entries on other players' slices

    Correct optimality metric MUST ignore those huge cross-player entries,
    because stationarity is only w.r.t. each player's local controls.
    """
    # Dimensions: N=2 players, K=1 step, nx=1, nu=2 (u_splits=[1,1])
    nlgame = _fake_nlgame_for_opt_test(nt=2, nx=1, u_splits=[1, 1])
    K = nlgame.nt - 1

    # stationarity w.r.t X[1:] is small
    dLdX = jnp.array([[[1e-4]], [[2e-4]]], dtype=jnp.float32)  # (N,K,nx)

    # stationarity w.r.t *joint* U has huge cross terms:
    # player 0 should only "care" about u0, but u1 entry is huge
    # player 1 should only "care" about u1, but u0 entry is huge
    dLdU = jnp.array(
        [
            [[1e-4, 1e+2]],   # player 0: local small, cross huge
            [[1e+2, 2e-4]],   # player 1: cross huge, local small
        ],
        dtype=jnp.float32,
    )  # (N,K,nu)

    dyn_res = jnp.array([[1e6]], dtype=jnp.float32)  # should be ignored anyway

    fake_res = SimpleNamespace(dLdX=dLdX, dLdU=dLdU, dyn_res=dyn_res)

    monkeypatch.setattr(pdg_alsolver, "compute_al_residual_struct_from_traj", lambda *a, **k: fake_res)

    out = pdg_alsolver.optimality_violation_inf(
        nlgame, SimpleNamespace(), SimpleNamespace(), discretize_method="rk2", ineq_activation="altro"
    )

    # Expected: max over packed stationarity:
    # - dLdX contributes max 2e-4
    # - local control components are [1e-4, 2e-4] (cross terms ignored)
    # so overall expected is 2e-4
    assert out == pytest.approx(2e-4, rel=1e-8, abs=1e-9)


def test_optimality_violation_inf_matches_packed_stationarity_exact(monkeypatch):
    """
    Ensure optimality_violation_inf equals max(abs(stationarity_part_of_packed_residual)),
    i.e., compute the reference explicitly using pack_al_residual_1d().
    """
    nlgame = _fake_nlgame_for_opt_test(nt=3, nx=2, u_splits=[2, 1, 3])  # N=3, nu=6
    K = nlgame.nt - 1
    N, nx, nu = nlgame.N, nlgame.nx, nlgame.nu

    # Make a residual struct with arbitrary values
    dLdX = jnp.arange(N * K * nx, dtype=jnp.float32).reshape(N, K, nx) * 0.01
    dLdU = jnp.arange(N * K * nu, dtype=jnp.float32).reshape(N, K, nu) * 0.001
    dyn_res = jnp.ones((K, nx), dtype=jnp.float32) * 123.0  # ignored

    fake_res = SimpleNamespace(dLdX=dLdX, dLdU=dLdU, dyn_res=dyn_res)
    monkeypatch.setattr(pdg_alsolver, "compute_al_residual_struct_from_traj", lambda *a, **k: fake_res)

    # Function under test
    out = pdg_alsolver.optimality_violation_inf(
        nlgame, SimpleNamespace(), SimpleNamespace(), 
        discretize_method="rk2",
        ineq_activation="altro"
    )

    # Reference: pack then strip dynamics block
    g_flat = pdg_alsolver.pack_al_residual_1d(fake_res, u_splits=nlgame.u_splits)

    dyn_block = (K * nx)
    sta = g_flat[:-dyn_block] if dyn_block > 0 else g_flat
    ref = float(jnp.max(jnp.abs(sta))) if sta.size else 0.0

    assert out == pytest.approx(ref, rel=0.0, abs=0.0)


def test_optimality_violation_inf_ignores_dyn_res_even_if_huge(monkeypatch):
    """
    The metric must ignore dynamics feasibility residuals (dyn_res),
    even if they are enormous.
    """
    nlgame = _fake_nlgame_for_opt_test(nt=2, nx=1, u_splits=[1])

    fake_res = SimpleNamespace(
        dLdX=jnp.array([[[1e-3]]], dtype=jnp.float32),
        dLdU=jnp.array([[[2e-3]]], dtype=jnp.float32),
        dyn_res=jnp.array([[1e9]], dtype=jnp.float32),
    )
    monkeypatch.setattr(pdg_alsolver, "compute_al_residual_struct_from_traj", lambda *a, **k: fake_res)

    out = pdg_alsolver.optimality_violation_inf(
        nlgame, SimpleNamespace(), SimpleNamespace(),
        discretize_method="rk2",
        ineq_activation="altro"
    )
    assert out == pytest.approx(2e-3, rel=1e-8, abs=1e-9)


def test_optimality_violation_inf_handles_empty_stationarity(monkeypatch):
    """
    If there are no decision steps (K=0, i.e. nt=1), stationarity is empty.
    The metric should return 0.0 and not crash packing.
    """

    # Choose any N,nx,nu you like; emptiness comes from K=0 (nt=1)
    N, nt, nx, nu = 1, 1, 2, 3
    K = nt - 1
    u_splits = jnp.array([nu], dtype=jnp.int32)

    nlgame = SimpleNamespace(nt=nt, nx=nx, nu=nu, N=N, u_splits=u_splits)

    fake_res = SimpleNamespace(
        dLdX=jnp.zeros((N, K, nx), dtype=jnp.float32),   # (1,0,2)
        dLdU=jnp.zeros((N, K, nu), dtype=jnp.float32),   # (1,0,3)
        dyn_res=jnp.zeros((K, nx), dtype=jnp.float32),   # (0,2)
    )

    monkeypatch.setattr(pdg_alsolver, "compute_al_residual_struct_from_traj", lambda *a, **k: fake_res)

    out = pdg_alsolver.optimality_violation_inf(
        nlgame, SimpleNamespace(), SimpleNamespace(),
        discretize_method="rk2",
        ineq_activation="altro")
    assert out == pytest.approx(0.0, rel=0.0, abs=0.0)


def test_solve_newton_system_tikhonov_basic_solution():
    """
    Well-conditioned system should solve with reg=reg0.
    """
    H = jnp.array([[3.0, 1.0],
                   [1.0, 2.0]])
    g = jnp.array([1.0, -1.0])

    out = pdg_alsolver.solve_newton_system_tikhonov(H, g, reg0=1e-12, reg1_min=1e-12, reg_increase=10.0, reg_max=1e6, max_iters=64)

    assert out.ok
    assert out.dz is not None
    assert out.reg == pytest.approx(1e-12)

    # Check it actually satisfies (H + reg I) dz ≈ -g
    dz = out.dz
    resid = (H + out.reg * jnp.eye(2, dtype=H.dtype)) @ dz + g
    np.testing.assert_allclose(np.asarray(resid), np.zeros(2), atol=1e-6, rtol=1e-6)

def test_solve_newton_system_tikhonov_singular_succeeds_with_regularization():
    """
    Singular H (rank-1) cannot be solved at reg=0, but becomes invertible with reg>0.
    """
    H = jnp.array([[1.0, 2.0],
                   [2.0, 4.0]])  # singular
    g = jnp.array([1.0, 1.0])

    out = pdg_alsolver.solve_newton_system_tikhonov(H, g, reg0=0.0, reg1_min=1e-12, reg_increase=10.0, reg_max=1e6, max_iters=64)

    assert out.ok
    assert out.dz is not None
    assert out.reg > 0.0

    # verify regularized equation holds
    dz = out.dz
    resid = (H + out.reg * jnp.eye(2, dtype=H.dtype)) @ dz + g
    np.testing.assert_allclose(np.asarray(resid), np.zeros(2), atol=1e-6, rtol=1e-6)


def test_solve_newton_system_tikhonov_reg_increases_until_success(monkeypatch):
    """
    Force first attempt to "fail" by monkeypatching jnp.linalg.solve to return non-finite dz
    on the first call, then a valid solve on the second call.
    This verifies reg escalation behavior deterministically.
    """
    call = {"n": 0}
    real_solve = jnp.linalg.solve

    def fake_solve(A, b):
        call["n"] += 1
        if call["n"] == 1:
            return jnp.array([jnp.nan, jnp.nan], dtype=A.dtype)
        return real_solve(A, b)

    monkeypatch.setattr(pdg_alsolver.jnp.linalg, "solve", fake_solve)

    H = jnp.array([[2.0, 0.0],
                   [0.0, 3.0]])
    g = jnp.array([1.0, -2.0])

    out = pdg_alsolver.solve_newton_system_tikhonov(H, g, reg0=1e-6, reg1_min=1e-12, reg_increase=10.0, reg_max=1e3, max_iters=64)

    assert out.ok
    assert call["n"] >= 2
    # reg should have been increased at least once
    assert out.reg == pytest.approx(1e-5)


def test_solve_newton_system_tikhonov_fails_when_reg_exceeds_max(monkeypatch):
    """
    Force all solves to fail (return non-finite). Ensure we stop when reg > reg_max.
    """
    def always_bad_solve(A, b):
        return jnp.array([jnp.nan, jnp.nan], dtype=A.dtype)

    monkeypatch.setattr(pdg_alsolver.jnp.linalg, "solve", always_bad_solve)

    H = jnp.array([[2.0, 0.0],
                   [0.0, 3.0]])
    g = jnp.array([1.0, -2.0])

    out = pdg_alsolver.solve_newton_system_tikhonov(H, g, reg0=1e-2, reg1_min=1e-12, reg_increase=10.0, reg_max=1e-1, max_iters=64)

    assert not out.ok
    assert out.dz is None
    assert out.reg > 1e-1

def test_linesearch_fixedratio_accepts_full_step_when_residual_decreases():
    """
    G(z) = z. Starting at z0, dz = -z0 is perfect Newton direction.
    alpha=1 gives G(z0 + dz)=0 so should accept immediately.
    """
    def G(z):
        return z

    z0 = jnp.array([1.0, -2.0])
    dz = -z0
    g0 = G(z0)

    out = pdg_alsolver.backtracking_linesearch_fixedratio(
        G, z0, dz, g0, alpha0=1.0, beta=0.5, max_iters=10, accept_ratio=0.99
    )

    assert out.accepted
    assert out.alpha == pytest.approx(1.0)
    assert out.ls_iters == 1
    assert out.best_norm == pytest.approx(0.0, abs=1e-12)


def test_linesearch_fixedratio_backtracks_then_accepts():
    """
    Construct a residual where alpha=1 overshoots, but smaller alpha helps.

    Let z scalar. Define G(z) = z + 10 z^3.
    Starting at z0=0.5 and dz=-0.5, alpha=1 -> z=0 => residual 0 (actually accepts).
    So instead choose dz that overshoots: dz=-1.0, alpha=1 -> z=-0.5 which increases cubic term.
    Smaller alpha reduces magnitude.
    """
    def G(z):
        return z + 10.0 * (z ** 3)

    z0 = jnp.array([0.5])
    dz = jnp.array([-1.0])   # overshoot past 0
    g0 = G(z0)

    out = pdg_alsolver.backtracking_linesearch_fixedratio(
        G, z0, dz, g0, alpha0=1.0, beta=0.5, max_iters=10, accept_ratio=0.99
    )

    assert out.ls_iters >= 1
    assert out.best_norm <= float(jnp.linalg.norm(g0))  # should find something no worse
    # likely accepts at some alpha < 1.0
    assert out.best_alpha <= 1.0
    assert out.best_alpha > 0.0


def test_linesearch_fixedratio_rejects_when_direction_is_bad():
    """
    If dz points uphill for residual norm, line search should fail to meet accept criterion.

    Use G(z)=z, pick dz = +z0 so any alpha increases norm.
    """
    def G(z):
        return z

    z0 = jnp.array([1.0, 2.0])
    dz = +z0
    g0 = G(z0)

    out = pdg_alsolver.backtracking_linesearch_fixedratio(
        G, z0, dz, g0, alpha0=1.0, beta=0.5, max_iters=5, accept_ratio=0.99
    )

    assert not out.accepted
    # best_alpha should likely remain 0.0 (no improvement), best_norm = ||g0||
    assert out.best_norm == pytest.approx(float(jnp.linalg.norm(g0)))


def test_linesearch_fixedratio_accept_ratio_controls_strictness():
    """
    Show that stricter accept_ratio can reject steps that a looser criterion accepts.
    """
    def G(z):
        return z

    z0 = jnp.array([1.0])
    dz = jnp.array([-0.1])  # small improvement

    g0 = G(z0)
    # trial at alpha=1 gives norm 0.9, which is a 10% reduction

    out_strict = pdg_alsolver.backtracking_linesearch_fixedratio(
        G, z0, dz, g0, alpha0=1.0, beta=0.5, max_iters=1, accept_ratio=0.5
    )
    assert not out_strict.accepted  # requires 50% reduction, not achieved

    out_loose = pdg_alsolver.backtracking_linesearch_fixedratio(
        G, z0, dz, g0, alpha0=1.0, beta=0.5, max_iters=1, accept_ratio=0.95
    )
    assert out_loose.accepted
    assert out_loose.alpha == pytest.approx(1.0)


# def _make_linear_G(A: jnp.ndarray, b: jnp.ndarray):
#     """
#     Linear residual map G(z) = A z + b.
#     Useful because we can reason about norms and monotonicity.
#     """
#     def G(z):
#         return A @ z + b
#     return G


def test_linesearch_armijo_rejects_invalid_params():
    G = lambda z: z
    z0 = jnp.array([0.0])
    dz = jnp.array([1.0])
    g0 = G(z0)

    with pytest.raises(ValueError, match="alpha0"):
        pdg_alsolver.backtracking_linesearch_armijo(
            G, z0, dz, g0, alpha0=0.0, tau=0.5, beta=0.1, max_iters=5, normkind="l1_mean"
        )

    with pytest.raises(ValueError, match="tau"):
        pdg_alsolver.backtracking_linesearch_armijo(
            G, z0, dz, g0, alpha0=1.0, tau=1.0, beta=0.1, max_iters=5, normkind="l1_mean"
        )

    with pytest.raises(ValueError, match="beta"):
        pdg_alsolver.backtracking_linesearch_armijo(
            G, z0, dz, g0, alpha0=1.0, tau=0.5, beta=0.5, max_iters=5, normkind="l1_mean"
        )

    with pytest.raises(ValueError, match="max_iters"):
        pdg_alsolver.backtracking_linesearch_armijo(
            G, z0, dz, g0, alpha0=1.0, tau=0.5, beta=0.1, max_iters=0, normkind="l1_mean"
        )

    with pytest.raises(ValueError, match="norm must be"):
        pdg_alsolver.backtracking_linesearch_armijo(
            G, z0, dz, g0, alpha0=1.0, tau=0.5, beta=0.1, max_iters=5, normkind="l0"
        )


def test_linesearch_armijo_accepts_immediately_for_exact_root_step_l2():
    # G(z) = z - 1, z0=0, dz=1 -> z_trial = 1 gives G=0
    G = lambda z: z - 1.0
    z0 = jnp.array([0.0])
    dz = jnp.array([1.0])
    g0 = G(z0)

    out = pdg_alsolver.backtracking_linesearch_armijo(
        G, z0, dz, g0,
        alpha0=1.0, tau=0.5, beta=0.25, max_iters=10, normkind="l2"
    )

    assert out.accepted is True
    assert out.alpha == pytest.approx(1.0)
    assert out.ls_iters == 1
    assert out.best_alpha == pytest.approx(1.0)
    assert out.best_norm == pytest.approx(0.0, abs=1e-12)


def test_linesearch_armijo_backtracks_until_armijo_satisfied():
    # Construct a case where alpha=1 overshoots, but smaller alpha helps.
    # G(z) = z (so norm is |z|). z0=1, dz=-2:
    # alpha=1 -> z=-1, ||G||=1 (no improvement)
    # alpha=0.5 -> z=0,  ||G||=0 (improvement)
    G = lambda z: z
    z0 = jnp.array([1.0])
    dz = jnp.array([-2.0])
    g0 = G(z0)

    out = pdg_alsolver.backtracking_linesearch_armijo(
        G, z0, dz, g0,
        alpha0=1.0, tau=0.5, beta=0.25, max_iters=10, normkind="l2"
    )

    assert out.accepted is True
    # Should accept at alpha=0.5 on the second trial
    assert out.alpha == pytest.approx(0.5)
    assert out.ls_iters == 2
    assert out.best_alpha == pytest.approx(0.5)
    assert out.best_norm == pytest.approx(0.0, abs=1e-12)


def test_linesearch_armijo_returns_best_even_if_not_accepted():
    # Make a G where norm never decreases for any alpha (constant residual)
    G = lambda z: jnp.array([1.0, -2.0])
    z0 = jnp.array([0.0])
    dz = jnp.array([1.0])
    g0 = G(z0)

    out = pdg_alsolver.backtracking_linesearch_armijo(
        G, z0, dz, g0,
        alpha0=1.0, tau=0.5, beta=0.25, max_iters=4, normkind="l2"
    )

    assert out.accepted is False
    assert out.alpha == 0.0  # per API when not accepted
    # best_alpha stays 0.0 because norm never improves
    assert out.best_alpha == pytest.approx(0.0)
    assert out.best_norm == pytest.approx(float(jnp.linalg.norm(g0, ord=2)))
    assert out.ls_iters == 4


def test_linesearch_armijo_l1_mean_norm_is_scaled_by_length():
    # L1 norm is mean absolute value: ||g||_1 / len(g)
    # For g=[2,-2], L1/len = (4)/2 = 2
    G = lambda z: jnp.array([2.0, -2.0])
    z0 = jnp.array([0.0])
    dz = jnp.array([1.0])  # unused
    g0 = G(z0)

    out = pdg_alsolver.backtracking_linesearch_armijo(
        G, z0, dz, g0,
        alpha0=1.0, tau=0.5, beta=0.25, max_iters=1, normkind="l1_mean"
    )

    # One trial; never accepted because constant, but we can verify computed norm via best_norm
    assert out.accepted is False
    assert out.best_norm == pytest.approx(2.0)  # (|2|+|−2|)/2 = 2


def test_linesearch_armijo_l1_accepts_when_armijo_holds_with_scaling():
    # Use G(z) = [z, z] so L1/len = |z|.
    # z0=1, dz=-1, alpha=1 => z=0, norm=0 => accept.
    G = lambda z: jnp.array([z[0], z[0]])
    z0 = jnp.array([1.0])
    dz = jnp.array([-1.0])
    g0 = G(z0)

    out = pdg_alsolver.backtracking_linesearch_armijo(
        G, z0, dz, g0,
        alpha0=1.0, tau=0.5, beta=0.25, max_iters=5, normkind="l1_mean"
    )

    assert out.accepted is True
    assert out.alpha == pytest.approx(1.0)
    assert out.best_norm == pytest.approx(0.0, abs=1e-12)
    assert out.ls_iters == 1


def test_linesearch_armijo_becomes_easier_as_alpha_shrinks_eventually_accepts():
    """
    Construct a scenario where:
      - The trial residual norm decreases only slightly for alpha=1 (not enough for Armijo),
      - But as alpha shrinks, the Armijo RHS approaches ||g0||, and the same slight decrease
        becomes sufficient, so the step is eventually accepted.

    We define a 1D residual map that is almost flat in the direction dz:
        G(z) = 1 + eps*z
    With z0=0, dz=-1, the residual at alpha is:
        g_trial = 1 - eps*alpha
    So g_norm_trial = 1 - eps*alpha (tiny decrease).

    Armijo condition:
        1 - eps*alpha <= (1 - beta*alpha) * 1  =>  eps >= beta
    That would never accept if eps < beta, *except* we choose a different construction:

    Instead we use:
        G(z) = 1 - eps + eps * exp(-z^2)   (smooth, near-constant around z=0)
    and pick dz so that alpha=1 moves to a point where exp(-z^2) drops a tiny amount,
    giving a tiny decrease. For smaller alpha, z is closer to 0, so decrease shrinks,
    BUT the Armijo RHS also loosens linearly with alpha and can be satisfied.

    In practice, to make this robust and simple, we hard-code a G_of_z that returns:
      - g0 norm = 1.0 at z0
      - g_trial norms that are slightly less than 1.0 for all alpha > 0,
        but only *just* less, so alpha-dependent Armijo is needed.
    """
    # Define a "nearly constant" residual norm: g_norm(alpha) = 1 - eps (independent of alpha),
    # but only after moving at all. This mimics numerical noise floor improvements.
    eps = 1e-6

    def G(z):
        # If we haven't moved (exactly), return 1.0; else return 1.0 - eps.
        moved = jnp.any(z != 0.0)
        return jnp.where(moved, jnp.array([1.0 - eps]), jnp.array([1.0]))

    z0 = jnp.array([0.0])
    dz = jnp.array([1.0])
    g0 = G(z0)

    # With a fixed accept_ratio rule, this would basically never accept unless accept_ratio is ~1.
    # With Armijo (1 - alpha*beta), it will accept once alpha is small enough that:
    #   (1 - alpha*beta) >= 1 - eps  => alpha <= eps / beta
    beta = 0.25
    tau = 0.5

    out = pdg_alsolver.backtracking_linesearch_armijo(
        G, z0, dz, g0,
        alpha0=1.0,
        tau=tau,
        beta=beta,
        max_iters=60,
        normkind="l2",
    )

    assert out.accepted is True

    # It should not accept at alpha=1, because 1-eps <= 1-beta is false for beta=0.25.
    # It should accept only after alpha shrinks below eps/beta.
    # eps/beta = 4e-6. With tau=0.5, alpha sequence is 1, 0.5, 0.25, ..., so it needs
    # about n s.t. 2^-n <= 4e-6 => n >= log2(2.5e5) ~ 17.9 -> 18 steps.
    assert out.ls_iters >= 10  # loose lower bound to avoid brittleness
    assert out.alpha > 0.0
    expected_norm = float(jnp.asarray(1.0 - eps, dtype=jnp.float32))
    assert out.best_norm == pytest.approx(expected_norm, rel=0.0, abs=0.0)


def test_newton_step_autodiff_rejects_when_linear_solve_fails(monkeypatch):
    tg = "dummy_tg"
    op = SimpleNamespace(tg=tg)
    nlgame = SimpleNamespace(tg=tg)
    alstate = SimpleNamespace()

    # pack -> z0
    monkeypatch.setattr(pdg_alsolver, "pack_decision_vars_1d", lambda op_: jnp.array([1.0, 2.0]))
    # residual and jacobian (won't matter much)
    monkeypatch.setattr(pdg_alsolver, "compute_al_residual_flat_from_decision_vars", lambda *args, **kw: jnp.array([3.0, 4.0]))
    monkeypatch.setattr(pdg_alsolver, "jacobian_al_residual_flat_autodiff", lambda *args, **kw: jnp.eye(2))

    # force solve failure
    monkeypatch.setattr(
        pdg_alsolver,
        "solve_newton_system_tikhonov",
        lambda H, g, **kw: altypes.RegularizedSolveResult(dz=None, reg=1.0, ok=False),
    )

    op_new, diag = pdg_alsolver.newton_step_autodiff(nlgame, op, alstate,
        step_rtol = 1e-7,
        step_atol = 1e-8,
        discretize_method = "rk2",
        ineq_activation = "altro",
        reg0 = 0.0,
        reg1_min = 1e-12,
        reg_increase = 10.0,
        reg_max = 1e8,
        reg_max_iters = 64,
        ls_alpha0 = 1.0,
        ls_tau = 0.5,
        ls_beta = 0.25,
        ls_max_iters = 20,
        normkind="l1_mean",
    )

    assert op_new is op
    assert not diag.accepted
    assert not diag.solve_ok
    assert diag.reg == pytest.approx(1.0)


def test_newton_step_autodiff_accepts_noop_if_step_tiny(monkeypatch):
    tg = "dummy_tg"
    op = SimpleNamespace(tg=tg)
    nlgame = SimpleNamespace(tg=tg)
    alstate = SimpleNamespace()

    monkeypatch.setattr(pdg_alsolver, "pack_decision_vars_1d", lambda op_: jnp.array([1.0, 2.0]))
    monkeypatch.setattr(pdg_alsolver, "compute_al_residual_flat_from_decision_vars", lambda *args, **kw: jnp.array([1.0, 0.0]))
    monkeypatch.setattr(pdg_alsolver, "jacobian_al_residual_flat_autodiff", lambda *args, **kw: jnp.eye(2))

    # tiny dz
    tiny = jnp.array([1e-20, 0.0])
    monkeypatch.setattr(
        pdg_alsolver,
        "solve_newton_system_tikhonov",
        lambda H, g, **kw: altypes.RegularizedSolveResult(dz=tiny, reg=0.0, ok=True),
    )

    # should not even call line search; but if it did, fail test
    def _fail_ls(*args, **kw):
        raise AssertionError("line search should not be called for tiny step")
    monkeypatch.setattr(pdg_alsolver, "backtracking_linesearch_armijo", _fail_ls)

    op_new, diag = pdg_alsolver.newton_step_autodiff(nlgame, op, alstate,
        step_rtol = 1e-7,
        step_atol = 1e-8,
        discretize_method = "rk2",
        ineq_activation = "altro",
        reg0 = 0.0,
        reg1_min = 1e-12,
        reg_increase = 10.0,
        reg_max = 1e8,
        reg_max_iters = 64,
        ls_alpha0 = 1.0,
        ls_tau=0.5,
        ls_beta = 0.25,
        ls_max_iters = 20,
        normkind="l1_mean",                                       
    )

    assert op_new is op
    assert diag.accepted
    assert diag.alpha == 0.0
    assert diag.step_norm < 1e-12


def test_newton_step_autodiff_rejects_when_line_search_rejects(monkeypatch):
    tg = "dummy_tg"
    op = SimpleNamespace(tg=tg)
    nlgame = SimpleNamespace(tg=tg)
    alstate = SimpleNamespace()

    z0 = jnp.array([1.0, 2.0])
    monkeypatch.setattr(pdg_alsolver, "pack_decision_vars_1d", lambda op_: z0)
    monkeypatch.setattr(pdg_alsolver, "compute_al_residual_flat_from_decision_vars", lambda *args, **kw: jnp.array([1.0, 0.0]))
    monkeypatch.setattr(pdg_alsolver, "jacobian_al_residual_flat_autodiff", lambda *args, **kw: jnp.eye(2))

    dz = jnp.array([-1.0, -2.0])
    monkeypatch.setattr(
        pdg_alsolver,
        "solve_newton_system_tikhonov",
        lambda H, g, **kw: altypes.RegularizedSolveResult(dz=dz, reg=0.0, ok=True),
    )

    monkeypatch.setattr(
        pdg_alsolver,
        "backtracking_linesearch_armijo",
        lambda *args, **kw: altypes.LineSearchResult(
            accepted=False,
            alpha=0.0,
            g_norm_trial=999.0,
            ls_iters=5,
            best_alpha=0.25,
            best_norm=0.9,
        ),
    )

    op_new, diag = pdg_alsolver.newton_step_autodiff(nlgame, op, alstate,
        step_rtol = 1e-7,
        step_atol = 1e-8,
        discretize_method = "rk2",
        ineq_activation = "altro",
        reg0 = 0.0,
        reg1_min = 1e-12,
        reg_increase = 10.0,
        reg_max = 1e8,
        reg_max_iters = 64,
        ls_alpha0 = 1.0,
        ls_tau = 0.5,
        ls_beta = 0.25,
        ls_max_iters = 20,
        normkind="l1_mean",
    )

    assert op_new is op
    assert not diag.accepted
    assert diag.ls_iters == 5


def test_newton_step_autodiff_accepts_and_unpacks_on_success(monkeypatch):
    tg = "dummy_tg"
    op = SimpleNamespace(tg=tg)
    nlgame = SimpleNamespace(tg=tg)
    alstate = SimpleNamespace()

    z0 = jnp.array([1.0, 2.0])
    dz = jnp.array([-0.5, -0.5])

    monkeypatch.setattr(pdg_alsolver, "pack_decision_vars_1d", lambda op_: z0)
    # residual norm at z0
    monkeypatch.setattr(pdg_alsolver, "compute_al_residual_flat_from_decision_vars", lambda *args, **kw: jnp.array([3.0, 4.0]))
    monkeypatch.setattr(pdg_alsolver, "jacobian_al_residual_flat_autodiff", lambda *args, **kw: jnp.eye(2))
    monkeypatch.setattr(
        pdg_alsolver,
        "solve_newton_system_tikhonov",
        lambda H, g, **kw: altypes.RegularizedSolveResult(dz=dz, reg=1e-6, ok=True),
    )
    monkeypatch.setattr(
        pdg_alsolver,
        "backtracking_linesearch_armijo",
        lambda *args, **kw: altypes.LineSearchResult(
            accepted=True,
            alpha=1.0,
            g_norm_trial=0.1,
            ls_iters=1,
            best_alpha=1.0,
            best_norm=0.1,
        ),
    )

    # verify unpack called with z_new = z0 + dz
    called = {"z": None}
    def fake_unpack(z_new, template, check_length=True):
        called["z"] = np.asarray(z_new)
        return "OP_NEW"
    monkeypatch.setattr(pdg_alsolver, "unpack_decision_vars", fake_unpack)

    op_new, diag = pdg_alsolver.newton_step_autodiff(nlgame, op, alstate,
        step_rtol = 1e-7,
        step_atol = 1e-8,
        discretize_method = "rk2",
        ineq_activation = "altro",
        reg0 = 0.0,
        reg1_min = 1e-12,
        reg_increase = 10.0,
        reg_max = 1e8,
        reg_max_iters = 64,
        ls_alpha0 = 1.0,
        ls_tau = 0.5,
        ls_beta = 0.25,
        ls_max_iters = 20,
        normkind="l1_mean",
    )

    assert op_new == "OP_NEW"
    np.testing.assert_allclose(called["z"], np.asarray(z0 + dz))
    assert diag.accepted
    assert diag.alpha == pytest.approx(1.0)
    assert diag.reg == pytest.approx(1e-6)


def test_newton_step_autodiff_reduces_residual_float32():
    # ---- small problem ----
    tg = TimeGrid(nt=4, dt=0.2, t0=0.0)
    nt = tg.nt
    K = nt - 1
    nx = 2
    N = 2
    u_splits = jnp.array([1, 1], dtype=jnp.int32)
    nu = int(np.sum(np.asarray(u_splits)))

    dtype = jnp.float32

    # ---- linear continuous dynamics ----
    A = jnp.array([[0.1, -0.2],
                   [0.3,  0.0]], dtype=dtype)
    B = jnp.array([[1.0, 0.0],
                   [0.0, 2.0]], dtype=dtype)

    def f_cont(t, x, u):
        return A @ x + B @ u

    cs = SampledContinuousSystemType1(tg=tg, nx=nx, nu=nu, dynamics=f_cont)

    # ---- quadratic costs ----
    Q0 = jnp.diag(jnp.array([1.0, 2.0], dtype=dtype))
    R0 = jnp.array([[0.7]], dtype=dtype)
    Qf0 = jnp.diag(jnp.array([1.5, 0.5], dtype=dtype))

    Q1 = jnp.diag(jnp.array([0.5, 1.0], dtype=dtype))
    R1 = jnp.array([[1.2]], dtype=dtype)
    Qf1 = jnp.diag(jnp.array([0.8, 1.1], dtype=dtype))

    def running0(t, x, u0):
        return 0.5 * (x @ (Q0 @ x)) + 0.5 * (u0 @ (R0 @ u0))
    def terminal0(t, x):
        return 0.5 * (x @ (Qf0 @ x))

    def running1(t, x, u1):
        return 0.5 * (x @ (Q1 @ x)) + 0.5 * (u1 @ (R1 @ u1))
    def terminal1(t, x):
        return 0.5 * (x @ (Qf1 @ x))

    costs = [
        PlayerCostSpecContinuous(running=running0, terminal=terminal0, control_domain=CostControlDomain.LOCAL, control_coupling=CostControlStructure.LOCAL_ONLY),
        PlayerCostSpecContinuous(running=running1, terminal=terminal1, control_domain=CostControlDomain.LOCAL, control_coupling=CostControlStructure.LOCAL_ONLY)
    ]

    # ---- no auxiliary constraints ----
    constraints = GameConstraintGridMap(ineq_blocks=(), eq_blocks=())
    alstate = altypes.JointAugmentedLagrangianState(
        lam_ineq=jnp.zeros((0,), dtype=dtype),
        rho_ineq=jnp.zeros((0,), dtype=dtype),
        lam_eq=jnp.zeros((0,), dtype=dtype),
        rho_eq=jnp.zeros((0,), dtype=dtype),
    )

    nlgame = NonlinearGameType2(
        cs=cs, N=N, costs=costs, constraints=constraints, u_splits=u_splits
    )

    # ---- initial trajectory (deterministic, not random) ----
    # Deterministic values reduce flakiness vs PRNG float32 differences.
    xs = (jnp.arange(nt * nx, dtype=dtype).reshape(nt, nx) - 2.0) * 0.1
    us = (jnp.arange(K * nu, dtype=dtype).reshape(K, nu) - 1.0) * 0.05
    ls = jnp.zeros((K, N, nx), dtype=dtype)
    op0 = FixedStepPrimalDualTrajectory(tg=tg, xs=xs, us=us, ls=ls)

    z0 = pdg_alsolver.pack_decision_vars_1d(op0)
    g0 = pdg_alsolver.compute_al_residual_flat_from_decision_vars(
        nlgame, z0, op0, alstate, discretize_method="euler", ineq_activation="none"
    )
    g0_norm = float(jnp.linalg.norm(g0))
    assert g0_norm > 0.0

    op1, diag = pdg_alsolver.newton_step_autodiff(
        nlgame, op0, alstate,
        step_rtol = 1e-7,
        step_atol = 1e-8,
        discretize_method="euler",
        ineq_activation="none",
        ls_max_iters=12,
        reg0=0.0,
        reg1_min=1e-6,          # float32-friendly min reg1
        reg_increase = 10.0,
        reg_max = 1e8,
        reg_max_iters = 64,
        ls_alpha0 = 1.0,
        ls_tau = 0.5,
        ls_beta = 0.25,
        normkind = "l1_mean",
    )

    assert diag.solve_ok
    assert diag.accepted
    assert diag.alpha > 0.0

    z1 = pdg_alsolver.pack_decision_vars_1d(op1)
    g1 = pdg_alsolver.compute_al_residual_flat_from_decision_vars(
        nlgame, z1, op0, alstate, discretize_method="euler", ineq_activation="none"
    )
    g1_norm = float(jnp.linalg.norm(g1))

    # Require a clear decrease (but not absurdly strict)
    assert g1_norm <= 0.5 * g0_norm


def _stub_game_and_op_same_tg():
    tg = TimeGrid(nt=4, dt=0.1, t0=0.0)
    nlgame = SimpleNamespace(tg=tg)
    op = SimpleNamespace(tg=tg)
    alstate = SimpleNamespace()
    return nlgame, op, alstate


def test_newton_step_autodiff_rejects_invalid_linesearch_params(monkeypatch):
    nlgame, op, alstate = _stub_game_and_op_same_tg()

    # Should fail before touching pack/residual
    monkeypatch.setattr(pdg_alsolver, "pack_decision_vars_1d", lambda *_: (_ for _ in ()).throw(AssertionError("should not call pack")))

    with pytest.raises(ValueError, match="tau"):
        pdg_alsolver.newton_step_autodiff(
            nlgame, op, alstate,
            step_rtol = 1e-7,
            step_atol = 1e-8,
            discretize_method="euler",
            ineq_activation="none",
            reg0=0.0, reg1_min=1e-6, reg_increase=10.0, reg_max=1e2, reg_max_iters=10,
            ls_alpha0=1.0, ls_tau=1.5, ls_beta=0.25, ls_max_iters=10, normkind="l1_mean",
        )


def test_newton_step_autodiff_rejects_invalid_reg_params(monkeypatch):
    nlgame, op, alstate = _stub_game_and_op_same_tg()
    monkeypatch.setattr(pdg_alsolver, "pack_decision_vars_1d", lambda *_: (_ for _ in ()).throw(AssertionError("should not call pack")))

    with pytest.raises(ValueError, match="reg_increase"):
        pdg_alsolver.newton_step_autodiff(
            nlgame, op, alstate,
            step_rtol = 1e-7,
            step_atol = 1e-8,
            discretize_method="euler",
            ineq_activation="none",
            reg0=0.0, reg1_min=1e-6, reg_increase=1.0, reg_max=1e2, reg_max_iters=10,
            ls_alpha0=1.0, ls_tau=0.5, ls_beta=0.25, ls_max_iters=10, normkind="l1mean",
        )


def test_newton_step_autodiff_rejects_timegrid_mismatch(monkeypatch):
    tg_game = TimeGrid(nt=4, dt=0.1, t0=0.0)
    tg_op = TimeGrid(nt=4, dt=0.1, t0=1.0)
    nlgame = SimpleNamespace(tg=tg_game)
    op = SimpleNamespace(tg=tg_op)
    alstate = SimpleNamespace()

    monkeypatch.setattr(pdg_alsolver, "pack_decision_vars_1d", lambda *_: (_ for _ in ()).throw(AssertionError("should not call pack")))

    with pytest.raises(ValueError, match="TimeGrid mismatch"):
        pdg_alsolver.newton_step_autodiff(
            nlgame, op, alstate,
            step_rtol = 1e-7,
            step_atol = 1e-8,
            discretize_method="euler",
            ineq_activation="none",
            reg0=0.0, reg1_min=1e-6, reg_increase=10.0, reg_max=1e2, reg_max_iters=10,
            ls_alpha0=1.0, ls_tau=0.5, ls_beta=0.25, ls_max_iters=10, normkind="l1_mean",
        )


def test_newton_step_autodiff_handles_nonfinite_residual(monkeypatch):
    """
    If G(z0) contains NaN/Inf, step should return (op, diag) with solve_ok=False
    and NOT attempt Jacobian/solve/line-search.
    """
    nlgame, op, alstate = _stub_game_and_op_same_tg()

    monkeypatch.setattr(pdg_alsolver, "pack_decision_vars_1d", lambda *_: jnp.array([1.0, 2.0], dtype=jnp.float32))

    # Make compute_al_residual_flat_from_decision_vars return non-finite
    monkeypatch.setattr(
        pdg_alsolver,
        "compute_al_residual_flat_from_decision_vars",
        lambda *args, **kwargs: jnp.array([jnp.nan, 1.0], dtype=jnp.float32),
    )

    # Guard: none of these should be called
    monkeypatch.setattr(
        pdg_alsolver,
        "jacobian_al_residual_flat_autodiff",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not compute Jacobian")),
    )
    monkeypatch.setattr(
        pdg_alsolver,
        "solve_newton_system_tikhonov",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not solve")),
    )
    monkeypatch.setattr(
        pdg_alsolver,
        "backtracking_linesearch_armijo",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not line-search")),
    )

    op_new, diag = pdg_alsolver.newton_step_autodiff(
        nlgame, op, alstate,
        step_rtol = 1e-7,
        step_atol = 1e-8,
        discretize_method="euler",
        ineq_activation="none",
        reg0=0.0, reg1_min=1e-6, reg_increase=10.0, reg_max=1e2, reg_max_iters=10,
        ls_alpha0=1.0, ls_tau=0.5, ls_beta=0.25, ls_max_iters=10, normkind="l1_mean",
    )

    assert op_new is op
    assert diag.solve_ok is False
    assert diag.accepted is False

def _stub_game_op_alstate_for_solve():
    tg = TimeGrid(nt=4, dt=0.1, t0=0.0)
    nlgame = SimpleNamespace(tg=tg)
    op0 = SimpleNamespace(tg=tg)
    alstate = SimpleNamespace()
    return nlgame, op0, alstate


def _stub_game_op_alstate_for_stationarity():
    tg = TimeGrid(nt=4, dt=0.1, t0=0.0)
    cs = SimpleNamespace()
    nlgame = SimpleNamespace(tg=tg, cs=cs, u_splits=jnp.array([1], dtype=jnp.int32))
    op0 = SimpleNamespace(tg=tg, name="op0")
    # op0 = SimpleNamespace(tg=tg)
    alstate = SimpleNamespace()
    return nlgame, op0, alstate


def _stationarity_metrics_residual(*, opt=1.0, dyn=0.0):
    return altypes.ALResidualStruct(
        dLdX=jnp.array([[[opt]]], dtype=jnp.float32),
        dLdU=jnp.zeros((1, 1, 1), dtype=jnp.float32),
        dyn_res=jnp.array([[dyn]], dtype=jnp.float32),
    )


def test_newton_solve_stationarity_rejects_invalid_params():
    nlgame, op0, alstate = _stub_game_op_alstate_for_stationarity()

    with pytest.raises(ValueError, match="max_iters"):
        pdg_alsolver.newton_solve_stationarity_autodiff(
            nlgame, op0, alstate,
            discretize_method="rk2",
            ineq_activation="altro",
            opt_tol=1e-3,
            dyn_tol=1e-3,
            max_iters=-1,
            max_rejects=5,
            step_rtol=1e-7, step_atol=1e-8,
            reg0=0.0, reg1_min=1e-12, reg_increase=10.0, reg_max=1e8, reg_max_iters=64,
            ls_alpha0=1.0, ls_tau=0.5, ls_beta=0.25, ls_max_iters=20,
            normkind="l1_mean",
        )

    with pytest.raises(ValueError, match="opt_tol"):
        pdg_alsolver.newton_solve_stationarity_autodiff(
            nlgame, op0, alstate,
            discretize_method="rk2",
            ineq_activation="altro",
            opt_tol=0.0,
            dyn_tol=1e-3,
            max_iters=5,
            max_rejects=5,
            step_rtol=1e-7, step_atol=1e-8,
            reg0=0.0, reg1_min=1e-12, reg_increase=10.0, reg_max=1e8, reg_max_iters=64,
            ls_alpha0=1.0, ls_tau=0.5, ls_beta=0.25, ls_max_iters=20,
            normkind="l1_mean",
        )

    with pytest.raises(ValueError, match="dyn_tol"):
        pdg_alsolver.newton_solve_stationarity_autodiff(
            nlgame, op0, alstate,
            discretize_method="rk2",
            ineq_activation="altro",
            opt_tol=1e-3,
            dyn_tol=0.0,
            max_iters=5,
            max_rejects=5,
            step_rtol=1e-7, step_atol=1e-8,
            reg0=0.0, reg1_min=1e-12, reg_increase=10.0, reg_max=1e8, reg_max_iters=64,
            ls_alpha0=1.0, ls_tau=0.5, ls_beta=0.25, ls_max_iters=20,
            normkind="l1_mean",
        )


def test_newton_solve_stationarity_rejects_timegrid_mismatch(monkeypatch):
    tg_game = TimeGrid(nt=4, dt=0.1, t0=0.0)
    tg_op = TimeGrid(nt=4, dt=0.1, t0=1.0)
    nlgame = SimpleNamespace(tg=tg_game)
    op0 = SimpleNamespace(tg=tg_op)
    alstate = SimpleNamespace()

    # Should fail before any packing/evals
    monkeypatch.setattr(
        pdg_alsolver,
        "pack_decision_vars_1d",
        lambda *_: (_ for _ in ()).throw(AssertionError("should not pack")),
    )

    with pytest.raises(ValueError, match="TimeGrid mismatch"):
        pdg_alsolver.newton_solve_stationarity_autodiff(
            nlgame, op0, alstate,
            discretize_method="rk2",
            ineq_activation="altro",
            opt_tol=1e-3,
            dyn_tol=1e-3,
            max_iters=5,
            max_rejects=5,
            step_rtol=1e-7, step_atol=1e-8,
            reg0=0.0, reg1_min=1e-12, reg_increase=10.0, reg_max=1e8, reg_max_iters=64,
            ls_alpha0=1.0, ls_tau=0.5, ls_beta=0.25, ls_max_iters=20,
            normkind="l1_mean",
        )


def test_newton_solve_stationarity_returns_reason_on_nonfinite_residual_at_start(monkeypatch):
    nlgame, op0, alstate = _stub_game_op_alstate_for_stationarity()

    # Non-finite stationarity should fail before any Newton step.
    monkeypatch.setattr(
        pdg_alsolver.systypes,
        "residual_discrete_dynamics_trajectory",
        lambda *a, **ka: (_ for _ in ()).throw(AssertionError("should use residual struct")),
    )
    monkeypatch.setattr(
        pdg_alsolver,
        "compute_al_residual_struct_from_traj",
        lambda *a, **k: _stationarity_metrics_residual(opt=jnp.inf, dyn=0.0),
    )

    op_out, diag = pdg_alsolver.newton_solve_stationarity_autodiff(
        nlgame, op0, alstate,
        discretize_method="rk2",
        ineq_activation="altro",
        opt_tol=1e-3,
        dyn_tol=1e-3,
        max_iters=5,
        max_rejects=5,
        step_rtol=1e-7, step_atol=1e-8,
        reg0=0.0, reg1_min=1e-12, reg_increase=10.0, reg_max=1e8, reg_max_iters=64,
        ls_alpha0=1.0, ls_tau=0.5, ls_beta=0.25, ls_max_iters=20,
        normkind="l1_mean",
    )

    assert op_out is op0
    assert diag.converged is False
    assert diag.iters == 0
    assert diag.reason == "nonfinite_residual_at_start"

    monkeypatch.setattr(
        pdg_alsolver,
        "compute_al_residual_struct_from_traj",
        lambda *a, **k: _stationarity_metrics_residual(opt=0.0, dyn=jnp.inf),
    )

    op_out, diag = pdg_alsolver.newton_solve_stationarity_autodiff(
        nlgame, op0, alstate,
        discretize_method="rk2",
        ineq_activation="altro",
        opt_tol=1e-3,
        dyn_tol=1e-3,
        max_iters=5,
        max_rejects=5,
        step_rtol=1e-7, step_atol=1e-8,
        reg0=0.0, reg1_min=1e-12, reg_increase=10.0, reg_max=1e8, reg_max_iters=64,
        ls_alpha0=1.0, ls_tau=0.5, ls_beta=0.25, ls_max_iters=20,
        normkind="l1_mean",
    )

    assert op_out is op0
    assert diag.converged is False
    assert diag.iters == 0
    assert diag.reason == "nonfinite_residual_at_start"


def test_newton_solve_stationarity_opt_tol_at_start_does_not_step(monkeypatch):
    nlgame, op0, alstate = _stub_game_op_alstate_for_stationarity()

    monkeypatch.setattr(
        pdg_alsolver,
        "compute_al_residual_struct_from_traj",
        lambda *a, **k: _stationarity_metrics_residual(opt=1e-6, dyn=0.0),
    )
    monkeypatch.setattr(
        pdg_alsolver.systypes,
        "residual_discrete_dynamics_trajectory",
        lambda *a, **ka: (_ for _ in ()).throw(AssertionError("should use residual struct")),
    )

    monkeypatch.setattr(
        pdg_alsolver,
        "newton_step_autodiff",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not call newton_step_autodiff")),
    )

    op_out, diag = pdg_alsolver.newton_solve_stationarity_autodiff(
        nlgame, op0, alstate,
        discretize_method="rk2",
        ineq_activation="altro",
        opt_tol=1e-3,
        dyn_tol=1e-3,
        max_iters=10,
        max_rejects=5,
        step_rtol=1e-7, step_atol=1e-8,
        reg0=0.0, reg1_min=1e-12, reg_increase=10.0, reg_max=1e8, reg_max_iters=64,
        ls_alpha0=1.0, ls_tau=0.5, ls_beta=0.25, ls_max_iters=20,
        normkind="l1_mean",
    )

    assert op_out is op0
    assert diag.converged
    assert diag.iters == 0
    assert diag.reason == "opt_dyn_tol_at_start"
    assert diag.opt_vios == pytest.approx((1e-6,))


def test_newton_solve_stationarity_metrics_reuse_structured_residual_at_start(monkeypatch):
    nlgame, op0, alstate = _stub_game_op_alstate_for_stationarity()
    calls = {"residual_struct": 0}

    def fake_residual_struct(*args, **kwargs):
        calls["residual_struct"] += 1
        return _stationarity_metrics_residual(opt=0.0, dyn=0.0)

    monkeypatch.setattr(pdg_alsolver, "compute_al_residual_struct_from_traj", fake_residual_struct)
    monkeypatch.setattr(
        pdg_alsolver,
        "compute_al_residual_flat_from_decision_vars",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should pack existing residual struct")),
    )
    monkeypatch.setattr(
        pdg_alsolver.systypes,
        "residual_discrete_dynamics_trajectory",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should use residual struct dyn_res")),
    )
    monkeypatch.setattr(
        pdg_alsolver,
        "newton_step_autodiff",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not step")),
    )

    op_out, diag = pdg_alsolver.newton_solve_stationarity_autodiff(
        nlgame, op0, alstate,
        discretize_method="rk2",
        ineq_activation="altro",
        opt_tol=1e-3,
        dyn_tol=1e-3,
        max_iters=10,
        max_rejects=5,
        step_rtol=1e-7, step_atol=1e-8,
        reg0=0.0, reg1_min=1e-12, reg_increase=10.0, reg_max=1e8, reg_max_iters=64,
        ls_alpha0=1.0, ls_tau=0.5, ls_beta=0.25, ls_max_iters=20,
        normkind="l1_mean",
    )

    assert op_out is op0
    assert diag.converged
    assert diag.reason == "opt_dyn_tol_at_start"
    assert calls == {"residual_struct": 1}


def test_newton_solve_stationarity_reject_streak_returns_last_accepted(monkeypatch):
    tg = TimeGrid(nt=4, dt=0.1, t0=0.0)
    nlgame = SimpleNamespace(tg=tg, cs=None, u_splits=jnp.array([1], dtype=jnp.int32))
    op0 = SimpleNamespace(tg=tg, name="op0")
    alstate = SimpleNamespace()

    monkeypatch.setattr(
        pdg_alsolver,
        "compute_al_residual_struct_from_traj",
        lambda *a, **k: _stationarity_metrics_residual(opt=1.0, dyn=0.0),
    )

    monkeypatch.setattr(
        pdg_alsolver.systypes,
        "residual_discrete_dynamics_trajectory",
        lambda *a, **ka: (_ for _ in ()).throw(AssertionError("should use residual struct")),
    )

    calls = {"k": 0}
    def fake_step(nlgame_, op, alstate_, **kwargs):
        calls["k"] += 1
        if calls["k"] == 1:
            opA = SimpleNamespace(tg=tg, name="opA")
            diag = altypes.NewtonStepDiag(
                accepted=True, alpha=1.0,
                g_norm0=1.0, g_norm_trial=0.8,
                step_norm=1.0, reg=0.0,
                ls_iters=1, solve_ok=True,
            )
            return opA, diag
        else:
            diag = altypes.NewtonStepDiag(
                accepted=False, alpha=0.0,
                g_norm0=1.0, g_norm_trial=1.0,
                step_norm=1.0, reg=0.0,
                ls_iters=2, solve_ok=True,
            )
            return op, diag

    monkeypatch.setattr(pdg_alsolver, "newton_step_autodiff", fake_step)

    op_out, diag = pdg_alsolver.newton_solve_stationarity_autodiff(
        nlgame, op0, alstate,
        discretize_method="rk2",
        ineq_activation="altro",
        opt_tol=1e-3,
        dyn_tol=1e-3,
        max_iters=10,
        max_rejects=3,
        return_last_accepted=True,
        step_rtol=1e-7, step_atol=1e-8,
        reg0=0.0, reg1_min=1e-12, reg_increase=10.0, reg_max=1e8, reg_max_iters=64,
        ls_alpha0=1.0, ls_tau=0.5, ls_beta=0.25, ls_max_iters=20,
        normkind="l1_mean",
    )

    assert not diag.converged
    assert diag.reason == "too_many_rejected_steps"
    assert op_out.name == "opA"
    assert diag.iters == 1 + 3
    assert diag.accepted[0] is True
    assert all(a is False for a in diag.accepted[1:])


def test_newton_solve_stationarity_step_stall_before_opt_tol(monkeypatch):
    tg = TimeGrid(nt=4, dt=0.1, t0=0.0)
    nlgame = SimpleNamespace(tg=tg, cs=None, u_splits=jnp.array([1], dtype=jnp.int32))
    op0 = SimpleNamespace(tg=tg, name="op0")
    alstate = SimpleNamespace()

    monkeypatch.setattr(
        pdg_alsolver,
        "compute_al_residual_struct_from_traj",
        lambda *a, **k: _stationarity_metrics_residual(opt=1.0, dyn=0.0),
    )

    monkeypatch.setattr(
        pdg_alsolver.systypes,
        "residual_discrete_dynamics_trajectory",
        lambda *a, **ka: (_ for _ in ()).throw(AssertionError("should use residual struct")),
    )

    def fake_step(*a, **k):
        diag = altypes.NewtonStepDiag(
            accepted=True, alpha=0.0,          # stall signature
            g_norm0=1.0, g_norm_trial=1.0,
            step_norm=1e-12, reg=0.0,
            ls_iters=0, solve_ok=True,
        )
        return op0, diag  # unchanged

    monkeypatch.setattr(pdg_alsolver, "newton_step_autodiff", fake_step)

    op_out, diag = pdg_alsolver.newton_solve_stationarity_autodiff(
        nlgame, op0, alstate,
        discretize_method="rk2",
        ineq_activation="altro",
        opt_tol=1e-3,
        dyn_tol=1e-3,
        max_iters=10,
        max_rejects=5,
        step_rtol=1e-7, step_atol=1e-8,
        reg0=0.0, reg1_min=1e-12, reg_increase=10.0, reg_max=1e8, reg_max_iters=64,
        ls_alpha0=1.0, ls_tau=0.5, ls_beta=0.25, ls_max_iters=20,
        normkind="l1_mean",
    )

    assert op_out is op0
    assert diag.converged is False
    assert diag.reason == "step_stall_before_opt_dyn_tol"
    assert diag.iters == 1  # one iteration executed
    assert diag.accepted == (True,)
    assert diag.alphas == (0.0,)


def test_newton_solve_stationarity_autodiff_nonmock_converges_float32():
    """
    Non-mocked integration test:
    - 1 player
    - small horizon
    - unconstrained (no auxiliary constraints)
    - quadratic cost with unique minimizer
    Expect stationarity solver to reduce opt_vio_inf and converge.
    """
    dtype = jnp.float32

    # ---- tiny horizon ----
    tg = TimeGrid(nt=4, dt=0.2, t0=0.0)   # K = 3 steps
    nt = tg.nt
    K = nt - 1

    # ---- dims ----
    N = 1
    nx = 2
    nu = 1
    u_splits = jnp.array([nu], dtype=jnp.int32)

    # ---- linear continuous dynamics: xdot = A x + B u ----
    A = jnp.array([[0.0, 1.0],
                   [-0.5, -0.1]], dtype=dtype)
    B = jnp.array([[0.0],
                   [1.0]], dtype=dtype)

    def f_cont(t, x, u):
        # x: (2,), u: (1,)
        return A @ x + (B @ u)

    cs = SampledContinuousSystemType1(tg=tg, dynamics=f_cont, nx=nx, nu=nu)

    # ---- player cost: quadratic in x and u, plus terminal cost ----
    Q = jnp.diag(jnp.array([1.0, 0.5], dtype=dtype))
    R = jnp.array([[0.2]], dtype=dtype)
    Qf = jnp.diag(jnp.array([2.0, 1.0], dtype=dtype))

    def running0(t, x, u0):
        # u0 is LOCAL control, shape (1,)
        return 0.5 * (x @ (Q @ x)) + 0.5 * (u0 @ (R @ u0))

    def terminal0(t, x):
        return 0.5 * (x @ (Qf @ x))

    costs = [
        PlayerCostSpecContinuous(
            running=running0,
            terminal=terminal0,
            control_domain=CostControlDomain.LOCAL,
            control_coupling=CostControlStructure.LOCAL_ONLY,
        )
    ]

    # ---- no auxiliary constraints ----
    constraints = GameConstraintGridMap(ineq_blocks=(), eq_blocks=())
    alstate = altypes.JointAugmentedLagrangianState(
        lam_ineq=jnp.zeros((0,), dtype=dtype),
        rho_ineq=jnp.zeros((0,), dtype=dtype),
        lam_eq=jnp.zeros((0,), dtype=dtype),
        rho_eq=jnp.zeros((0,), dtype=dtype),
    )

    nlgame = NonlinearGameType2(
        cs=cs,
        N=N,
        costs=costs,
        constraints=constraints,
        u_splits=u_splits,
    )

    # ---- initial primal-dual trajectory guess (deterministic) ----
    x0 = jnp.array([1.0, -0.5], dtype=dtype)
    xs0 = jnp.vstack([
        x0,
        x0 * 0.8,
        x0 * 0.6,
        x0 * 0.4,
    ]).astype(dtype)                              # (nt,nx)

    us0 = jnp.full((K, nu), 0.3, dtype=dtype)     # (K,nu)
    ls0 = jnp.zeros((K, N, nx), dtype=dtype)      # (K,N,nx)

    op0 = FixedStepPrimalDualTrajectory(tg=tg, xs=xs0, us=us0, ls=ls0)

    # ---- solve ----
    opt_tol = 1e-3  # float32-friendly
    op_out, diag = pdg_alsolver.newton_solve_stationarity_autodiff(
        nlgame,
        op0,
        alstate,
        discretize_method="euler",
        ineq_activation="none",
        opt_tol=opt_tol,
        dyn_tol=1e-3,
        max_iters=15,
        max_rejects=5,
        step_rtol=1e-7,
        step_atol=1e-8,
        reg0=0.0,
        reg1_min=1e-6,
        reg_increase=10.0,
        reg_max=1e6,
        reg_max_iters=32,
        ls_alpha0=1.0,
        ls_tau=0.5,
        ls_beta=0.25,
        ls_max_iters=20,
        normkind="l1_mean",
        return_last_accepted=True,
    )

    # ---- assertions ----
    assert len(diag.opt_vios) == diag.iters + 1
    assert np.isfinite(diag.opt_vios[0])
    assert diag.opt_vios[0] > 0.0

    # must reduce at least once (avoid brittleness: allow plateau after reductions)
    assert min(diag.opt_vios) < diag.opt_vios[0]

    # should converge on this small convex-ish problem
    assert diag.converged, f"Did not converge: reason={diag.reason}, last_opt={diag.opt_vios[-1]}"
    assert diag.opt_vios[-1] <= opt_tol * 1.1  # small slack for float32

    # sanity: returned trajectory has the right shapes
    assert op_out.xs.shape == (nt, nx)
    assert op_out.us.shape == (K, nu)
    assert op_out.ls.shape == (K, N, nx)

    # extra: verify helper agrees with final metric
    opt_final = pdg_alsolver.optimality_violation_inf(
        nlgame, op_out, alstate, discretize_method="euler", ineq_activation="none"
    )
    assert float(opt_final) == pytest.approx(diag.opt_vios[-1], rel=1e-6, abs=1e-6)

def _make_infeasible_op_and_game_for_dyn_check():
    # Minimal "game-like" stubs: newton_solve_stationarity_autodiff only needs nlgame.tg and nlgame.cs
    tg = TimeGrid(nt=2, dt=0.1, t0=0.0)  # K=1
    nx = 1
    nu = 1

    # Simple continuous dynamics that, with Euler, imply: x1 = x0 + dt*u0
    def f_cont(t, x, u):
        return u  # dx/dt = u

    cs = SampledContinuousSystemType1(tg=tg, dynamics=f_cont, nx=nx, nu=nu)

    class _GameStub:
        pass

    nlgame = _GameStub()
    nlgame.tg = tg
    nlgame.cs = cs
    nlgame.nt = tg.nt
    nlgame.nx = nx
    nlgame.nu = nu
    nlgame.u_splits = jnp.array([1], dtype=jnp.int32)  # not used here, but often expected elsewhere

    # Construct an *infeasible* trajectory: x0=0, u0=0, but x1=10 (should be 0 with Euler)
    xs = jnp.array([[0.0], [10.0]], dtype=jnp.float32)   # (nt=2,nx=1)
    us = jnp.array([[0.0]], dtype=jnp.float32)           # (K=1,nu=1)
    ls = jnp.zeros((1, 1, 1), dtype=jnp.float32)         # (K=1,N=1,nx=1) dummy

    op = FixedStepPrimalDualTrajectory(tg=tg, xs=xs, us=us, ls=ls)

    # Empty AL state is fine for this test (no aux constraints)
    alstate = altypes.JointAugmentedLagrangianState(
        lam_ineq=jnp.zeros((0,), dtype=jnp.float32),
        rho_ineq=jnp.zeros((0,), dtype=jnp.float32),
        lam_eq=jnp.zeros((0,), dtype=jnp.float32),
        rho_eq=jnp.zeros((0,), dtype=jnp.float32),
    )

    return nlgame, op, alstate


def test_newton_solve_stationarity_autodiff_does_not_converge_at_start_if_dynamics_infeasible(monkeypatch):
    """
    Regression test for the logical bug:
    The inner solver must NOT declare convergence solely because opt_vio is small if dyn_vio is huge.
    """
    nlgame, op0, alstate = _make_infeasible_op_and_game_for_dyn_check()

    monkeypatch.setattr(
        pdg_alsolver,
        "compute_al_residual_struct_from_traj",
        lambda *a, **k: _stationarity_metrics_residual(opt=0.0, dyn=10.0),
    )

    # If your solver is fixed per Option A, it should check dynamics feasibility and refuse opt_dyn_tol_at_start.
    op_out, diag = pdg_alsolver.newton_solve_stationarity_autodiff(
        nlgame, op0, alstate,
        discretize_method="euler",
        ineq_activation="none",
        opt_tol=1e-6,
        # NEW: assume you add this to the function signature
        dyn_tol=1e-6,
        max_iters=0,          # important: forces "at start" logic only
        max_rejects=0,
        step_rtol=1e-7,
        step_atol=1e-8,
        reg0=0.0,
        reg1_min=1e-6,
        reg_increase=10.0,
        reg_max=1e6,
        reg_max_iters=8,
        ls_alpha0=1.0,
        ls_tau=0.5,
        ls_beta=0.25,
        ls_max_iters=5,
        normkind="l1_mean",
        return_last_accepted=True,
    )

    # Must NOT claim convergence at start if dynamics are wildly violated
    assert diag.converged is False
    assert diag.reason != "opt_dyn_tol_at_start"

    # Sanity: dynamics are indeed infeasible for this op
    D = residual_discrete_dynamics_trajectory(nlgame.cs, op0, method="euler")
    assert float(jnp.max(jnp.abs(D))) > 1.0

def test_newton_solve_stationarity_nonfinite_dyn_residual_does_not_converge(monkeypatch):
    tg = TimeGrid(nt=3, dt=0.1, t0=0.0)
    nlgame = SimpleNamespace(tg=tg, nt=tg.nt, nx=1, cs=SimpleNamespace(), u_splits=jnp.array([1]))
    op0 = SimpleNamespace(tg=tg)
    alstate = SimpleNamespace()

    monkeypatch.setattr(
        pdg_alsolver,
        "compute_al_residual_struct_from_traj",
        lambda *a, **k: _stationarity_metrics_residual(opt=0.0, dyn=jnp.nan),
    )

    monkeypatch.setattr(
        pdg_alsolver.systypes,
        "residual_discrete_dynamics_trajectory",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should use residual struct")),
    )

    # Make stepping unreachable so we only test iteration-0 convergence decision
    monkeypatch.setattr(
        pdg_alsolver,
        "newton_step_autodiff",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not step")),
    )

    op_out, diag = pdg_alsolver.newton_solve_stationarity_autodiff(
        nlgame, op0, alstate,
        discretize_method="euler",
        ineq_activation="none",
        opt_tol=1e-3,
        dyn_tol=1e-3,
        max_iters=5,
        max_rejects=2,
        step_rtol=1e-7,
        step_atol=1e-8,
        reg0=0.0,
        reg1_min=1e-6,
        reg_increase=10.0,
        reg_max=1e6,
        reg_max_iters=16,
        ls_alpha0=1.0,
        ls_tau=0.5,
        ls_beta=0.25,
        ls_max_iters=10,
        normkind="l1_mean",
    )

    assert op_out is op0
    assert diag.converged is False
    # pick whatever string you standardize on:
    assert "nonfinite" in diag.reason or "dyn" in diag.reason
    assert not np.isfinite(diag.dyn_vios[0])

def test_stationarity_newton_reports_dyn_violation_metric(monkeypatch):
    """
    Optional: encourages you to record dyn_vio in StationarityNewtonDiag for debugging
    (mirrors ALGAMES separating opt_vio and dyn_vio).
    """
    nlgame, op0, alstate = _make_infeasible_op_and_game_for_dyn_check()

    monkeypatch.setattr(
        pdg_alsolver,
        "compute_al_residual_struct_from_traj",
        lambda *a, **k: _stationarity_metrics_residual(opt=0.0, dyn=10.0),
    )

    op_out, diag = pdg_alsolver.newton_solve_stationarity_autodiff(
        nlgame, op0, alstate,
        discretize_method="euler",
        ineq_activation="none",
        opt_tol=1e-6,
        dyn_tol=1e-6,
        max_iters=0,
        max_rejects=0,
        step_rtol=1e-7,
        step_atol=1e-8,
        reg0=0.0,
        reg1_min=1e-6,
        reg_increase=10.0,
        reg_max=1e6,
        reg_max_iters=8,
        ls_alpha0=1.0,
        ls_tau=0.5,
        ls_beta=0.25,
        ls_max_iters=5,
        normkind="l1_mean",
        return_last_accepted=True,
    )

    # This assertion assumes you add dyn_vios to the diag; if you don’t, skip this test.
    assert hasattr(diag, "dyn_vios")
    assert diag.dyn_vios[0] > 1.0


def test_dual_ascent_update_updates_eq_and_projects_ineq(monkeypatch):
    """
    Inequality: lam <- max(0, lam + rho*c)
    Equality:   lam <- lam + rho*c
    """
    # Fake constraints and op (only nc_ineq/nc_eq used directly here)
    constraints = SimpleNamespace(nc_ineq=3, nc_eq=2)
    op = SimpleNamespace()

    # Build fake linearizations with explicit slices + values
    # ineq stack: size 3
    ineq_lins = (
        SimpleNamespace(c=jnp.array([+0.5], dtype=jnp.float32), sl=slice(0, 1)),
        SimpleNamespace(c=jnp.array([-2.0, +1.0], dtype=jnp.float32), sl=slice(1, 3)),
    )
    # eq stack: size 2
    eq_lins = (
        SimpleNamespace(c=jnp.array([+3.0, -4.0], dtype=jnp.float32), sl=slice(0, 2)),
    )

    monkeypatch.setattr(pdg_alsolver.contypes, "build_constraint_step_linearizations", lambda *_: (ineq_lins, eq_lins))

    al0 = _make_alstate_from_lists(
        lam_ineq=[0.1, 0.2, 0.3],
        rho_ineq=[2.0, 2.0, 2.0],
        lam_eq=[-1.0, 1.0],
        rho_eq=[0.5, 0.5],
        dtype=jnp.float32,
    )

    al1 = pdg_alsolver.dual_ascent_update(constraints, op, al0)

    # expected:
    # c_ineq = [0.5, -2.0, 1.0]
    # lam_ineq_new = max(0, [0.1,0.2,0.3] + 2*[0.5,-2,1]) = max(0, [1.1, -3.8, 2.3]) = [1.1, 0.0, 2.3]
    expected_ineq = np.array([1.1, 0.0, 2.3], dtype=np.float32)
    np.testing.assert_allclose(np.asarray(al1.lam_ineq), expected_ineq, atol=1e-6, rtol=1e-6)

    # c_eq = [3.0, -4.0]
    # lam_eq_new = [-1,1] + 0.5*[3,-4] = [-1+1.5, 1-2] = [0.5, -1]
    expected_eq = np.array([0.5, -1.0], dtype=np.float32)
    np.testing.assert_allclose(np.asarray(al1.lam_eq), expected_eq, atol=1e-6, rtol=1e-6)


def test_dual_ascent_update_does_not_mutate_original(monkeypatch):
    """
    Ensure immutability: al0 unchanged, al1 is a new object.
    """
    constraints = SimpleNamespace(nc_ineq=1, nc_eq=0)
    op = SimpleNamespace()

    ineq_lins = (SimpleNamespace(c=jnp.array([1.0], dtype=jnp.float32), sl=slice(0, 1)),)
    eq_lins = ()
    monkeypatch.setattr(pdg_alsolver.contypes, "build_constraint_step_linearizations", lambda *_: (ineq_lins, eq_lins))

    al0 = _make_alstate_from_lists(
        lam_ineq=[0.0], rho_ineq=[1.0],
        lam_eq=[], rho_eq=[],
        dtype=jnp.float32
    )

    al1 = pdg_alsolver.dual_ascent_update(constraints, op, al0)

    assert al1 is not al0
    # original unchanged
    np.testing.assert_allclose(np.asarray(al0.lam_ineq), np.array([0.0], dtype=np.float32))
    # new changed
    np.testing.assert_allclose(np.asarray(al1.lam_ineq), np.array([1.0], dtype=np.float32))


def test_dual_ascent_update_shape_mismatch_raises(monkeypatch):
    """
    If constraints.nc_* doesn't match alstate shapes, should raise (validate_shapes=True).
    """
    constraints = SimpleNamespace(nc_ineq=2, nc_eq=0)
    op = SimpleNamespace()

    # but linearizations only fill 1 element => c_ineq gets shape (2,) but only one set; still OK,
    # mismatch we actually want is alstate.lam_ineq has wrong length
    ineq_lins = (SimpleNamespace(c=jnp.array([1.0], dtype=jnp.float32), sl=slice(0, 1)),)
    eq_lins = ()
    monkeypatch.setattr(pdg_alsolver.contypes, "build_constraint_step_linearizations", lambda *_: (ineq_lins, eq_lins))

    al0 = _make_alstate_from_lists(
        lam_ineq=[0.0, 0.0, 0.0],  # WRONG length (3) vs constraints.nc_ineq (2)
        rho_ineq=[1.0, 1.0, 1.0],
        lam_eq=[], rho_eq=[],
        dtype=jnp.float32
    )

    with pytest.raises(ValueError, match="c_ineq shape"):
        pdg_alsolver.dual_ascent_update(constraints, op, al0, validate_shapes=True)


def test_rho_increase_schedule_multiplies_and_caps():
    al0 = _make_alstate_from_lists(
        lam_ineq=[100.0, 200.0], rho_ineq=[1.0, 20.0],
        lam_eq=[300.0],        rho_eq=[3.0],
        dtype=jnp.float32,
    )

    al1 = pdg_alsolver.rho_increase_schedule(al0, rho_increase=10.0, rho_max=50.0)

    # rho_ineq: [1*10=10, 20*10=200 cap->50]
    np.testing.assert_allclose(np.asarray(al1.rho_ineq), np.array([10.0, 50.0], dtype=np.float32), atol=1e-6)
    # rho_eq: [3*10=30]
    np.testing.assert_allclose(np.asarray(al1.rho_eq), np.array([30.0], dtype=np.float32), atol=1e-6)

    # lambdas unchanged
    np.testing.assert_allclose(np.asarray(al1.lam_ineq), np.asarray(al0.lam_ineq))
    np.testing.assert_allclose(np.asarray(al1.lam_eq), np.asarray(al0.lam_eq))


def test_rho_increase_schedule_returns_new_instance():
    al0 = _make_alstate_from_lists(
        lam_ineq=[1.0], rho_ineq=[2.0],
        lam_eq=[], rho_eq=[],
        dtype=jnp.float32,
    )
    al1 = pdg_alsolver.rho_increase_schedule(al0, rho_increase=2.0, rho_max=100.0)

    assert al1 is not al0
    # original unchanged
    np.testing.assert_allclose(np.asarray(al0.rho_ineq), np.array([2.0], dtype=np.float32))
    # new updated
    np.testing.assert_allclose(np.asarray(al1.rho_ineq), np.array([4.0], dtype=np.float32))


def test_rho_increase_schedule_preserves_dtype():
    al0 = _make_alstate_from_lists(
        lam_ineq=[0.0], rho_ineq=[1.0],
        lam_eq=[0.0],  rho_eq=[1.0],
        dtype=jnp.float32,
    )
    al1 = pdg_alsolver.rho_increase_schedule(al0, rho_increase=3.0, rho_max=10.0)
    assert al1.rho_ineq.dtype == jnp.float32
    assert al1.rho_eq.dtype == jnp.float32


def test_rho_increase_schedule_rejects_bad_inputs():
    al0 = _make_alstate_from_lists(
        lam_ineq=[], rho_ineq=[],
        lam_eq=[],  rho_eq=[],
        dtype=jnp.float32,
    )

    with pytest.raises(ValueError, match="rho_increase"):
        pdg_alsolver.rho_increase_schedule(al0, rho_increase=0.9, rho_max=10.0)

    with pytest.raises(ValueError, match="rho_max"):
        pdg_alsolver.rho_increase_schedule(al0, rho_increase=2.0, rho_max=0.0)


def test_constraint_violation_metrics_empty():
    ineq_v, eq_v = pdg_alsolver._constraint_violation_metrics(
        jnp.zeros((0,), dtype=jnp.float32),
        jnp.zeros((0,), dtype=jnp.float32),
    )
    assert ineq_v == 0.0
    assert eq_v == 0.0


def test_constraint_violation_metrics_ineq_and_eq():
    c_ineq = jnp.array([-2.0, 0.0, 1.5, -0.1], dtype=jnp.float32)
    c_eq = jnp.array([0.2, -3.0, 1.0], dtype=jnp.float32)

    ineq_v, eq_v = pdg_alsolver._constraint_violation_metrics(c_ineq, c_eq)

    assert ineq_v == pytest.approx(1.5)
    assert eq_v == pytest.approx(3.0)


def test_collect_constraint_stacks_from_linearizations():
    constraints = SimpleNamespace(nc_ineq=4, nc_eq=3)

    # ineq stack: [a, b0, b1, c]
    ineq_lins = (
        SimpleNamespace(c=jnp.array([1.0], dtype=jnp.float32), sl=slice(0, 1)),
        SimpleNamespace(c=jnp.array([2.0, 3.0], dtype=jnp.float32), sl=slice(1, 3)),
        SimpleNamespace(c=jnp.array([4.0], dtype=jnp.float32), sl=slice(3, 4)),
    )

    # eq stack: [d0, d1, e]
    eq_lins = (
        SimpleNamespace(c=jnp.array([10.0, 20.0], dtype=jnp.float32), sl=slice(0, 2)),
        SimpleNamespace(c=jnp.array([30.0], dtype=jnp.float32), sl=slice(2, 3)),
    )

    c_ineq, c_eq = pdg_alsolver._collect_constraint_stacks_from_linearizations(
        constraints, ineq_lins, eq_lins, dtype=jnp.float32
    )

    np.testing.assert_allclose(np.asarray(c_ineq), np.array([1, 2, 3, 4], dtype=np.float32))
    np.testing.assert_allclose(np.asarray(c_eq), np.array([10, 20, 30], dtype=np.float32))


def test_al_solve_autodiff_outer_updates_lambda_and_rho(monkeypatch):
    """
    Fully mocked inner solve + constraint evaluation:
      - Dynamics residual is zero => dyn feasible
      - Constraints are constant and known
      - Verify one outer iteration applies dual ascent and rho schedule.
    """
    tg = TimeGrid(nt=4, dt=0.1, t0=0.0)

    # Minimal nlgame/op shapes used by al_solve_autodiff
    nlgame = SimpleNamespace(
        tg=tg,
        cs=SimpleNamespace(),  # only passed through to residual_discrete_dynamics_trajectory
        constraints=SimpleNamespace(nc_ineq=3, nc_eq=2, ineq_blocks=(), eq_blocks=(), tg=tg),
    )
    op0 = SimpleNamespace(tg=tg, nx=3)

    # Initial AL state
    al0 = _make_alstate_from_lists(
        lam_ineq=[0.1, 0.2, 0.3],
        rho_ineq=[2.0, 2.0, 2.0],
        lam_eq=[-1.0, 1.0],
        rho_eq=[0.5, 0.5],
        dtype=jnp.float32,
    )

    # ---- Mock 1: inner Newton solve (returns same op + diag) ----
    fake_newton_diag = SimpleNamespace(
        converged=True,
        iters=3,
        reason="residual_tolerance_met",
        merit_norms=(10.0, 1.0, 0.1),
        opt_vios=(10.0, 1.0, 0.1),
        dyn_vios=(10.0, 1.0, 0.1),
        accepted=(True, True, True),
    )

    def fake_newton_solve(*args, **kwargs):
        # returns (op, newton_diag)
        return op0, fake_newton_diag

    monkeypatch.setattr(pdg_alsolver, "newton_solve_stationarity_autodiff", fake_newton_solve)

    # ---- Mock 2: stationarity conditions are nonzero ----
    def fake_grad_aug_lag_traj(*a, **ka):
        return jnp.ones((1, tg.nt, 2)), jnp.ones((1, tg.nt-1, 1))
    
    monkeypatch.setattr(pdg_alsolver, "gradient_aug_lagrangian_trajectory", fake_grad_aug_lag_traj)

    # ---- Mock 3: dynamics residual is zero ----
    def fake_dyn_residual(cs, op, method):
        return jnp.zeros((tg.nt - 1, 2), dtype=jnp.float32)  # shape doesn't matter except nonempty

    monkeypatch.setattr(pdg_alsolver.systypes, "residual_discrete_dynamics_trajectory", fake_dyn_residual)

    # ---- Mock 4: constraint linearizations (only c and sl are used for stacks) ----
    ineq_lins = (
        SimpleNamespace(c=jnp.array([0.5], dtype=jnp.float32), sl=slice(0, 1)),
        SimpleNamespace(c=jnp.array([-2.0, 1.0], dtype=jnp.float32), sl=slice(1, 3)),
    )
    eq_lins = (
        SimpleNamespace(c=jnp.array([3.0, -4.0], dtype=jnp.float32), sl=slice(0, 2)),
    )

    monkeypatch.setattr(
        pdg_alsolver.contypes,
        "build_constraint_step_linearizations",
        lambda constraints, op: (ineq_lins, eq_lins),
    )

    # ---- Run outer loop for exactly 1 iteration ----
    op_out, al_out, diag = pdg_alsolver.al_solve_autodiff(
        nlgame,
        op0,
        al0,
        discretize_method="euler",
        ineq_activation="none",
        max_iters=1,
        rho_increase=10.0,
        rho_max=50.0,
        # ensure it doesn't early-exit due to convergence criteria:
        opt_tol=1e-12,
        dyn_tol=0.0,
        ineq_tol=0.0,
        eq_tol=0.0,
        newton_max_iters=1,
    )

    # op unchanged (since fake_newton returns op0)
    assert op_out is op0

    # Check lambda update:
    # c_ineq = [0.5, -2.0, 1.0]
    # lam_ineq_new = max(0, [0.1,0.2,0.3] + 2*[0.5,-2,1]) = max(0, [1.1, -3.8, 2.3]) = [1.1, 0, 2.3]
    expected_lam_ineq = np.array([1.1, 0.0, 2.3], dtype=np.float32)
    np.testing.assert_allclose(np.asarray(al_out.lam_ineq), expected_lam_ineq, atol=1e-6, rtol=1e-6)

    # c_eq = [3, -4]
    # lam_eq_new = [-1,1] + 0.5*[3,-4] = [0.5, -1]
    expected_lam_eq = np.array([0.5, -1.0], dtype=np.float32)
    np.testing.assert_allclose(np.asarray(al_out.lam_eq), expected_lam_eq, atol=1e-6, rtol=1e-6)

    # rho schedule:
    # rho_ineq = min(2*10, 50) => 20
    # rho_eq = min(0.5*10, 50) => 5
    np.testing.assert_allclose(np.asarray(al_out.rho_ineq), np.array([20.0, 20.0, 20.0], dtype=np.float32))
    np.testing.assert_allclose(np.asarray(al_out.rho_eq), np.array([5.0, 5.0], dtype=np.float32))

    # diag
    assert diag.converged is False
    assert diag.iters == 1
    assert diag.reason == "max_outer_iters"
    assert len(diag.history) == 1
    assert diag.history[0].outer_iter == 0

def test_al_solve_autodiff_rho_caps(monkeypatch):
    tg = TimeGrid(nt=4, dt=0.1, t0=0.0)

    nlgame = SimpleNamespace(
        tg=tg,
        cs=SimpleNamespace(),
        constraints=SimpleNamespace(nc_ineq=1, nc_eq=0, ineq_blocks=(), eq_blocks=(), tg=tg),
    )
    op0 = SimpleNamespace(tg=tg)

    al0 = _make_alstate_from_lists(
        lam_ineq=[0.0], rho_ineq=[2.0],
        lam_eq=[], rho_eq=[],
        dtype=jnp.float32,
    )

    fake_newton_diag = SimpleNamespace(converged=True, iters=1, reason="residual_tolerance_met", merit_norms=(1.0,), opt_vios=(1.0,), dyn_vios=(1.0,))
    monkeypatch.setattr(pdg_alsolver, "newton_solve_stationarity_autodiff", lambda *a, **k: (op0, fake_newton_diag))
    def fake_grad_aug_lag_traj(*a, **ka):
        return jnp.ones((1, tg.nt, 2)), jnp.ones((1, tg.nt-1, 1))
    monkeypatch.setattr(pdg_alsolver, "gradient_aug_lagrangian_trajectory", fake_grad_aug_lag_traj)
    monkeypatch.setattr(pdg_alsolver.systypes, "residual_discrete_dynamics_trajectory",
                        lambda cs, op, method: jnp.zeros((tg.nt - 1, 1), dtype=jnp.float32))

    # constant violated inequality c=+1
    ineq_lins = (SimpleNamespace(c=jnp.array([1.0], dtype=jnp.float32), sl=slice(0, 1)),)
    monkeypatch.setattr(pdg_alsolver.contypes, "build_constraint_step_linearizations",
                        lambda constraints, op: (ineq_lins, ()))

    # 3 outer iterations: rho: 2 -> 10 -> 10 -> 10 (cap at 10)
    op_out, al_out, diag = pdg_alsolver.al_solve_autodiff(
        nlgame,
        op0,
        al0,
        discretize_method="euler",
        ineq_activation="none",
        max_iters=3,
        rho_increase=10.0,
        rho_max=10.0,
        # prevent early convergence
        opt_tol=1e-12,
        dyn_tol=0.0,
        ineq_tol=0.0,
        eq_tol=0.0,
        newton_max_iters=1,
    )

    assert op_out is op0
    np.testing.assert_allclose(np.asarray(al_out.rho_ineq), np.array([10.0], dtype=np.float32))
    assert len(diag.history) == 3
    assert diag.reason == "max_outer_iters"

def test_al_solve_autodiff_converges_early_and_does_not_update_alstate(monkeypatch):
    nt = 4
    N, nx, nu = 1, 1, 1
    tg = TimeGrid(nt=4, dt=0.1, t0=0.0)

    nlgame = SimpleNamespace(
        tg=tg,
        nt=nt,
        nx=nx,
        cs=SimpleNamespace(),
        constraints=SimpleNamespace(nc_ineq=1, nc_eq=1, ineq_blocks=(), eq_blocks=(), tg=tg),
        u_splits=jnp.array([1])
    )
    op0 = SimpleNamespace(tg=tg)

    al0 = _make_alstate_from_lists(
        lam_ineq=[1.0], rho_ineq=[2.0],
        lam_eq=[-1.0], rho_eq=[3.0],
        dtype=jnp.float32,
    )

    # inner solve returns a tiny residual norm
    fake_newton_diag = SimpleNamespace(converged=True, iters=2, reason="residual_tolerance_met", merit_norms=(1e-9,), opt_vios=(1e-9,), dyn_vios=(1e-9,))
    monkeypatch.setattr(pdg_alsolver, "newton_solve_stationarity_autodiff", lambda *a, **k: (op0, fake_newton_diag))

    # optimality/stationarity conditions met (smaller than opt_vio but non-zero)
    monkeypatch.setattr(pdg_alsolver, "gradient_aug_lagrangian_trajectory", lambda *a, **k: (1e-9*jnp.ones((N, nt, nx)), 1e-9*jnp.ones((N, nt-1, nu))))

    # dyn residual is zero (feasible)
    monkeypatch.setattr(
        pdg_alsolver.systypes, "residual_discrete_dynamics_trajectory",
        lambda cs, op, method: jnp.zeros((tg.nt - 1, 1), dtype=jnp.float32)
    )

    # constraints are feasible: ineq <= 0, eq == 0
    ineq_lins = (SimpleNamespace(c=jnp.array([-1.0], dtype=jnp.float32), sl=slice(0, 1)),)
    eq_lins = (SimpleNamespace(c=jnp.array([0.0], dtype=jnp.float32), sl=slice(0, 1)),)
    monkeypatch.setattr(pdg_alsolver.contypes, "build_constraint_step_linearizations", lambda *_: (ineq_lins, eq_lins))

    # With tight tolerances, should exit at k=0 before updating alstate
    op_out, al_out, diag = pdg_alsolver.al_solve_autodiff(
        nlgame, op0, al0,
        discretize_method="euler",
        ineq_activation="none",
        max_iters=10,
        opt_tol=1e-6,
        dyn_tol=1e-6,
        ineq_tol=1e-6,
        eq_tol=1e-6,
        # schedules irrelevant if it converges early
        rho_increase=10.0, rho_max=50.0,
    )

    assert op_out is op0
    assert diag.converged is True
    assert diag.reason == "converged"
    assert diag.iters == 1
    assert len(diag.history) == 1

    # alstate unchanged if convergence triggers before updates
    np.testing.assert_allclose(np.asarray(al_out.lam_ineq), np.asarray(al0.lam_ineq))
    np.testing.assert_allclose(np.asarray(al_out.rho_ineq), np.asarray(al0.rho_ineq))
    np.testing.assert_allclose(np.asarray(al_out.lam_eq), np.asarray(al0.lam_eq))
    np.testing.assert_allclose(np.asarray(al_out.rho_eq), np.asarray(al0.rho_eq))
    

def test_al_solve_autodiff_does_not_converge_if_inner_reports_nonfinite_dyn(monkeypatch):
    nt = 4
    tg = TimeGrid(nt=nt, dt=0.1, t0=0.0)

    nlgame = SimpleNamespace(
        tg=tg,
        nt=nt,
        nx=1,
        cs=SimpleNamespace(),
        constraints=SimpleNamespace(nc_ineq=0, nc_eq=0, ineq_blocks=(), eq_blocks=(), tg=tg),
        u_splits=jnp.array([1], dtype=jnp.int32),
    )
    op0 = SimpleNamespace(tg=tg)

    al0 = altypes.JointAugmentedLagrangianState(
        lam_ineq=jnp.zeros((0,), dtype=jnp.float32),
        rho_ineq=jnp.zeros((0,), dtype=jnp.float32),
        lam_eq=jnp.zeros((0,), dtype=jnp.float32),
        rho_eq=jnp.zeros((0,), dtype=jnp.float32),
    )

    fake_newton_diag = altypes.StationarityNewtonDiag(
        converged=False,
        iters=0,
        opt_vios=(0.0,),
        dyn_vios=(float("nan"),),
        merit_norms=(0.0,),
        step_norms=tuple(),
        alphas=tuple(),
        regs=tuple(),
        accepted=tuple(),
        solve_ok=tuple(),
        reason="nonfinite_dyn_vio",
    )

    monkeypatch.setattr(
        pdg_alsolver,
        "newton_solve_stationarity_autodiff",
        lambda *a, **k: (op0, fake_newton_diag),
    )

    # empty constraints
    monkeypatch.setattr(pdg_alsolver.contypes, "build_constraint_step_linearizations", lambda *a, **k: ((), ()))

    op_out, al_out, diag = pdg_alsolver.al_solve_autodiff(
        nlgame, op0, al0,
        discretize_method="euler",
        ineq_activation="none",
        max_iters=2,
        opt_tol=1e6,
        dyn_tol=1e6,
        ineq_tol=1e6,
        eq_tol=1e6,
    )

    assert op_out is op0
    assert diag.converged is False
    assert diag.reason == "max_outer_iters"
    assert len(diag.history) == 2
    assert not np.isfinite(diag.history[0].dyn_vio_inf)

def test_al_solve_autodiff_nonmock_no_constraints_reduces_residual():
    tg = TimeGrid(nt=4, dt=0.2, t0=0.0)
    nt = tg.nt
    K = nt - 1
    nx = 2
    N = 2
    u_splits = jnp.array([1, 1], dtype=jnp.int32)
    nu = int(np.sum(np.asarray(u_splits)))
    dtype = jnp.float32

    # ---- linear continuous dynamics ----
    A = jnp.array([[0.1, -0.2],
                   [0.3,  0.0]], dtype=dtype)
    B = jnp.array([[1.0, 0.0],
                   [0.0, 2.0]], dtype=dtype)

    def f_cont(t, x, u):
        return A @ x + B @ u

    cs = SampledContinuousSystemType1(tg=tg, dynamics=f_cont, nx=nx, nu=nu)

    # ---- quadratic costs ----
    Q0 = jnp.diag(jnp.array([1.0, 2.0], dtype=dtype))
    R0 = jnp.array([[0.7]], dtype=dtype)
    Qf0 = jnp.diag(jnp.array([1.5, 0.5], dtype=dtype))

    Q1 = jnp.diag(jnp.array([0.5, 1.0], dtype=dtype))
    R1 = jnp.array([[1.2]], dtype=dtype)
    Qf1 = jnp.diag(jnp.array([0.8, 1.1], dtype=dtype))

    def running0(t, x, u0): return 0.5 * (x @ (Q0 @ x)) + 0.5 * (u0 @ (R0 @ u0))
    def terminal0(t, x):    return 0.5 * (x @ (Qf0 @ x))
    def running1(t, x, u1): return 0.5 * (x @ (Q1 @ x)) + 0.5 * (u1 @ (R1 @ u1))
    def terminal1(t, x):    return 0.5 * (x @ (Qf1 @ x))

    costs = [
        PlayerCostSpecContinuous(running=running0, terminal=terminal0, control_domain=CostControlDomain.LOCAL, control_coupling=CostControlStructure.LOCAL_ONLY),
        PlayerCostSpecContinuous(running=running1, terminal=terminal1, control_domain=CostControlDomain.LOCAL, control_coupling=CostControlStructure.LOCAL_ONLY)
    ]

    # ---- no auxiliary constraints ----
    constraints = GameConstraintGridMap(ineq_blocks=(), eq_blocks=())
    al0 = altypes.JointAugmentedLagrangianState(
        lam_ineq=jnp.zeros((0,), dtype=dtype),
        rho_ineq=jnp.zeros((0,), dtype=dtype),
        lam_eq=jnp.zeros((0,), dtype=dtype),
        rho_eq=jnp.zeros((0,), dtype=dtype),
    )

    nlgame = NonlinearGameType2(cs=cs, N=N, costs=costs, constraints=constraints, u_splits=u_splits)

    # ---- deterministic initial guess ----
    xs = (jnp.arange(nt * nx, dtype=dtype).reshape(nt, nx) - 2.0) * 0.1
    us = (jnp.arange(K * nu, dtype=dtype).reshape(K, nu) - 1.0) * 0.05
    ls = jnp.zeros((K, N, nx), dtype=dtype)
    op0 = FixedStepPrimalDualTrajectory(tg=tg, xs=xs, us=us, ls=ls)

    # ---- measure initial residual norm ----
    z0 = pdg_alsolver.pack_decision_vars_1d(op0)
    g0 = pdg_alsolver.compute_al_residual_flat_from_decision_vars(
        nlgame, z0, op0, al0,
        discretize_method="euler",
        ineq_activation="none",
    )
    g0_norm = float(jnp.linalg.norm(g0))
    assert np.isfinite(g0_norm)
    assert g0_norm > 0.0

    # ---- run outer AL solve (few outer iters; no constraints so λ/ρ won't matter) ----
    op_out, al_out, diag = pdg_alsolver.al_solve_autodiff(
        nlgame, op0, al0,
        discretize_method="euler",
        ineq_activation="none",
        max_iters=2,
        # keep tolerances modest for float32
        opt_tol=1e-4,
        dyn_tol=1e-4,
        ineq_tol=1e-6,
        eq_tol=1e-6,
        # inner solve controls
        newton_max_iters=8,
        newton_max_rejects=4,
        reg_init=0.0,
        reg_min_on_fail=1e-6,
        reg_increase=10.0,
        reg_max=1e6,
        reg_max_iters=32,
        ls_max_iters=12,
    )

    assert len(diag.history) >= 1
    # diagnostics should be finite (at least for this simple case)
    assert np.isfinite(diag.history[-1].dyn_vio_inf)
    assert np.isfinite(diag.history[-1].residual_norm_final)

    # AL state should remain empty and unchanged in size
    assert al_out.lam_ineq.shape == (0,)
    assert al_out.lam_eq.shape == (0,)
    assert al_out.rho_ineq.shape == (0,)
    assert al_out.rho_eq.shape == (0,)

    # ---- measure final residual norm (should improve) ----
    z1 = pdg_alsolver.pack_decision_vars_1d(op_out)
    g1 = pdg_alsolver.compute_al_residual_flat_from_decision_vars(
        nlgame, z1, op_out, al_out,
        discretize_method="euler",
        ineq_activation="none",
    )
    g1_norm = float(jnp.linalg.norm(g1))
    assert np.isfinite(g1_norm)

    # Strong but robust expectation: should reduce noticeably
    assert g1_norm <= 0.5 * g0_norm


def test_al_solve_autodiff_always_violated_ineq_updates_lambda_and_rho():
    tg = TimeGrid(nt=4, dt=0.2, t0=0.0)
    nt = tg.nt
    K = nt - 1
    nx = 2
    N = 2
    u_splits = jnp.array([1, 1], dtype=jnp.int32)
    nu = int(np.sum(np.asarray(u_splits)))
    dtype = jnp.float32

    # ---- simple linear continuous dynamics ----
    A = jnp.array([[0.1, -0.2],
                   [0.3,  0.0]], dtype=dtype)
    B = jnp.array([[1.0, 0.0],
                   [0.0, 2.0]], dtype=dtype)

    def f_cont(t, x, u):
        return A @ x + B @ u

    cs = SampledContinuousSystemType1(tg=tg, dynamics=f_cont, nx=nx, nu=nu)

    # ---- simple quadratic costs (same as earlier tests) ----
    Q = jnp.diag(jnp.array([1.0, 1.0], dtype=dtype))
    R = jnp.array([[1.0]], dtype=dtype)
    Qf = jnp.diag(jnp.array([1.0, 1.0], dtype=dtype))

    def running0(t, x, u0): return 0.5 * (x @ (Q @ x)) + 0.5 * (u0 @ (R @ u0))
    def terminal0(t, x):    return 0.5 * (x @ (Qf @ x))
    def running1(t, x, u1): return 0.5 * (x @ (Q @ x)) + 0.5 * (u1 @ (R @ u1))
    def terminal1(t, x):    return 0.5 * (x @ (Qf @ x))

    costs = [
        PlayerCostSpecContinuous(running=running0, terminal=terminal0, control_domain=CostControlDomain.LOCAL, control_coupling=CostControlStructure.LOCAL_ONLY),
        PlayerCostSpecContinuous(running=running1, terminal=terminal1, control_domain=CostControlDomain.LOCAL, control_coupling=CostControlStructure.LOCAL_ONLY)
    ]

    # ---- always-violated inequality constraint: c(t,x,u) = +1 (so c<=0 is violated) ----
    def c_always_violated(t, x, u):
        return jnp.array(1.0, dtype=x.dtype)  # scalar => normalized to (1,) in your linearization code

    # active at all stage steps by default when active_steps=None and terminal=False
    b = ConstraintBlockGridMap(
        tg=tg,
        func=c_always_violated,
        cdim_out_step=1,
        active_steps=None,     # defaults to range(nt-1)
        iseq=False,
        terminal=False,
    )
    constraints = GameConstraintGridMap(ineq_blocks=(b,), eq_blocks=())

    # ---- AL state sizes must match total scalar constraints ----
    # here: nc_ineq = K * 1 = 3, nc_eq = 0
    al0 = altypes.JointAugmentedLagrangianState(
        lam_ineq=jnp.zeros((K,), dtype=dtype),
        rho_ineq=jnp.ones((K,), dtype=dtype),
        lam_eq=jnp.zeros((0,), dtype=dtype),
        rho_eq=jnp.zeros((0,), dtype=dtype),
    )

    nlgame = NonlinearGameType2(cs=cs, N=N, costs=costs, constraints=constraints, u_splits=u_splits)

    # ---- deterministic initial guess ----
    xs = (jnp.arange(nt * nx, dtype=dtype).reshape(nt, nx) - 2.0) * 0.1
    us = (jnp.arange(K * nu, dtype=dtype).reshape(K, nu) - 1.0) * 0.05
    ls = jnp.zeros((K, N, nx), dtype=dtype)
    op0 = FixedStepPrimalDualTrajectory(tg=tg, xs=xs, us=us, ls=ls)

    # ---- run outer loop ----
    rho_increase = 10.0
    rho_max = 50.0
    outer_iters = 3

    op_out, al_out, diag = pdg_alsolver.al_solve_autodiff(
        nlgame, op0, al0,
        discretize_method="euler",
        ineq_activation="none",   # doesn't matter here
        max_iters=outer_iters,
        rho_increase=rho_increase,
        rho_max=rho_max,
        # prevent early convergence (ineq violation is always 1 anyway)
        opt_tol=1e-12,
        dyn_tol=1e-12,
        ineq_tol=1e-12,
        eq_tol=1e-12,
        # inner controls (keep small but stable)
        newton_max_iters=6,
        newton_max_rejects=3,
        reg_init=0.0,
        reg_min_on_fail=1e-6,
        reg_increase=10.0,
        reg_max=1e6,
        reg_max_iters=32,
        ls_max_iters=10,
    )

    assert diag.converged is False
    assert diag.reason == "max_outer_iters"
    assert len(diag.history) == outer_iters

    # ---- expected closed-form λ and ρ updates for this constant constraint ----
    # Outer loop does:
    #   λ <- max(0, λ + ρ*c)  with c=1
    #   ρ <- min(ρ*rho_increase, rho_max)
    #
    # Start: λ0=0, ρ0=1
    # iter1: λ1 = 0 + 1 = 1;   ρ1 = 10
    # iter2: λ2 = 1 + 10 = 11; ρ2 = 50 (cap)
    # iter3: λ3 = 11 + 50 = 61; ρ3 = 50
    expected_rho = np.array([50.0, 50.0, 50.0], dtype=np.float32)
    expected_lam = np.array([61.0, 61.0, 61.0], dtype=np.float32)

    np.testing.assert_allclose(np.asarray(al_out.rho_ineq), expected_rho, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(al_out.lam_ineq), expected_lam, atol=1e-6, rtol=1e-6)

    # equality stacks remain empty
    assert al_out.lam_eq.shape == (0,)
    assert al_out.rho_eq.shape == (0,)


def notest_al_vs_lqgame_unconstrained_2player_running_cost_only():
    # this test was meant to define a linear-quadratic game and then cross-validate
    # the solution of the AL solver with that of the LQ solver in pydgens.
    # However, this has been set to "notest" because the test is not passing and it is
    # expected that this is due to the fact that LQ solver is computing the 
    # feedback Nash solution while the AL solver is computing the open-loop solution
    # which are not guaranteed to match, even for simple problems
    # To better implement this test, we would need to implement the open-loop LQ solution
    # described here: 
    # https://github.com/HJReachability/ilqgames/blob/master/derivations/open_loop_lq_nash.pdf
    # but this is extra work that doesn't have use beyond this test at this point

    # ----- dimensions -----
    dtype = jnp.float32
    N = 2
    nx = 2
    u_splits = jnp.array([1, 1], dtype=jnp.int32)
    nu = int(np.sum(np.asarray(u_splits)))

    tg = TimeGrid(nt=6, dt=0.2, t0=0.0)
    nt = tg.nt
    K = nt - 1

    # ----- continuous dynamics: xdot = A_c x + B_c u -----
    A_c = jnp.array([[0.0, 1.0],
                     [-1.0, -0.4]], dtype=dtype)
    B_c = jnp.array([[1.0, 0.0],
                     [0.0, 1.0]], dtype=dtype)

    def f_cont(t, x, u):
        return A_c @ x + B_c @ u

    cs_cont = SampledContinuousSystemType1(tg=tg, dynamics=f_cont, nx=nx, nu=nu)

    # ----- discrete dynamics for LQ solver: Euler discretization -----
    dt = float(tg.dt)
    A_d = jnp.eye(nx, dtype=dtype) + dt * A_c
    B_d = dt * B_c

    A = jnp.tile(A_d[None, :, :], (nt, 1, 1))   # (nt,nx,nx)
    B = jnp.tile(B_d[None, :, :], (nt, 1, 1))   # (nt,nx,nu)
    cs_disc = LinearDiscreteSystemType1(tg=tg, nx=nx, nu=nu, A=A, B=B)

    # ----- costs (running only) -----
    Q0 = jnp.diag(jnp.array([4.0, 0.5], dtype=dtype))
    Q1 = jnp.diag(jnp.array([0.5, 4.0], dtype=dtype))
    r0 = jnp.array([[0.5]], dtype=dtype)  # scalar u0 cost
    r1 = jnp.array([[0.5]], dtype=dtype)  # scalar u1 cost

    # LQ tensors: Q(t,i), R(t,i) with block diagonal structure
    Q = jnp.zeros((nt, N, nx, nx), dtype=dtype)
    R = jnp.zeros((nt, N, nu, nu), dtype=dtype)
    q = jnp.zeros((nt, N, nx), dtype=dtype)
    r = jnp.zeros((nt, N, nu), dtype=dtype)

    for t in range(nt):
        Q = Q.at[t, 0].set(Q0)
        Q = Q.at[t, 1].set(Q1)

        # player 0 cost depends only on u0 (first control component)
        R = R.at[t, 0].set(jnp.diag(jnp.array([float(r0[0, 0]), 0.0], dtype=dtype)))
        # player 1 cost depends only on u1 (second control component)
        R = R.at[t, 1].set(jnp.diag(jnp.array([0.0, float(r1[0, 0])], dtype=dtype)))

    # IMPORTANT: last control is "meaningless" in your FixedStepSystemTrajectory.
    # To align, set R at terminal step to zero so LQ feedback does not care about u[nt-1].
    R = R.at[nt - 1].set(jnp.zeros((N, nu, nu), dtype=dtype))
    r = r.at[nt - 1].set(jnp.zeros((N, nu), dtype=dtype))

    lqgame = LinearQuadraticGameType1(
        cs=cs_disc,
        N=N,
        Q=Q, q=q,
        R=R, r=r,
        u_splits=u_splits,
    )

    # ----- solve feedback LQ game + propagate open-loop trajectory -----
    strategy = solve_lqgame_feedback(lqgame)

    x0 = jnp.array([1.0, -0.5], dtype=dtype)
    traj_lq = propagate_system_trajectory(lqgame.cs, x0=x0, strategy=strategy)

    # traj_lq: FixedStepSystemTrajectory with xs (nt,nx) and us (nt,nu)
    xs_lq = traj_lq.xs
    np.testing.assert_allclose(xs_lq[0], x0)    # check that x0 encoded into trajectory as expected
    us_lq = traj_lq.us[:-1]  # compare only first K controls

    # ----- build AL game (no constraints, running only, no terminal cost) -----
    constraints = GameConstraintGridMap(ineq_blocks=(), eq_blocks=())
    alstate0 = altypes.JointAugmentedLagrangianState(
        lam_ineq=jnp.zeros((0,), dtype=dtype),
        rho_ineq=jnp.zeros((0,), dtype=dtype),
        lam_eq=jnp.zeros((0,), dtype=dtype),
        rho_eq=jnp.zeros((0,), dtype=dtype),
    )

    # Running cost callables for AL: depend on (t, x, u_i)
    def running0(t, x, u0_):
        return 0.5 * (x @ (Q0 @ x)) + 0.5 * (u0_ @ (r0 @ u0_))

    def running1(t, x, u1_):
        return 0.5 * (x @ (Q1 @ x)) + 0.5 * (u1_ @ (r1 @ u1_))
    
    def terminal_zero(t, x):
        return jnp.array(0.0, dtype=x.dtype)

    costs = [
        PlayerCostSpecContinuous(running=running0, terminal=terminal_zero, control_domain=CostControlDomain.LOCAL, control_coupling=CostControlStructure.LOCAL_ONLY),
        PlayerCostSpecContinuous(running=running1, terminal=terminal_zero, control_domain=CostControlDomain.LOCAL, control_coupling=CostControlStructure.LOCAL_ONLY)
    ]

    nlgame = NonlinearGameType2(
        cs=cs_cont,
        N=N,
        costs=costs,
        constraints=constraints,
        u_splits=u_splits,
    )

    # ----- initial guess for AL: simple (x0 then zeros) -----
    xs0 = jnp.zeros((nt, nx), dtype=dtype).at[0].set(x0)
    us0 = jnp.zeros((K, nu), dtype=dtype)
    ls0 = jnp.zeros((K, N, nx), dtype=dtype)
    op0 = FixedStepPrimalDualTrajectory(tg=tg, xs=xs0, us=us0, ls=ls0)

    # ----- run AL outer loop (no constraints => outer loop largely irrelevant) -----
    op_out, al_out, diag = pdg_alsolver.al_solve_autodiff(
        nlgame,
        op0,
        alstate0,
        discretize_method="euler",
        ineq_activation="none",
        max_iters=1,          # no aux constraints => 1 outer iter is enough
        residual_tol=1e-6,
        dyn_tol=1e-6,
        ineq_tol=1e-7,
        eq_tol=1e-7,
        newton_max_iters=32,
        newton_max_rejects=4,
        newton_step_tol=1e-10,
        reg_init=0.0,
        reg_min_on_fail=1e-6,
        reg_increase=10.0,
        reg_max=1e6,
        reg_max_iters=32,
        ls_max_iters=12,
    )
    assert diag.converged 
    assert diag.iters == 1
    assert diag.history[0].newton_converged
    assert diag.history[0].newton_iters == 1

    # ----- compare AL open-loop to LQ-induced open-loop -----
    xs_al = op_out.xs
    us_al = op_out.us

    # sanity shapes
    assert xs_al.shape == (nt, nx)
    assert us_al.shape == (K, nu)
    assert xs_lq.shape == (nt, nx)
    assert us_lq.shape == (K, nu)

    # manual propagation checks for dynamics matching
    np.testing.assert_allclose(np.asarray(us_al[0]), np.asarray(us_lq[0]), atol=5e-2, rtol=5e-2)
    x1_al_manual = A_d @ x0 + B_d @ us_al[0]
    np.testing.assert_allclose(xs_al[1], x1_al_manual, atol=1e-6, rtol=1e-6)
    x1_lq_manual = A_d @ x0 + B_d @ us_lq[0]
    np.testing.assert_allclose(xs_lq[1], x1_lq_manual, atol=1e-6, rtol=1e-6)

    # "close enough" tolerances for float32 and solver differences
    # Start with loose tolerances; tighten once you see typical error magnitudes.
    np.testing.assert_allclose(np.asarray(us_al), np.asarray(us_lq), atol=5e-2, rtol=5e-2)
    np.testing.assert_allclose(np.asarray(xs_al), np.asarray(xs_lq), atol=5e-2, rtol=5e-2)

    # AL state should remain empty
    assert al_out.lam_ineq.shape == (0,)
    assert al_out.lam_eq.shape == (0,)


def test_al_solve_autodiff_lq_openloop_residual_affine_one_newton_step_solves():
    """
    For a deterministic linear dynamics + quadratic costs game (no auxiliary constraints),
    the AL residual map G(z) should be affine in z. Therefore, one exact Newton step
    using the autodiff Jacobian should drive ||G|| close to zero (up to float32 error).
    """
    dtype = jnp.float32
    N = 2
    nx = 2
    u_splits = jnp.array([1, 1], dtype=jnp.int32)
    nu = int(np.sum(np.asarray(u_splits)))

    tg = TimeGrid(nt=6, dt=0.2, t0=0.0)
    nt = tg.nt
    K = nt - 1

    # ---- continuous linear dynamics (Euler discretization used under the hood) ----
    A_c = jnp.array([[0.0, 1.0],
                     [-1.0, -0.4]], dtype=dtype)
    B_c = jnp.array([[1.0, 0.0],
                     [0.0, 1.0]], dtype=dtype)

    def f_cont(t, x, u):
        return A_c @ x + B_c @ u

    cs_cont = SampledContinuousSystemType1(tg=tg, dynamics=f_cont, nx=nx, nu=nu)

    # ---- quadratic running costs; terminal cost = 0 ----
    Q0 = jnp.diag(jnp.array([4.0, 0.5], dtype=dtype))
    Q1 = jnp.diag(jnp.array([0.5, 4.0], dtype=dtype))
    R0 = jnp.array([[0.5]], dtype=dtype)  # u0 scalar cost
    R1 = jnp.array([[0.5]], dtype=dtype)  # u1 scalar cost

    def running0(t, x, u0_):
        return 0.5 * (x @ (Q0 @ x)) + 0.5 * (u0_ @ (R0 @ u0_))

    def running1(t, x, u1_):
        return 0.5 * (x @ (Q1 @ x)) + 0.5 * (u1_ @ (R1 @ u1_))

    def terminal_zero(t, x):
        return jnp.array(0.0, dtype=x.dtype)

    costs = [
        PlayerCostSpecContinuous(running=running0, terminal=terminal_zero, control_domain=CostControlDomain.LOCAL, control_coupling=CostControlStructure.LOCAL_ONLY),
        PlayerCostSpecContinuous(running=running1, terminal=terminal_zero, control_domain=CostControlDomain.LOCAL, control_coupling=CostControlStructure.LOCAL_ONLY)
    ]

    # ---- no auxiliary constraints ----
    constraints = GameConstraintGridMap(ineq_blocks=(), eq_blocks=())
    alstate0 = altypes.JointAugmentedLagrangianState(
        lam_ineq=jnp.zeros((0,), dtype=dtype),
        rho_ineq=jnp.zeros((0,), dtype=dtype),
        lam_eq=jnp.zeros((0,), dtype=dtype),
        rho_eq=jnp.zeros((0,), dtype=dtype),
    )

    nlgame = NonlinearGameType2(
        cs=cs_cont,
        N=N,
        costs=costs,
        constraints=constraints,
        u_splits=u_splits,
    )

    # ---- build a deterministic initial primal-dual trajectory ----
    x0 = jnp.array([1.0, -0.5], dtype=dtype)

    xs0 = (jnp.arange(nt * nx, dtype=dtype).reshape(nt, nx) * 0.1).at[0].set(x0)
    us0 = jnp.linspace(-0.2, 0.3, K * nu, dtype=dtype).reshape(K, nu)
    ls0 = jnp.zeros((K, N, nx), dtype=dtype)
    op0 = FixedStepPrimalDualTrajectory(tg=tg, xs=xs0, us=us0, ls=ls0)

    # ---- helper: compute packed residual g(z) via unpack -> residual_struct -> pack ----
    def G_of_z(z: jnp.ndarray) -> jnp.ndarray:
        op = pdg_alsolver.unpack_decision_vars(z, op0, check_length=True)
        Gs = pdg_alsolver.compute_al_residual_struct_from_traj(
            nlgame,
            op,
            alstate0,
            discretize_method="euler",
            ineq_activation="none",
        )
        return pdg_alsolver.pack_al_residual_1d(Gs, nlgame.u_splits)

    z0 = pdg_alsolver.pack_decision_vars_1d(op0)
    z0 = jnp.asarray(z0, dtype=dtype)

    g0 = G_of_z(z0)
    g0_norm = float(jnp.linalg.norm(g0))
    assert np.isfinite(g0_norm) and g0_norm > 0.0

    # ---- autodiff Jacobian at z0 ----
    H0 = pdg_alsolver.jacobian_al_residual_flat_autodiff(
        nlgame,
        z0,
        op0,
        alstate0,
        discretize_method="euler",
        ineq_activation="none",
        mode="jacfwd",
    )

    # ---- one Newton step ----
    # Use a small Tikhonov reg to avoid occasional singular/ill-conditioned solves in float32
    reg = 1e-6
    I = jnp.eye(H0.shape[0], dtype=H0.dtype)
    dz = jnp.linalg.solve(H0 + reg * I, -g0)
    z1 = z0 + dz

    g1 = G_of_z(z1)
    g1_norm = float(jnp.linalg.norm(g1))

    # For a truly affine G, one step should essentially solve it (up to numerical error)
    # Choose tolerance appropriate for float32 and conditioning.
    assert g1_norm <= 1e-3 * g0_norm
    assert g1_norm <= 1e-4  # absolute “near zero” threshold for this small problem

    # ---- optional: Jacobian invariance check (affine residual => constant Jacobian) ----
    H1 = pdg_alsolver.jacobian_al_residual_flat_autodiff(
        nlgame,
        z1,
        op0,
        alstate0,
        discretize_method="euler",
        ineq_activation="none",
        mode="jacfwd",
    )
    np.testing.assert_allclose(np.asarray(H1), np.asarray(H0), atol=5e-4, rtol=5e-4)


def test_al_solve_autodiff_one_step_box_constrained_quadratic_clips_to_bound():
    # This represents a static optimization problem (i.e. not really a "game") with constraints
    # where the optimal solution is on the control constraint bounds
    dtype = jnp.float32

    # ---- problem constants ----
    a = 2.0          # unconstrained minimizer
    u_max = 0.7      # constraint bound
    u_star = min(max(a, -u_max), u_max)  # clip

    # ---- dimensions ----
    N = 1
    nx = 1
    nu = 1
    u_splits = jnp.array([1], dtype=jnp.int32)

    tg = TimeGrid(nt=2, dt=0.2, t0=0.0)   # K=1 step
    dt = float(tg.dt)

    # ---- dynamics: Euler step yields x1 = u0 ----
    # xdot = (u - x)/dt  -> x_next = x + dt*(u-x)/dt = u
    def f_cont(t, x, u):
        return (u - x) / dt

    cs = SampledContinuousSystemType1(tg=tg, dynamics=f_cont, nx=nx, nu=nu)

    # ---- cost: running only, depends on local control u ----
    def running0(t, x, u0):
        # u0 is shape (1,)
        return 0.5 * (u0[0] - a) ** 2

    def terminal_zero(t, x):
        return jnp.array(0.0, dtype=x.dtype)

    # costs = [SimpleNamespace(running=running0, terminal=terminal_zero)]
    costs = [
        PlayerCostSpecContinuous(running=running0, terminal=terminal_zero, control_domain=CostControlDomain.LOCAL, control_coupling=CostControlStructure.LOCAL_ONLY),
    ]

    # ---- inequality constraint block: |u| <= u_max at k=0 ----
    # Returns shape (2,)
    def u_box_constraint(t, x, u):
        # u shape (1,)
        return jnp.array([u[0] - u_max, -u[0] - u_max], dtype=u.dtype)

    b_ineq = ConstraintBlockGridMap(
        tg=tg,
        func=u_box_constraint,
        cdim_out_step=2,
        active_steps=(0,),   # only step is k=0
        iseq=False,
        terminal=False,
    )
    constraints = GameConstraintGridMap(ineq_blocks=(b_ineq,), eq_blocks=())

    # ---- AL state: nc_ineq = 2, nc_eq = 0 ----
    alstate0 = altypes.JointAugmentedLagrangianState(
        lam_ineq=jnp.zeros((2,), dtype=dtype),
        rho_ineq=jnp.ones((2,), dtype=dtype),
        lam_eq=jnp.zeros((0,), dtype=dtype),
        rho_eq=jnp.zeros((0,), dtype=dtype),
    )

    nlgame = NonlinearGameType2(
        cs=cs,
        N=N,
        costs=costs,
        constraints=constraints,
        u_splits=u_splits,
    )

    # ---- initial trajectory guess ----
    # x0 fixed; x1 decision var, u0 decision var; ls is dynamics multiplier (K,N,nx) = (1,1,1)
    x0 = jnp.array([0.0], dtype=dtype)
    xs0 = jnp.stack([x0, jnp.array([0.0], dtype=dtype)], axis=0)  # (nt=2,nx=1)
    us0 = jnp.array([[0.0]], dtype=dtype)                         # (K=1,nu=1)
    ls0 = jnp.zeros((1, 1, 1), dtype=dtype)                      # (K,N,nx)

    op0 = FixedStepPrimalDualTrajectory(tg=tg, xs=xs0, us=us0, ls=ls0)

    # ---- solve ----
    op_out, al_out, diag = pdg_alsolver.al_solve_autodiff(
        nlgame,
        op0,
        alstate0,
        discretize_method="euler",  # euler is important for this test, though rk2 is usually the default
        ineq_activation="altro"
    )

    u0_sol = float(op_out.us[0, 0])
    x1_sol = float(op_out.xs[1, 0])

    # for i, h in enumerate(diag.history):
    #     print(f"iter {i}\n{h}\n")
    # print(f"Final Result: \nx1 = {x1_sol}\nu0 = {u0_sol}\n")

    assert diag.converged, f"AL did not converge: reason={diag.reason}, last={diag.history[-1]}"

    # ---- check solution ----

    # Since x1 = u0 under our Euler-engineered dynamics, both should match u_star
    assert u0_sol == pytest.approx(u_star, abs=5e-3, rel=5e-3)
    assert x1_sol == pytest.approx(u_star, abs=5e-3, rel=5e-3)

    # feasibility: constraint values at solution should be <= 0 within tolerance
    c_val = np.asarray(u_box_constraint(0.0, op_out.xs[0], op_out.us[0]))
    assert np.max(c_val) <= 5e-3


def test_al_solve_autodiff_two_player_mixed_constraints_converges_and_satisfies_constraints():
    dtype = jnp.float32

    # ---- dimensions ----
    N = 2
    nx = 2
    u_splits = jnp.array([1, 1], dtype=jnp.int32)
    nu = int(np.sum(np.asarray(u_splits)))
    assert nu == 2

    tg = TimeGrid(nt=6, dt=0.2, t0=0.0)  # K=5
    K = tg.nt - 1

    # ---- continuous dynamics (simple stable-ish linear system) ----
    A = jnp.array([[0.0, 1.0],
                   [-0.5, -0.2]], dtype=dtype)
    B = jnp.array([[1.0, 0.0],
                   [0.0, 1.0]], dtype=dtype)

    def f_cont(t, x, u):
        return A @ x + B @ u

    cs = SampledContinuousSystemType1(tg=tg, dynamics=f_cont, nx=nx, nu=nu)

    # ---- costs: both players want x near 0, but have different control effort ----
    Q0 = jnp.diag(jnp.array([2.0, 0.5], dtype=dtype))
    Q1 = jnp.diag(jnp.array([0.5, 2.0], dtype=dtype))
    R0 = jnp.array([[0.2]], dtype=dtype)
    R1 = jnp.array([[0.4]], dtype=dtype)

    def running0(t, x, u0):
        return 0.5 * (x @ (Q0 @ x)) + 0.5 * (u0 @ (R0 @ u0))

    def running1(t, x, u1):
        return 0.5 * (x @ (Q1 @ x)) + 0.5 * (u1 @ (R1 @ u1))

    def terminal_zero(t, x):
        return jnp.array(0.0, dtype=x.dtype)

    costs = [
        PlayerCostSpecContinuous(
            running=running0,
            terminal=terminal_zero,
            control_domain=CostControlDomain.LOCAL,
            control_coupling=CostControlStructure.LOCAL_ONLY,
        ),
        PlayerCostSpecContinuous(
            running=running1,
            terminal=terminal_zero,
            control_domain=CostControlDomain.LOCAL,
            control_coupling=CostControlStructure.LOCAL_ONLY,
        ),
    ]

    # ---- constraints ----
    # control bounds per-player at all stage steps (k=0..K-1)
    u0_max = 0.6
    u1_max = 0.4

    def u0_box(t, x, u):
        # c <= 0: u0 - umax <= 0 AND -u0 - umax <= 0
        return jnp.array([u[0] - u0_max, -u[0] - u0_max], dtype=u.dtype)

    def u1_box(t, x, u):
        return jnp.array([u[1] - u1_max, -u[1] - u1_max], dtype=u.dtype)

    b_u0 = ConstraintBlockGridMap(
        tg=tg,
        func=u0_box,
        cdim_out_step=2,
        active_steps=tuple(range(K)),
        iseq=False,
        terminal=False,
    )
    b_u1 = ConstraintBlockGridMap(
        tg=tg,
        func=u1_box,
        cdim_out_step=2,
        active_steps=tuple(range(K)),
        iseq=False,
        terminal=False,
    )

    # terminal equality: x_T[0] == target
    xT_target = 0.25

    def terminal_eq(t, x):
        return jnp.array([x[0] - xT_target], dtype=x.dtype)

    b_xT = ConstraintBlockGridMap(
        tg=tg,
        func=terminal_eq,
        cdim_out_step=1,
        active_steps=None,   # will default to (nt-1,) due to terminal=True in your class
        iseq=True,
        terminal=True,
    )

    constraints = GameConstraintGridMap(
        ineq_blocks=(b_u0, b_u1),
        eq_blocks=(b_xT,),
    )

    # ---- AL state dims ----
    # Each box block contributes: 2 * K constraints
    nc_ineq = 2 * K + 2 * K  # b_u0 + b_u1
    nc_eq = 1  # terminal equality
    alstate0 = altypes.JointAugmentedLagrangianState(
        lam_ineq=jnp.zeros((nc_ineq,), dtype=dtype),
        rho_ineq=jnp.ones((nc_ineq,), dtype=dtype),
        lam_eq=jnp.zeros((nc_eq,), dtype=dtype),
        rho_eq=jnp.ones((nc_eq,), dtype=dtype),
    )

    nlgame = NonlinearGameType2(
        cs=cs,
        N=N,
        costs=costs,
        constraints=constraints,
        u_splits=u_splits,
    )

    # ---- initial guess trajectory ----
    x0 = jnp.array([1.0, -0.5], dtype=dtype)
    xs0 = jnp.zeros((tg.nt, nx), dtype=dtype).at[0].set(x0)
    us0 = jnp.zeros((K, nu), dtype=dtype)
    ls0 = jnp.zeros((K, N, nx), dtype=dtype)

    op0 = FixedStepPrimalDualTrajectory(tg=tg, xs=xs0, us=us0, ls=ls0)

    # ---- solve ----
    op_out, al_out, diag = pdg_alsolver.al_solve_autodiff(
        nlgame,
        op0,
        alstate0,
        discretize_method="rk2",
        ineq_activation="altro",
        # max_iters=10,
        # rho_increase=10.0,
        # rho_max=1e6,
        # # match your new convergence style (ALGAMES-like)
        opt_tol=1e-3,
        dyn_tol=1e-4,
        ineq_tol=1e-4,
        eq_tol=1e-4,
        # # inner
        # newton_max_iters=20,
        # newton_max_rejects=6,
        # newton_step_rtol=1e-7,
        # newton_step_atol=1e-8,
        # reg_init=0.0,
        # reg_min_on_fail=1e-6,
        # reg_increase=10.0,
        # reg_max=1e6,
        # reg_max_iters=32,
        # ls_max_iters=25,
        # ls_alpha0=1.0,
        # ls_tau=0.5,
        # ls_beta=0.25,
        # normkind="l1_mean",
    )

    # for i, h in enumerate(diag.history):
    #     print(f"iter {i}\n{h}\n")
    # # print(f"Final Result: \nx1 = {x1_sol}\nu0 = {u0_sol}\n")

    assert diag.converged, f"did not converge: reason={diag.reason}, last={diag.history[-1]}"

    # ---- feasibility checks (explicit) ----
    # dynamics residual
    D = residual_discrete_dynamics_trajectory(cs, op_out, method="rk2")
    dyn_vio = float(jnp.max(jnp.abs(D))) if D.size else 0.0
    assert dyn_vio <= 5e-4

    # constraint values
    ineq_lins, eq_lins = build_constraint_step_linearizations(constraints, op_out)
    c_ineq, c_eq = pdg_alsolver._collect_constraint_stacks_from_linearizations(
        constraints, ineq_lins, eq_lins, dtype=dtype
    )
    ineq_vio, eq_vio = pdg_alsolver._constraint_violation_metrics(c_ineq, c_eq)
    assert ineq_vio <= 5e-4
    assert eq_vio <= 5e-4

    # terminal equality specifically
    xT0 = float(op_out.xs[-1, 0])
    assert xT0 == pytest.approx(xT_target, abs=5e-3)

    # control bounds explicitly
    u = np.asarray(op_out.us)
    assert np.max(np.abs(u[:, 0])) <= u0_max + 5e-4
    assert np.max(np.abs(u[:, 1])) <= u1_max + 5e-4
