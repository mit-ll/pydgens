# Copyright 2026 MIT Lincoln Laboratory
# SPDX-License-Identifier: MIT

import pytest
import jax.numpy as jnp
import numpy as np

from dataclasses import FrozenInstanceError

# import pydgens helper funcs and classes
from pydgens.ir.timetypes import TimeGrid
from pydgens.ir.trajectorytypes import FixedStepPrimalDualTrajectory

# import module under test
import pydgens.ir.constrainttypes as pdg_con


# -------------------------
# Helpers
# -------------------------

def _step_con_scalar(t, x, u):
    return x[0] + u[0] + 0.0 * t  # scalar

def _step_con_vec2(t, x, u):
    return x[:2] + u[:2] + 0.0 * t  # (2,)

def _term_con_scalar(t, x):
    return x[0] + 0.0 * t  # scalar


# -------------------------
# ConstraintBlockGridMap tests
# -------------------------

def test_con_block_nonterminal_defaults_active_steps_to_0_to_nt_minus_2():
    tg = TimeGrid(nt=5, dt=0.1, t0=0.0)

    b = pdg_con.ConstraintBlockGridMap(
        tg=tg,
        func=_step_con_scalar,
        cdim_out_step=1,
        active_steps=None,
        iseq=False,
        terminal=False,
    )

    assert b.active_steps == (0, 1, 2, 3)  # nt-1 steps
    assert b.n_active_steps == 4
    assert b.nc_block == 1 * 4


def test_con_block_terminal_defaults_active_steps_to_terminal_only():
    tg = TimeGrid(nt=6, dt=0.1, t0=0.0)

    b = pdg_con.ConstraintBlockGridMap(
        tg=tg,
        func=_term_con_scalar,
        cdim_out_step=1,
        active_steps=None,
        iseq=True,
        terminal=True,
    )

    assert b.active_steps == (tg.nt - 1,)
    assert b.n_active_steps == 1
    assert b.nc_block == 1


def test_con_block_terminal_rejects_nonterminal_active_steps():
    tg = TimeGrid(nt=6, dt=0.1, t0=0.0)

    with pytest.raises(ValueError, match="terminal=True implies active_steps"):
        pdg_con.ConstraintBlockGridMap(
            tg=tg,
            func=_term_con_scalar,
            cdim_out_step=1,
            active_steps=(0,),  # illegal for terminal=True
            iseq=True,
            terminal=True,
        )


def test_con_block_rejects_bad_tg_type():
    with pytest.raises(TypeError, match="tg must be TimeGrid"):
        pdg_con.ConstraintBlockGridMap(
            tg="not_a_timegrid",
            func=_step_con_scalar,
            cdim_out_step=1,
            active_steps=None,
        )


def test_con_block_rejects_noncallable_func():
    tg = TimeGrid(nt=4, dt=0.1, t0=0.0)

    with pytest.raises(TypeError, match="func must be callable"):
        pdg_con.ConstraintBlockGridMap(
            tg=tg,
            func=123,
            cdim_out_step=1,
            active_steps=None,
        )


@pytest.mark.parametrize("bad_cdim", [0, -1, 1.5, "2"])
def test_con_block_rejects_bad_cdim_out_step(bad_cdim):
    tg = TimeGrid(nt=4, dt=0.1, t0=0.0)

    with pytest.raises(ValueError, match="cdim_out_step must be a positive int"):
        pdg_con.ConstraintBlockGridMap(
            tg=tg,
            func=_step_con_scalar,
            cdim_out_step=bad_cdim,
            active_steps=None,
        )


def test_con_block_rejects_active_steps_not_tuple():
    tg = TimeGrid(nt=5, dt=0.1, t0=0.0)

    with pytest.raises(TypeError, match="active_steps must be a tuple"):
        pdg_con.ConstraintBlockGridMap(
            tg=tg,
            func=_step_con_scalar,
            cdim_out_step=1,
            active_steps=[0, 1],  # must be tuple[int,...]
            terminal=False,
        )


def test_con_block_rejects_active_steps_non_int():
    tg = TimeGrid(nt=5, dt=0.1, t0=0.0)

    with pytest.raises(TypeError, match="active_steps entries must be int"):
        pdg_con.ConstraintBlockGridMap(
            tg=tg,
            func=_step_con_scalar,
            cdim_out_step=1,
            active_steps=(0, "1"),
            terminal=False,
        )


def test_con_block_rejects_active_steps_out_of_range():
    tg = TimeGrid(nt=5, dt=0.1, t0=0.0)

    with pytest.raises(ValueError, match=r"active_steps entry .* out of range"):
        pdg_con.ConstraintBlockGridMap(
            tg=tg,
            func=_step_con_scalar,
            cdim_out_step=1,
            active_steps=(0, 5),  # valid indices are 0..4
            terminal=False,
        )


def test_con_block_rejects_active_steps_not_strictly_increasing():
    tg = TimeGrid(nt=6, dt=0.1, t0=0.0)

    # unsorted
    with pytest.raises(ValueError, match="active_steps must be strictly increasing"):
        pdg_con.ConstraintBlockGridMap(
            tg=tg,
            func=_step_con_scalar,
            cdim_out_step=1,
            active_steps=(2, 1),
            terminal=False,
        )

    # duplicate
    with pytest.raises(ValueError, match="active_steps must be strictly increasing"):
        pdg_con.ConstraintBlockGridMap(
            tg=tg,
            func=_step_con_scalar,
            cdim_out_step=1,
            active_steps=(1, 1),
            terminal=False,
        )


def test_con_block_is_frozen():
    tg = TimeGrid(nt=5, dt=0.1, t0=0.0)

    b = pdg_con.ConstraintBlockGridMap(
        tg=tg, func=_step_con_scalar, cdim_out_step=1, terminal=False
    )

    with pytest.raises(FrozenInstanceError):
        b.cdim_out_step = 99


# -------------------------
# GameConstraintGridMap tests
# -------------------------

def test_game_constraint_map_enforces_block_classification():
    tg = TimeGrid(nt=5, dt=0.1, t0=0.0)

    b_bad_for_ineq = pdg_con.ConstraintBlockGridMap(
        tg=tg, func=_step_con_scalar, cdim_out_step=1, iseq=True, terminal=False
    )
    with pytest.raises(ValueError, match="All blocks in ineq_blocks must have iseq=False"):
        pdg_con.GameConstraintGridMap(ineq_blocks=(b_bad_for_ineq,), eq_blocks=())

    b_bad_for_eq = pdg_con.ConstraintBlockGridMap(
        tg=tg, func=_step_con_scalar, cdim_out_step=1, iseq=False, terminal=False
    )
    with pytest.raises(ValueError, match="All blocks in eq_blocks must have iseq=True"):
        pdg_con.GameConstraintGridMap(ineq_blocks=(), eq_blocks=(b_bad_for_eq,))


def test_game_constraint_map_enforces_same_timegrid():
    tg1 = TimeGrid(nt=5, dt=0.1, t0=0.0)
    tg2 = TimeGrid(nt=6, dt=0.1, t0=0.0)

    b1 = pdg_con.ConstraintBlockGridMap(
        tg=tg1, func=_step_con_scalar, cdim_out_step=1, iseq=False, terminal=False
    )
    b2 = pdg_con.ConstraintBlockGridMap(
        tg=tg2, func=_step_con_scalar, cdim_out_step=1, iseq=False, terminal=False
    )

    with pytest.raises(ValueError, match="All blocks must share the same TimeGrid"):
        pdg_con.GameConstraintGridMap(ineq_blocks=(b1, b2), eq_blocks=())


def test_game_constraint_map_counts_constraints_correctly():
    tg = TimeGrid(nt=6, dt=0.1, t0=0.0)
    # nt=6 => stage default active_steps length is nt-1 = 5

    # Two ineq blocks:
    # - block A outputs 2 dims per step, active at steps (0,2,4) => 2*3 = 6 constraints
    # - block B outputs 1 dim per step, active default (0..4) => 1*5 = 5 constraints
    bA = pdg_con.ConstraintBlockGridMap(
        tg=tg,
        func=_step_con_vec2,
        cdim_out_step=2,
        active_steps=(0, 2, 4),
        iseq=False,
        terminal=False,
    )
    bB = pdg_con.ConstraintBlockGridMap(
        tg=tg,
        func=_step_con_scalar,
        cdim_out_step=1,
        active_steps=None,
        iseq=False,
        terminal=False,
    )

    # One eq block: terminal-only, scalar => 1 constraint total
    bT = pdg_con.ConstraintBlockGridMap(
        tg=tg,
        func=_term_con_scalar,
        cdim_out_step=1,
        active_steps=None,
        iseq=True,
        terminal=True,
    )

    C = pdg_con.GameConstraintGridMap(ineq_blocks=(bA, bB), eq_blocks=(bT,))

    assert C.nc_blocks_ineq == 2
    assert C.nc_blocks_eq == 1

    assert C.nc_ineq == 6 + 5
    assert C.nc_eq == 1
    assert C.nc_all == (6 + 5 + 1)


def test_game_constraint_map_is_frozen():
    tg = TimeGrid(nt=5, dt=0.1, t0=0.0)

    b = pdg_con.ConstraintBlockGridMap(
        tg=tg, func=_step_con_scalar, cdim_out_step=1, iseq=False, terminal=False
    )
    C = pdg_con.GameConstraintGridMap(ineq_blocks=(b,), eq_blocks=())

    with pytest.raises(FrozenInstanceError):
        C.ineq_blocks = ()


def _dummy_step_con(t, x, u):
    # scalar constraint
    return jnp.array(0.0)


def _dummy_term_con(t, x):
    return jnp.array(0.0)


def test_game_constraints_tg_none_when_no_blocks():
    gc = pdg_con.GameConstraintGridMap()
    assert gc.tg is None


def test_game_constraints_tg_cached_from_ineq_blocks():
    tg = TimeGrid(nt=5, dt=0.1, t0=0.0)

    b = pdg_con.ConstraintBlockGridMap(
        tg=tg,
        func=_dummy_step_con,
        cdim_out_step=1,
        active_steps=(0, 1, 2),   # arbitrary
        iseq=False,
        terminal=False,
    )

    gc = pdg_con.GameConstraintGridMap(ineq_blocks=(b,), eq_blocks=())
    assert gc.tg == tg
    assert gc.tg is tg  # if you reuse the same object, this should hold


def test_game_constraints_tg_cached_from_eq_blocks_only():
    tg = TimeGrid(nt=6, dt=0.2, t0=1.0)

    b = pdg_con.ConstraintBlockGridMap(
        tg=tg,
        func=_dummy_term_con,
        cdim_out_step=1,
        active_steps=None,   # terminal=True should default to (nt-1,)
        iseq=True,
        terminal=True,
    )

    gc = pdg_con.GameConstraintGridMap(ineq_blocks=(), eq_blocks=(b,))
    assert gc.tg == tg


def test_game_constraints_rejects_mismatched_timegrids():
    tg1 = TimeGrid(nt=5, dt=0.1, t0=0.0)
    tg2 = TimeGrid(nt=5, dt=0.1, t0=0.5)  # different t0 (or dt/nt)

    b1 = pdg_con.ConstraintBlockGridMap(
        tg=tg1, func=_dummy_step_con, cdim_out_step=1, active_steps=(0,), iseq=False, terminal=False
    )
    b2 = pdg_con.ConstraintBlockGridMap(
        tg=tg2, func=_dummy_step_con, cdim_out_step=1, active_steps=(0,), iseq=False, terminal=False
    )

    with pytest.raises(ValueError, match="TimeGrid"):
        pdg_con.GameConstraintGridMap(ineq_blocks=(b1, b2))


def test_game_constraints_tg_is_immutable():
    tg = TimeGrid(nt=5, dt=0.1, t0=0.0)
    b = pdg_con.ConstraintBlockGridMap(
        tg=tg, func=_dummy_step_con, cdim_out_step=1, active_steps=(0,), iseq=False, terminal=False
    )
    gc = pdg_con.GameConstraintGridMap(ineq_blocks=(b,))

    with pytest.raises((AttributeError, TypeError)):
        gc.tg = TimeGrid(nt=5, dt=0.1, t0=1.0)


def test_build_constraint_step_linearizations_empty_returns_empty():
    nt, dt, nx, nu, N = 5, 0.1, 3, 2, 2
    tg = TimeGrid(nt=nt, dt=dt, t0=0.0)
    xs = jnp.zeros((tg.nt, nx), dtype=jnp.float32)
    us = jnp.zeros((tg.nt - 1, nu), dtype=jnp.float32)
    ls = jnp.zeros((tg.nt - 1, N, nx), dtype=jnp.float32)   # this shouldn't matter
    op = FixedStepPrimalDualTrajectory(tg=tg, xs=xs, us=us, ls=ls)

    constraints = pdg_con.GameConstraintGridMap(ineq_blocks=(), eq_blocks=())

    ineq_lins, eq_lins = pdg_con.build_constraint_step_linearizations(constraints, op)

    assert isinstance(ineq_lins, tuple) and isinstance(eq_lins, tuple)
    assert len(ineq_lins) == 0
    assert len(eq_lins) == 0


def test_build_constraint_step_linearizations_orders_blocks_and_steps_and_slices():
    """
    Checks canonical order:
      - inequality blocks in order, expanded by active_steps order
      - then equality blocks similarly (returned separately)

    Also checks slices march forward correctly by cdim_out_step.
    """
    tg = TimeGrid(nt=6, dt=0.1, t0=0.0)
    nt, nx, nu = tg.nt, 2, 2
    K = nt - 1
    N = 1

    xs = jnp.ones((nt, nx), dtype=jnp.float32)
    us = jnp.ones((K, nu), dtype=jnp.float32)
    ls = jnp.ones((K, N, nx), dtype=jnp.float32)    # this shouldn't matter
    op = FixedStepPrimalDualTrajectory(tg=tg, xs=xs, us=us, ls=ls)

    # block A (ineq): cdim=1, active steps (0,2) => contributes 2 entries
    def cA(t, x, u):
        return x[0] + u[0]
    bA = pdg_con.ConstraintBlockGridMap(tg=tg, func=cA, cdim_out_step=1, active_steps=(0, 2), iseq=False, terminal=False)

    # block B (ineq): cdim=2, active steps (1,) => contributes 2 entries
    def cB(t, x, u):
        return jnp.array([x[0], u[1]])
    bB = pdg_con.ConstraintBlockGridMap(tg=tg, func=cB, cdim_out_step=2, active_steps=(1,), iseq=False, terminal=False)

    # block C (eq): terminal cdim=1 => contributes 1 entry at k=nt-1
    def cT(t, x):
        return x[0] ** 2
    bC = pdg_con.ConstraintBlockGridMap(tg=tg, func=cT, cdim_out_step=1, active_steps=None, iseq=True, terminal=True)

    constraints = pdg_con.GameConstraintGridMap(ineq_blocks=(bA, bB), eq_blocks=(bC,))

    ineq_lins, eq_lins = pdg_con.build_constraint_step_linearizations(constraints, op)

    # Expect 3 instances in ineq (A@0, A@2, B@1) and 1 in eq (C@nt-1)
    assert len(ineq_lins) == 3
    assert len(eq_lins) == 1

    # Check ordering by (block, step):
    assert ineq_lins[0].k == 0 and ineq_lins[0].cdim == 1
    assert ineq_lins[1].k == 2 and ineq_lins[1].cdim == 1
    assert ineq_lins[2].k == 1 and ineq_lins[2].cdim == 2
    assert ineq_lins[0].func is cA
    assert ineq_lins[1].func is cA
    assert ineq_lins[2].func is cB

    assert eq_lins[0].k == nt - 1
    assert eq_lins[0].terminal is True
    assert eq_lins[0].Ju is None
    assert eq_lins[0].func is cT

    # Check slices: A@0 consumes [0:1], A@2 consumes [1:2], B@1 consumes [2:4]
    assert ineq_lins[0].sl == slice(0, 1)
    assert ineq_lins[1].sl == slice(1, 2)
    assert ineq_lins[2].sl == slice(2, 4)

    # Eq slice should be slice(0,1) within the eq stack (eq has its own pointer)
    assert eq_lins[0].sl == slice(0, 1)


def test_build_constraint_step_linearizations_raises_on_timegrid_mismatch():
    tg_op = TimeGrid(nt=5, dt=0.1, t0=0.0)
    tg_c  = TimeGrid(nt=5, dt=0.1, t0=0.5)

    xs = jnp.zeros((tg_op.nt, 2), dtype=jnp.float32)
    us = jnp.zeros((tg_op.nt - 1, 2), dtype=jnp.float32)
    ls = jnp.zeros((tg_op.nt - 1, 2, 2), dtype=jnp.float32)
    op = FixedStepPrimalDualTrajectory(tg=tg_op, xs=xs, us=us, ls=ls)

    def cA(t, x, u):
        return x[0]
    bA = pdg_con.ConstraintBlockGridMap(tg=tg_c, func=cA, cdim_out_step=1, active_steps=(0,), iseq=False, terminal=False)
    constraints = pdg_con.GameConstraintGridMap(ineq_blocks=(bA,), eq_blocks=())

    with pytest.raises(ValueError, match="TimeGrid"):
        pdg_con.build_constraint_step_linearizations(constraints, op)


def test_constraint_block_raises_on_nonterminal_at_terminal_step():
    tg = TimeGrid(nt=5, dt=0.1, t0=0.0)

    def cA(t, x, u):
        return x[0]

    # illegal by convention: a non-terminal block cannot be active at the
    # terminal node because there is no corresponding u[k] there.
    with pytest.raises(ValueError, match="stage grid"):
        pdg_con.ConstraintBlockGridMap(
            tg=tg,
            func=cA,
            cdim_out_step=1,
            active_steps=(tg.nt - 1,),
            iseq=False,
            terminal=False,
        )


def test_accumulate_Jt_weighted_vector_matches_manual_sum():
    """
    This test isolates accumulate_Jt_weighted_vector by constructing a couple
    ConstraintStepLinearization objects directly with known Jx/Ju and slices.
    """
    nt, nx, nu = 5, 2, 3
    K = nt - 1

    # Two non-terminal instances at k=1 and k=3, cdim=2 each
    Jx1 = jnp.array([[1.0, 0.0],
                     [0.0, 2.0]], dtype=jnp.float32)   # (2,2)
    Ju1 = jnp.array([[1.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0]], dtype=jnp.float32)  # (2,3)

    Jx3 = jnp.array([[3.0, 1.0],
                     [-1.0, 0.5]], dtype=jnp.float32)
    Ju3 = jnp.array([[0.0, 2.0, 0.0],
                     [0.0, 0.0, -1.0]], dtype=jnp.float32)

    # w_flat has 4 entries: first 2 correspond to instance 1, next 2 to instance 3
    w_flat = jnp.array([10.0, -2.0, 5.0, 7.0], dtype=jnp.float32)

    l1 = pdg_con.ConstraintStepLinearization(
        kind="ineq", k=1, terminal=False, cdim=2,
        c=jnp.zeros((2,), dtype=jnp.float32),
        Jx=Jx1, Ju=Ju1,
        sl=slice(0, 2),
    )
    l3 = pdg_con.ConstraintStepLinearization(
        kind="ineq", k=3, terminal=False, cdim=2,
        c=jnp.zeros((2,), dtype=jnp.float32),
        Jx=Jx3, Ju=Ju3,
        sl=slice(2, 4),
    )
    assert l1.func is None
    assert l3.func is None

    dX, dU = pdg_con.accumulate_Jt_weighted_vector(
        lins=(l1, l3),
        w_flat=w_flat,
        nt=nt,
        nx=nx,
        nu=nu,
        dtype=jnp.float32,
    )

    # Manual reference:
    refX = np.zeros((nt, nx), dtype=np.float32)
    refU = np.zeros((K, nu), dtype=np.float32)

    w1 = np.array([10.0, -2.0], dtype=np.float32)
    w3 = np.array([5.0, 7.0], dtype=np.float32)

    refX[1] += np.array(Jx1).T @ w1
    refU[1] += np.array(Ju1).T @ w1

    refX[3] += np.array(Jx3).T @ w3
    refU[3] += np.array(Ju3).T @ w3

    np.testing.assert_allclose(np.array(dX), refX, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(np.array(dU), refU, atol=1e-6, rtol=1e-6)


def test_accumulate_Jt_weighted_vector_terminal_instance_only_updates_dX():
    """
    Terminal instances should not contribute to dU (Ju=None).
    """
    nt, nx, nu = 6, 2, 2
    K = nt - 1

    JxT = jnp.array([[1.0, 2.0]], dtype=jnp.float32)  # (cdim=1, nx=2)
    w_flat = jnp.array([3.0], dtype=jnp.float32)

    lT = pdg_con.ConstraintStepLinearization(
        kind="eq", k=nt - 1, terminal=True, cdim=1,
        c=jnp.zeros((1,), dtype=jnp.float32),
        Jx=JxT, Ju=None,
        sl=slice(0, 1),
    )
    assert lT.func is None

    dX, dU = pdg_con.accumulate_Jt_weighted_vector(
        lins=(lT,),
        w_flat=w_flat,
        nt=nt, nx=nx, nu=nu,
        dtype=jnp.float32,
    )

    refX = np.zeros((nt, nx), dtype=np.float32)
    refU = np.zeros((K, nu), dtype=np.float32)
    refX[nt - 1] += np.array(JxT).T @ np.array([3.0], dtype=np.float32)

    np.testing.assert_allclose(np.array(dX), refX, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(np.array(dU), refU, atol=1e-6, rtol=1e-6)


# -------------------------------
# ARCHIVED TESTS FROM UNUSED CODE
# -------------------------------
# def test_basic_constraint_scalar_output_normalized_to_vector():
#     def c(t, x, u):
#         return x[0] + u[0] + t  # scalar

#     con = pdg_con.BasicConstraint(func=c, iseq=False)

#     t = 0.5
#     x = jnp.array([1.0, 2.0])
#     u = jnp.array([3.0])

#     val = pdg_con.evaluate_constraint_step(con, t, x, u)

#     assert isinstance(val, jnp.ndarray)
#     assert val.shape == (1,)
#     assert jnp.allclose(val, jnp.array([1.0 + 3.0 + 0.5]))

# def test_basic_constraint_vector_output_preserved():
#     def c(t, x, u):
#         return jnp.array([x[0] - 1.0, u[0] + t])  # shape (2,)

#     con = pdg_con.BasicConstraint(func=c, iseq=False)

#     t = 0.25
#     x = jnp.array([2.0])
#     u = jnp.array([0.5])

#     val = pdg_con.evaluate_constraint_step(con, t, x, u)

#     assert val.shape == (2,)
#     assert jnp.allclose(val, jnp.array([1.0, 0.75]))

# def test_basic_constraint_matrix_output_raises():
#     def c(t, x, u):
#         return jnp.zeros((2, 2))  # invalid: 2D

#     con = pdg_con.BasicConstraint(func=c, iseq=False)

#     t = 0.0
#     x = jnp.array([1.0])
#     u = jnp.array([1.0])

#     with pytest.raises(ValueError, match="Constraint func must return scalar or 1D array"):
#         _ = pdg_con.evaluate_constraint_step(con, t, x, u)

# def test_basic_constraint_iseq_default_false():
#     def c(t, x, u):
#         return 0.0

#     con = pdg_con.BasicConstraint(func=c)
#     assert con.iseq is False


# def test_basic_constraint_python_float_return_ok():
#     def c(t, x, u):
#         return 1.234  # plain float, not jnp scalar

#     con = pdg_con.BasicConstraint(func=c, iseq=True)

#     val = pdg_con.evaluate_constraint_step(con, 0.0, jnp.array([0.0]), jnp.array([0.0]))

#     assert val.shape == (1,)
#     assert jnp.allclose(val, jnp.array([1.234]))

# def test_constraint_jacobian_scalar_shapes_and_values():
#     # c(t,x,u) = x0 + 2*u1 + t  (scalar)
#     def c(t, x, u):
#         return x[0] + 2.0 * u[1] + t

#     con = pdg_con.BasicConstraint(func=c, iseq=False)

#     t = 0.3
#     x = jnp.array([1.0, -2.0, 0.5])     # nx=3
#     u = jnp.array([0.1, 0.2, 0.3, 0.4]) # nu=4

#     dc_dx, dc_du = pdg_con.constraint_jacobian_step_no_checks(con, t, x, u)

#     assert dc_dx.shape == (1, 3)
#     assert dc_du.shape == (1, 4)

#     dc_dx_expected = jnp.array([[1.0, 0.0, 0.0]])
#     dc_du_expected = jnp.array([[0.0, 2.0, 0.0, 0.0]])

#     assert jnp.allclose(jnp.array(dc_dx), dc_dx_expected, atol=1e-7, rtol=1e-7)
#     assert jnp.allclose(jnp.array(dc_du), dc_du_expected, atol=1e-7, rtol=1e-7)

# def test_constraint_jacobian_vector_shapes_and_values():
#     # c1 = x0 + u0
#     # c2 = x1*u1 + t
#     def c(t, x, u):
#         return jnp.array([x[0] + u[0], x[1] * u[1] + t])

#     con = pdg_con.BasicConstraint(func=c, iseq=False)

#     t = 0.5
#     x = jnp.array([2.0, -3.0])   # nx=2
#     u = jnp.array([0.25, 4.0])   # nu=2

#     dc_dx, dc_du = pdg_con.constraint_jacobian_step_no_checks(con, t, x, u)

#     assert dc_dx.shape == (2, 2)
#     assert dc_du.shape == (2, 2)

#     # Row 0: d(x0+u0)/dx = [1,0], /du = [1,0]
#     # Row 1: d(x1*u1+t)/dx = [0,u1], /du = [0,x1]
#     dc_dx_expected = jnp.array([[1.0, 0.0],
#                                [0.0, 4.0]])
#     dc_du_expected = jnp.array([[1.0, 0.0],
#                                [0.0, -3.0]])

#     assert jnp.allclose(jnp.array(dc_dx), dc_dx_expected, atol=1e-7, rtol=1e-7)
#     assert jnp.allclose(jnp.array(dc_du), dc_du_expected, atol=1e-7, rtol=1e-7)

# def test_constraint_jacobian_time_dependence():
#     # c = sin(t) * x0 + u0  (scalar)
#     def c(t, x, u):
#         return jnp.sin(t) * x[0] + u[0]

#     con = pdg_con.BasicConstraint(func=c, iseq=False)

#     x = jnp.array([2.0, 0.0])
#     u = jnp.array([3.0])

#     t1 = 0.0
#     t2 = 0.7

#     dc_dx1, dc_du1 = pdg_con.constraint_jacobian_step_no_checks(con, t1, x, u)
#     dc_dx2, dc_du2 = pdg_con.constraint_jacobian_step_no_checks(con, t2, x, u)

#     # dc/dx0 = sin(t), dc/du0 = 1
#     assert jnp.allclose(jnp.array(dc_du1), jnp.array([[1.0]]), atol=1e-7, rtol=1e-7)
#     assert jnp.allclose(jnp.array(dc_du2), jnp.array([[1.0]]), atol=1e-7, rtol=1e-7)

#     assert jnp.allclose(jnp.array(dc_dx1), jnp.array([[0.0, 0.0]]), atol=1e-7, rtol=1e-7)
#     assert jnp.allclose(jnp.array(dc_dx2), jnp.array([[jnp.sin(t2), 0.0]]), atol=1e-7, rtol=1e-7)

# def test_constraint_jacobian_u_independent_gives_zero_dc_du():
#     # c = [x0^2, x1^2] (vector), independent of u
#     def c(t, x, u):
#         return jnp.array([x[0] ** 2, x[1] ** 2])

#     con = pdg_con.BasicConstraint(func=c, iseq=False)

#     t = 0.1
#     x = jnp.array([3.0, -4.0])
#     u = jnp.array([1.0, 2.0, 3.0])

#     dc_dx, dc_du = pdg_con.constraint_jacobian_step_no_checks(con, t, x, u)

#     # dc_dx should be [[2x0,0],[0,2x1]]
#     dc_dx_expected = jnp.array([[6.0, 0.0],
#                                [0.0, -8.0]])
#     assert jnp.allclose(jnp.array(dc_dx), dc_dx_expected, atol=1e-7, rtol=1e-7)

#     # dc_du should be zeros
#     assert dc_du.shape == (2, 3)
#     assert jnp.allclose(jnp.array(dc_du), jnp.zeros((2, 3)), atol=1e-7, rtol=1e-7)

# def test_constraint_jacobian_jittable():
#     # simple scalar constraint
#     def c(t, x, u):
#         return x[0] + u[0] + t

#     con = pdg_con.BasicConstraint(func=c, iseq=False)

#     t = 0.2
#     x = jnp.array([1.0, 2.0])
#     u = jnp.array([0.5])

#     # jit a wrapper (con and t captured as Python objects; fine for now)
#     jit_fn = jax.jit(lambda x_, u_: pdg_con.constraint_jacobian_step_no_checks(con, t, x_, u_))
#     A1, B1 = pdg_con.constraint_jacobian_step_no_checks(con, t, x, u)
#     A2, B2 = jit_fn(x, u)

#     assert jnp.allclose(jnp.array(A1), jnp.array(A2), atol=1e-7, rtol=1e-7)
#     assert jnp.allclose(jnp.array(B1), jnp.array(B2), atol=1e-7, rtol=1e-7)

# def test_constraint_cost_expansion_equality_linear_constraint():
#     # c(z) = A z - b, J = A (constant)
#     A = jnp.array([[1.0, 2.0],
#                    [3.0, 4.0]])      # q=2, nz=2
#     z = jnp.array([0.5, -1.0])
#     b = jnp.array([0.1, -0.2])
#     c = A @ z - b

#     lam = jnp.array([0.7, -0.3])
#     rho = 2.0

#     grad, hess = pdg_con.constraint_cost_expansion_equality(c, A, lam, rho)

#     lam_bar = lam + rho * c
#     grad_expected = A.T @ lam_bar
#     hess_expected = rho * (A.T @ A)

#     assert jnp.allclose(jnp.array(grad), jnp.array(grad_expected), atol=1e-8, rtol=1e-8)
#     assert jnp.allclose(jnp.array(hess), jnp.array(hess_expected), atol=1e-8, rtol=1e-8)

# def test_constraint_cost_expansion_inequality_active_mask_behavior():
#     # Simple J = I, so grad/hess are easy to reason about
#     J = jnp.eye(3)              # q=nz=3
#     c = jnp.array([-0.5, 0.2, -0.1])
#     lam = jnp.array([-1.0, -0.2, 0.3])  # last one has lam > 0 -> active even if c < 0
#     rho = 10.0

#     grad, hess, a = pdg_con.constraint_cost_expansion_inequality(c, J, lam, rho)

#     # Active rule: a = (c >= 0) | (lam > 0)
#     a_expected = jnp.array([False, True, True], dtype=bool)
#     assert jnp.all(a.astype(bool) == a_expected)

#     # With J=I: grad = lam + rho*(a*c)
#     grad_expected = lam + rho * (a.astype(J.dtype) * c)
#     assert jnp.allclose(jnp.array(grad), jnp.array(grad_expected), atol=1e-8, rtol=1e-8)

#     # With J=I: hess = rho * diag(a)
#     hess_expected = rho * jnp.diag(a_expected.astype(float))
#     assert jnp.allclose(jnp.array(hess), hess_expected, atol=1e-8, rtol=1e-8)
