# Copyright 2026 MIT Lincoln Laboratory
# SPDX-License-Identifier: MIT

import pytest
import jax
import jax.numpy as jnp

import pydgens as pdg

import pydgens.ir.timetypes as irtime


def test_time_grid_public_api():
    tg = pdg.time_grid(nt=51, dt=0.1)
    assert isinstance(tg, irtime.TimeGrid)

    assert tg.nt == 51
    assert tg.nsteps == 50
    assert tg.dt == 0.1
    assert tg.t0 == 0.0

def test_valid_construction():
    t = irtime.TimeGrid(nt=10, dt=0.1, t0=1.0)
    assert t.nt == 10
    assert t.nsteps == 9
    assert t.dt == 0.1
    assert t.t0 == 1.0

def test_default_t0():
    t = irtime.TimeGrid(nt=5, dt=0.5)
    assert t.t0 == 0.0

def test_zero_step():
    t = irtime.TimeGrid(nt=1, dt=0.1)
    assert t.nsteps == 0

def test_invalid_nt_type():
    with pytest.raises(TypeError):
        irtime.TimeGrid(nt=5.0, dt=0.1)

def test_invalid_nt_value():
    with pytest.raises(ValueError):
        irtime.TimeGrid(nt=0, dt=0.1)

def test_invalid_dt_type():
    with pytest.raises(TypeError):
        irtime.TimeGrid(nt=5, dt="0.1")

def test_invalid_dt_value():
    with pytest.raises(ValueError):
        irtime.TimeGrid(nt=5, dt=0.0)

def test_invalid_t0_type():
    with pytest.raises(TypeError):
        irtime.TimeGrid(nt=5, dt=0.1, t0="start")

def test_immutable_fields():
    t = irtime.TimeGrid(nt=5, dt=0.1)
    with pytest.raises(AttributeError):
        t.nt = 20  # flax.struct.dataclass is frozen

def test_compute_ts_basic():
    tg = irtime.TimeGrid(nt=5, dt=1.0, t0=0.0)
    ts = irtime.compute_ts(tg)
    expected = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
    assert jnp.allclose(ts, expected)


def test_compute_ts_with_offset():
    tg = irtime.TimeGrid(nt=4, dt=0.5, t0=1.0)
    ts = irtime.compute_ts(tg)
    expected = jnp.array([1.0, 1.5, 2.0, 2.5])
    assert jnp.allclose(ts, expected)


def test_compute_ts_small_dt():
    tg = irtime.TimeGrid(nt=3, dt=0.001, t0=0.0)
    ts = irtime.compute_ts(tg)
    expected = jnp.array([0.0, 0.001, 0.002])
    assert jnp.allclose(ts, expected)


def test_compute_ts_negative_t0():
    tg = irtime.TimeGrid(nt=3, dt=2.0, t0=-2.0)
    ts = irtime.compute_ts(tg)
    expected = jnp.array([-2.0, 0.0, 2.0])
    assert jnp.allclose(ts, expected)


def test_compute_ts_length_matches_nt():
    tg = irtime.TimeGrid(nt=10, dt=0.1, t0=0.0)
    ts = irtime.compute_ts(tg)
    assert ts.shape == (tg.nt,)


def test_compute_ts_zero_step_case():
    tg = irtime.TimeGrid(nt=1, dt=0.1, t0=2.5)
    ts = irtime.compute_ts(tg)
    assert ts.shape == (1,)
    assert jnp.allclose(ts, jnp.array([2.5]))


def test_disc2cont_basic():
    tg = irtime.TimeGrid(nt=5, dt=0.1, t0=0.0)
    assert irtime.disc2cont(0, tg) == pytest.approx(0.0)
    assert irtime.disc2cont(1, tg) == pytest.approx(0.1)
    assert irtime.disc2cont(4, tg) == pytest.approx(0.4)


def test_disc2cont_zero_step_case():
    tg = irtime.TimeGrid(nt=1, dt=0.1, t0=3.0)
    assert irtime.disc2cont(0, tg) == pytest.approx(3.0)


def test_disc2cont_with_offset():
    tg = irtime.TimeGrid(nt=5, dt=0.2, t0=1.0)
    assert irtime.disc2cont(0, tg) == pytest.approx(1.0)
    assert irtime.disc2cont(2, tg) == pytest.approx(1.4)

def test_cont2disc_basic():
    tg = irtime.TimeGrid(nt=5, dt=0.1, t0=0.0)
    assert irtime.cont2disc(0.0, tg) == 0
    assert irtime.cont2disc(0.1, tg) == 1
    assert irtime.cont2disc(0.4, tg) == 4


def test_cont2disc_zero_step_case():
    tg = irtime.TimeGrid(nt=1, dt=0.1, t0=3.0)
    assert irtime.cont2disc(3.0, tg) == 0
    assert irtime.cont2disc(2.0, tg) == 0
    assert irtime.cont2disc(5.0, tg) == 0


def test_non_exact_cont_value_1():
    tg = irtime.TimeGrid(nt=5, dt=0.1, t0=0.0)
    # Slightly off from 0.3 due to floating-point
    t_val = 0.3000000001
    k = irtime.cont2disc(t_val, tg)
    assert k == 3  # should still map correctly

def test_cont2disc_exact_boundary_1():
    tg = irtime.TimeGrid(nt=5, dt=1.0, t0=0.0)
    assert irtime.cont2disc(0.0, tg) == 0
    assert irtime.cont2disc(1.0, tg) == 1
    assert irtime.cont2disc(4.0, tg) == 4
    
def test_cont2disc_exact_boundary_2():
    tg = irtime.TimeGrid(nt=5, dt=0.2, t0=1.0)
    # left edge → bucket 0
    assert irtime.cont2disc(1.0, tg) == 0
    # right edge of bucket 0 (should go to bucket 1)
    assert irtime.cont2disc(1.2, tg) == 1
    # middle of bucket
    assert irtime.cont2disc(1.35, tg) == 1
    # last edge
    assert irtime.cont2disc(1.8, tg) == 4

def test_cont2disc_within_buckets():
    tg = irtime.TimeGrid(nt=5, dt=1.0, t0=0.0)
    assert irtime.cont2disc(0.9, tg) == 0   # falls in bucket [0,1)
    assert irtime.cont2disc(1.1, tg) == 1   # falls in bucket [1,2)
    assert irtime.cont2disc(3.9999, tg) == 3
    assert irtime.cont2disc(3.99999999999999, tg) == 4 # falls into bucket [4,5) because of epsilon

def test_cont2disc_with_offset_2():
    tg = irtime.TimeGrid(nt=5, dt=1.0, t0=-2.0)
    assert irtime.cont2disc(-2.0, tg) == 0
    assert irtime.cont2disc(-1.5, tg) == 0  # still bucket 0
    assert irtime.cont2disc(-1.0, tg) == 1
    assert irtime.cont2disc(0.0, tg) == 2
    assert irtime.cont2disc(2.0, tg) == 4

def test_cont2disc_with_offset_small_dt():
    tg = irtime.TimeGrid(nt=5, dt=0.2, t0=1.0)
    assert irtime.cont2disc(1.0, tg) == 0
    assert irtime.cont2disc(1.2, tg) == 1
    assert irtime.cont2disc(1.8, tg) == 4

# def test_cont2disc_out_of_bounds():
#     tg = irtime.TimeGrid(nt=5, dt=1.0, t0=0.0)
#     with pytest.raises(ValueError):
#         cont2disc(-0.1, tg)
#     with pytest.raises(ValueError):
#         cont2disc(5.0, tg)  # exactly beyond last bucket

def test_cont2disc_below_bounds(capsys):
    tg = irtime.TimeGrid(t0=0.0, dt=0.1, nt=5)
    idx = irtime.cont2disc(-0.05, tg)
    assert idx == 0  # clipped
    # out, _ = capsys.readouterr()
    # assert "out-of-bounds" in out

def test_cont2disc_above_bounds(capsys):
    tg = irtime.TimeGrid(t0=0.0, dt=0.1, nt=5)
    idx = irtime.cont2disc(1.0, tg)
    assert idx == 4  # clipped
    # out, _ = capsys.readouterr()
    # assert "out-of-bounds" in out

def test_cont2disc_jax_array_runs():
    tg = irtime.TimeGrid(t0=0.0, dt=0.1, nt=5)
    # Make sure function works with jax arrays too
    t = jnp.array(0.25)
    idx = irtime.cont2disc(t, tg)
    assert idx == 2

def test_cont2disc_jit_out_of_bounds(capsys):
    tg = irtime.TimeGrid(t0=0.0, dt=0.1, nt=5)

    # Wrap cont2disc in a jit-compiled function
    @jax.jit
    def wrapped(t):
        return irtime.cont2disc(t, tg)

    # Trigger out-of-bounds (too small)
    idx = wrapped(jnp.array(-0.05))
    assert int(idx) == 0
    # out, _ = capsys.readouterr()
    # assert "out-of-bounds" in out

    # Trigger out-of-bounds (too large)
    idx = wrapped(jnp.array(1.0))
    assert int(idx) == 4
    # out, _ = capsys.readouterr()
    # assert "out-of-bounds" in out

def test_cont2disc_jit_vmap(capsys):
    tg = irtime.TimeGrid(t0=0.0, dt=0.1, nt=5)

    # jit + vmap wrapper
    @jax.jit
    def wrapped(ts):
        return jax.vmap(lambda t: irtime.cont2disc(t, tg))(ts)

    # includes values inside and outside grid
    ts = jnp.array([-0.05, 0.0, 0.25, 0.35, 0.9, 1.0])
    idxs = wrapped(ts)

    # Expected: clip to [0, nt-1] while warning
    assert idxs.tolist() == [0, 0, 2, 3, 4, 4]

    # out, _ = capsys.readouterr()
    # # We should see at least one warning about out-of-bounds
    # assert "out-of-bounds" in out

def test_cont2disc_floating_point_tolerance():
    tg = irtime.TimeGrid(nt=5, dt=0.1, t0=0.0)
    assert irtime.cont2disc(0.2999999, tg) == 2  # still bucket 2
    assert irtime.cont2disc(0.29999999, tg) == 3  # now tipping over to bucket 3 based on eps
    assert irtime.cont2disc(0.3000000001, tg) == 3  # bucket 3

def test_round_trip_disc_to_cont_1():
    tg = irtime.TimeGrid(nt=10, dt=0.5, t0=2.0)
    for k in range(tg.nt):
        t = irtime.disc2cont(k, tg)
        assert irtime.cont2disc(t, tg) == k

def test_round_trip_disc_to_cont_2():
    tg = irtime.TimeGrid(nt=10, dt=0.25, t0=-1.0)
    for k in range(tg.nt):
        t = irtime.disc2cont(k, tg)
        assert irtime.cont2disc(t, tg) == k

def test_timegrid_equality_same_object():
    tg = irtime.TimeGrid(nt=5, dt=0.1, t0=0.0)
    assert tg == tg  # should be True for the same object

def test_timegrid_equality_different_objects_equal():
    tg1 = irtime.TimeGrid(nt=5, dt=0.1, t0=0.0)
    tg2 = irtime.TimeGrid(nt=5, dt=0.1, t0=0.0)
    assert tg1 == tg2  # should be True for numerically equal objects

def test_timegrid_equality_different_nt():
    tg1 = irtime.TimeGrid(nt=5, dt=0.1, t0=0.0)
    tg2 = irtime.TimeGrid(nt=6, dt=0.1, t0=0.0)
    assert tg1 != tg2

def test_timegrid_equality_different_dt():
    tg1 = irtime.TimeGrid(nt=5, dt=0.1, t0=0.0)
    tg2 = irtime.TimeGrid(nt=5, dt=0.1000001, t0=0.0)  # slightly different
    assert tg1 != tg2

def test_timegrid_equality_different_t0():
    tg1 = irtime.TimeGrid(nt=5, dt=0.1, t0=0.0)
    tg2 = irtime.TimeGrid(nt=5, dt=0.1, t0=0.00001)
    assert tg1 != tg2
