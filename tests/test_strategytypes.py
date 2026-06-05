# Copyright 2026 MIT Lincoln Laboratory
# SPDX-License-Identifier: MIT

import pytest
import jax.numpy as jnp

from pydgens.ir.timetypes import TimeGrid

# module under test
import pydgens.ir.strategytypes as irstrat

def test_affine_strategy_valid_init():
    tg = TimeGrid(nt=6, dt=0.1)
    P = jnp.ones((5, 2, 3))
    alpha = jnp.ones((5, 2))
    strategy = irstrat.FixedStepAffineStrategies(tg, P, alpha)
    assert strategy.nt == 6
    assert strategy.nsteps == 5
    assert strategy.nu == 2
    assert strategy.nx == 3

def test_affine_strategy_valid_update():
    tg = TimeGrid(nt=5, dt=0.1)
    strategy = irstrat.FixedStepAffineStrategies(tg, jnp.ones((4, 2, 2)), jnp.ones((4, 2)))
    strategy = irstrat.update_affine_strategy(strategy, jnp.zeros((4, 2, 2)), None)     # Valid P update
    strategy = irstrat.update_affine_strategy(strategy, None, jnp.ones((4, 2)) * 3)     # Valid alpha update
    assert jnp.allclose(strategy.P, 0)
    assert jnp.allclose(strategy.alpha, 3)

def test_affine_strategy_valid_zero_step_case():
    tg = TimeGrid(nt=1, dt=0.1)
    P = jnp.empty((0, 2, 3))
    alpha = jnp.empty((0, 2))
    strategy = irstrat.FixedStepAffineStrategies(tg, P, alpha)
    assert strategy.nt == 1
    assert strategy.nsteps == 0
    assert strategy.nu == 2
    assert strategy.nx == 3
    assert strategy.P.shape == (0, 2, 3)
    assert strategy.alpha.shape == (0, 2)

def test_affine_strategy_valid_zero_step_update():
    tg = TimeGrid(nt=1, dt=0.1)
    strategy = irstrat.FixedStepAffineStrategies(
        tg,
        jnp.empty((0, 2, 3)),
        jnp.empty((0, 2)),
    )
    strategy = irstrat.update_affine_strategy(
        strategy,
        jnp.zeros((0, 2, 3)),
        jnp.zeros((0, 2)),
    )
    assert strategy.P.shape == (0, 2, 3)
    assert strategy.alpha.shape == (0, 2)

def test_affine_strategy_invalid_P_shape():
    with pytest.raises(ValueError, match="P must be 3D"):
        irstrat.FixedStepAffineStrategies(TimeGrid(nt=5, dt=0.1), jnp.ones((4, 2)), jnp.ones((4, 2)))

def test_affine_strategy_invalid_alpha_shape():
    with pytest.raises(ValueError, match="alpha must be 2D"):
        irstrat.FixedStepAffineStrategies(TimeGrid(nt=5, dt=0.1), jnp.ones((4, 2, 2)), jnp.ones((4, 2, 1)))

def test_affine_strategy_inconsistent_T_init():
    with pytest.raises(ValueError, match="nstep dimension mismatch in P"):
        irstrat.FixedStepAffineStrategies(TimeGrid(nt=5, dt=0.1), jnp.ones((5, 2, 2)), jnp.ones((4, 2)))

    with pytest.raises(ValueError, match="nstep dimension mismatch in alpha"):
        irstrat.FixedStepAffineStrategies(TimeGrid(nt=5, dt=0.1), jnp.ones((4, 2, 2)), jnp.ones((5, 2)))

def test_affine_strategy_inconsistent_T_set_P():
    strategy = irstrat.FixedStepAffineStrategies(TimeGrid(nt=7, dt=0.1), jnp.ones((6, 2, 2)), jnp.ones((6, 2)))
    with pytest.raises(ValueError, match="nstep dimension mismatch in P"):
        strategy = irstrat.update_affine_strategy(strategy, jnp.ones((7, 2, 2)))

def test_affine_strategy_inconsistent_T_set_alpha():
    strategy = irstrat.FixedStepAffineStrategies(TimeGrid(nt=7, dt=0.1), jnp.ones((6, 2, 2)), jnp.ones((6, 2)))
    with pytest.raises(ValueError, match="nstep dimension mismatch in alpha"):
        strategy = irstrat.update_affine_strategy(strategy, None, jnp.ones((7, 2)))

def test_affine_strategy_inconsistent_T_zero_step_init():
    tg = TimeGrid(nt=1, dt=0.1)
    with pytest.raises(ValueError, match="nstep dimension mismatch in P"):
        irstrat.FixedStepAffineStrategies(tg, jnp.ones((1, 2, 2)), jnp.empty((0, 2)))

    with pytest.raises(ValueError, match="nstep dimension mismatch in alpha"):
        irstrat.FixedStepAffineStrategies(tg, jnp.empty((0, 2, 2)), jnp.ones((1, 2)))

@pytest.fixture
def example_strat():
    tg = TimeGrid(nt=7, dt=0.1)
    P = jnp.ones((6, 3, 2))    # nsteps=6, nu=3, nx=2
    alpha = jnp.zeros((6, 3))  # nsteps=6, nu=3
    return irstrat.FixedStepAffineStrategies(tg, P, alpha)

def test_affine_strategy_immutable(example_strat):
    strategy = example_strat
    with pytest.raises(AttributeError, match="cannot assign to field 'P'"):
        strategy.P = jnp.ones((6, 3, 2))
    with pytest.raises(AttributeError, match="cannot assign to field 'alpha'"):
        strategy.alpha = jnp.zeros((6, 3))

def test_T_property(example_strat):
    strategy = example_strat
    assert strategy.nt == 7
    with pytest.raises(AttributeError, match="cannot assign to field 'nt'"):
        strategy.nt = 10

def test_nsteps_property(example_strat):
    strategy = example_strat
    assert strategy.nsteps == 6
    with pytest.raises(AttributeError, match="cannot assign to field 'nsteps'"):
        strategy.nsteps = 10

def test_m_property(example_strat):
    strategy = example_strat
    assert strategy.nu == 3
    with pytest.raises(AttributeError, match="cannot assign to field 'nu'"):
        strategy.nu = 5

def test_n_property(example_strat):
    strategy = example_strat
    assert strategy.nx == 2
    with pytest.raises(AttributeError, match="cannot assign to field 'nx'"):
        strategy.nx = 7
