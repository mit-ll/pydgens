# Copyright 2026 MIT Lincoln Laboratory
# SPDX-License-Identifier: MIT

import jax.numpy as jnp
import pytest

import pydgens as pdg
from pydgens.ir.constrainttypes import (
    ConstraintBlockGridMap,
    GameConstraintGridMap,
)


def test_control_bounds_factory_constructs_control_bounds():

    con = pdg.control_bounds(
        lower=-1.0,
        upper=1.0,
        indices=[0],
    )

    assert isinstance(
        con,
        pdg.constraints.ControlBounds,
    )


def test_state_bounds_factory_constructs_state_bounds():

    con = pdg.state_bounds(
        lower=0.0,
        indices=[1],
    )

    assert isinstance(
        con,
        pdg.constraints.StateBounds,
    )


def test_bounds_require_at_least_one_side():

    with pytest.raises(ValueError, match="At least one of `lower` or `upper`"):
        pdg.control_bounds()

    with pytest.raises(ValueError, match="At least one of `lower` or `upper`"):
        pdg.state_bounds()


def test_control_bounds_indices_none_expands_to_all_joint_controls():

    tg = pdg.time_grid(
        nt=4,
        dt=0.1,
    )

    con = pdg.control_bounds(
        upper=1.0,
    )

    blocks = con.to_ir_blocks(
        tg=tg,
        nx=3,
        nu=2,
    )

    assert len(blocks) == 1
    assert blocks[0].cdim_out_step == 2

    x = jnp.array([10.0, 20.0, 30.0])
    u = jnp.array([0.5, -2.0])
    out = blocks[0].func(0.0, x, u)

    assert jnp.allclose(out, jnp.array([-0.5, -3.0]))


def test_state_bounds_indices_none_expands_to_all_joint_states():

    tg = pdg.time_grid(
        nt=4,
        dt=0.1,
    )

    con = pdg.state_bounds(
        lower=0.0,
        include_terminal=False,
    )

    blocks = con.to_ir_blocks(
        tg=tg,
        nx=3,
        nu=2,
    )

    assert len(blocks) == 1
    assert blocks[0].cdim_out_step == 3

    x = jnp.array([2.0, -1.0, 4.0])
    u = jnp.array([7.0, 8.0])
    out = blocks[0].func(0.0, x, u)

    assert jnp.allclose(out, jnp.array([-2.0, 1.0, -4.0]))


def test_bounds_preserve_explicit_step_subset_in_lowered_block():

    tg = pdg.time_grid(
        nt=6,
        dt=0.1,
    )

    con = pdg.control_bounds(
        lower=-1.0,
        upper=1.0,
        indices=[0],
        steps=[1, 3],
    )

    blocks = con.to_ir_blocks(
        tg=tg,
        nx=2,
        nu=2,
    )

    assert len(blocks) == 1
    assert blocks[0].active_steps == (1, 3)


def test_control_bounds_empty_steps_lowers_to_no_blocks():

    tg = pdg.time_grid(
        nt=5,
        dt=0.1,
    )

    con = pdg.control_bounds(
        upper=1.0,
        indices=[0],
        steps=[],
    )

    blocks = con.to_ir_blocks(
        tg=tg,
        nx=2,
        nu=2,
    )

    assert blocks == ()


def test_state_bounds_include_terminal_false_omits_terminal_block():

    tg = pdg.time_grid(
        nt=4,
        dt=0.1,
    )

    con = pdg.state_bounds(
        lower=0.0,
        indices=[1],
        include_terminal=False,
    )

    blocks = con.to_ir_blocks(
        tg=tg,
        nx=3,
        nu=2,
    )

    assert len(blocks) == 1
    assert blocks[0].terminal is False


def test_control_bounds_vector_bounds_are_preserved_in_selected_order():

    tg = pdg.time_grid(
        nt=4,
        dt=0.1,
    )

    con = pdg.control_bounds(
        lower=jnp.array([-3.0, -1.0]),
        upper=jnp.array([2.0, 4.0]),
        indices=[2, 0],
    )

    blocks = con.to_ir_blocks(
        tg=tg,
        nx=2,
        nu=3,
    )

    x = jnp.array([10.0, 20.0])
    u = jnp.array([1.0, 99.0, -2.0])
    out = blocks[0].func(0.0, x, u)

    expected = jnp.array([
        -2.0 - 2.0,
        1.0 - 4.0,
        -3.0 - (-2.0),
        -1.0 - 1.0,
    ])
    assert jnp.allclose(out, expected)


def test_control_bounds_vector_length_mismatch_raises():

    tg = pdg.time_grid(
        nt=4,
        dt=0.1,
    )

    con = pdg.control_bounds(
        upper=jnp.array([1.0, 2.0]),
        indices=[0],
    )

    with pytest.raises(ValueError, match="`upper` must be scalar or have shape"):
        con.to_ir_blocks(
            tg=tg,
            nx=2,
            nu=3,
        )


def test_constraints_reject_non_constraint_items():

    with pytest.raises(TypeError, match="must inherit from AbstractConstraintSpec"):
        pdg.constraint_set("not a constraint")


def test_empty_constraint_set_lowers_to_empty_ir_map():

    tg = pdg.time_grid(
        nt=3,
        dt=0.1,
    )

    cons = pdg.constraint_set()

    ir_cons = cons.to_ir(
        tg=tg,
        nx=2,
        nu=2,
    )

    assert isinstance(ir_cons, GameConstraintGridMap)
    assert ir_cons.ineq_blocks == ()
    assert ir_cons.eq_blocks == ()
    assert ir_cons.nc_ineq == 0


def test_control_bounds_lower_to_single_path_block():

    tg = pdg.time_grid(
        nt=5,
        dt=0.1,
    )

    con = pdg.control_bounds(
        lower=-2.0,
        upper=3.0,
        indices=[0, 2],
    )

    blocks = con.to_ir_blocks(
        tg=tg,
        nx=4,
        nu=3,
    )

    assert len(blocks) == 1
    assert isinstance(blocks[0], ConstraintBlockGridMap)
    assert blocks[0].terminal is False
    assert blocks[0].active_steps == tuple(range(tg.nsteps))
    assert blocks[0].cdim_out_step == 4

    x = jnp.array([10.0, 20.0, 30.0, 40.0])
    u = jnp.array([1.0, 99.0, -4.0])

    # ordering is [upper residuals ; lower residuals]
    out = blocks[0].func(0.0, x, u)
    expected = jnp.array([
        1.0 - 3.0,
        -4.0 - 3.0,
        -2.0 - 1.0,
        -2.0 - (-4.0),
    ])
    assert jnp.allclose(out, expected)


def test_state_bounds_lower_to_path_and_terminal_blocks():

    tg = pdg.time_grid(
        nt=4,
        dt=0.2,
    )

    con = pdg.state_bounds(
        lower=0.0,
        upper=5.0,
        indices=[1],
        include_terminal=True,
    )

    blocks = con.to_ir_blocks(
        tg=tg,
        nx=3,
        nu=2,
    )

    assert len(blocks) == 2

    path_block, term_block = blocks

    assert path_block.terminal is False
    assert path_block.active_steps == tuple(range(tg.nsteps))
    assert path_block.cdim_out_step == 2

    assert term_block.terminal is True
    assert term_block.active_steps == (tg.nt - 1,)
    assert term_block.cdim_out_step == 2

    x = jnp.array([10.0, 7.0, -1.0])
    u = jnp.array([1.0, 2.0])

    path_out = path_block.func(0.0, x, u)
    term_out = term_block.func(0.6, x)
    expected = jnp.array([
        7.0 - 5.0,
        0.0 - 7.0,
    ])

    assert jnp.allclose(path_out, expected)
    assert jnp.allclose(term_out, expected)


def test_constraints_container_lowers_to_game_constraint_map():

    tg = pdg.time_grid(
        nt=3,
        dt=0.1,
    )

    cons = pdg.constraint_set(
        pdg.control_bounds(
            lower=-1.0,
            upper=1.0,
            indices=[0],
        ),
        pdg.state_bounds(
            lower=0.0,
            indices=[1],
            include_terminal=True,
        ),
    )

    ir_cons = cons.to_ir(
        tg=tg,
        nx=2,
        nu=2,
    )

    assert isinstance(ir_cons, GameConstraintGridMap)
    assert len(ir_cons.ineq_blocks) == 3
    assert len(ir_cons.eq_blocks) == 0
    assert ir_cons.nc_ineq == 7
