# Copyright 2026 MIT Lincoln Laboratory
# SPDX-License-Identifier: MIT

# tests/frontend/test_costs.py

import pytest
import jax.numpy as jnp

# module under test
import pydgens as pdg
from pydgens.ir.costtypes import (
    ControlDomain,
    PlayerCostSpecContinuous,
)
# from pydgens.frontend.costs import (
#     AbstractPlayerCost,
#     QuadraticPlayerCost,
# )


# ---------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------


def test_player_cost_factory_constructs_continuous_player_cost():

    cost = pdg.player_cost(
        running=lambda t, x, u: x[0] ** 2 + u[0] ** 2,
    )

    assert isinstance(
        cost,
        pdg.costs.ContinuousPlayerCost,
    )


def test_player_cost_factory_uses_joint_control_and_defaults_terminal_to_none():

    cost = pdg.player_cost(
        running=lambda t, x, u: x[0] ** 2 + u[0] ** 2,
    )

    assert cost.running is not None
    assert cost.terminal is None


def test_player_cost_factory_supports_terminal_cost():

    running = lambda t, x, u: x[0] ** 2 + u[0] ** 2
    terminal = lambda t, x: x[0] ** 2

    cost = pdg.player_cost(
        running=running,
        terminal=terminal,
    )

    assert cost.running is running
    assert cost.terminal is terminal


def test_player_cost_rejects_noncallable_running():

    with pytest.raises(TypeError, match="`running` must be callable"):
        pdg.player_cost(
            running=3.14,
        )


def test_player_cost_rejects_noncallable_terminal():

    with pytest.raises(TypeError, match="`terminal` must be callable or None"):
        pdg.player_cost(
            running=lambda t, x, u: 0.0,
            terminal=3.14,
        )


def test_continuous_player_cost_lowers_to_ir_cost_spec():

    running = lambda t, x, u: t + x[0] ** 2 + u[0] ** 2
    terminal = lambda t, x: t + x[0] ** 2

    cost = pdg.player_cost(
        running=running,
        terminal=terminal,
    )

    ir_cost = cost.to_ir()

    assert isinstance(ir_cost, PlayerCostSpecContinuous)
    assert ir_cost.running is running
    assert ir_cost.terminal is terminal
    assert ir_cost.control_domain == ControlDomain.JOINT


def test_quadratic_player_cost_constructs():

    cost = pdg.costs.QuadraticPlayerCost(
        nx=4,
        nu=2,
    )

    assert cost.nx == 4
    assert cost.nu == 2

    assert cost.Qp.shape == (4, 4)
    assert cost.Rp.shape == (2, 2)

    assert cost.x_ref.shape == (4,)
    assert cost.u_ref.shape == (2,)


def test_quadratic_cost_factory_constructs_quadratic_player_cost():

    cost = pdg.quadratic_cost(
        nx=3,
        nu=2,
    )

    assert isinstance(
        cost,
        pdg.costs.QuadraticPlayerCost,
    )


def test_quadratic_cost_factory_applies_weights_and_targets():

    cost = pdg.quadratic_cost(
        nx=3,
        nu=2,
        state_weights=[2.0, -4.0],
        state_indices=[0, 2],
        state_target=[10.0, 20.0, 30.0],
        control_weights=[5.0],
        control_indices=[1],
        control_target=[1.0, -1.0],
    )

    assert jnp.allclose(
        cost.Qp,
        jnp.diag(jnp.array([2.0, 0.0, -4.0])),
    )
    assert jnp.allclose(
        cost.Rp,
        jnp.diag(jnp.array([0.0, 5.0])),
    )
    assert jnp.allclose(
        cost.x_ref,
        jnp.array([10.0, 20.0, 30.0]),
    )
    assert jnp.allclose(
        cost.u_ref,
        jnp.array([1.0, -1.0]),
    )


def test_quadratic_cost_factory_defaults_to_zero_cost_and_targets():

    cost = pdg.quadratic_cost(
        nx=2,
        nu=3,
    )

    assert jnp.allclose(
        cost.Qp,
        jnp.zeros((2, 2)),
    )
    assert jnp.allclose(
        cost.Rp,
        jnp.zeros((3, 3)),
    )
    assert jnp.allclose(
        cost.x_ref,
        jnp.zeros(2),
    )
    assert jnp.allclose(
        cost.Qp_terminal,
        jnp.zeros((2, 2)),
    )
    assert jnp.allclose(
        cost.x_ref_terminal,
        jnp.zeros(2),
    )
    assert jnp.allclose(
        cost.u_ref,
        jnp.zeros(3),
    )


def test_matrix_quadratic_cost_factory_accepts_full_state_matrix():

    Q = jnp.array([
        [1.0, -0.5],
        [-0.5, 2.0],
    ])
    R = jnp.eye(2)

    cost = pdg.matrix_quadratic_cost(
        nx=2,
        nu=2,
        state_matrix=Q,
        control_matrix=R,
    )

    assert jnp.allclose(cost.Qp, Q)
    assert jnp.allclose(cost.Rp, R)


def test_matrix_quadratic_cost_factory_rejects_asymmetric_state_matrix():

    Q = jnp.array([
        [1.0, 2.0],
        [0.0, 1.0],
    ])

    with pytest.raises(ValueError, match="symmetric"):
        pdg.matrix_quadratic_cost(
            nx=2,
            nu=2,
            state_matrix=Q,
        )


def test_matrix_quadratic_cost_factory_rejects_indefinite_control_matrix():

    R = jnp.diag(jnp.array([1.0, -1.0]))

    with pytest.raises(ValueError, match="positive semidefinite"):
        pdg.matrix_quadratic_cost(
            nx=2,
            nu=2,
            control_matrix=R,
        )


def test_matrix_quadratic_cost_wrappers_delegate_to_property_setters():

    cost = pdg.costs.QuadraticPlayerCost(
        nx=2,
        nu=2,
    )

    Q = jnp.array([
        [1.0, -2.0],
        [-2.0, 1.0],
    ])
    Q_terminal = 2.0 * Q
    R = jnp.array([
        [2.0, 0.5],
        [0.5, 1.0],
    ])

    cost.set_state_matrix(Q)
    cost.set_terminal_state_matrix(Q_terminal)
    cost.set_control_matrix(R)

    assert jnp.allclose(cost.Qp, Q)
    assert jnp.allclose(cost.Qp_terminal, Q_terminal)
    assert jnp.allclose(cost.Rp, R)


def test_quadratic_cost_factory_applies_terminal_state_weights_and_target():

    cost = pdg.quadratic_cost(
        nx=3,
        nu=2,
        terminal_state_weights=[2.0, -4.0],
        terminal_state_indices=[0, 2],
        terminal_state_target=[10.0, 20.0, 30.0],
    )

    assert jnp.allclose(
        cost.Qp_terminal,
        jnp.diag(jnp.array([2.0, 0.0, -4.0])),
    )
    assert jnp.allclose(
        cost.x_ref_terminal,
        jnp.array([10.0, 20.0, 30.0]),
    )


def test_quadratic_cost_factory_rejects_terminal_state_indices_without_weights():

    with pytest.raises(ValueError, match="terminal_state_indices"):

        pdg.quadratic_cost(
            nx=3,
            nu=2,
            terminal_state_indices=[0, 1],
        )


def test_quadratic_cost_factory_rejects_state_indices_without_weights():

    with pytest.raises(ValueError, match="state_indices"):

        pdg.quadratic_cost(
            nx=3,
            nu=2,
            state_indices=[0, 1],
        )


def test_quadratic_cost_factory_rejects_control_indices_without_weights():

    with pytest.raises(ValueError, match="control_indices"):

        pdg.quadratic_cost(
            nx=3,
            nu=2,
            control_indices=[0],
        )


def test_default_Qp_is_zero():

    cost = pdg.costs.QuadraticPlayerCost(
        nx=3,
        nu=2,
    )

    assert jnp.allclose(
        cost.Qp,
        jnp.zeros((3, 3)),
    )


def test_default_Rp_is_zero():

    cost = pdg.costs.QuadraticPlayerCost(
        nx=3,
        nu=2,
    )

    assert jnp.allclose(
        cost.Rp,
        jnp.zeros((2, 2)),
    )


def test_default_x_ref_is_zero():

    cost = pdg.costs.QuadraticPlayerCost(
        nx=3,
        nu=2,
    )

    assert jnp.allclose(
        cost.x_ref,
        jnp.zeros(3),
    )


def test_default_u_ref_is_zero():

    cost = pdg.costs.QuadraticPlayerCost(
        nx=3,
        nu=2,
    )

    assert jnp.allclose(
        cost.u_ref,
        jnp.zeros(2),
    )


# ---------------------------------------------------------------------
# Qp validation
# ---------------------------------------------------------------------


def test_Qp_accepts_valid_symmetric_matrix():

    cost = pdg.costs.QuadraticPlayerCost(
        nx=3,
        nu=2,
    )

    Qp = jnp.array([
        [1.0, -0.5, 0.0],
        [-0.5, 2.0, 0.25],
        [0.0, 0.25, 3.0],
    ])

    cost.Qp = Qp

    assert jnp.allclose(cost.Qp, Qp)


def test_Qp_must_be_2d():

    cost = pdg.costs.QuadraticPlayerCost(
        nx=3,
        nu=2,
    )

    with pytest.raises(ValueError, match="`Qp` must be a 2D array"):

        cost.Qp = jnp.array([1.0, 2.0, 3.0])


def test_Qp_must_have_correct_shape():

    cost = pdg.costs.QuadraticPlayerCost(
        nx=3,
        nu=2,
    )

    with pytest.raises(ValueError, match="`Qp` must have shape"):

        cost.Qp = jnp.eye(2)


def test_Qp_must_be_symmetric():

    cost = pdg.costs.QuadraticPlayerCost(
        nx=3,
        nu=2,
    )

    Qp = jnp.array([
        [1.0, 1.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 3.0],
    ])

    with pytest.raises(ValueError, match="symmetric"):

        cost.Qp = Qp


def test_Qp_allows_indefinite_state_matrix():

    cost = pdg.costs.QuadraticPlayerCost(
        nx=3,
        nu=2,
    )

    Qp = jnp.diag(jnp.array([1.0, -1.0, 2.0]))

    cost.Qp = Qp

    assert jnp.allclose(cost.Qp, Qp)


def test_failed_Qp_assignment_does_not_mutate_state():

    cost = pdg.costs.QuadraticPlayerCost(
        nx=3,
        nu=2,
    )

    original_Qp = cost.Qp

    with pytest.raises(ValueError):

        cost.Qp = jnp.array([
            [1.0, 1.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ])

    assert jnp.allclose(cost.Qp, original_Qp)


# ---------------------------------------------------------------------
# Qp_terminal validation
# ---------------------------------------------------------------------


def test_Qp_terminal_accepts_valid_symmetric_matrix():

    cost = pdg.costs.QuadraticPlayerCost(
        nx=3,
        nu=2,
    )

    Qp_terminal = jnp.array([
        [1.0, -0.5, 0.0],
        [-0.5, 2.0, 0.25],
        [0.0, 0.25, 3.0],
    ])

    cost.Qp_terminal = Qp_terminal

    assert jnp.allclose(cost.Qp_terminal, Qp_terminal)


def test_Qp_terminal_must_be_symmetric():

    cost = pdg.costs.QuadraticPlayerCost(
        nx=3,
        nu=2,
    )

    Qp_terminal = jnp.array([
        [1.0, 1.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 3.0],
    ])

    with pytest.raises(ValueError, match="symmetric"):

        cost.Qp_terminal = Qp_terminal


def test_Qp_terminal_allows_indefinite_state_matrix():

    cost = pdg.costs.QuadraticPlayerCost(
        nx=3,
        nu=2,
    )

    Qp_terminal = jnp.diag(jnp.array([1.0, -1.0, 2.0]))

    cost.Qp_terminal = Qp_terminal

    assert jnp.allclose(cost.Qp_terminal, Qp_terminal)


# ---------------------------------------------------------------------
# Rp validation
# ---------------------------------------------------------------------


def test_Rp_accepts_valid_positive_semidefinite_matrix():

    cost = pdg.costs.QuadraticPlayerCost(
        nx=3,
        nu=2,
    )

    Rp = jnp.array([
        [2.0, 0.5],
        [0.5, 1.0],
    ])

    cost.Rp = Rp

    assert jnp.allclose(cost.Rp, Rp)


def test_Rp_must_be_2d():

    cost = pdg.costs.QuadraticPlayerCost(
        nx=3,
        nu=2,
    )

    with pytest.raises(ValueError, match="`Rp` must be a 2D array"):

        cost.Rp = jnp.array([1.0, 2.0])


def test_Rp_must_have_correct_shape():

    cost = pdg.costs.QuadraticPlayerCost(
        nx=3,
        nu=2,
    )

    with pytest.raises(ValueError, match="`Rp` must have shape"):

        cost.Rp = jnp.eye(3)


def test_Rp_must_be_symmetric():

    cost = pdg.costs.QuadraticPlayerCost(
        nx=3,
        nu=2,
    )

    Rp = jnp.array([
        [1.0, 1.0],
        [0.0, 2.0],
    ])

    with pytest.raises(ValueError, match="symmetric"):

        cost.Rp = Rp


def test_Rp_must_be_positive_semidefinite():

    cost = pdg.costs.QuadraticPlayerCost(
        nx=3,
        nu=2,
    )

    Rp = jnp.diag(jnp.array([1.0, -1.0]))

    with pytest.raises(ValueError, match="positive semidefinite"):

        cost.Rp = Rp


def test_failed_Rp_assignment_does_not_mutate_state():

    cost = pdg.costs.QuadraticPlayerCost(
        nx=3,
        nu=2,
    )

    original_Rp = cost.Rp

    with pytest.raises(ValueError):

        cost.Rp = jnp.array([
            [1.0, 1.0],
            [0.0, 2.0],
        ])

    assert jnp.allclose(cost.Rp, original_Rp)


# ---------------------------------------------------------------------
# x_ref validation
# ---------------------------------------------------------------------


def test_x_ref_accepts_valid_vector():

    cost = pdg.costs.QuadraticPlayerCost(
        nx=3,
        nu=2,
    )

    x_ref = jnp.array([1.0, 2.0, 3.0])

    cost.x_ref = x_ref

    assert jnp.allclose(cost.x_ref, x_ref)


def test_x_ref_must_have_correct_shape():

    cost = pdg.costs.QuadraticPlayerCost(
        nx=3,
        nu=2,
    )

    with pytest.raises(ValueError, match="`x_ref` must have shape"):

        cost.x_ref = jnp.array([1.0, 2.0])


# ---------------------------------------------------------------------
# u_ref validation
# ---------------------------------------------------------------------


def test_u_ref_accepts_valid_vector():

    cost = pdg.costs.QuadraticPlayerCost(
        nx=3,
        nu=2,
    )

    u_ref = jnp.array([1.0, 2.0])

    cost.u_ref = u_ref

    assert jnp.allclose(cost.u_ref, u_ref)


def test_u_ref_must_have_correct_shape():

    cost = pdg.costs.QuadraticPlayerCost(
        nx=3,
        nu=2,
    )

    with pytest.raises(ValueError, match="`u_ref` must have shape"):

        cost.u_ref = jnp.array([1.0])


# ---------------------------------------------------------------------
# add_state_cost
# ---------------------------------------------------------------------


def test_add_state_cost_all_dimensions():

    cost = pdg.costs.QuadraticPlayerCost(
        nx=3,
        nu=2,
    )

    cost.add_state_cost(
        weights=[1.0, 2.0, 3.0],
    )

    expected = jnp.diag(jnp.array([1.0, 2.0, 3.0]))

    assert jnp.allclose(cost.Qp, expected)


def test_add_state_cost_selected_indices():

    cost = pdg.costs.QuadraticPlayerCost(
        nx=4,
        nu=2,
    )

    cost.add_state_cost(
        weights=[10.0, 20.0],
        indices=[1, 3],
    )

    expected = jnp.diag(jnp.array([0.0, 10.0, 0.0, 20.0]))

    assert jnp.allclose(cost.Qp, expected)


def test_add_state_cost_accumulates():

    cost = pdg.costs.QuadraticPlayerCost(
        nx=3,
        nu=2,
    )

    cost.add_state_cost(
        weights=[1.0, 2.0, 3.0],
    )

    cost.add_state_cost(
        weights=[4.0],
        indices=[1],
    )

    expected = jnp.diag(jnp.array([1.0, 6.0, 3.0]))

    assert jnp.allclose(cost.Qp, expected)


def test_add_state_cost_allows_negative_weights():

    cost = pdg.costs.QuadraticPlayerCost(
        nx=3,
        nu=2,
    )

    cost.add_state_cost(
        weights=[1.0, -1.0, 2.0],
    )

    expected = jnp.diag(jnp.array([1.0, -1.0, 2.0]))

    assert jnp.allclose(cost.Qp, expected)


def test_add_state_cost_indices_and_weights_must_match():

    cost = pdg.costs.QuadraticPlayerCost(
        nx=3,
        nu=2,
    )

    with pytest.raises(ValueError, match="equal length"):

        cost.add_state_cost(
            weights=[1.0, 2.0],
            indices=[0],
        )


# ---------------------------------------------------------------------
# add_control_cost
# ---------------------------------------------------------------------


def test_add_control_cost_all_dimensions():

    cost = pdg.costs.QuadraticPlayerCost(
        nx=3,
        nu=2,
    )

    cost.add_control_cost(
        weights=[0.1, 0.2],
    )

    expected = jnp.diag(jnp.array([0.1, 0.2]))

    assert jnp.allclose(cost.Rp, expected)


def test_add_control_cost_selected_indices():

    cost = pdg.costs.QuadraticPlayerCost(
        nx=3,
        nu=4,
    )

    cost.add_control_cost(
        weights=[1.0, 2.0],
        indices=[0, 3],
    )

    expected = jnp.diag(jnp.array([1.0, 0.0, 0.0, 2.0]))

    assert jnp.allclose(cost.Rp, expected)


def test_add_control_cost_accumulates():

    cost = pdg.costs.QuadraticPlayerCost(
        nx=3,
        nu=2,
    )

    cost.add_control_cost(
        weights=[1.0, 2.0],
    )

    cost.add_control_cost(
        weights=[3.0],
        indices=[1],
    )

    expected = jnp.diag(jnp.array([1.0, 5.0]))

    assert jnp.allclose(cost.Rp, expected)


def test_add_control_cost_weights_must_be_nonnegative():

    cost = pdg.costs.QuadraticPlayerCost(
        nx=3,
        nu=2,
    )

    with pytest.raises(ValueError, match="nonnegative"):

        cost.add_control_cost(
            weights=[1.0, -1.0],
        )


def test_add_control_cost_indices_and_weights_must_match():

    cost = pdg.costs.QuadraticPlayerCost(
        nx=3,
        nu=2,
    )

    with pytest.raises(ValueError, match="equal length"):

        cost.add_control_cost(
            weights=[1.0],
            indices=[0, 1],
        )


# ---------------------------------------------------------------------
# add_terminal_state_cost
# ---------------------------------------------------------------------


def test_add_terminal_state_cost_allows_negative_weights():

    cost = pdg.costs.QuadraticPlayerCost(
        nx=3,
        nu=2,
    )

    cost.add_terminal_state_cost(
        weights=[1.0, -1.0],
        indices=[0, 2],
    )

    expected = jnp.diag(jnp.array([1.0, 0.0, -1.0]))

    assert jnp.allclose(cost.Qp_terminal, expected)


# ---------------------------------------------------------------------
# Convenience target setters
# ---------------------------------------------------------------------


def test_set_target_state():

    cost = pdg.costs.QuadraticPlayerCost(
        nx=3,
        nu=2,
    )

    x_ref = jnp.array([1.0, 2.0, 3.0])

    cost.set_target_state(x_ref)

    assert jnp.allclose(cost.x_ref, x_ref)


def test_set_target_control():

    cost = pdg.costs.QuadraticPlayerCost(
        nx=3,
        nu=2,
    )

    u_ref = jnp.array([4.0, 5.0])

    cost.set_target_control(u_ref)

    assert jnp.allclose(cost.u_ref, u_ref)

# ---------------------------------------------------------------------
# nx / nu validation
# ---------------------------------------------------------------------


def test_nx_must_be_positive():

    with pytest.raises(ValueError, match="`nx`"):

        pdg.costs.QuadraticPlayerCost(
            nx=0,
            nu=1,
        )


def test_nx_must_be_integer():

    with pytest.raises(ValueError, match="`nx`"):

        pdg.costs.QuadraticPlayerCost(
            nx=3.5,
            nu=1,
        )


def test_nu_must_be_positive():

    with pytest.raises(ValueError, match="`nu`"):

        pdg.costs.QuadraticPlayerCost(
            nx=1,
            nu=0,
        )


def test_nu_must_be_integer():

    with pytest.raises(ValueError, match="`nu`"):

        pdg.costs.QuadraticPlayerCost(
            nx=1,
            nu=2.5,
        )


def test_nx_is_read_only():

    cost = pdg.costs.QuadraticPlayerCost(
        nx=3,
        nu=2,
    )

    with pytest.raises(AttributeError):

        cost.nx = 10


def test_nu_is_read_only():

    cost = pdg.costs.QuadraticPlayerCost(
        nx=3,
        nu=2,
    )

    with pytest.raises(AttributeError):

        cost.nu = 10
