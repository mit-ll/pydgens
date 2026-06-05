# Copyright 2026 MIT Lincoln Laboratory
# SPDX-License-Identifier: MIT

# tests/frontend/test_players.py

import pytest

# direct import classes/functions that support tests
from pydgens.frontend.costs import (
    AbstractPlayerCost,
    QuadraticPlayerCost,
)

# module under test (via public api)
import pydgens as pdg

# ---------------------------------------------------------------------
# Generic Player construction
# ---------------------------------------------------------------------


def test_player_constructs():

    cost = QuadraticPlayerCost(
        nx=4,
        nu=3,
    )

    player = pdg.players.Player(
        cost=cost,
        joint_ctrl_slice=slice(0, 2),
        name="player1",
    )

    assert player.cost is cost
    assert player.joint_ctrl_slice == slice(0, 2)
    assert player.name == "player1"


def test_player_factory_returns_lqplayer_for_quadratic_cost():

    cost = QuadraticPlayerCost(
        nx=4,
        nu=3,
    )
    cost.add_control_cost(
        weights=[1.0, 2.0],
        indices=[0, 1],
    )

    player = pdg.player(
        cost=cost,
        joint_ctrl_slice=slice(0, 2),
        name="player1",
        state_view=[1, 3],
    )

    assert isinstance(player, pdg.players.LQPlayer)
    assert player.cost is cost
    assert player.joint_ctrl_slice == slice(0, 2)
    assert player.name == "player1"
    assert player.state_view == (1, 3)


def test_player_factory_returns_generic_player_for_unknown_cost_type():

    class FakeCost(AbstractPlayerCost):
        pass

    cost = FakeCost()

    player = pdg.player(
        cost=cost,
        joint_ctrl_slice=slice(1, 3),
        name="generic",
    )

    assert isinstance(player, pdg.players.Player)
    assert not isinstance(player, pdg.players.LQPlayer)
    assert player.cost is cost
    assert player.joint_ctrl_slice == slice(1, 3)
    assert player.name == "generic"


def test_player_normalizes_sequence_control_slice():

    cost = QuadraticPlayerCost(
        nx=4,
        nu=3,
    )

    player = pdg.players.Player(
        cost=cost,
        joint_ctrl_slice=(1, 3),
    )

    assert player.joint_ctrl_slice == slice(1, 3)


def test_player_ctrl_dim():

    cost = QuadraticPlayerCost(
        nx=4,
        nu=5,
    )

    player = pdg.players.Player(
        cost=cost,
        joint_ctrl_slice=slice(2, 5),
    )

    assert player.ctrl_dim == 3


def test_player_accepts_state_view():

    cost = QuadraticPlayerCost(
        nx=6,
        nu=3,
    )

    player = pdg.players.Player(
        cost=cost,
        joint_ctrl_slice=slice(0, 2),
        state_view=[1, 3, 5],
    )

    assert player.state_view == (1, 3, 5)


# ---------------------------------------------------------------------
# Player type validation
# ---------------------------------------------------------------------


def test_player_requires_abstract_player_cost():

    class FakeCost:
        pass

    with pytest.raises(TypeError, match="AbstractPlayerCost"):

        pdg.players.Player(
            cost=FakeCost(),
            joint_ctrl_slice=slice(0, 1),
        )


# ---------------------------------------------------------------------
# joint_ctrl_slice validation
# ---------------------------------------------------------------------


def test_joint_ctrl_slice_requires_integer_start():

    cost = QuadraticPlayerCost(
        nx=4,
        nu=3,
    )

    with pytest.raises(TypeError, match="start"):

        pdg.players.Player(
            cost=cost,
            joint_ctrl_slice=slice(0.5, 2),
        )


def test_joint_ctrl_slice_requires_integer_stop():

    cost = QuadraticPlayerCost(
        nx=4,
        nu=3,
    )

    with pytest.raises(TypeError, match="stop"):

        pdg.players.Player(
            cost=cost,
            joint_ctrl_slice=slice(0, 2.5),
        )


def test_joint_ctrl_slice_requires_nonnegative_start():

    cost = QuadraticPlayerCost(
        nx=4,
        nu=3,
    )

    with pytest.raises(ValueError, match="nonnegative"):

        pdg.players.Player(
            cost=cost,
            joint_ctrl_slice=slice(-1, 2),
        )


def test_joint_ctrl_slice_requires_stop_greater_than_start():

    cost = QuadraticPlayerCost(
        nx=4,
        nu=3,
    )

    with pytest.raises(ValueError, match="greater than start"):

        pdg.players.Player(
            cost=cost,
            joint_ctrl_slice=slice(2, 2),
        )


def test_joint_ctrl_slice_rejects_nonunit_step():

    cost = QuadraticPlayerCost(
        nx=4,
        nu=3,
    )

    with pytest.raises(ValueError, match="step"):

        pdg.players.Player(
            cost=cost,
            joint_ctrl_slice=slice(0, 4, 2),
        )


def test_joint_ctrl_slice_requires_explicit_start():

    cost = QuadraticPlayerCost(
        nx=4,
        nu=3,
    )

    with pytest.raises(ValueError, match="explicit"):

        pdg.players.Player(
            cost=cost,
            joint_ctrl_slice=slice(None, 2),
        )


def test_joint_ctrl_slice_requires_explicit_stop():

    cost = QuadraticPlayerCost(
        nx=4,
        nu=3,
    )

    with pytest.raises(ValueError, match="explicit"):

        pdg.players.Player(
            cost=cost,
            joint_ctrl_slice=slice(0, None),
        )


def test_joint_ctrl_slice_sequence_must_have_length_two():

    cost = QuadraticPlayerCost(
        nx=4,
        nu=3,
    )

    with pytest.raises(ValueError, match="length-2"):

        pdg.players.Player(
            cost=cost,
            joint_ctrl_slice=(0, 1, 2),
        )


# ---------------------------------------------------------------------
# state_view validation
# ---------------------------------------------------------------------


def test_state_view_requires_nonnegative_indices():

    cost = QuadraticPlayerCost(
        nx=4,
        nu=3,
    )

    with pytest.raises(ValueError, match="nonnegative"):

        pdg.players.Player(
            cost=cost,
            joint_ctrl_slice=slice(0, 2),
            state_view=[0, -1],
        )


def test_state_view_converts_to_tuple():

    cost = QuadraticPlayerCost(
        nx=4,
        nu=3,
    )

    player = pdg.players.Player(
        cost=cost,
        joint_ctrl_slice=slice(0, 2),
        state_view=[1, 2],
    )

    assert isinstance(player.state_view, tuple)


# ---------------------------------------------------------------------
# LQPlayer construction
# ---------------------------------------------------------------------


def test_lqplayer_constructs():

    cost = QuadraticPlayerCost(
        nx=4,
        nu=3,
    )

    cost.add_control_cost(
        weights=[1.0, 2.0],
        indices=[0, 1],
    )

    player = pdg.players.LQPlayer(
        cost=cost,
        joint_ctrl_slice=slice(0, 2),
        name="lq_player",
    )

    assert player.cost is cost
    assert player.ctrl_dim == 2


# ---------------------------------------------------------------------
# LQPlayer cost validation
# ---------------------------------------------------------------------


def test_lqplayer_requires_quadratic_player_cost():

    class FakeCost(AbstractPlayerCost):
        pass

    with pytest.raises(TypeError, match="QuadraticPlayerCost"):

        pdg.players.LQPlayer(
            cost=FakeCost(),
            joint_ctrl_slice=slice(0, 1),
        )


def test_lqplayer_requires_positive_owned_control_penalties():

    cost = QuadraticPlayerCost(
        nx=4,
        nu=4,
    )

    # Penalize controls 0 and 2 only
    cost.add_control_cost(
        weights=[1.0, 1.0],
        indices=[0, 2],
    )

    # Player owns controls 0 and 1
    # Control 1 has zero penalty -> invalid
    with pytest.raises(ValueError, match="strictly positive"):

        pdg.players.LQPlayer(
            cost=cost,
            joint_ctrl_slice=slice(0, 2),
        )


def test_lqplayer_accepts_positive_owned_control_penalties():

    cost = QuadraticPlayerCost(
        nx=4,
        nu=4,
    )

    cost.add_control_cost(
        weights=[1.0, 2.0],
        indices=[0, 1],
    )

    player = pdg.players.LQPlayer(
        cost=cost,
        joint_ctrl_slice=slice(0, 2),
    )

    assert isinstance(player, pdg.players.LQPlayer)


def test_lqplayer_allows_unpenalized_unowned_controls():

    cost = QuadraticPlayerCost(
        nx=4,
        nu=4,
    )

    # Penalize only controls owned by player
    cost.add_control_cost(
        weights=[1.0, 2.0],
        indices=[0, 1],
    )

    # Controls 2 and 3 remain unpenalized
    # This should still be valid
    player = pdg.players.LQPlayer(
        cost=cost,
        joint_ctrl_slice=slice(0, 2),
    )

    assert isinstance(player, pdg.players.LQPlayer)


def test_lqplayer_allows_cross_player_penalties():

    cost = QuadraticPlayerCost(
        nx=4,
        nu=4,
    )

    # Penalize all controls jointly
    cost.add_control_cost(
        weights=[1.0, 2.0, 3.0, 4.0],
    )

    player = pdg.players.LQPlayer(
        cost=cost,
        joint_ctrl_slice=slice(1, 3),
    )

    assert isinstance(player, pdg.players.LQPlayer)
