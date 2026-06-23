# Copyright 2026 MIT Lincoln Laboratory
# SPDX-License-Identifier: MIT

# Frontend semantic player definitions for dynamic games.
from __future__ import annotations

import jax.numpy as jnp

from abc import ABC
from collections.abc import Sequence
from typing import TypeAlias

from pydgens.frontend.costs import (
    AbstractPlayerCost,
    QuadraticPlayerCost,
)

SliceLike: TypeAlias = slice | Sequence[int]


def player(
    *,
    cost: AbstractPlayerCost,
    joint_ctrl_slice: SliceLike,
    name: str | None = None,
    state_view: Sequence[int] | None = None,
) -> AbstractPlayer:
    """
    Create a frontend player object from semantic inputs.

    A player owns a contiguous slice of the joint control vector and carries
    the cost model used to evaluate that player's objective. This factory
    chooses the most specific known frontend player type compatible with the
    supplied cost object.

    Current dispatch rules
    ----------------------
    - ``QuadraticPlayerCost`` -> ``LQPlayer``
    - any other ``AbstractPlayerCost`` -> generic ``Player``

    Parameters
    ----------
    cost:
        Player-specific frontend cost model created by ``player_cost(...)``,
        ``quadratic_cost(...)``, or another frontend cost factory.

    joint_ctrl_slice:
        Contiguous block of the joint control vector owned by this player.
        This may be a ``slice(start, stop)`` or a length-2 sequence such as
        ``(start, stop)``.

    name:
        Optional player name used for diagnostics and solution access.

    state_view:
        Optional joint-state indices associated with this player for
        plotting and diagnostics. This is metadata only; it does not imply
        state ownership.

    Returns
    -------
    AbstractPlayer
        The most specific known frontend player object compatible with
        ``cost``.

    Notes
    -----
    This factory intentionally hides some frontend type selection from
    beginners. For example, quadratic costs currently imply an
    ``LQPlayer`` because that is the structurally appropriate frontend
    player type for linear-quadratic games.

    Advanced users may still instantiate ``Player`` or ``LQPlayer``
    directly when they want precise control over the concrete type.
    """

    # Dispatch on cost type so the beginner API can stay small while the
    # concrete frontend player hierarchy grows over time.
    if isinstance(cost, QuadraticPlayerCost):
        return LQPlayer(
            cost=cost,
            joint_ctrl_slice=joint_ctrl_slice,
            name=name,
            state_view=state_view,
        )

    return Player(
        cost=cost,
        joint_ctrl_slice=joint_ctrl_slice,
        name=name,
        state_view=state_view,
    )


class AbstractPlayer(ABC):
    """
    Abstract base class for frontend semantic player definitions.

    A player defines:
    - a player-specific cost model
    - ownership of a contiguous block of the joint control vector
    - optional semantic metadata used for diagnostics and visualization

    Frontend player objects are semantic modeling objects rather than
    executable runtime representations.
    """

    pass


class Player(AbstractPlayer):
    """
    Generic semantic player definition for dynamic games.

    Parameters
    ----------
    cost:
        Player-specific cost model defined over the joint state and
        joint control spaces.

    joint_ctrl_slice:
        Contiguous block of the joint control vector owned by this player.

        This may be provided either as a Python slice,

            slice(0, 2)

        or as a length-2 sequence interpreted as ``(start, stop)``,

            (0, 2)
            [0, 2]

        In all cases, the stored value is normalized to
        ``slice(start, stop)``.

    name:
        Optional player name used for diagnostics and solution access.

    state_view:
        Optional joint-state indices associated with this player for
        plotting and diagnostics.

        This does not imply state ownership.
    """

    def __init__(
        self,
        cost: AbstractPlayerCost,
        joint_ctrl_slice: SliceLike,
        name: str | None = None,
        state_view: Sequence[int] | None = None,
        **kwargs
    ):
        
        super().__init__(**kwargs)

        if not isinstance(cost, AbstractPlayerCost):
            raise TypeError(
                "`cost` must inherit from AbstractPlayerCost."
            )

        self.cost = cost

        self.joint_ctrl_slice = _normalize_joint_ctrl_slice(
            joint_ctrl_slice
        )

        self.name = name

        if state_view is not None:

            state_view = tuple(int(i) for i in state_view)

            if any(i < 0 for i in state_view):
                raise ValueError(
                    "`state_view` indices must be nonnegative."
                )

        self.state_view = state_view

    @property
    def ctrl_dim(self) -> int:
        """
        Dimension of this player's control subvector.
        """
        return (
            self.joint_ctrl_slice.stop
            -
            self.joint_ctrl_slice.start
        )


class LQPlayer(Player):
    """
    Linear-quadratic (LQ) player definition.

    This specialization assumes:
    - quadratic player costs
    - continuous-time, time-invariant frontend semantics
    - joint-state and joint-control cost definitions

    Additional structural validation is performed to ensure compatibility
    with downstream LQ game solvers.
    """

    def __init__(
        self,
        cost: QuadraticPlayerCost,
        joint_ctrl_slice: SliceLike,
        name: str | None = None,
        state_view: Sequence[int] | None = None,
        **kwargs
    ):
    

        if not isinstance(cost, QuadraticPlayerCost):
            raise TypeError(
                "`cost` must be a QuadraticPlayerCost."
            )

        super().__init__(
            cost=cost,
            joint_ctrl_slice=joint_ctrl_slice,
            name=name,
            state_view=state_view,
            **kwargs
        )

        self._validate_cost_structure()

    def _validate_cost_structure(self):
        """
        Validate structural assumptions required by downstream
        LQ game solvers.

        In particular, the player's owned control dimensions must
        appear in the player's control penalty matrix.
        """

        start = self.joint_ctrl_slice.start
        stop = self.joint_ctrl_slice.stop

        owned_diag = jnp.diag(self.cost.Rp)[start:stop]

        if jnp.any(owned_diag <= 0):
            raise ValueError(
                "All owned control dimensions must have strictly "
                "positive control penalties in Rp."
            )
    

def _normalize_joint_ctrl_slice(joint_ctrl_slice: SliceLike) -> slice:
    if isinstance(joint_ctrl_slice, slice):
        start = joint_ctrl_slice.start
        stop = joint_ctrl_slice.stop
        step = joint_ctrl_slice.step

        if step not in (None, 1):
            raise ValueError("`joint_ctrl_slice.step` must be None or 1.")

        if start is None or stop is None:
            raise ValueError("`joint_ctrl_slice` must have explicit start and stop.")

    else:
        if len(joint_ctrl_slice) != 2:
            raise ValueError(
                "`joint_ctrl_slice` must be a slice or a length-2 sequence "
                "like `(start, stop)`."
            )

        start, stop = joint_ctrl_slice
        step = None

    if not isinstance(start, int):
        raise TypeError("`joint_ctrl_slice.start` must be an integer.")

    if not isinstance(stop, int):
        raise TypeError("`joint_ctrl_slice.stop` must be an integer.")

    if start < 0:
        raise ValueError("`joint_ctrl_slice.start` must be nonnegative.")

    if stop <= start:
        raise ValueError("`joint_ctrl_slice.stop` must be greater than start.")

    return slice(start, stop)
