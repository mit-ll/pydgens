# Copyright 2026 MIT Lincoln Laboratory
# SPDX-License-Identifier: MIT

# Frontend game definitions

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Literal

import jax.numpy as jnp


from pydgens.ir.timetypes import TimeGrid
from pydgens.ir.gametypes import (
    LinearQuadraticGameType1,
    NonlinearGameType1,
    NonlinearGameType2,
)
from pydgens.ir.costtypes import (
    ControlDomain as CostControlDomain,
    ControlStructure as CostControlStructure,
    PlayerCostSpecContinuous,
)
from pydgens.frontend.costs import (
    ContinuousPlayerCost,
)
from pydgens.frontend.constraints import ConstraintSet
from pydgens.frontend.players import (
    LQPlayer,
    Player,
)
from pydgens.frontend.dynamics import (
    LTIContinuousSystem,
    NonlinearContinuousSystem,
)


def game(
    *,
    tg: TimeGrid,
    dynamics,
    players: Sequence,
    constraints: ConstraintSet | None = None,
    discretization: Literal["zoh", "euler"] = "zoh",
):
    """
    Create a frontend game object from semantic modeling inputs.

    This is the beginner-facing entry point for constructing games. It
    chooses the most specific known frontend game type compatible with the
    supplied dynamics and player objects.

    Current dispatch rules
    ----------------------
    - ``LTIContinuousSystem`` with all ``LQPlayer`` objects -> ``LQGame``
    - ``NonlinearContinuousSystem`` with all generic ``Player`` objects
      backed by ``ContinuousPlayerCost`` -> ``NonlinearGame``
    - same nonlinear ingredients plus ``ConstraintSet`` ->
      ``ConstrainedNonlinearGame``

    Parameters
    ----------
    tg:
        Time grid used to sample the game.

    dynamics:
        Frontend dynamics object.

    players:
        Sequence of frontend player objects.

    constraints:
        Optional frontend constraint set. Supplying constraints currently
        selects the constrained nonlinear frontend game path.

    discretization:
        Method used to discretize continuous-time dynamics when lowering
        to solver IR. Currently relevant for ``LQGame`` construction.

    Returns
    -------
    object
        The most specific known frontend game object compatible with the
        supplied inputs.

    Notes
    -----
    This factory intentionally hides some frontend type selection from
    beginners. For example, linear continuous-time dynamics together with
    quadratic players currently imply ``LQGame`` because that is the
    structurally appropriate frontend game type for that combination.

    Advanced users may still instantiate concrete game classes directly
    when they want precise control over the frontend type.
    """

    # Keep the beginner API small by dispatching on semantic ingredients
    # rather than asking users to choose concrete frontend game classes.
    if (
        isinstance(dynamics, LTIContinuousSystem)
        and
        all(isinstance(p, LQPlayer) for p in players)
    ):
        if constraints is not None:
            raise NotImplementedError(
                "Frontend constrained LQ games are not yet supported."
            )
        return LQGame(
            tg=tg,
            dynamics=dynamics,
            players=players,
            discretization=discretization,
        )

    if (
        isinstance(dynamics, NonlinearContinuousSystem)
        and
        all(
            isinstance(p, Player)
            and not isinstance(p, LQPlayer)
            and isinstance(p.cost, ContinuousPlayerCost)
            for p in players
        )
    ):
        if constraints is not None:
            return ConstrainedNonlinearGame(
                tg=tg,
                dynamics=dynamics,
                players=players,
                constraints=constraints,
            )
        return NonlinearGame(
            tg=tg,
            dynamics=dynamics,
            players=players,
        )

    raise NotImplementedError(
        "No frontend game factory rule matches the supplied `dynamics` "
        "and `players`. Currently supported: `LTIContinuousSystem` with "
        "all players of type `LQPlayer`, and "
        "`NonlinearContinuousSystem` with generic `Player` objects backed "
        "by `ContinuousPlayerCost`."
    )


class AbstractGame(ABC):
    """
    Thin abstract base class for frontend semantic game definitions.

    This base class owns only the pieces that are truly common across the
    current frontend game hierarchy:

    - sampled time grid
    - frontend dynamics object
    - player sequence
    - a few dimension/accessor properties
    - shared validation helpers for player coverage of the joint control
      vector

    Subclasses remain responsible for their own structural validation and
    IR lowering logic.
    """

    def __init__(
        self,
        *,
        tg: TimeGrid,
        dynamics,
        players: Sequence,
    ):
        self.tg = tg
        self.dynamics = dynamics
        self.players = tuple(players)

    @property
    def nx(self) -> int:
        """Joint state dimension."""
        return self.dynamics.nx

    @property
    def nu(self) -> int:
        """Joint control dimension."""
        return self.dynamics.nu

    @property
    def num_players(self) -> int:
        """Number of players."""
        return len(self.players)

    def _validate_nonempty_players(self):
        if len(self.players) == 0:
            raise ValueError(
                "`players` must contain at least one player."
            )

    def _validate_contiguous_control_coverage(self):
        """
        Validate that player-owned control slices are contiguous, ordered,
        and cover the full joint control vector.

        Subclasses are responsible for checking player and cost types before
        calling this helper.
        """
        expected_start = 0

        for i, p in enumerate(self.players):
            expected_stop = expected_start + p.ctrl_dim
            expected_slice = slice(expected_start, expected_stop)

            label = (
                p.name
                if p.name is not None
                else f"players[{i}]"
            )

            if p.joint_ctrl_slice != expected_slice:
                raise ValueError(
                    f"Inconsistent control ordering for "
                    f"{label!r}. Expected {expected_slice}, got "
                    f"{p.joint_ctrl_slice}."
                )

            expected_start = expected_stop

        if expected_start != self.nu:
            raise ValueError(
                "Player control slices do not cover the "
                "full joint control vector."
            )

    @abstractmethod
    def to_ir(self):
        """
        Lower the frontend semantic game into solver-facing IR.
        """
        raise NotImplementedError


class LQGame(AbstractGame):
    """
    Frontend definition of a finite-horizon, unconstrained, linear-quadratic game.

    This frontend game object stores:
    - linear time-invariant (LTI) continuous-time system dynamics
    - Player objects with quadratic cost functions
    - time discretization information

    Parameters
    ----------
    tg:
        Time grid used to sample the game.

    dynamics:
        Continuous-time linear dynamics created by ``linear_dynamics(...)``.

    players:
        Sequence of player definitions. Each player must define a cost function
        and a contiguous slice of the joint control vector.

    discretization:
        Method used to discretize the continuous-time dynamics. ``"zoh"`` uses
        zero-order hold. ``"euler"`` uses forward Euler.

    Notes
    -----
    The executable solver representation is constructed through
    ``to_ir()``.
    """

    def __init__(
        self,
        *,
        tg: TimeGrid,
        dynamics: LTIContinuousSystem,
        players: Sequence[LQPlayer],
        discretization: Literal["zoh", "euler"] = "zoh",
    ):
        super().__init__(
            tg=tg,
            dynamics=dynamics,
            players=players,
        )
        self.discretization = discretization

        self._validate()

    # -----------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------

    def _validate(self):
        self._validate_nonempty_players()

        if self.discretization not in {"zoh", "euler"}:
            raise ValueError(
                "`discretization` must be one of "
                "{'zoh', 'euler'}."
            )

        for i, p in enumerate(self.players):

            if not isinstance(p, LQPlayer):
                raise TypeError(
                    f"players[{i}] must be an LQPlayer."
                )

            if p.cost.nx != self.nx:
                label = (
                    p.name
                    if p.name is not None
                    else f"players[{i}]"
                )
                raise ValueError(
                    f"{label!r} cost nx={p.cost.nx} "
                    f"does not match game nx={self.nx}."
                )

            if p.cost.nu != self.nu:
                label = (
                    p.name
                    if p.name is not None
                    else f"players[{i}]"
                )
                raise ValueError(
                    f"{label!r} cost nu={p.cost.nu} "
                    f"does not match game nu={self.nu}."
                )

        self._validate_contiguous_control_coverage()

    # -----------------------------------------------------------------
    # IR lowering
    # -----------------------------------------------------------------

    def to_ir(
        self,
    ) -> LinearQuadraticGameType1:
        """
        Lower the frontend semantic game definition into the
        executable IR representation used by LQ solvers.
        """

        disc_sys = self.dynamics.discretize_to_ir(
            tg = self.tg,
            method=self.discretization
        )

        nsteps = self.tg.nsteps

        Qs = []
        qs = []
        Qfs = []
        qfs = []

        Rs = []
        rs = []

        u_splits = []

        for p in self.players:

            cost = p.cost

            # ---------------------------------------------------------
            # Frontend:
            #
            #   (x - x_ref)^T Qp (x - x_ref)
            #
            # IR:
            #
            #   1/2 x^T Q x + q^T x
            #
            # Matching coefficients gives:
            #
            #   Q = 2 Qp
            #   q = -2 Qp x_ref
            # ---------------------------------------------------------

            Qi = 2.0 * cost.Qp
            qi = -2.0 * cost.Qp @ cost.x_ref
            Qfi = 2.0 * cost.Qp_terminal
            qfi = -2.0 * cost.Qp_terminal @ cost.x_ref_terminal

            Ri = 2.0 * cost.Rp
            ri = -2.0 * cost.Rp @ cost.u_ref

            # Broadcast over time grid points
            Qs.append(jnp.broadcast_to(Qi, (nsteps, *Qi.shape)))
            qs.append(jnp.broadcast_to(qi, (nsteps, *qi.shape)))
            Qfs.append(Qfi)
            qfs.append(qfi)
            Rs.append(jnp.broadcast_to(Ri, (nsteps, *Ri.shape)))
            rs.append(jnp.broadcast_to(ri, (nsteps, *ri.shape)))

            u_splits.append(p.ctrl_dim)

        return LinearQuadraticGameType1(
            cs=disc_sys,
            N=self.num_players,
            Q=jnp.stack(Qs, axis=1),
            q=jnp.stack(qs, axis=1),
            R=jnp.stack(Rs, axis=1),
            r=jnp.stack(rs, axis=1),
            u_splits=jnp.asarray(u_splits),
            Qf=jnp.stack(Qfs, axis=0),
            qf=jnp.stack(qfs, axis=0),
        )


class NonlinearGame(AbstractGame):
    """
    Frontend definition of a finite-horizon, unconstrained, nonlinear game.

    This frontend game object stores:
    - nonlinear continuous-time dynamics
    - generic player objects with continuous-time callable costs
    - time sampling information

    Notes
    -----
    The executable solver representation is constructed through ``to_ir()``.
    The current lowering target is ``NonlinearGameType1``, which is the
    unconstrained nonlinear game IR used by the iLQ solver path.
    """

    def __init__(
        self,
        *,
        tg: TimeGrid,
        dynamics: NonlinearContinuousSystem,
        players: Sequence[Player],
    ):
        super().__init__(
            tg=tg,
            dynamics=dynamics,
            players=players,
        )

        self._validate()

    def _validate(self):
        self._validate_nonempty_players()

        for i, p in enumerate(self.players):
            if not isinstance(p, Player) or isinstance(p, LQPlayer):
                raise TypeError(
                    f"players[{i}] must be a generic nonlinear `Player`."
                )

            if not isinstance(p.cost, ContinuousPlayerCost):
                raise TypeError(
                    f"players[{i}] cost must be a ContinuousPlayerCost."
                )

        self._validate_contiguous_control_coverage()

    def to_ir(self) -> NonlinearGameType1:
        """
        Lower the frontend nonlinear game into the IR used by iLQ solvers.
        """
        cs = self.dynamics.to_ir(tg=self.tg)
        costs = [p.cost.to_ir() for p in self.players]
        u_splits = jnp.asarray(
            [p.ctrl_dim for p in self.players],
            dtype=jnp.int32,
        )

        return NonlinearGameType1(
            cs=cs,
            N=self.num_players,
            costs=costs,
            u_splits=u_splits,
        )


class ConstrainedNonlinearGame(AbstractGame):
    """
    Frontend definition of a finite-horizon, constrained, nonlinear game.

    This frontend game mirrors ``NonlinearGame`` but also carries a semantic
    frontend constraint set. It lowers to ``NonlinearGameType2``, the
    constrained nonlinear IR used by the augmented Lagrangian solver path.

    Notes
    -----
    Frontend nonlinear player costs are written in JOINT control coordinates,
    while ``NonlinearGameType2`` expects LOCAL control costs. This frontend
    lowering therefore wraps each player's running cost into that player's
    owned control slice. The current prototype assumes those frontend costs
    are AL-compatible, i.e. they depend only on the owning player's control.
    """

    def __init__(
        self,
        *,
        tg: TimeGrid,
        dynamics: NonlinearContinuousSystem,
        players: Sequence[Player],
        constraints: ConstraintSet,
    ):
        super().__init__(
            tg=tg,
            dynamics=dynamics,
            players=players,
        )
        self.constraints = constraints

        self._validate()

    def _validate(self):
        self._validate_nonempty_players()

        if not isinstance(self.constraints, ConstraintSet):
            raise TypeError(
                "`constraints` must be a frontend ConstraintSet."
            )

        for i, p in enumerate(self.players):
            if not isinstance(p, Player) or isinstance(p, LQPlayer):
                raise TypeError(
                    f"players[{i}] must be a generic nonlinear `Player`."
                )

            if not isinstance(p.cost, ContinuousPlayerCost):
                raise TypeError(
                    f"players[{i}] cost must be a ContinuousPlayerCost."
                )

        self._validate_contiguous_control_coverage()

    def _make_local_running_cost(
        self,
        *,
        player: Player,
    ):
        """
        Wrap a frontend joint-control running cost into player-local form.

        The returned callable has the ``(t, x, u_i)`` signature required by
        ``NonlinearGameType2``. It rebuilds a joint control vector by placing
        ``u_i`` into the owning player's control block and zeroing all other
        control entries.
        """
        start = player.joint_ctrl_slice.start
        stop = player.joint_ctrl_slice.stop
        running_joint = player.cost.running
        nu = self.nu

        def running_local(t, x, u_i):
            u_joint = jnp.zeros((nu,), dtype=jnp.result_type(x, u_i))
            u_joint = u_joint.at[start:stop].set(u_i)
            return running_joint(t, x, u_joint)

        return running_local

    def _make_terminal_cost(
        self,
        *,
        player: Player,
    ):
        """
        Ensure each lowered player cost has a terminal callable.

        ``NonlinearGameType2`` currently requires terminal costs, so a missing
        frontend terminal cost is interpreted as the zero terminal cost.
        """
        if player.cost.terminal is not None:
            return player.cost.terminal

        def zero_terminal(t, x):
            del t
            return jnp.asarray(0.0, dtype=x.dtype)

        return zero_terminal

    def to_ir(self) -> NonlinearGameType2:
        """
        Lower the constrained frontend nonlinear game into AL-facing IR.
        """
        cs = self.dynamics.to_ir(tg=self.tg)
        constraints = self.constraints.to_ir(
            tg=self.tg,
            nx=self.nx,
            nu=self.nu,
        )
        u_splits = jnp.asarray(
            [p.ctrl_dim for p in self.players],
            dtype=jnp.int32,
        )

        costs = []
        for p in self.players:
            costs.append(
                PlayerCostSpecContinuous(
                    running=self._make_local_running_cost(player=p),
                    terminal=self._make_terminal_cost(player=p),
                    control_domain=CostControlDomain.LOCAL,
                    control_structure=CostControlStructure.LOCAL_ONLY,
                )
            )

        return NonlinearGameType2(
            cs=cs,
            N=self.num_players,
            costs=costs,
            constraints=constraints,
            u_splits=u_splits,
        )
