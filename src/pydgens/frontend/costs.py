# Copyright 2026 MIT Lincoln Laboratory
# SPDX-License-Identifier: MIT

# User-facing class definitions and factor functions for defining cost functions

from __future__ import annotations

from abc import ABC

import jax.numpy as jnp

from pydgens.ir.costtypes import (
    ControlDomain,
    PlayerCostSpecContinuous,
)


def player_cost(
    *,
    running,
    terminal=None,
) -> ContinuousPlayerCost:
    """
    Create a continuous-time player cost from user-provided callables.

    This is the generic beginner-facing frontend cost factory for
    nonlinear games. It stores Python callables describing a player's
    objective in continuous time, while hiding the lower-level IR cost
    dataclass from the public modeling workflow.

    Parameters
    ----------
    running:
        Running cost callable of the form

            running(t, x, u_joint) -> scalar

    terminal:
        Optional terminal cost callable of the form

            terminal(t, x) -> scalar

    Returns
    -------
    ContinuousPlayerCost
        Generic continuous-time player cost object suitable for nonlinear
        frontend games.

    Notes
    -----
    The frontend nonlinear-cost contract is intentionally narrower than
    the lower-level IR: running costs are defined over the joint control
    vector, just as quadratic frontend costs are. Structural conditions
    such as player-wise control separability are validated later at the
    game/solver layer, where player ownership information is available.
    """
    return ContinuousPlayerCost(
        running=running,
        terminal=terminal,
    )


def quadratic_cost(
    *,
    nx: int,
    nu: int,
    state_weights=None,
    state_indices: list[int] | None = None,
    state_target=None,
    terminal_state_weights=None,
    terminal_state_indices: list[int] | None = None,
    terminal_state_target=None,
    control_weights=None,
    control_indices: list[int] | None = None,
    control_target=None,
) -> QuadraticPlayerCost:
    """
    Create a quadratic player cost using semantic frontend arguments.

    This factory is the beginner-facing way to define quadratic running
    costs over the joint state and joint control spaces. It wraps
    ``QuadraticPlayerCost`` while exposing a more tutorial-friendly API
    based on:

    - which state dimensions are penalized
    - which control dimensions are penalized
    - what state target is desired
    - what control target is desired

    Conceptually, the returned cost represents

        (x - x_ref)^T Qp (x - x_ref)
        +
        (u - u_ref)^T Rp (u - u_ref)
        +
        (x_T - x_ref_terminal)^T Qp_terminal (x_T - x_ref_terminal)

    where ``Qp`` and ``Rp`` are diagonal frontend penalty matrices built
    from the provided weights.

    Parameters
    ----------
    nx:
        Joint state dimension.

    nu:
        Joint control dimension.

    state_weights:
        Optional nonnegative weights for quadratic state penalties.

    state_indices:
        Optional joint-state indices to penalize. If omitted, the state
        weights apply to all state dimensions.

    state_target:
        Optional desired joint-state reference ``x_ref``.

    terminal_state_weights:
        Optional nonnegative weights for terminal quadratic state penalties.

    terminal_state_indices:
        Optional joint-state indices to penalize at the terminal state. If
        omitted, the terminal state weights apply to all state dimensions.

    terminal_state_target:
        Optional desired terminal joint-state reference ``x_ref_terminal``.

    control_weights:
        Optional nonnegative weights for quadratic control penalties.

    control_indices:
        Optional joint-control indices to penalize. If omitted, the
        control weights apply to all control dimensions.

    control_target:
        Optional desired joint-control reference ``u_ref``.

    Returns
    -------
    QuadraticPlayerCost
        Configured quadratic player cost object.

    """

    cost = QuadraticPlayerCost(
        nx=nx,
        nu=nu,
    )

    # Guard against a common beginner mistake: specifying which entries
    # to penalize but forgetting to provide the corresponding weights.
    if state_indices is not None and state_weights is None:
        raise ValueError(
            "`state_indices` requires `state_weights`."
        )

    if terminal_state_indices is not None and terminal_state_weights is None:
        raise ValueError(
            "`terminal_state_indices` requires `terminal_state_weights`."
        )

    if control_indices is not None and control_weights is None:
        raise ValueError(
            "`control_indices` requires `control_weights`."
        )

    # Apply penalties before targets so the returned object reads like a
    # fully assembled semantic cost model when inspected.
    if state_weights is not None:
        cost.add_state_cost(
            weights=state_weights,
            indices=state_indices,
        )

    if control_weights is not None:
        cost.add_control_cost(
            weights=control_weights,
            indices=control_indices,
        )

    if state_target is not None:
        cost.set_target_state(state_target)

    if terminal_state_weights is not None:
        cost.add_terminal_state_cost(
            weights=terminal_state_weights,
            indices=terminal_state_indices,
        )

    if terminal_state_target is not None:
        cost.set_terminal_target_state(terminal_state_target)

    if control_target is not None:
        cost.set_target_control(control_target)

    return cost


def matrix_quadratic_cost(
    *,
    nx: int,
    nu: int,
    state_matrix=None,
    state_target=None,
    terminal_state_matrix=None,
    terminal_state_target=None,
    control_matrix=None,
    control_target=None,
) -> QuadraticPlayerCost:
    """
    Create an advanced quadratic player cost from explicit full matrices.

    This is the LQ companion to ``quadratic_cost(...)`` for games that need
    coupled state terms such as ``||p_guard - alpha p_bandit||^2`` or
    indefinite state rewards/penalties. The simpler ``quadratic_cost(...)``
    remains the recommended beginner-facing factory for diagonal,
    nonnegative weights.
    """
    cost = QuadraticPlayerCost(
        nx=nx,
        nu=nu,
    )

    if state_matrix is not None:
        cost.set_state_matrix(state_matrix)

    if state_target is not None:
        cost.set_target_state(state_target)

    if terminal_state_matrix is not None:
        cost.set_terminal_state_matrix(terminal_state_matrix)

    if terminal_state_target is not None:
        cost.set_terminal_target_state(terminal_state_target)

    if control_matrix is not None:
        cost.set_control_matrix(control_matrix)

    if control_target is not None:
        cost.set_target_control(control_target)

    return cost


class AbstractPlayerCost(ABC):
    """
    Abstract base class for player-specific cost models.

    Player costs are defined over the joint state and joint control spaces.

    Frontend cost objects are semantic modeling objects rather than
    executable runtime representations.
    """

    def __init__(self, **kwargs):
        super().__init__()


class ContinuousPlayerCost(AbstractPlayerCost):
    """
    Generic continuous-time player cost defined by Python callables.

    This frontend cost object is intended for nonlinear games. It stores a
    running cost and optional terminal cost in semantic form and can lower
    itself into ``PlayerCostSpecContinuous`` for solver-facing IR.

    Parameters
    ----------
    running:
        Running cost callable of the form

            running(t, x, u_joint) -> scalar

    terminal:
        Optional terminal cost callable ``terminal(t, x)``.

    Notes
    -----
    This frontend object does not try to encode player ownership of
    control variables. It therefore always treats the running cost as a
    function of the full joint control vector. Stronger structural
    assumptions, such as player-wise control separability, belong at the
    player/game layer where control partitions are known.
    """

    def __init__(
        self,
        *,
        running,
        terminal=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if not callable(running):
            raise TypeError(
                f"`running` must be callable. Got {type(running)}."
            )

        if terminal is not None and not callable(terminal):
            raise TypeError(
                f"`terminal` must be callable or None. Got {type(terminal)}."
            )

        self.running = running
        self.terminal = terminal

    def to_ir(self) -> PlayerCostSpecContinuous:
        """
        Lower this frontend cost object into solver-facing continuous-time
        cost IR.
        """
        return PlayerCostSpecContinuous(
            running=self.running,
            terminal=self.terminal,
            control_domain=ControlDomain.JOINT,
        )


class QuadraticPlayerCost(AbstractPlayerCost):
    """
    Time-invariant quadratic running and terminal state cost defined over the
    joint state and joint control spaces.

    Conceptually represents costs of the form

        (x - x_ref)^T Qp (x - x_ref)
        +
        (u - u_ref)^T Rp (u - u_ref)
        +
        (x_T - x_ref_terminal)^T Qp_terminal (x_T - x_ref_terminal)

    where:
    - x is the joint state vector
    - u is the joint control vector

    Notes
    -----
    The matrices ``Qp`` and ``Rp`` are semantic/frontend penalty matrices.

    These are intentionally distinguished from canonical affine-quadratic
    forms such as

        1/2 x^T Q x + q^T x

    which may use different internal matrix scalings during lowering into
    executable IR representations.

    The beginner-facing ``quadratic_cost(...)`` factory builds diagonal costs
    from nonnegative scalar weights for simplicity and clarity. Direct matrix
    assignment through ``Qp``, ``Rp``, ``Qp_terminal``, or the explicit
    ``set_*_matrix`` methods supports full matrices where the solver supports
    them. State matrices must be symmetric and may be indefinite; control
    matrices must be symmetric positive semidefinite.

    Attributes
    ----------
    Qp:
        Joint-state quadratic penalty matrix.

    Rp:
        Joint-control quadratic penalty matrix. Must be symmetric positive
        semidefinite. Additional solver-specific structure, such as
        block-diagonal control coupling by player, is checked when the game is
        lowered or solved.

    x_ref:
        Desired joint-state reference target.

    Qp_terminal:
        Joint-state quadratic terminal penalty matrix.

    x_ref_terminal:
        Desired terminal joint-state reference target.

    u_ref:
        Desired joint-control reference target.

    nx:
        Joint state dimension.

    nu:
        Joint control dimension.
    """

    def __init__(
        self,
        nx: int,
        nu: int,
        **kwargs
    ):
        super().__init__(**kwargs)

        if not isinstance(nx, int) or nx <= 0:
            raise ValueError(f"`nx` must be positive integer. Got {nx}.")

        if not isinstance(nu, int) or nu <= 0:
            raise ValueError(f"`nu` must be positive integer. Got {nu}.")
        
        self._nx = nx
        self._nu = nu

        self._Qp = None
        self._Rp = None
        self._Qp_terminal = None

        self._x_ref = None
        self._x_ref_terminal = None
        self._u_ref = None

        # Initialize through validated setters
        self.Qp = jnp.zeros((nx, nx))
        self.Rp = jnp.zeros((nu, nu))
        self.Qp_terminal = jnp.zeros((nx, nx))

        self.x_ref = jnp.zeros(nx)
        self.x_ref_terminal = jnp.zeros(nx)
        self.u_ref = jnp.zeros(nu)

    @property
    def nx(self) -> int:
        """
        Joint state dimension.
        """
        return self._nx


    @property
    def nu(self) -> int:
        """
        Joint control dimension.
        """
        return self._nu

    # -----------------------------------------------------------------
    # Qp
    # -----------------------------------------------------------------

    @property
    def Qp(self) -> jnp.ndarray:
        """
        Joint-state quadratic penalty matrix.
        """
        return self._Qp

    @Qp.setter
    def Qp(self, value: jnp.ndarray):

        self._Qp = _validate_symmetric_matrix(
            value,
            shape=(self.nx, self.nx),
            name="Qp",
        )

    # -----------------------------------------------------------------
    # Qp_terminal
    # -----------------------------------------------------------------

    @property
    def Qp_terminal(self) -> jnp.ndarray:
        """
        Joint-state quadratic terminal penalty matrix.
        """
        return self._Qp_terminal

    @Qp_terminal.setter
    def Qp_terminal(self, value: jnp.ndarray):

        self._Qp_terminal = _validate_symmetric_matrix(
            value,
            shape=(self.nx, self.nx),
            name="Qp_terminal",
        )

    # -----------------------------------------------------------------
    # Rp
    # -----------------------------------------------------------------

    @property
    def Rp(self) -> jnp.ndarray:
        """
        Joint-control quadratic penalty matrix.
        """
        return self._Rp

    @Rp.setter
    def Rp(self, value: jnp.ndarray):

        self._Rp = _validate_positive_semidefinite_matrix(
            value,
            shape=(self.nu, self.nu),
            name="Rp",
        )

    # -----------------------------------------------------------------
    # x_ref
    # -----------------------------------------------------------------

    @property
    def x_ref(self) -> jnp.ndarray:
        """
        Desired joint-state reference target.
        """
        return self._x_ref

    @x_ref.setter
    def x_ref(self, value: jnp.ndarray):

        value = jnp.asarray(value)

        if value.shape != (self.nx,):
            raise ValueError(
                f"`x_ref` must have shape ({self.nx},). "
                f"Got {value.shape}."
            )

        self._x_ref = value

    # -----------------------------------------------------------------
    # x_ref_terminal
    # -----------------------------------------------------------------

    @property
    def x_ref_terminal(self) -> jnp.ndarray:
        """
        Desired terminal joint-state reference target.
        """
        return self._x_ref_terminal

    @x_ref_terminal.setter
    def x_ref_terminal(self, value: jnp.ndarray):

        value = jnp.asarray(value)

        if value.shape != (self.nx,):
            raise ValueError(
                f"`x_ref_terminal` must have shape ({self.nx},). "
                f"Got {value.shape}."
            )

        self._x_ref_terminal = value

    # -----------------------------------------------------------------
    # u_ref
    # -----------------------------------------------------------------

    @property
    def u_ref(self) -> jnp.ndarray:
        """
        Desired joint-control reference target.
        """
        return self._u_ref

    @u_ref.setter
    def u_ref(self, value: jnp.ndarray):

        value = jnp.asarray(value)

        if value.shape != (self.nu,):
            raise ValueError(
                f"`u_ref` must have shape ({self.nu},). "
                f"Got {value.shape}."
            )

        self._u_ref = value

    # -----------------------------------------------------------------
    # State costs
    # -----------------------------------------------------------------

    def add_state_cost(
        self,
        weights,
        indices: list[int] | None = None,
    ):
        """
        Add diagonal quadratic penalties on joint-state variables.

        Parameters
        ----------
        weights:
            Nonnegative penalty weights.

        indices:
            Joint-state indices to penalize.

            If omitted, penalties are applied to all state dimensions.
        """

        weights = jnp.asarray(weights)

        if jnp.any(weights < 0):
            raise ValueError(
                "`weights` must be nonnegative."
            )

        if indices is None:

            if weights.shape != (self.nx,):
                raise ValueError(
                    f"Expected weights shape ({self.nx},). "
                    f"Got {weights.shape}."
                )

            indices = list(range(self.nx))

        if len(indices) != len(weights):
            raise ValueError(
                "`indices` and `weights` must have equal length."
            )

        Qp = self.Qp

        for i, w in zip(indices, weights):
            Qp = Qp.at[i, i].add(w)

        self.Qp = Qp

    def set_state_matrix(self, matrix):
        """
        Set a full joint-state matrix for advanced LQ games.

        This is a descriptive wrapper around the ``Qp`` property setter.
        """
        self.Qp = matrix

    # -----------------------------------------------------------------
    # Control costs
    # -----------------------------------------------------------------

    def add_control_cost(
        self,
        weights,
        indices: list[int] | None = None,
    ):
        """
        Add diagonal quadratic penalties on joint-control variables.

        Parameters
        ----------
        weights:
            Nonnegative penalty weights.

        indices:
            Joint-control indices to penalize.

            If omitted, penalties are applied to all control dimensions.
        """

        weights = jnp.asarray(weights)

        if jnp.any(weights < 0):
            raise ValueError(
                "`weights` must be nonnegative."
            )

        if indices is None:

            if weights.shape != (self.nu,):
                raise ValueError(
                    f"Expected weights shape ({self.nu},). "
                    f"Got {weights.shape}."
                )

            indices = list(range(self.nu))

        if len(indices) != len(weights):
            raise ValueError(
                "`indices` and `weights` must have equal length."
            )

        Rp = self.Rp

        for i, w in zip(indices, weights):
            Rp = Rp.at[i, i].add(w)

        self.Rp = Rp

    def set_control_matrix(self, matrix):
        """
        Set a full joint-control matrix for advanced LQ games.

        This is a descriptive wrapper around the ``Rp`` property setter.
        """
        self.Rp = matrix

    # -----------------------------------------------------------------
    # Terminal state costs
    # -----------------------------------------------------------------

    def add_terminal_state_cost(
        self,
        weights,
        indices: list[int] | None = None,
    ):
        """
        Add diagonal quadratic penalties on terminal joint-state variables.
        """

        weights = jnp.asarray(weights)

        if jnp.any(weights < 0):
            raise ValueError(
                "`weights` must be nonnegative."
            )

        if indices is None:

            if weights.shape != (self.nx,):
                raise ValueError(
                    f"Expected weights shape ({self.nx},). "
                    f"Got {weights.shape}."
                )

            indices = list(range(self.nx))

        if len(indices) != len(weights):
            raise ValueError(
                "`indices` and `weights` must have equal length."
            )

        Qp_terminal = self.Qp_terminal

        for i, w in zip(indices, weights):
            Qp_terminal = Qp_terminal.at[i, i].add(w)

        self.Qp_terminal = Qp_terminal

    def set_terminal_state_matrix(self, matrix):
        """
        Set a full terminal joint-state matrix for advanced LQ games.

        This is a descriptive wrapper around the ``Qp_terminal`` property
        setter.
        """
        self.Qp_terminal = matrix

    # -----------------------------------------------------------------
    # Reference targets
    # -----------------------------------------------------------------

    def set_target_state(
        self,
        x_ref,
    ):
        """
        Convenience wrapper for setting ``x_ref``.
        """
        self.x_ref = x_ref

    def set_target_control(
        self,
        u_ref,
    ):
        """
        Convenience wrapper for setting ``u_ref``.
        """
        self.u_ref = u_ref

    def set_terminal_target_state(
        self,
        x_ref_terminal,
    ):
        """
        Convenience wrapper for setting ``x_ref_terminal``.
        """
        self.x_ref_terminal = x_ref_terminal


def _validate_symmetric_matrix(
    matrix,
    *,
    shape: tuple[int, int],
    name: str,
):
    matrix = jnp.asarray(matrix)

    if matrix.ndim != 2:
        raise ValueError(
            f"`{name}` must be a 2D array. Got shape {matrix.shape}."
        )

    if matrix.shape != shape:
        raise ValueError(
            f"`{name}` must have shape {shape}. Got {matrix.shape}."
        )

    if not jnp.allclose(matrix, matrix.T):
        raise ValueError(
            f"`{name}` must be symmetric."
        )

    return matrix


def _validate_positive_semidefinite_matrix(
    matrix,
    *,
    shape: tuple[int, int],
    name: str,
):
    matrix = _validate_symmetric_matrix(
        matrix,
        shape=shape,
        name=name,
    )

    eigvals = jnp.linalg.eigvalsh(matrix)
    if jnp.any(eigvals < -1e-10):
        raise ValueError(
            f"`{name}` must be positive semidefinite."
        )

    return matrix
