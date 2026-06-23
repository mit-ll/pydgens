# Copyright 2026 MIT Lincoln Laboratory
# SPDX-License-Identifier: MIT

# Frontend semantic constraint definitions for dynamic games.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import jax.numpy as jnp

from pydgens.ir.constrainttypes import (
    ConstraintBlockGridMap,
    GameConstraintGridMap,
)
from pydgens.ir.timetypes import TimeGrid


def control_bounds(
    *,
    lower=None,
    upper=None,
    indices: Sequence[int] | None = None,
    steps: Sequence[int] | None = None,
) -> ControlBounds:
    """
    Create path-wise bounds on the joint control vector.

    Bounds may apply to selected joint-control dimensions and selected
    control intervals. For example, this can express constraints such as:

        ``-1 <= u[0] <= 1``
        ``-2 <= u[1] <= 2``

    over some or all control intervals.

    Parameters
    ----------
    lower:
        Lower bounds for the selected control dimensions. May be a scalar or
        a length-matched sequence. ``None`` means no lower bound.

    upper:
        Upper bounds for the selected control dimensions. May be a scalar or
        a length-matched sequence. ``None`` means no upper bound.

    indices:
        Joint-control indices to constrain. If omitted, the bounds apply to
        all joint-control dimensions when lowered against a game.

    steps:
        Optional control-interval indices where the bounds are active. If
        omitted, the bounds apply on all control intervals.

    Returns
    -------
    ControlBounds
        Frontend control-bound specification suitable for
        ``constraint_set(...)``.
    """
    return ControlBounds(
        lower=lower,
        upper=upper,
        indices=indices,
        steps=steps,
    )


def state_bounds(
    *,
    lower=None,
    upper=None,
    indices: Sequence[int] | None = None,
    steps: Sequence[int] | None = None,
    include_terminal: bool = True,
) -> StateBounds:
    """
    Create bounds on the joint state vector.

    Bounds may apply to selected joint-state dimensions and selected path
    intervals. For example, this can express constraints such as:

        ``px >= 0``
        ``-5 <= v <= 5``

    with the option to also enforce the same bound at the terminal state.

    Parameters
    ----------
    lower:
        Lower bounds for the selected state dimensions. May be a scalar or
        a length-matched sequence. ``None`` means no lower bound.

    upper:
        Upper bounds for the selected state dimensions. May be a scalar or
        a length-matched sequence. ``None`` means no upper bound.

    indices:
        Joint-state indices to constrain. If omitted, the bounds apply to
        all joint-state dimensions when lowered against a game.

    steps:
        Optional control-interval indices where the path portion of the
        bounds is active. If omitted, the bounds apply on all control
        intervals.

    include_terminal:
        Whether the same bounds should also be enforced at the terminal node.

    Returns
    -------
    StateBounds
        Frontend state-bound specification suitable for
        ``constraint_set(...)``.
    """
    return StateBounds(
        lower=lower,
        upper=upper,
        indices=indices,
        steps=steps,
        include_terminal=include_terminal,
    )


def constraint_set(
    *items: AbstractConstraintSpec,
) -> ConstraintSet:
    """
    Bundle frontend constraint specifications into a single set.

    Parameters
    ----------
    *items:
        Constraint specifications such as ``control_bounds(...)`` or
        ``state_bounds(...)``.

    Returns
    -------
    ConstraintSet
        Frontend constraint collection that can be passed to ``game(...)``.

    Examples
    --------
    >>> cons = constraint_set(
    ...     control_bounds(lower=-1.0, upper=1.0, indices=[0]),
    ...     state_bounds(lower=0.0, indices=[1]),
    ... )
    """
    return ConstraintSet(items=items)


class AbstractConstraintSpec(ABC):
    """
    Abstract base class for frontend semantic constraint specifications.

    Frontend constraint objects are semantic modeling objects rather than
    executable runtime representations. They lower into one or more IR
    ``ConstraintBlockGridMap`` objects once a time grid and joint state/control
    dimensions are known.
    """

    @abstractmethod
    def to_ir_blocks(
        self,
        *,
        tg: TimeGrid,
        nx: int,
        nu: int,
    ) -> tuple[ConstraintBlockGridMap, ...]:
        """
        Lower this frontend constraint into one or more IR constraint blocks.
        """
        raise NotImplementedError


class ConstraintSet:
    """
    Collection of frontend constraint specifications.

    This container is intentionally small: it groups a set of semantic
    constraint specifications and lowers them into a single
    ``GameConstraintGridMap`` when the surrounding game dimensions are known.
    """

    def __init__(
        self,
        *,
        items: Sequence[AbstractConstraintSpec],
    ):
        self.items = tuple(items)

        for i, item in enumerate(self.items):
            if not isinstance(item, AbstractConstraintSpec):
                raise TypeError(
                    f"items[{i}] must inherit from AbstractConstraintSpec."
                )

    def to_ir(
        self,
        *,
        tg: TimeGrid,
        nx: int,
        nu: int,
    ) -> GameConstraintGridMap:
        """
        Lower this frontend constraint set into the IR constraint container.
        """
        ineq_blocks: list[ConstraintBlockGridMap] = []

        for item in self.items:
            ineq_blocks.extend(
                item.to_ir_blocks(
                    tg=tg,
                    nx=nx,
                    nu=nu,
                )
            )

        return GameConstraintGridMap(
            ineq_blocks=tuple(ineq_blocks),
            eq_blocks=(),
        )


class BoundsConstraint(AbstractConstraintSpec):
    """
    Shared implementation for simple lower/upper bounds on a selected set of
    joint coordinates.

    Subclasses decide whether the constrained variable is the joint state or
    joint control vector, and whether a terminal-only block is also needed.
    """

    def __init__(
        self,
        *,
        lower=None,
        upper=None,
        indices: Sequence[int] | None = None,
        steps: Sequence[int] | None = None,
    ):
        if lower is None and upper is None:
            raise ValueError(
                "At least one of `lower` or `upper` must be provided."
            )

        self.lower = lower
        self.upper = upper
        self.indices = None if indices is None else tuple(int(i) for i in indices)
        self.steps = None if steps is None else tuple(int(k) for k in steps)

        if self.indices is not None:
            if any(i < 0 for i in self.indices):
                raise ValueError("`indices` entries must be nonnegative.")

        if self.steps is not None:
            if any(k < 0 for k in self.steps):
                raise ValueError("`steps` entries must be nonnegative.")

    @property
    @abstractmethod
    def variable_name(self) -> str:
        """Human-readable name of the joint variable being constrained."""
        raise NotImplementedError

    @abstractmethod
    def _select_vector(
        self,
        *,
        x: jnp.ndarray,
        u: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Pick the ambient joint vector this bound acts on.

        Subclasses share the generic bound-building logic in this base class
        and specialize only which vector is constrained: the joint state
        ``x`` or the joint control ``u``.
        """
        raise NotImplementedError

    @abstractmethod
    def _full_dim(
        self,
        *,
        nx: int,
        nu: int,
    ) -> int:
        """
        Return the size of the ambient joint vector being constrained.

        This lets the base class interpret omitted ``indices`` as "all
        coordinates" and validate any explicit coordinate selection.
        """
        raise NotImplementedError

    def _resolve_indices(
        self,
        *,
        nx: int,
        nu: int,
    ) -> tuple[int, ...]:
        """
        Expand the user-facing ``indices`` selection into a concrete tuple of
        joint-coordinate indices.

        If the user omitted ``indices``, this means "apply the bound to every
        coordinate of the relevant joint vector" and we materialize that
        convention here.
        """
        full_dim = self._full_dim(nx=nx, nu=nu)

        if self.indices is None:
            return tuple(range(full_dim))

        for i in self.indices:
            if i >= full_dim:
                raise ValueError(
                    f"{self.variable_name} index {i} out of range [0, {full_dim - 1}]."
                )

        return self.indices

    def _resolve_steps(
        self,
        *,
        tg: TimeGrid,
    ) -> tuple[int, ...]:
        """
        Expand the user-facing ``steps`` selection into explicit stage indices.

        Frontend bounds are stage-oriented by default, so omitting ``steps``
        means "enforce on every control interval."
        """
        if self.steps is None:
            return tuple(range(tg.nsteps))

        for k in self.steps:
            if k >= tg.nsteps:
                raise ValueError(
                    f"`steps` entry {k} out of range [0, {tg.nsteps - 1}]."
                )

        return self.steps

    def _normalize_bound_values(
        self,
        value,
        *,
        size: int,
        label: str,
    ) -> jnp.ndarray | None:
        """
        Normalize a lower/upper bound specification to a length-``size`` vector.

        This lets the public API accept either:

        - a scalar bound applied uniformly to each selected coordinate, or
        - an explicit per-coordinate vector of bounds
        """
        if value is None:
            return None

        arr = jnp.asarray(value)

        if arr.ndim == 0:
            return jnp.broadcast_to(arr, (size,))

        if arr.shape == (size,):
            return arr

        raise ValueError(
            f"`{label}` must be scalar or have shape ({size},). Got {arr.shape}."
        )

    def _make_bound_kernel(
        self,
        *,
        indices: tuple[int, ...],
        lower: jnp.ndarray | None,
        upper: jnp.ndarray | None,
    ):
        """
        Build the IR-style inequality kernel for the selected bounds.

        The returned callable always produces values in ``c(...) <= 0`` form,
        which is the convention expected by the lower-level constraint IR and
        AL solver stack.
        """
        def kernel(t, x, u):
            del t
            z = self._select_vector(x=x, u=u)
            z_sel = z[jnp.asarray(indices)]

            parts = []
            if upper is not None:
                parts.append(z_sel - upper)
            if lower is not None:
                parts.append(lower - z_sel)

            return jnp.concatenate(parts, axis=0)

        return kernel


class ControlBounds(BoundsConstraint):
    """
    Bounds on the joint control vector enforced across selected control
    intervals.
    """

    @property
    def variable_name(self) -> str:
        return "control"

    def _select_vector(
        self,
        *,
        x: jnp.ndarray,
        u: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Select the joint vector this bound acts on.

        Control bounds ignore the state and constrain entries of ``u``.
        """
        del x
        return u

    def _full_dim(
        self,
        *,
        nx: int,
        nu: int,
    ) -> int:
        """
        Return the ambient dimension of the constrained joint vector.

        For control bounds this is simply the joint control dimension ``nu``.
        """
        del nx
        return nu

    def to_ir_blocks(
        self,
        *,
        tg: TimeGrid,
        nx: int,
        nu: int,
    ) -> tuple[ConstraintBlockGridMap, ...]:
        """
        Lower control bounds into a single path-wise inequality block.

        Controls live on the stage grid, so control bounds never need a
        separate terminal block.
        """
        indices = self._resolve_indices(nx=nx, nu=nu)
        steps = self._resolve_steps(tg=tg)

        lower = self._normalize_bound_values(
            self.lower,
            size=len(indices),
            label="lower",
        )
        upper = self._normalize_bound_values(
            self.upper,
            size=len(indices),
            label="upper",
        )

        # Each selected coordinate contributes one scalar inequality for an
        # upper bound and/or one scalar inequality for a lower bound.
        cdim = len(indices) * int(lower is not None) + len(indices) * int(upper is not None)

        if cdim == 0 or len(steps) == 0:
            # Avoid emitting zero-width or inactive IR blocks.
            return ()

        block = ConstraintBlockGridMap(
            tg=tg,
            func=self._make_bound_kernel(
                indices=indices,
                lower=lower,
                upper=upper,
            ),
            cdim_out_step=cdim,
            active_steps=steps,
            iseq=False,
            terminal=False,
        )

        return (block,)


class StateBounds(BoundsConstraint):
    """
    Bounds on the joint state vector enforced across selected control
    intervals, with optional enforcement at the terminal node.
    """

    def __init__(
        self,
        *,
        lower=None,
        upper=None,
        indices: Sequence[int] | None = None,
        steps: Sequence[int] | None = None,
        include_terminal: bool = True,
    ):
        super().__init__(
            lower=lower,
            upper=upper,
            indices=indices,
            steps=steps,
        )
        self.include_terminal = include_terminal

    @property
    def variable_name(self) -> str:
        return "state"

    def _select_vector(
        self,
        *,
        x: jnp.ndarray,
        u: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Select the joint vector this bound acts on.

        State bounds ignore the control and constrain entries of ``x``.
        """
        del u
        return x

    def _full_dim(
        self,
        *,
        nx: int,
        nu: int,
    ) -> int:
        """
        Return the ambient dimension of the constrained joint vector.

        For state bounds this is the joint state dimension ``nx``.
        """
        del nu
        return nx

    def to_ir_blocks(
        self,
        *,
        tg: TimeGrid,
        nx: int,
        nu: int,
    ) -> tuple[ConstraintBlockGridMap, ...]:
        """
        Lower state bounds into path and optional terminal inequality blocks.

        The path block acts on the stage grid, while the optional terminal
        block reuses the same numeric bounds at the final state node.
        """
        indices = self._resolve_indices(nx=nx, nu=nu)
        steps = self._resolve_steps(tg=tg)

        lower = self._normalize_bound_values(
            self.lower,
            size=len(indices),
            label="lower",
        )
        upper = self._normalize_bound_values(
            self.upper,
            size=len(indices),
            label="upper",
        )

        # As with control bounds, each lower/upper side contributes one scalar
        # inequality per selected coordinate.
        cdim = len(indices) * int(lower is not None) + len(indices) * int(upper is not None)

        if cdim == 0:
            return ()

        blocks: list[ConstraintBlockGridMap] = []

        if len(steps) > 0:
            # Path-state bounds are represented as a non-terminal block active
            # on the selected control intervals.
            blocks.append(
                ConstraintBlockGridMap(
                    tg=tg,
                    func=self._make_bound_kernel(
                        indices=indices,
                        lower=lower,
                        upper=upper,
                    ),
                    cdim_out_step=cdim,
                    active_steps=steps,
                    iseq=False,
                    terminal=False,
                )
            )

        if self.include_terminal:
            # Terminal bounds use the same canonical c(x_T) <= 0 convention,
            # but the IR expects a dedicated terminal-only callable with
            # signature (t, x_terminal).
            def terminal_kernel(t, x):
                z_sel = x[jnp.asarray(indices)]

                parts = []
                if upper is not None:
                    parts.append(z_sel - upper)
                if lower is not None:
                    parts.append(lower - z_sel)

                return jnp.concatenate(parts, axis=0)

            blocks.append(
                ConstraintBlockGridMap(
                    tg=tg,
                    func=terminal_kernel,
                    cdim_out_step=cdim,
                    active_steps=None,
                    iseq=False,
                    terminal=True,
                )
            )

        return tuple(blocks)
