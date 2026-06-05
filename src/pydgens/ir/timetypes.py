# Copyright 2026 MIT Lincoln Laboratory
# SPDX-License-Identifier: MIT

from __future__ import annotations

import jax.numpy as jnp
import flax.struct

# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------

### PUBLIC API CLASSES AND FUNCTIONS


def time_grid(
    *,
    nt: int,
    dt: float,
    t0: float = 0.0,
) -> TimeGrid:
    """
    Create a uniform time grid for a dynamic game.

    A ``TimeGrid`` defines the discrete times at which the game dynamics,
    controls, strategies, costs, and constraints are evaluated.

    The grid contains ``nt`` time points:

        t0, t0 + dt, t0 + 2 dt, ..., t0 + (nt - 1) dt

    This corresponds to nt-1 steps in the grid. i.e. 
    ``nsteps`` = nt-1

    Parameters
    ----------
    nt:
        Number of discrete time points in the grid. Must be greater than zero.

        If ``nt`` states are stored along a trajectory, then there are typically
        ``nt - 1`` control intervals between them.

    dt:
        Time step between consecutive grid points. Must be greater than zero.

    t0:
        Initial time. Defaults to ``0.0``.

    Returns
    -------
    TimeGrid
        Validated time-grid object used by control systems and games.

    Examples
    --------
    >>> tg = time_grid(nt=51, dt=0.1)
    >>> tg.nt
    51
    >>> tg.dt
    0.1

    For a one-stage game with an initial and final state:

    >>> tg = time_grid(nt=2, dt=1.0)
    """
    return TimeGrid(nt=nt, dt=dt, t0=t0)


# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------

### INTERNALS - CLASSES AND FUNCTIONS

@flax.struct.dataclass
class TimeGrid:
    """
    Represents a discrete time domain specification for dynamical systems.

    Attributes
    ----------
    nt : int
        Number of discrete grid points, i.e. nodes, in time grid. Must be > 0.
    dt : float
        Time step size between consecutive points. Must be > 0.
    t0 : float
        Initial time (start of the grid). Defaults to `0.0`
    nsteps : int, inferred
        Number of time steps between time nodes
    
    Notes
    -----
    - The implied time grid is [t0, t0 + dt, t0 + 2*dt, ..., t0 + (nt-1)*dt].
    - Objects that compose a `Time` instance are guaranteed to share
      consistent temporal discretization if they reference the same
      `Time` object in memory.
    """

    nt: int
    dt: float
    t0: float = 0.0

    def __post_init__(self):
        if not isinstance(self.nt, int):
            raise TypeError(f"nt must be an int, got {type(self.nt)}")
        if self.nt <= 0:
            raise ValueError(f"nt must be > 0, got {self.nt}")
        if not isinstance(self.dt, (float, int)):  # allow ints, cast if needed
            raise TypeError(f"dt must be a float, got {type(self.dt)}")
        if float(self.dt) <= 0.0:
            raise ValueError(f"dt must be > 0, got {self.dt}")
        if not isinstance(self.t0, (float, int)):
            raise TypeError(f"t0 must be a float, got {type(self.t0)}")
        
    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, TimeGrid):
            return False
        if self.nt != other.nt:
            return False
        # Use jnp.isclose for floating point comparisons
        if not jnp.isclose(self.dt, other.dt, rtol=1e-9, atol=0.0):
            return False
        if not jnp.isclose(self.t0, other.t0, rtol=1e-9, atol=0.0):
            return False
        return True
    
    @property
    def nsteps(self):
        # number of time steps between time nodes
        return self.nt-1
    
def compute_ts(tg: TimeGrid) -> jnp.ndarray:
    """
    Compute the array of time points for a given TimeGrid.

    Args:
        tg (TimeGrid): A TimeGrid object with fields (nt, dt, t0)

    Returns:
        jnp.ndarray: Array of shape (nt,) containing evenly spaced time points:
            [t0, t0 + dt, t0 + 2*dt, ..., t0 + (nt-1)*dt].
    """
    return jnp.linspace(tg.t0, tg.t0 + (tg.nt - 1) * tg.dt, tg.nt)

def disc2cont(k, tg: TimeGrid) -> float:
    """
    Convert discrete index to continuous time value.

    Args:
        k (int or array): Discrete index (0 <= k < nt).
        tg (TimeGrid): Time grid.

    Returns:
        float: Continuous time (t0 + k*dt)
    """
    # k = jnp.asarray(k)

    # def in_bounds(k_):
    #     return tg.t0 + k_ * tg.dt

    # def out_of_bounds(_):
    #     return jnp.nan

    # return jax.lax.cond(
    #     (k < 0) | (k >= tg.nt),
    #     out_of_bounds,
    #     in_bounds,
    #     k,
    # )
    return tg.t0 + k * tg.dt

def cont2disc(t: float, tg: TimeGrid) -> int:
    """
    JAX-compatible Convert continuous time t into a discrete time step index k.

    Returns the index of the "bucket" such that:
        ts[k] <= t < ts[k+1], where ts[k] = t0 + k*dt
    
    Note that, to maintain jax-compatibility with tracing by avoiding
    conditional error statements that require explicit knowledge of k
    rather than a tracer, k is clipped to the bounds [0, nt-1]

    Args:
        t (float): Continuous time value
        tg (Time): Time grid object

    Returns:
        int: Discrete index k
    """
    # small epsilon to be applied to avoid numerical drift that can produce wrong answers at grid bounds
    eps = 1e-12 * tg.dt

    k = jnp.floor((t - tg.t0) / tg.dt + eps).astype(int)

    return jnp.clip(k, 0, tg.nt - 1)