# Copyright 2026 MIT Lincoln Laboratory
# SPDX-License-Identifier: MIT

# Class definitions for various system trajectory types

import jax.numpy as jnp

from flax import struct
from functools import singledispatch

from pydgens.ir.timetypes import TimeGrid

@struct.dataclass
class FixedStepSystemTrajectory:
    """
    Trajectory of state and control vectors at a fixed time step dt

    States live on time-grid nodes and controls live on the intervals
    between them.

    Therefore:

    - ``xs`` has length ``nt`` because it includes the initial and terminal
      states
    - ``us`` has length ``nsteps = nt - 1`` because each control acts over
      one interval and propagates the system to the next state

    Special case: when ``nt == 1`` there are zero control intervals, so the
    control trajectory must still be provided as a 2D array with shape
    ``(0, nu)``. This keeps the control dimension ``nu`` explicit and avoids
    ambiguity from a rank-1 empty array like ``[]``.

    Properties:
    - tg : TimeGrid 
        time characteristics of trajectory (nt, dt, t0)
    - xs : jnp.ndarray of shape (nt,nx)
        The sequence of system's joint states vector at each time t
        including the terminal state xs[-1]=xs[nt-1]
    - us : jnp.ndarray of shape (nt-1,nu)
        The sequence of system's joint control vector at each time t
        exluding a control at the terminal state
    """
    tg: TimeGrid
    xs: jnp.ndarray
    us: jnp.ndarray

    # input checking
    def __post_init__(self):

        if not isinstance(self.tg, TimeGrid):
            raise TypeError(f"tg must be TimeGrid, got type {type(self.tg)}")
        
        if self.xs.ndim != 2:
            raise ValueError(f"xs must be 2-dimensional (nt, nx), got {self.xs.ndim}")
        if self.xs.shape[0] != self.tg.nt:
            raise ValueError(f"Inconsistent number of state steps in trajectory. Expected {self.tg.nt}, got {self.xs.shape[0]}")
        
        if self.us.ndim != 2:
            raise ValueError(
                "us must be 2-dimensional (nsteps, nu). "
                f"Got ndim={self.us.ndim}. For nt=1, pass an empty control "
                "trajectory with shape (0, nu), not a rank-1 empty array."
            )
        if self.us.shape[0] != self.nsteps:
            raise ValueError(f"Inconsistent number of control steps in trajectory. Expected {self.tg.nsteps}, got {self.us.shape[0]}")
        
    @property
    def nt(self):
        # number of time nodes
        return self.tg.nt
    
    @property
    def nsteps(self):
        # number of time steps between time nodes
        return self.tg.nsteps
    
    @property
    def dt(self):
        return self.tg.dt
    
    @property
    def t0(self):
        return self.tg.t0

    @property
    def nx(self):
        return self.xs.shape[1]

    @property
    def nu(self):
        # Even when nt == 1 and nsteps == 0, validation requires `us` to
        # remain rank-2 with shape (0, nu), so the control dimension stays
        # explicit and can be read from axis 1.
        return self.us.shape[1]


@struct.dataclass
class FixedStepPrimalDualTrajectory:
    """
    Trajectory of state, control, and Lagrange multiplier vectors on a fixed grid.

    This follows the node/stage convention used throughout the solver stack:
    - ``xs`` is node-indexed with length ``nt`` because it includes the initial
      and terminal states.
    - ``us`` is stage-indexed with length ``nsteps = nt - 1`` because each
      control acts over one interval between consecutive state nodes.
    - ``ls`` is also stage-indexed with length ``nsteps`` because each dynamics
      multiplier corresponds to one discrete dynamics residual
      ``x[k+1] - f_d(x[k], u[k])``.

    Note that these dynamics multipliers correspond to ``mu`` in
    Le Cleac'h et al ALGames. The do NOT include the augmented
    lagrangian multipliers, lambda and rho, which are considered
    separately from-, not intrinsically part of the primal dual 
    trajectory even though they are related. This separate treatment
    comes from how they are used in the AL game solver where the 
    augmented lagrange multipliers are updated in an outer loop
    (Alg 3, Le Cleac'h et al) where the dynamics multipliers 
    are updated in the newton root finding (Alg 2, Le Cleac'h),
    and the fact that the augmented multipliers have apply 
    across players and have distinctly different vector shapes
    based on the total number of auxilliary constraints to enforce
    rather than on the number of players and time steps

    Properties:
    - tg : TimeGrid 
        time characteristics of trajectory (nt, dt, t0)
    - xs : jnp.ndarray of shape (nt,nx)
        The sequence of system's joint states vector at each time t.
    - us : jnp.ndarray of shape (nsteps,nu)
        The sequence of system's joint control vectors, one per control interval.
    - ls : jnp.ndarray of shape (nsteps,N,nx)
        The sequence of Lagrange multiplier vectors (for dynamics), one per
        player and control interval.
        the N players
    """
    tg: TimeGrid
    xs: jnp.ndarray
    us: jnp.ndarray
    ls: jnp.ndarray

    # input checking
    def __post_init__(self):

        if not isinstance(self.tg, TimeGrid):
            raise TypeError(f"tg must be TimeGrid, got type {type(self.tg)}")
        
        if self.xs.ndim != 2:
            raise ValueError(f"xs must be 2-dimensional (nt, nx), got {self.xs.ndim}")
        if self.xs.shape[0] != self.tg.nt:
            raise ValueError(f"Inconsistent number of state steps in trajectory. Expected {self.tg.nt}, got {self.xs.shape[0]}")
        
        if self.us.ndim != 2:
            raise ValueError(f"us must be 2-dimensional (nsteps, nu), got {self.us.ndim}")
        if self.us.shape[0] != self.tg.nsteps:
            raise ValueError(f"Inconsistent number of control steps in trajectory. Expected {self.tg.nsteps}, got {self.us.shape[0]}")
        
        if self.ls.ndim != 3:
            raise ValueError(f"ls must be 3-dimensional (nsteps, N, nx), got {self.ls.ndim}")
        if self.ls.shape[0] != self.tg.nsteps:
            raise ValueError(f"Inconsistent number of Lagrange multipliers in trajectory. Expected {self.tg.nsteps}, got {self.ls.shape[0]}")
        if self.ls.shape[2] != self.xs.shape[1]:
            raise ValueError(f"Inconsistent Lagrange multiplier shape. Expected {self.xs.shape[1]}, got {self.ls.shape[2]}")
        
    @property
    def nt(self):
        # number of time nodes
        return self.tg.nt
    
    @property
    def nsteps(self):
        # number of control intervals between time nodes
        return self.tg.nsteps
    
    @property
    def dt(self):
        # time spacing between adjacent nodes
        return self.tg.dt
    
    @property
    def t0(self):
        # initial time
        return self.tg.t0

    @property
    def nx(self):
        # dimension of joint state space
        return self.xs.shape[1]

    @property
    def nu(self):
        # dimension of joint control space
        return self.us.shape[1]
    
    @property
    def N(self):
        # number of players
        return self.ls.shape[1]
    
def make_system_trajectory(tg, xs, us) -> FixedStepSystemTrajectory:
    """
    Factory function that coerces inputs into jnp.ndarrays and
    enforces consistency rules before creating a SystemTrajectory.
    
    Parameters
    ----------
    tg : TimeGrid 
        time characteristics of trajectory (nt, dt, t0)
    xs : array-like
        State trajectory (shape (nt, nx)).
    us : array-like
        Control trajectory (shape (nsteps, nu)).

        If ``tg.nt == 1``, then ``us`` must still be a 2D empty array with
        shape ``(0, nu)`` so that the control dimension remains explicit.
    """
    xs = jnp.asarray(xs)
    us = jnp.asarray(us)

    if tg.nsteps == 0 and us.ndim == 1 and us.shape[0] == 0:
        raise ValueError(
            "For nt=1, `us` must be an empty 2D array with shape (0, nu). "
            "A rank-1 empty array does not preserve the control dimension."
        )

    return FixedStepSystemTrajectory(tg=tg, xs=xs, us=us)

@singledispatch
def are_xs_close(
    traj1, 
    traj2: FixedStepSystemTrajectory, 
    max_elwise_diff: float
) -> bool:
    """Check if two trajectories are elementwise-close in `xs` under inf-norm."""
    raise NotImplementedError

@are_xs_close.register(FixedStepSystemTrajectory)
def _are_xs_close(
    traj1: FixedStepSystemTrajectory, 
    traj2: FixedStepSystemTrajectory, 
    max_elwise_diff: float
) -> bool:
    """Check if two trajectories are elementwise-close in `xs` under inf-norm."""
    if traj1.xs.shape != traj2.xs.shape:
        raise ValueError(f"State trajectories xs shapes do not match. got traj1.xs.shape={traj1.xs.shape}, traj2.xs.shape={traj2.xs.shape}")
    
    if max_elwise_diff < 0.0:
        raise ValueError(f"max_elwise_diff must be non-negative, got {max_elwise_diff}")
    
    if traj1.tg != traj2.tg:
        raise ValueError(f"Inconsistent time characteristics, got traj1.tg={traj1.tg}, traj2.tg={traj2.tg}")

    # Check all time steps
    diffs = jnp.max(jnp.abs(traj1.xs - traj2.xs), axis=1)  # (T,)
    return jnp.all(diffs < max_elwise_diff)

@singledispatch
def get_player_control_trajectory(
    traj,
    player_i: int,
    u_splits: jnp.ndarray,
) -> jnp.ndarray:
    """parse player-i's control trajectory from joint control trajectory
    
    Parameters:
    - traj : Trajectory Type
        joint trajectory from which player control trajectory is parsed
    - player_i : int
        player index i for parsing player i's controls from joint control trajectory (0-indexed)
    - u_splits: jnp.ndarray of length N. 
        Lengths of each u_j block defining each player's portion of the joint control vector
        therefore u_splits[player_i] = nu_i

    Returns:
    - us_i : jnp.ndarray
        player-i's control trajectory. Shape depends on type of trajectory
        e.g. FixedStepSystemTrajectory will have shape (nt, nu_i)
        e.g. FixedStepPrimalDualTrajectory will have shape (nt-1, nu_i)
    """
    raise NotImplementedError

@get_player_control_trajectory.register(FixedStepSystemTrajectory)
def _get_player_control_trajectory(
    traj: FixedStepSystemTrajectory,
    player_i: int,
    u_splits: jnp.ndarray
):
    return _get_us_i(traj=traj, player_i=player_i, u_splits=u_splits)

@get_player_control_trajectory.register(FixedStepPrimalDualTrajectory)
def _get_player_control_trajectory(
    traj: FixedStepPrimalDualTrajectory,
    player_i: int,
    u_splits: jnp.ndarray
):
    return _get_us_i(traj=traj, player_i=player_i, u_splits=u_splits)

def _get_us_i(
    traj,
    player_i: int,
    u_splits: jnp.ndarray
):
    
    # rename for simplicity
    i = player_i

    # input checking
    assert u_splits.ndim == 1
    N = len(u_splits)
    assert i < N

    cum = jnp.cumsum(u_splits)          # (N,)
    end = cum[i]                        # end index for player i (exclusive)
    start = 0 if i == 0 else cum[i-1]   # start index for player i (inclusive)
    return traj.us[:, start:end]
    
