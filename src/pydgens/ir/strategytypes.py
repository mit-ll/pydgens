# Copyright 2026 MIT Lincoln Laboratory
# SPDX-License-Identifier: MIT

# Class definitions for various strategy types
import jax.numpy as jnp
from flax import struct

from pydgens.ir.timetypes import TimeGrid

@struct.dataclass 
class FixedStepAffineStrategies:
    """
    Represents an affine control strategy of the form u(x) = -P @ x - alpha.

    Attributes:
    - tg : TimeGrid 
        time characteristics of trajectory (nt, dt, t0)
    - P : jnp.ndarray of shape (nt-1,nu,nx) 
        feedback Nash equilibrium linear term
        P[t] is a size (nu,nx) matrix describing the linear term of the joint affine policy
        at discrete time t such that
        u[t] = -P_t @ x[t] - a_t
    - alpha : jnp.ndarray of shape (nt-1,nu) 
        feedback Nash equilibrium bias term
        alpha[t] is a size (nu,) vector describing the bias term of the joint affine policy 
        at discrete time t such that
        u[t] = -P_t @ x[t] - a_t
    """
    tg: TimeGrid
    P: jnp.ndarray
    alpha: jnp.ndarray

    def __post_init__(self):
        _validate_affine_strategy_inputs(tg=self.tg, P=self.P, alpha=self.alpha)

    @property
    def nt(self) -> int:
        # number of time nodes
        return self.tg.nt
    
    @property
    def nsteps(self):
        # number of time steps between time nodes
        return self.tg.nsteps

    @property 
    def nu(self) -> int:
        """Joint control dimension"""
        return self.P.shape[1]

    @property
    def nx(self) -> int:
        """Joint state dimension"""
        return self.P.shape[2]

def _validate_affine_strategy_inputs(tg: TimeGrid, P: jnp.ndarray, alpha: jnp.ndarray) -> None:
    """Validate inputs for AffineStrategy construction.
    
    Args:
        tg: TimeGrid time characteristics of trajectory (nt, dt, t0)
        P: Linear feedback term, shape (nsteps, nu, nx)
        alpha: Bias term, shape (nsteps, nu)
        
    Raises:
        ValueError: If dimensions are incompatible
    """
    if P.ndim != 3:
        raise ValueError(f"P must be 3D, got shape {P.shape}")
    if alpha.ndim != 2:
        raise ValueError(f"alpha must be 2D, got shape {alpha.shape}")
    if P.shape[0] != tg.nsteps:
        raise ValueError(f"nstep dimension mismatch in P: expected {tg.nsteps}, got P.shape[0]={P.shape[0]}")
    if alpha.shape[0] != tg.nsteps:
        raise ValueError(f"nstep dimension mismatch in alpha: expected {tg.nsteps}, got alpha.shape[0]={alpha.shape[0]}")
    if P.shape[1] != alpha.shape[1]:
        raise ValueError(f"nu dimension mismatch: P has {P.shape[1]}, alpha has {alpha.shape[1]}")

def update_affine_strategy(strategy: FixedStepAffineStrategies, P: jnp.ndarray = None, alpha: jnp.ndarray = None) -> FixedStepAffineStrategies:
    """Update an AffineStrategy with validation.

    Note that this is largely just a wrapper for dataclass.replace to add validation
    
    Args:
        strategy: Existing strategy
        P: New P matrix, shape (nsteps, nu, nx) (optional)
        alpha: New alpha vector, shape (nsteps, nu) (optional)
        
    Returns:
        New validated AffineStrategy instance
        
    Raises:
        ValueError: If dimensions are incompatible
    """
    new_P = P if P is not None else strategy.P
    new_alpha = alpha if alpha is not None else strategy.alpha
    
    # Manual validation since replace() doesn't call __post_init__
    _validate_affine_strategy_inputs(strategy.tg, new_P, new_alpha)
    return strategy.replace(P=new_P, alpha=new_alpha)
