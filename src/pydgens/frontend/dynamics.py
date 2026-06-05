# Copyright 2026 MIT Lincoln Laboratory
# SPDX-License-Identifier: MIT

# class definitions for various control systems that define system dynamics
from __future__ import annotations

import jax.numpy as jnp

from abc import ABC, abstractmethod
from typing import Literal
from jax.scipy.linalg import expm

import pydgens.ir.timetypes as irtime
import pydgens.ir.systemtypes as irsys


def linear_dynamics(
    *,
    A: jnp.ndarray,
    B: jnp.ndarray,
) -> LTIContinuousSystem:
    """
    Create continuous-time linear dynamics for a dynamic game.

    The dynamics are

        dx/dt = A x + B u

    in the joint state and joint control coordinates.

    Parameters
    ----------
    A:
        State matrix with shape ``(nx, nx)``.

    B:
        Control matrix with shape ``(nx, nu)``.

    Returns
    -------
    LinearTimeInvariantContinuousSystem
        Continuous-time linear system with time-invariant dynamics, 
        with inferred state and control dimensions.

    Examples
    --------
    A scalar system with two players, each controlling one scalar input:

    >>> dyn = linear_dynamics(
    ...     A=jnp.array([[0.0]]),
    ...     B=jnp.array([[1.0, 1.0]]),
    ... )
    >>> dyn.nx
    1
    >>> dyn.nu
    2
    """
    return LTIContinuousSystem(
        A=jnp.asarray(A),
        B=jnp.asarray(B),
    )


def nonlinear_dynamics(
    *,
    nx: int,
    nu: int,
    dynamics,
) -> NonlinearContinuousSystem:
    """
    Create continuous-time nonlinear dynamics for a dynamic game.

    The beginner-facing nonlinear frontend expects a dynamics function of
    the form

        dx/dt = f(t, x, u)

    in the joint state and joint control coordinates.

    Parameters
    ----------
    nx:
        Joint state dimension.

    nu:
        Joint control dimension.

    dynamics:
        Callable of the form ``dynamics(t, x, u) -> dxdt``.

        The returned derivative must have shape ``(nx,)``. Time invariance
        is not assumed at the frontend layer, which keeps this interface
        aligned with common ODE-solver conventions.

    Returns
    -------
    NonlinearContinuousSystem
        Continuous-time nonlinear system with fixed joint state and joint
        control dimensions.
    """
    return NonlinearContinuousSystem(
        nx=nx,
        nu=nu,
        dynamics=dynamics,
    )


class DynamicalSystem(ABC):
    """
    Abstract base class for dynamical systems.

    A dynamical system defines the evolution of a state vector ``x`` under
    a control input ``u`` and (optionally) time ``t``.

    Subclasses implement specific continuous-time or discrete-time dynamics.
    """

    def __init__(self, **kwargs):
        super().__init__()

    @property
    @abstractmethod
    def nx(self) -> int:
        """Number of joint state dimensions."""
        raise NotImplementedError

    @property
    @abstractmethod
    def nu(self) -> int:
        """Number of joint control dimensions."""
        raise NotImplementedError

    @abstractmethod
    def evaluate(
        self,
        t: float,
        x: jnp.ndarray,
        u: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Evaluate the system dynamics.

        For continuous-time systems, this typically returns ``dx/dt``.
        For discrete-time systems, this typically returns the next state.
        """
        raise NotImplementedError


class ContinuousSystem(DynamicalSystem):
    """
    Base class for continuous-time dynamical systems.

    Continuous-time systems define dynamics of the form

        dx/dt = f(t, x, u)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class NonlinearContinuousSystem(ContinuousSystem):
    """
    Continuous-time nonlinear control system.

    Represents dynamics of the form

        dx/dt = f(t, x, u)

    where:
    - ``x`` is the joint state vector
    - ``u`` is the joint control vector
    - ``f`` is a user-supplied nonlinear vector field
    """

    def __init__(
        self,
        *,
        nx: int,
        nu: int,
        dynamics,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if not isinstance(nx, int) or nx <= 0:
            raise ValueError(
                f"`nx` must be a positive integer. Got {nx}."
            )

        if not isinstance(nu, int) or nu <= 0:
            raise ValueError(
                f"`nu` must be a positive integer. Got {nu}."
            )

        if not callable(dynamics):
            raise TypeError(
                f"`dynamics` must be callable. Got {type(dynamics)}."
            )

        self._nx = nx
        self._nu = nu
        self.dynamics = dynamics

    @property
    def nx(self) -> int:
        """Number of joint state dimensions."""
        return self._nx

    @property
    def nu(self) -> int:
        """Number of joint control dimensions."""
        return self._nu

    def evaluate(
        self,
        t: float,
        x: jnp.ndarray,
        u: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Evaluate the continuous-time dynamics.

        The time argument ``t`` is forwarded to the user-supplied
        nonlinear vector field.
        """
        return self.dynamics(t, x, u)

    def to_ir(
        self,
        *,
        tg: irtime.TimeGrid,
    ) -> irsys.SampledContinuousSystemType1:
        """
        Lower the frontend nonlinear dynamics into sampled continuous-time IR.

        Parameters
        ----------
        tg:
            Time grid used to sample the continuous-time dynamics.

        Returns
        -------
        SampledContinuousSystemType1
            Sampled continuous-time system that carries the same nonlinear
            vector field.
        """
        if not isinstance(tg, irtime.TimeGrid):
            raise TypeError(
                f"`tg` must be a TimeGrid. Got {type(tg)}."
            )

        return irsys.SampledContinuousSystemType1(
            tg=tg,
            nx=self.nx,
            nu=self.nu,
            dynamics=self.evaluate,
        )


class LinearSystem(DynamicalSystem):
    """
    Base class for linear control systems.

    Linear systems evolve according to dynamics that are linear in the
    state and control variables.

    Attributes:
        A (ArrayLike): Joint state transition matrix or matrices.
        B (ArrayLike): Joint control input matrix or matrices.
        nx (int): Number of joint state dimensions (inherited/inferred).
        nu (int): Number of joint control input dimensions (inherited/inferred).
    """

    def __init__(self, A, B, **kwargs):
        super().__init__(**kwargs)

        self._A = None
        self._B = None

        self.A = A
        self.B = B

    # -----------------------------------------------------------------
    # Generic state matrix property
    # -----------------------------------------------------------------

    @property
    def A(self):
        return self._A

    @A.setter
    def A(self, value: jnp.ndarray):

        self._validate_A(value)
        self._A = value

        # Cross-validate if B already exists
        if self._B is not None:
            self._validate_matrix_compatability(self._A, self._B)

    # -----------------------------------------------------------------
    # Generic control matrix property
    # -----------------------------------------------------------------

    @property
    def B(self):
        return self._B
    
    @B.setter
    def B(self, value: jnp.ndarray):

        self._validate_B(value)
        self._B = value

        # Cross-validate if A already exists
        if self._A is not None:
            self._validate_matrix_compatability(self._A, self._B)

    # -----------------------------------------------------------------
    # Generic validation functions
    # -----------------------------------------------------------------

    @staticmethod
    def _validate_A(A):
        pass

    @staticmethod
    def _validate_B(B):
        pass

    @staticmethod
    def _validate_matrix_compatability(A, B):
        pass


class LinearContinuousSystem(ContinuousSystem, LinearSystem):
    """
    Base class for continuous-time linear control systems.

    Continuous-time linear systems define dynamics of the form

        dx/dt = A(t) x + B(t) u

    where the system matrices may optionally vary with time.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class LTIContinuousSystem(LinearContinuousSystem):
    """
    Continuous-time linear time-invariant (LTI) control system.

    Represents dynamics of the form

        dx/dt = A x + B u

    where:
    - ``x`` is the joint state vector
    - ``u`` is the joint control vector
    - ``A`` and ``B`` are constant system matrices

    Attributes:
        A (jnp.ndarray): State transition matrices of shape (nx, nx).
        B (jnp.ndarray): Control matrices of shape (nx, nu).
        nx (int): Number of joint state dimensions (inferred).
        nu (int): Number of joint control dimensions (inferred).

    This object defines only the continuous-time dynamics themselves.
    Time discretization and sampling are handled separately by solvers
    or time-grid utilities.
    """

    def __init__(
        self,
        A: jnp.ndarray,
        B: jnp.ndarray,
        **kwargs,
    ):
        super().__init__(A=A, B=B, **kwargs)

    # -----------------------------------------------------------------
    # Subclass-specific validation helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _validate_A(A: jnp.ndarray):

        if A.ndim != 2:
            raise ValueError(
                f"`A` must be a 2D array. Got shape {A.shape}."
            )

        if A.shape[0] != A.shape[1]:
            raise ValueError(
                f"`A` must be square with shape (nx, nx). "
                f"Got shape {A.shape}."
            )

    @staticmethod
    def _validate_B(B: jnp.ndarray):

        if B.ndim != 2:
            raise ValueError(
                f"`B` must be a 2D array. Got shape {B.shape}."
            )

    @staticmethod
    def _validate_matrix_compatability(A: jnp.ndarray, B: jnp.ndarray):

        if B.shape[0] != A.shape[0]:
            raise ValueError(
                "`B` must have shape (nx, nu), where nx matches "
                f"A.shape[0]. Got A.shape={A.shape}, "
                f"B.shape={B.shape}."
            )

    # -----------------------------------------------------------------
    # Dimensions
    # -----------------------------------------------------------------

    @property
    def nx(self) -> int:
        """Number of joint state dimensions."""
        return self.A.shape[0]

    @property
    def nu(self) -> int:
        """Number of joint control dimensions."""
        return self.B.shape[1]

    def evaluate(
        self,
        t: float,
        x: jnp.ndarray,
        u: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Evaluate the continuous-time dynamics.

        The time argument ``t`` is accepted for API consistency, although
        this time-invariant system does not explicitly depend on time.
        """
        return self.A @ x + self.B @ u
    
    def discretize_to_ir(
        self,
        *,
        tg: irtime.TimeGrid,
        method: Literal["zoh", "euler"] = "zoh",
    ) -> irsys.LinearDiscreteSystemType1:
        """
        Discretize continuous-time dynamics to intermediate representation (discrete,
        time-varying) used in game solvers.

        with constant matrices ``A`` and ``B`` into a discrete-time linear system

            x[k + 1] = A[k] x[k] + B[k] u[k]

        over the supplied time grid.

        This function is intended for solvers that require explicit discrete-time
        linear dynamics matrices, such as the LQ feedback Nash solver.

        Parameters
        ----------
        tg:
            Time grid used to sample the dynamics.

        method:
            Discretization method.

            ``"zoh"`` uses exact zero-order-hold discretization, assuming the
            control input is held constant over each time step.

            ``"euler"`` uses forward Euler discretization.

        Returns
        -------
        LinearDiscreteSystemType1
            Discrete-time linear system sampled over ``tg``.

        Notes
        -----

        For ``method="zoh"``, the returned matrices are exact under the assumption
        that ``u`` is constant over each time interval.
        """
        if not isinstance(tg, irtime.TimeGrid):
            raise TypeError(f"`tg` must be a TimeGrid. Got {type(tg)}.")

        if method == "zoh":
            A_d, B_d = _discretize_lti_zoh(self.A, self.B, tg.dt)
        elif method == "euler":
            A_d, B_d = _discretize_lti_euler(self.A, self.B, tg.dt)
        else:
            raise ValueError(
                f"Unknown LTI discretization method {method!r}. "
                "Expected 'zoh' or 'euler'."
            )

        # num_transitions = tg.nt - 1

        A_seq = jnp.broadcast_to(A_d, (tg.nt-1, self.nx, self.nx))
        B_seq = jnp.broadcast_to(B_d, (tg.nt-1, self.nx, self.nu))

        return irsys.LinearDiscreteSystemType1(
            tg=tg,
            nx=self.nx,
            nu=self.nu,
            A=A_seq,
            B=B_seq,
        )


def _discretize_lti_euler(
    A: jnp.ndarray,
    B: jnp.ndarray,
    dt: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Discretize LTI dynamics with forward Euler."""
    nx = A.shape[0]

    A_d = jnp.eye(nx, dtype=A.dtype) + dt * A
    B_d = dt * B

    return A_d, B_d


def _discretize_lti_zoh(
    A: jnp.ndarray,
    B: jnp.ndarray,
    dt: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Discretize LTI dynamics exactly under zero-order hold controls."""
    nx = A.shape[0]
    nu = B.shape[1]

    dtype = jnp.result_type(A, B, dt)

    block = jnp.zeros((nx + nu, nx + nu), dtype=dtype)
    block = block.at[:nx, :nx].set(A)
    block = block.at[:nx, nx:].set(B)

    block_exp = expm(block * dt)

    A_d = block_exp[:nx, :nx]
    B_d = block_exp[:nx, nx:]

    return A_d, B_d
