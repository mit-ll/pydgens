# Copyright 2026 MIT Lincoln Laboratory
# SPDX-License-Identifier: MIT

# Class definitions for various game types
import jax.numpy as jnp
import numpy as np

from dataclasses import dataclass
from typing import List
from functools import singledispatch

from pydgens.ir.systemtypes import (
    SampledContinuousSystemType1, 
    LinearDiscreteSystemType1,
    approx_linear_discrete_system
)

from pydgens.ir.trajectorytypes import (
    FixedStepSystemTrajectory
)

from pydgens.ir.costtypes import (
    PlayerCostSpecContinuous,
    quadraticize_cost_joint_ctrl_playerwise_trajectory,
)
from pydgens.ir.costtypes import ControlDomain as CostControlDomain
from pydgens.ir.costtypes import ControlStructure as CostControlStructure

from pydgens.ir.constrainttypes import GameConstraintGridMap


@dataclass(frozen=True)
class LinearQuadraticGameType1:
    """
    Represents a discrete-time, finite-horizon, linear-quadratic dynamic game with no state or control bounds.

    Attributes:
    - cs (LinearDiscreteSystemType1): control system defining dynamics, sampling time, and state and control dimesnions

    - N (int): Number of players.

    - Q : jnp.ndarray of size (nsteps,N,nx,nx)
        Q[k][i] is a size (nx,nx) symmetric matrix describing the quadratic term
        of player-i's running cost function with respect to joint state x
        (size (nx,)) on control interval k such that
        
        For symmetry constraint/assumption, see Basar and Olsder, 2nd Ed, Def 6.1 (pg 269). 

        Note that, I'm not sure if strick symmetry is required since Q is only evaluated
        in a quadratic term: x^T @ Q @ x, and the skew-symmetric component Q_n of a non-symmetric
        matrix Q (which can be decomposed into symetric and skew-symmetric components Q= Q_s + Q_n)
        has no effect on the quadratic term since x^T @ Q @ x = 1/2 * x^T @ (Q+Q^T) @ x = x^T @ Q_s @ x

        However, since Basar and Olsder explicitly assume symmetric Q, symmetry is enforce in this definition

    - q : jnp.ndarray of size (nsteps,N,nx)
        q[k][i] is a size (nx,) vector describing the linear term of player-i's
        running cost function with respect to joint state x (size (nx,))
        on control interval k

    - Qf : jnp.ndarray of size (N,nx,nx)
        Qf[i] is the quadratic term of player-i's terminal state cost evaluated
        at the terminal node x_K where K = nsteps.

    - qf : jnp.ndarray of size (N,nx)
        qf[i] is the linear term of player-i's terminal state cost evaluated at
        the terminal node x_K where K = nsteps.

    - R : jnp.ndarray of size (nsteps,N,nu,nu)
        R[k][i] is a size (nu,nu) block-diagnonal matrix describing the quadratic
        term of player-i's running cost function with respect to joint control
        vector u (size (m,)) on control interval k.

        Positive definite sub-block for player i: This algorithms assume that player i's cost MUST depend
        upon all of player i's control, thus a positive definite sub-block on the diagonal of R[t,i] of 
        size (nu_i,nu_i) at player i's sub block location.
        See Basar and Olsder, 2nd Ed, Def 6.1, pg 269
        
        Furthermore, player i's cost MAY depend upon player j's control (although this may be unlikely), 
        and thus you may have non-zero block-diagonal components of R[t,i] for diagonal sub blocks other than 
        player i.

        HOWEVER, the algorithm assumes that player i's cost CANNOT DEPEND upon cross terms between
        different player's control inputs; i.e. player-i cost cannot depend upon player-j and player-k's 
        cross terms. Thus the off diagonal blocks of R[t,i] must be zero for all t and i.
        This assumption is not made explicit in Basar and Olsder but rather implied by the construction 
        of R matrices in those sources. Basar and Olsder both define the cost function for player-i as 
        a sum over contributions from player-specific subvectors of the control space (allowing, 
        in general, for player-j's control to contribute to player-i's cost). That is, 
        Basar and Olsder and Fridovich-Keil define a set of matrices R_ij, each of which being 
        dimension (m_j, m_j). In contrast, I define a single consolidated matrix R_i of size (m,m) for 
        each player-i (at each time step t). However, mapping the sub-block matrices R_ij into R_i, only 
        block-diagonal subblocks could be nonzero. There is no meaning for off-diagonal sub-blocks based 
        on Basar and Olsder's construction.

    - r : jnp.ndarray of size (nsteps,N,nu)
        r[k][i] is a size (nu,) vector describing the linear term of player-i's
        running cost function with respect to joint control vector u (size (nu,))
        on control interval k
        
    - u_splits: list[int] of length N. 
        Lengths of each u_j block defining each player's portion of the joint control vector
    """
    cs: LinearDiscreteSystemType1
    N: int
    Q: jnp.ndarray
    q: jnp.ndarray
    R: jnp.ndarray
    r: jnp.ndarray
    u_splits: jnp.ndarray
    Qf: jnp.ndarray | None = None
    qf: jnp.ndarray | None = None

    # input checking
    def __post_init__(self):

        # type-check control system is a discrete linear system, as opposed to continuous linear system
        # for which the LQ solution is not compatible
        if not isinstance(self.cs, LinearDiscreteSystemType1):
            raise TypeError(f"Control System must be of type LinearDiscreteSystemType1. Got {type(self.cs)}")
        
        # Cost matrice shape checking
        if self.Q.shape != (self.cs.nsteps, self.N, self.cs.nx, self.cs.nx):
            raise ValueError(f"Q must have shape ({self.cs.nsteps}, {self.N}, {self.cs.nx}, {self.cs.nx}), got {self.Q.shape}")
        if self.q.shape != (self.cs.nsteps, self.N, self.cs.nx):
            raise ValueError(f"q must have shape ({self.cs.nsteps}, {self.N}, {self.cs.nx}), got {self.q.shape}")
        if self.R.shape != (self.cs.nsteps, self.N, self.cs.nu, self.cs.nu):
            raise ValueError(f"R must have shape ({self.cs.nsteps}, {self.N}, {self.cs.nu}, {self.cs.nu}), got {self.R.shape}")
        if self.r.shape != (self.cs.nsteps, self.N, self.cs.nu):
            raise ValueError(f"r must have shape ({self.cs.nsteps}, {self.N}, {self.cs.nu}), got {self.r.shape}")

        if self.Qf is None:
            object.__setattr__(
                self,
                "Qf",
                jnp.zeros((self.N, self.cs.nx, self.cs.nx), dtype=self.Q.dtype),
            )
        if self.qf is None:
            object.__setattr__(
                self,
                "qf",
                jnp.zeros((self.N, self.cs.nx), dtype=self.q.dtype),
            )

        if self.Qf.shape != (self.N, self.cs.nx, self.cs.nx):
            raise ValueError(f"Qf must have shape ({self.N}, {self.cs.nx}, {self.cs.nx}), got {self.Qf.shape}")
        if self.qf.shape != (self.N, self.cs.nx):
            raise ValueError(f"qf must have shape ({self.N}, {self.cs.nx}), got {self.qf.shape}")
        
        # check that u_splits is appropriate shape and dtype
        if self.u_splits.ndim != 1 or self.u_splits.shape[0] != self.N:
            raise ValueError(f"u_splits must be shape ({self.N},)")
        if jnp.sum(self.u_splits) != self.cs.nu:
            raise ValueError(f"u_splits must sum to {self.cs.nu}")
        if not np.issubdtype(self.u_splits.dtype, np.integer):
            raise TypeError(f"u_splits must be an integer array, got dtype {self.u_splits.dtype}")
        
    # convience properties to raise time characteristics to top-level

    @property
    def tg(self):
        # TimeGrid time characteristics
        return self.cs.tg
    
    @property
    def nt(self):
        # number of time nodes
        return self.cs.tg.nt

    @property
    def nsteps(self):
        # number of control intervals / stages
        return self.tg.nsteps
    
    @property
    def dt(self):
        # length of time step, [s] by default
        return self.cs.tg.dt
    
    @property
    def t0(self):
        # initial time, [s] by default
        return self.cs.tg.t0
    
    @property
    def nx(self):
        # number of joint-state dimensions
        return self.cs.nx
    
    @property
    def nu(self):
        # number of joint-control dimenstions
        return self.cs.nu
    
    @property
    def A(self):
        # Linear dynamics state transition matrices of shape (nsteps, nx, nx).
        return self.cs.A
    
    @property
    def B(self):
        # Linear dynamics control transition matrices of shape (nsteps, nx, nu)
        return self.cs.B

@dataclass(frozen=True)
class NonlinearGameType1:
    """
    Represents a time-sampled, finite-horizon, nonlinear dynamic game with no state or control bounds

    Attributes:
    - cs (SampledContinuousSystem): control system defining dynamics, sampling time, and state and control dimesnions
    - N (int): Number of players.
    - costs : List[PlayerCostSpecContinuous]
        list of cost specifications for each player in form function(t, x, u) -> scalar
        cost specs must use the JOINT control domain. For compatibility with
        the LQ approximation used by iLQ, running costs must not have
        ``GENERAL`` control structure. Joint-control costs that are
        ``LOCAL_ONLY`` or ``BLOCK_SEPARABLE`` are both compatible.
    - u_splits: jnp.ndarray of length N. 
        Lengths of each u_j block defining each player's portion of the joint control vector

    Notes:
    - Lightweight checks are applied to players' declared running-cost
      structure: costs must use the JOINT control domain and must not be
      explicitly labeled ``GENERAL``. This is a declaration-based check,
      not a full symbolic proof about the callable.
    """
    cs: SampledContinuousSystemType1 
    N: int
    costs: List[PlayerCostSpecContinuous]
    u_splits: jnp.ndarray

    # input checking
    def __post_init__(self):

        # type-check control system is a sampled continuous system, since ilq solver is only designed
        # for such a system
        if not isinstance(self.cs, SampledContinuousSystemType1):
            raise TypeError(f"Control System must be of type SampledContinuousSystemType1. Got {type(self.cs)}")
        
        # check that costs list is appropriate length, i.e. one entry for each player
        if len(self.costs) != self.N:
            raise ValueError(f"costs must have length N = {self.N}, got {len(self.costs)}")
        
        # check that costs is a list PlayerCostSpecsContinuous, with joint-
        # control running costs and no declared GENERAL mixed-control block
        # structure
        for i in range(self.N):
            if not isinstance(self.costs[i], PlayerCostSpecContinuous):
                raise TypeError(f"costs must be PlayerCostSpecContinuous. Got type {type(self.costs[i])} for player {i}")
            if self.costs[i].terminal is not None:
                raise ValueError(f"terminal costs not supported. Got terminal cost for player {i}")
            if self.costs[i].control_domain is not CostControlDomain.JOINT:
                raise ValueError(f"cost functions take joint control vectors. Got non-joint control domain for player {i}")
            if self.costs[i].control_structure is CostControlStructure.GENERAL:
                raise ValueError(
                    "cost functions must not have GENERAL control structure "
                    "for iLQ compatibility. Got general control structure "
                    f"for player {i}"
                )

        
        # check that u_splits is appropriate shape and dtype
        if self.u_splits.ndim != 1 or self.u_splits.shape[0] != self.N:
            raise ValueError(f"u_splits must be shape ({self.N},)")
        if jnp.sum(self.u_splits) != self.cs.nu:
            raise ValueError(f"u_splits must sum to {self.cs.nu}")
        if not np.issubdtype(self.u_splits.dtype, np.integer):
            raise TypeError(f"u_splits must be an integer array, got dtype {self.u_splits.dtype}")
        
    @property
    def tg(self):
        # TimeGrid time characteristics
        return self.cs.tg
    
    @property
    def nt(self):
        # number of time nodes
        return self.cs.tg.nt
    
    @property
    def nsteps(self):
        # number of control intervals / stages
        return self.tg.nsteps
    
    @property
    def dt(self):
        # length of time step, [s] by default
        return self.cs.tg.dt
    
    @property
    def t0(self):
        # initial time, [s] by default
        return self.cs.tg.t0
    
    @property
    def nx(self):
        # number of joint-state dimensions
        return self.cs.nx
    
    @property
    def nu(self):
        # number of joint-control dimenstions
        return self.cs.nu
    
    @property
    def dynamics(self):
        # callable game joint game dynamics
        return self.cs.dynamics
    
@dataclass(frozen=True)
class NonlinearGameType2:
    """
    Represents a time-sampled, finite-horizon, nonlinear dynamic game WITH constraints
    (e.g. state or control bounds), that can be used with an augmented lagrangian solver

    Attributes:
    - cs (SampledContinuousSystem): control system defining dynamics, sampling time, and state and control dimesnions
    - N (int): Number of players.
    - costs : List[PlayerCostSpecContinuous]
        list of cost specifications for each player in form:
            running_cost(t, x, u_i) -> scalar
            terminal_cost(t, x) -> scalar
        where running cost specs must use the LOCAL control domain. This is
        stronger than the iLQ requirement: the AL formulation assumes each
        player's running cost depends only on that player's own local
        control, i.e. ``LOCAL_ONLY`` control structure.
    - constraints : GameConstraintGridMap
        state and control constraint functions shared across players
    - u_splits: jnp.ndarray of length N. 
        Lengths of each u_j block defining each player's portion of the joint control vector

    Notes:
    - Lightweight checks are applied to players' declared running-cost
      structure: costs must use the LOCAL control domain and must be
      explicitly labeled ``LOCAL_ONLY``. This is a declaration-based check,
      not a full symbolic proof about the callable.
    """
    cs: SampledContinuousSystemType1 
    N: int
    costs: List[PlayerCostSpecContinuous]
    constraints : GameConstraintGridMap
    u_splits: jnp.ndarray

    # input checking
    def __post_init__(self):

        # type-check control system is a sampled continuous system, since aug lagrange solver is only designed
        # for such a system
        if not isinstance(self.cs, SampledContinuousSystemType1):
            raise TypeError(f"Control System must be of type SampledContinuousSystemType1. Got {type(self.cs)}")
        
        # check that costs list is appropriate length, i.e. one entry for each player
        if len(self.costs) != self.N:
            raise ValueError(f"costs must have length N = {self.N}, got {len(self.costs)}")
            
        # check that costs is a list PlayerCostSpecsContinuous, with local-
        # control running costs and declared LOCAL_ONLY structure
        for i in range(self.N):
            if not isinstance(self.costs[i], PlayerCostSpecContinuous):
                raise TypeError(f"costs must be PlayerCostSpecContinuous. Got type {type(self.costs[i])} for player {i}")
            if self.costs[i].terminal is None:
                raise ValueError(f"terminal costs functions must be defined. Got None for player {i}")
            if self.costs[i].control_domain is not CostControlDomain.LOCAL:
                raise ValueError(f"cost functions take local control vectors. Got non-local control domain for player {i}")
            if self.costs[i].control_structure is not CostControlStructure.LOCAL_ONLY:
                raise ValueError(
                    "cost functions must be explicitly declared LOCAL_ONLY "
                    "for the AL formulation. Got non-local-only or unknown "
                    f"control structure for player {i}"
                )
            
        # check that constraints are of appropriate type
        if not isinstance(self.constraints, GameConstraintGridMap):
            raise TypeError(f"constraints must be of type GameConstraintGridMap. Got {type(self.constraints)}")
        
        # check that u_splits is appropriate shape and dtype
        if self.u_splits.ndim != 1 or self.u_splits.shape[0] != self.N:
            raise ValueError(f"u_splits must be shape ({self.N},)")
        if jnp.sum(self.u_splits) != self.cs.nu:
            raise ValueError(f"u_splits must sum to {self.cs.nu}")
        if not np.issubdtype(self.u_splits.dtype, np.integer):
            raise TypeError(f"u_splits must be an integer array, got dtype {self.u_splits.dtype}")
        
    @property
    def tg(self):
        # TimeGrid time characteristics
        return self.cs.tg
    
    @property
    def nt(self):
        # number of time nodes
        return self.cs.tg.nt
    
    @property
    def nsteps(self):
        # number of control intervals / stages
        return self.tg.nsteps
    
    @property
    def dt(self):
        # length of time step, [s] by default
        return self.cs.tg.dt
    
    @property
    def t0(self):
        # initial time, [s] by default
        return self.cs.tg.t0
    
    @property
    def nx(self):
        # number of joint-state dimensions
        return self.cs.nx
    
    @property
    def nu(self):
        # number of joint-control dimenstions
        return self.cs.nu
    
    @property
    def dynamics(self):
        # callable game joint game dynamics
        return self.cs.dynamics

@singledispatch
def approx_linear_quadratic_game(nlgame, *args, **kwargs):
    """
    Approximate a linear-quadratic game from a nonlinear game.
    Generic entry point; specialized by type.
    """
    raise NotImplementedError(f"No implementation for {type(nlgame)}")


@approx_linear_quadratic_game.register(NonlinearGameType1)
def _approx_linear_quadratic_game(nlgame: NonlinearGameType1, op: FixedStepSystemTrajectory) -> LinearQuadraticGameType1:
    """
    Approximate a LinearQuadraticGameType1 from a NonlinearGameType1.

    Note that this approximation creates a subtle, yet meaningful, re-definition
    of the state and control varaible. If the state and control variables of 
    the SampledContinuousSystemType1 are (x, u); then the state and control 
    variables of LinearDiscreteSystemType1 approximation are (delx, delu) where
    delx = x - op.x
    delu = u - op.u
    delx_(t+1) = A @ delx_(t) + B @ delu_(t)

    This distinction is important for correctly interpreting the nash feedback 
    strategies of linear-quadratic games based upon the LinearDiscreteSystemType1
    approximate dynamics, and thus, correctly propagating trajectories based 
    upon these strategies. The distinction is subtle because misinterpreting
    these variables mostly leads to silent errors as they are all of
    consistent shapes

    Args:
    - nlgame : NonlinearGameType1
        nonlinear game to be approximated as linear-quadratic
    - op : SystemTrajectory
        operating point about which sytem is linearized, discretized, and quadraticized

    Returns:
    - lqgame : LinearQuadraticGameType1
        linearized-discretized-quadraticized game approximated about operating point
    """

    # Linearize and discretize the underlying control system
    # NOTE: the change in state and control variables to (delx, delu)
    cs = approx_linear_discrete_system(nlgame.cs, op=op)

    # Quadraticize cost for each player at discrete times around current operating point
    Q = jnp.zeros((nlgame.nsteps, nlgame.N, nlgame.nx, nlgame.nx))
    q = jnp.zeros((nlgame.nsteps, nlgame.N, nlgame.nx))
    R = jnp.zeros((nlgame.nsteps, nlgame.N, nlgame.nu, nlgame.nu))
    r = jnp.zeros((nlgame.nsteps, nlgame.N, nlgame.nu))
    for pidx in range(nlgame.N):
        Qp, qp, Rp, rp = quadraticize_cost_joint_ctrl_playerwise_trajectory(
            g_i = nlgame.costs[pidx].running, 
            op = op, 
            u_splits = nlgame.u_splits
        )
        Q = Q.at[:, pidx, :, :].set(Qp)
        q = q.at[:, pidx, :].set(qp)
        R = R.at[:, pidx, :, :].set(Rp)
        r = r.at[:, pidx, :].set(rp)

    # Construct LQGame around current operating point
    # The LQgame is formulated as the second order Taylor expansion
    # around the current operating point because the linearization and
    # quadratization compute the jacobians and hessians at that points
    # Therefore, the solution to the LQGame assumes the transformed
    # state and control variables delx = x - x_op and delu = u - u_op,
    # respectively
    lqgame = LinearQuadraticGameType1(
        cs = cs,
        N = nlgame.N,
        Q = Q,
        q = q,
        R = R,
        r = r,
        u_splits = nlgame.u_splits,
        Qf = jnp.zeros((nlgame.N, nlgame.nx, nlgame.nx), dtype=Q.dtype),
        qf = jnp.zeros((nlgame.N, nlgame.nx), dtype=q.dtype),
    )

    return lqgame
