# Copyright 2026 MIT Lincoln Laboratory
# SPDX-License-Identifier: MIT

"""
Advanced example: a linear-quadratic Lady-Bandit-Guard game in IR form.

This example is intentionally lower-level than the beginner examples. It shows
how to construct a ``LinearQuadraticGameType1`` directly from dynamics and
cost matrices, then solve it with the LQ solver.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp

from pydgens.ir.gametypes import LinearQuadraticGameType1
from pydgens.ir.systemtypes import (
    LinearDiscreteSystemType1,
    propagate_system_trajectory,
)
from pydgens.ir.timetypes import TimeGrid
from pydgens.solvers.lqsolver import solve_lqgame_feedback


DEFAULT_INITIAL_STATE = jnp.array([
    -10.0, 0.0, 1.0, 0.0,
    10.0, 0.0, -1.0, 0.0,
    0.0, 10.0, 0.0, -1.0,
])


class LadyBanditGuardLQ:
    '''
    3-player target guarding game where each player has 2D double-integrator
    dynamics and quadratic costs.

    Nomenclature comes from paper: 
    > Rusnak, Ilan. "The lady, the bandits and the body guards–a two team dynamic game." 
    > IFAC Proceedings Volumes 38, no. 1 (2005): 441-446.

    The game dynamics are discrete-time and linear:
        xi_{t+1} = A_game @ xi_t + B_game @ mu_t

    where:
    - nx (int) = 12: dimension of auxiliary joint game state vector
    - nu (int) = 6: dimension of auxiliary joint game control vector 
    - xi_t (jnp.ndarray size (n,)): is the joint auxiliary state,
        which is a concatenation of Bandit, Lady, and Guard (each size (4,)) states in that order
        - xi_t[0] = px_B : x-position of bandit at time t [m]
        - xi_t[1] = py_B : y-position of bandit at time t [m]
        - xi_t[2] = vx_B : x-velocity of bandit at time t [m/s]
        - xi_t[3] = vy_B : y-velocity of bandit at time t [m/s]
        - xi_t[4] = px_L : x-position of lady at time t [m]
        - xi_t[5] = py_L : y-position of lady at time t [m]
        - xi_t[6] = vx_L : x-velocity of lady at time t [m/s]
        - xi_t[7] = vy_L : y-velocity of lady at time t [m/s]
        - xi_t[8] = px_G : x-position of guard at time t [m]
        - xi_t[9] = py_G : y-position of guard at time t [m]
        - xi_t[10] = vx_G : x-velocity of guard at time t [m/s]
        - xi_t[11] = vy_G : y-velocity of guard at time t [m/s]
    - mu_t (jnp.ndarray size (m,)): is the joint auxiliary control,
        which is a concatenation of Bandit, Lady, and Guard (each size (2,)) controls in that order
        - mu_t[0] = ax_B : x-acceleration of bandit at time t [m/s/s]
        - mu_t[1] = ay_B : y-acceleration of bandit at time t [m/s/s]
        - mu_t[2] = ax_L : x-acceleration of lady at time t [m/s/s]
        - mu_t[3] = ay_L : y-acceleration of lady at time t [m/s/s]
        - mu_t[4] = ax_G : x-acceleration of guard at time t [m/s/s]
        - mu_t[5] = ay_G : y-acceleration of guard at time t [m/s/s]
    '''

    # Hard-coded, static parameters of game
    PARAMS = SimpleNamespace()
    PARAMS.N_PLAYERS = 3
    PARAMS.BANDIT_PLAYER_IDX = 0
    PARAMS.LADY_PLAYER_IDX = 1
    PARAMS.GUARD_PLAYER_IDX = 2

    # Auxiliary joint state space parameterization
    PARAMS.GAME_AUX_STATE = SimpleNamespace()
    PARAMS.GAME_AUX_STATE.NX = 12    # dimension of joint auxiliary state space
    PARAMS.GAME_AUX_STATE.NX_BANDIT = 4
    PARAMS.GAME_AUX_STATE.NX_LADY = 4
    PARAMS.GAME_AUX_STATE.NX_GUARD = 4
    PARAMS.GAME_AUX_STATE.I_BANDIT_PX = 0
    PARAMS.GAME_AUX_STATE.I_BANDIT_PY = 1
    PARAMS.GAME_AUX_STATE.I_BANDIT_VX = 2
    PARAMS.GAME_AUX_STATE.I_BANDIT_VY = 3
    PARAMS.GAME_AUX_STATE.I_LADY_PX = 4
    PARAMS.GAME_AUX_STATE.I_LADY_PY = 5
    PARAMS.GAME_AUX_STATE.I_LADY_VX = 6
    PARAMS.GAME_AUX_STATE.I_LADY_VY = 7
    PARAMS.GAME_AUX_STATE.I_GUARD_PX = 8
    PARAMS.GAME_AUX_STATE.I_GUARD_PY = 9
    PARAMS.GAME_AUX_STATE.I_GUARD_VX = 10
    PARAMS.GAME_AUX_STATE.I_GUARD_VY = 11

    # Auxiliary joint control space parameterization
    PARAMS.GAME_AUX_CTRL = SimpleNamespace()
    PARAMS.GAME_AUX_CTRL.NU = 6    # dimension of joint auxiliary control space
    PARAMS.GAME_AUX_CTRL.NU_BANDIT = 2
    PARAMS.GAME_AUX_CTRL.NU_LADY = 2
    PARAMS.GAME_AUX_CTRL.NU_GUARD = 2
    PARAMS.GAME_AUX_CTRL.I_BANDIT_AX = 0
    PARAMS.GAME_AUX_CTRL.I_BANDIT_AY = 1
    PARAMS.GAME_AUX_CTRL.I_LADY_AX = 2
    PARAMS.GAME_AUX_CTRL.I_LADY_AY = 3
    PARAMS.GAME_AUX_CTRL.I_GUARD_AX = 4
    PARAMS.GAME_AUX_CTRL.I_GUARD_AY = 5

    def package_cfg_vars(self, 
        px_target,
        py_target,
        w_b_bl_dist,
        w_b_gb_dist,
        w_b_lt_dist,
        w_b_acc,
        w_l_bl_dist,
        w_l_gb_dist,
        w_l_lt_dist,
        w_l_acc,
        w_g_bl_dist,
        w_g_gb_dist,
        w_g_lt_dist,
        w_g_acc,
    ):
        assert w_b_bl_dist > 0
        assert w_b_gb_dist > 0
        assert w_b_lt_dist > 0
        assert w_b_acc > 0
        assert w_l_bl_dist > 0
        assert w_l_gb_dist > 0
        assert w_l_lt_dist > 0
        assert w_l_acc > 0
        assert w_g_bl_dist > 0
        assert w_g_gb_dist > 0
        assert w_g_lt_dist > 0
        assert w_g_acc > 0


        # package instance attributes for ease of use
        self.cfg = SimpleNamespace()
        self.cfg.px_target = px_target
        self.cfg.py_target = py_target

        self.cfg.w_b_bl_dist = w_b_bl_dist
        self.cfg.w_b_gb_dist = w_b_gb_dist
        self.cfg.w_b_lt_dist = w_b_lt_dist
        self.cfg.w_b_acc = w_b_acc

        self.cfg.w_l_bl_dist = w_l_bl_dist
        self.cfg.w_l_gb_dist = w_l_gb_dist
        self.cfg.w_l_lt_dist = w_l_lt_dist
        self.cfg.w_l_acc = w_l_acc

        self.cfg.w_g_bl_dist = w_g_bl_dist
        self.cfg.w_g_gb_dist = w_g_gb_dist
        self.cfg.w_g_lt_dist = w_g_lt_dist
        self.cfg.w_g_acc = w_g_acc
    
    def compose_lq_game(self, nt, dt):

        # package time characteristics
        tg = TimeGrid(nt=nt, dt=dt)

        # Unpack parameters for ease of use
        N = self.PARAMS.N_PLAYERS
        nx = self.PARAMS.GAME_AUX_STATE.NX
        nu = self.PARAMS.GAME_AUX_CTRL.NU

        # Instantiate linear-quadratic game matrices
        A = jnp.zeros((tg.nt-1, nx, nx))
        B = jnp.zeros((tg.nt-1, nx, nu))
        Q = jnp.zeros((tg.nt-1, N, nx, nx))
        q = jnp.zeros((tg.nt-1, N, nx))
        R = jnp.zeros((tg.nt-1, N, nu, nu))
        r = jnp.zeros((tg.nt-1, N, nu))

        u_splits = self.PARAMS.N_PLAYERS*[None]
        u_splits[self.PARAMS.BANDIT_PLAYER_IDX] = self.PARAMS.GAME_AUX_CTRL.NU_BANDIT
        u_splits[self.PARAMS.LADY_PLAYER_IDX] = self.PARAMS.GAME_AUX_CTRL.NU_LADY
        u_splits[self.PARAMS.GUARD_PLAYER_IDX] = self.PARAMS.GAME_AUX_CTRL.NU_GUARD

        # Compile dynamics and cost matrices at each time node
        for tidx in range(tg.nt-1):

            # encode linear game dynamics
            At, Bt = self.fblin_dynamics(tg.dt, self.PARAMS)
            A = A.at[tidx].set(At)
            B = B.at[tidx].set(Bt)

            # encode bandit quadratic cost matrices
            Q, q, R, r = self.compose_bandit_costs(Q, q, R, r, tidx, self.PARAMS, self.cfg)

            # encode lady quadratic cost matrices
            Q, q, R, r = self.compose_lady_costs(Q, q, R, r, tidx, self.PARAMS, self.cfg)

            # encode guard quadratic cost matrices
            Q, q, R, r = self.compose_guard_costs(Q, q, R, r, tidx, self.PARAMS, self.cfg)

        # Instantiate control system
        cs = LinearDiscreteSystemType1(
            tg = tg,
            nx = nx,
            nu = nu,
            A = A,
            B = B
        )

        # Compose game object for solver
        self.game = LinearQuadraticGameType1(
            cs = cs,
            N = N,
            Q = Q,
            q = q,
            R = R, 
            r = r,
            u_splits = jnp.asarray(u_splits)
        )

        
    @staticmethod
    def fblin_dynamics(delta: float, params: SimpleNamespace):
        """
        Construct the discrete-time dynamics matrices for the full 3-player game.

        Each player's dynamics are governed by a double integrator in 2D:
            position_next = position + delta * velocity + 0.5 * delta^2 * acceleration
            velocity_next = velocity + delta * acceleration

        This function returns the block-diagonal system matrices A_game and B_game such that:
            xi_{t+1} = A_game @ xi_t + B_game @ mu_t

        Args:
            delta (float): Time step for discrete integration [sec].

        Returns:
            A_game (jax.numpy.ndarray): (12x12) Block-diagonal state transition matrix.
            B_game (jax.numpy.ndarray): (12x6) Block-diagonal control input matrix.
        """

        # unpack dimensions of state and control space for bookkeeping
        nx_b = params.GAME_AUX_STATE.NX_BANDIT
        nx_l = params.GAME_AUX_STATE.NX_LADY
        nx_g = params.GAME_AUX_STATE.NX_GUARD
        nu_b = params.GAME_AUX_CTRL.NU_BANDIT
        nu_l = params.GAME_AUX_CTRL.NU_LADY
        nu_g = params.GAME_AUX_CTRL.NU_GUARD

        A_d = jnp.array([
            [1.0, 0.0, delta, 0.0],
            [0.0, 1.0, 0.0, delta],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])

        B_d = jnp.array([
            [0.5 * delta**2, 0.0],
            [0.0, 0.5 * delta**2],
            [delta, 0.0],
            [0.0, delta]
        ])

        A_game = jnp.block([
            [A_d,        jnp.zeros((nx_b, nx_l + nx_g))],
            [jnp.zeros((nx_l, nx_b)), A_d,        jnp.zeros((nx_l, nx_g))],
            [jnp.zeros((nx_g, nx_b+nx_l)), A_d]
        ])

        B_game = jnp.block([
            [B_d,                    jnp.zeros((nx_b, nu_l + nu_g))],
            [jnp.zeros((nx_l, nu_b)), B_d, jnp.zeros((nx_l, nu_g))],
            [jnp.zeros((nx_g, nu_b + nu_l)), B_d]
        ])

        return A_game, B_game
    
    @staticmethod
    def cost_bandit_lady_distance(c_dBL: float, params) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Constructs the quadratic cost matrix Q and vector q for the
        Bandit-Lady distance cost. The cost is from the perspective of the
        Bandit and thus trying to minimize the Bandit-Lady distance

        The cost is in standard form
            J = 1/2 xi.T @ Q @ xi + q.T @ xi

        Args:
            c_dBL (float): Weighting parameter for the squared distance cost.

        Returns:
            Q (jax.numpy.ndarray): (12x12) quadratic cost matrix
            q (jax.numpy.ndarray): (12,) linear cost vector (zero in this case)
        """

        assert c_dBL > 0, f"weight c_dBL must be positive, got {c_dBL}"

        nx = params.GAME_AUX_STATE.NX
        Q = jnp.zeros((nx, nx))

        # unpack aux state indices for ease of bookkeeping
        i_px_b = params.GAME_AUX_STATE.I_BANDIT_PX
        i_py_b = params.GAME_AUX_STATE.I_BANDIT_PY
        i_px_l = params.GAME_AUX_STATE.I_LADY_PX
        i_py_l = params.GAME_AUX_STATE.I_LADY_PY

        # Diagonal terms
        Q = Q.at[i_px_b, i_px_b].set(2 * c_dBL)  # p_{x,B}
        Q = Q.at[i_py_b, i_py_b].set(2 * c_dBL)  # p_{y,B}
        Q = Q.at[i_px_l, i_px_l].set(2 * c_dBL)  # p_{x,L}
        Q = Q.at[i_py_l, i_py_l].set(2 * c_dBL)  # p_{y,L}
        # Cross terms
        Q = Q.at[i_px_b, i_px_l].set(-2 * c_dBL)
        Q = Q.at[i_px_l, i_px_b].set(-2 * c_dBL)
        Q = Q.at[i_py_b, i_py_l].set(-2 * c_dBL)
        Q = Q.at[i_py_l, i_py_b].set(-2 * c_dBL)

        q = jnp.zeros(nx)

        return Q, q

    @staticmethod
    def cost_guard_bandit_distance(c_dGB: float, params) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Constructs the quadratic cost matrix Q and vector q for the
        Guard-Bandit distance cost. The cost is from the perspective of the
        Bandit and thus trying to maximize the Bandit-Guard distance

        The cost is in standard form
            J = 1/2 xi.T @ Q @ xi + q.T @ xi

        Args:
            c_dGB (float): Weighting parameter for the squared distance cost.

        Returns:
            Q (jax.numpy.ndarray): (12x12) quadratic cost matrix
            q (jax.numpy.ndarray): (12,) linear cost vector (zero in this case)
        """

        assert c_dGB > 0, f"weight c_dGB must be positive, got {c_dGB}"

        nx = params.GAME_AUX_STATE.NX
        Q = jnp.zeros((nx, nx))

        # unpack aux state indices for ease of bookkeeping
        i_px_b = params.GAME_AUX_STATE.I_BANDIT_PX
        i_py_b = params.GAME_AUX_STATE.I_BANDIT_PY
        i_px_g = params.GAME_AUX_STATE.I_GUARD_PX
        i_py_g = params.GAME_AUX_STATE.I_GUARD_PY

        # Diagonal terms
        Q = Q.at[i_px_b, i_px_b].set(-2 * c_dGB)
        Q = Q.at[i_py_b, i_py_b].set(-2 * c_dGB)
        Q = Q.at[i_px_g, i_px_g].set(-2 * c_dGB)
        Q = Q.at[i_py_g, i_py_g].set(-2 * c_dGB)
        # Cross terms
        Q = Q.at[i_px_b, i_px_g].set(2 * c_dGB)
        Q = Q.at[i_px_g, i_px_b].set(2 * c_dGB)
        Q = Q.at[i_py_b, i_py_g].set(2 * c_dGB)
        Q = Q.at[i_py_g, i_py_b].set(2 * c_dGB)

        q = jnp.zeros(nx)

        return Q, q
    
    @staticmethod
    def cost_lady_target_distance(c_dLT: float, px_target, py_target, params) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Constructs the quadratic cost matrix Q and vector q for the
        Lady-Target distance cost. The cost is from the perspective of the
        Lady that is trying to minimize distance to fixed target

        The cost is in standard form
            J = 1/2 xi.T @ Q @ xi + q.T @ xi

        Args:
            c_dLT (float): Weighting parameter for the squared distance cost.
            px_target (float): x-position of fixed target [m]
            py_target (float): y-position of fixed target [m]

        Returns:
            Q (jax.numpy.ndarray): (12x12) quadratic cost matrix
            q (jax.numpy.ndarray): (12,) linear cost vector
        """

        assert c_dLT > 0, f"weight c_dLT must be positive, got {c_dLT}"

        nx = params.GAME_AUX_STATE.NX
        Q = jnp.zeros((nx, nx))

        # unpack aux state indices for ease of bookkeeping
        i_px_l = params.GAME_AUX_STATE.I_LADY_PX
        i_py_l = params.GAME_AUX_STATE.I_LADY_PY

        # Diagonal terms
        Q = Q.at[i_px_l, i_px_l].set(2 * c_dLT) 
        Q = Q.at[i_py_l, i_py_l].set(2 * c_dLT) 

        # Linear term
        q = jnp.zeros(nx)
        q = q.at[i_px_l].set(-2 * c_dLT * px_target)
        q = q.at[i_py_l].set(-2 * c_dLT * py_target)

        return Q, q

    # default time components
    DEFAULT_N_TIMENODES = 20
    DEFAULT_TIMESTEP_SIZE = 1.0

    # default bandit cost weights
    DEFAULT_B_BL_ALIGN_WEIGHT = 1.0
    DEFAULT_B_BL_DIST_WEIGHT = 1.0
    DEFAULT_B_GB_ALIGN_WEIGHT = 1.0
    DEFAULT_B_GB_DIST_WEIGHT = 1.0
    DEFAULT_B_LT_DIST_WEIGHT = 1.0
    DEFAULT_B_ACC_WEIGHT = 1.0

    # default lady cost weights
    DEFAULT_L_BL_ALIGN_WEIGHT = 1.0
    DEFAULT_L_BL_DIST_WEIGHT = 1.0
    DEFAULT_L_GB_ALIGN_WEIGHT = 1.0
    DEFAULT_L_GB_DIST_WEIGHT = 1.0
    DEFAULT_L_LT_DIST_WEIGHT = 1.0
    DEFAULT_L_ACC_WEIGHT = 1.0
    DEFAULT_TARGET_PX = 0.0
    DEFAULT_TARGET_PY = 0.0

    # default guard cost weights
    DEFAULT_G_BL_ALIGN_WEIGHT = 1.0
    DEFAULT_G_BL_DIST_WEIGHT = 1.0
    DEFAULT_G_GB_ALIGN_WEIGHT = 1.0
    DEFAULT_G_GB_DIST_WEIGHT = 1.0
    DEFAULT_G_LT_DIST_WEIGHT = 1.0
    DEFAULT_G_ACC_WEIGHT = 1.0

    def __init__(self, 
        nt: int=DEFAULT_N_TIMENODES,
        dt: float=DEFAULT_TIMESTEP_SIZE,
        px_target: float=DEFAULT_TARGET_PX,
        py_target: float=DEFAULT_TARGET_PY,
        w_b_bl_align: float=DEFAULT_B_BL_ALIGN_WEIGHT,
        w_b_bl_dist: float=DEFAULT_B_BL_DIST_WEIGHT,
        w_b_gb_align: float=DEFAULT_B_GB_ALIGN_WEIGHT,
        w_b_gb_dist: float=DEFAULT_B_GB_DIST_WEIGHT,
        w_b_lt_dist: float=DEFAULT_B_LT_DIST_WEIGHT,
        w_b_acc: float=DEFAULT_B_ACC_WEIGHT,
        w_l_bl_align: float=DEFAULT_L_BL_ALIGN_WEIGHT,
        w_l_bl_dist: float=DEFAULT_L_BL_DIST_WEIGHT,
        w_l_gb_align: float=DEFAULT_L_GB_ALIGN_WEIGHT,
        w_l_gb_dist: float=DEFAULT_L_GB_DIST_WEIGHT,
        w_l_lt_dist: float=DEFAULT_L_LT_DIST_WEIGHT,
        w_l_acc: float=DEFAULT_L_ACC_WEIGHT,
        w_g_bl_align: float=DEFAULT_G_BL_ALIGN_WEIGHT,
        w_g_bl_dist: float=DEFAULT_G_BL_DIST_WEIGHT,
        w_g_gb_align: float=DEFAULT_G_GB_ALIGN_WEIGHT,
        w_g_gb_dist: float=DEFAULT_G_GB_DIST_WEIGHT,
        w_g_lt_dist: float=DEFAULT_G_LT_DIST_WEIGHT,
        w_g_acc: float=DEFAULT_G_ACC_WEIGHT,
        ):
        """
        # Args:
        - tg (TimeGrid): time characteristics (nt, dt, t0)
        - px_target (float): x-position of lady's target [m]
        - py_target (float): y-position of lady's target [m]
        - w_b_bl_align (float): bandit's cost weight for aligning bandit heading to lady
        - w_b_bl_dist (float): bandit's cost weight for minimizing bandit distance to lady
        - w_b_gb_align (float): bandit's cost weight for misaligning guard heading to bandit 
        - w_b_gb_dist (float): bandit's cost weight for maximizing guard distance to bandit
        - w_b_lt_dist (float): bandit's cost weight for maximizing lady distance to target
        - w_b_acc (float): bandit's cost weight on control effort of acceleration
        - w_l_bl_align (float): lady's cost weight for misaligning bandit heading to lady
        - w_l_bl_dist (float): lady's cost weight for maximizing bandit distance to lady
        - w_l_gb_align (float): lady's cost weight for aligning guard heading to bandit 
        - w_l_gb_dist (float): lady's cost weight for minimizing guard distance to bandit
        - w_l_lt_dist (float): lady's cost weight for minimizing lady distance to target
        - w_l_acc (float): lady's cost weight on control effort of acceleration
        - w_g_bl_align (float): guard's cost weight for misaligning bandit heading to lady
        - w_g_bl_dist (float): guard's cost weight for maximizing bandit distance to lady
        - w_g_gb_align (float): guard's cost weight for aligning guard heading to bandit 
        - w_g_gb_dist (float): guard's cost weight for minimizing guard distance to bandit
        - w_g_lt_dist (float): guard's cost weight for minimizing lady distance to target
        - w_g_acc (float): guard's cost weight on control effort of acceleration
        Auxiliary Dynamics: xi[t+1] = A[t]xi[t] + B[t]mu[t]
        Cost: J[t,i] = 0.5 * xi[t].T @ Q[t,i] @ xi[t] + q[t,i].T @ xi[t] + 
                       0.5 * mu[t].T @ R[t,i] @ mu[t] + r[t,i].T @ mu[t]
        """

        # Store the scalar weights used while assembling Q, q, R, and r.
        self.package_cfg_vars(
            px_target = px_target,
            py_target = py_target,
            w_b_bl_dist = w_b_bl_dist,
            w_b_gb_dist = w_b_gb_dist,
            w_b_lt_dist = w_b_lt_dist,
            w_b_acc = w_b_acc,
            w_l_bl_dist = w_l_bl_dist,
            w_l_gb_dist = w_l_gb_dist,
            w_l_lt_dist = w_l_lt_dist,
            w_l_acc = w_l_acc,
            w_g_bl_dist = w_g_bl_dist,
            w_g_gb_dist = w_g_gb_dist,
            w_g_lt_dist = w_g_lt_dist,
            w_g_acc = w_g_acc,
        )

        # Store the alignment-proxy weights alongside the distance weights.
        assert w_b_bl_align > 0
        assert w_b_gb_align > 0
        assert w_l_bl_align > 0
        assert w_l_gb_align > 0
        assert w_g_bl_align > 0
        assert w_g_gb_align > 0
        self.cfg.w_b_bl_align = w_b_bl_align
        self.cfg.w_b_gb_align = w_b_gb_align
        self.cfg.w_l_bl_align = w_l_bl_align
        self.cfg.w_l_gb_align = w_l_gb_align
        self.cfg.w_g_bl_align = w_g_bl_align
        self.cfg.w_g_gb_align = w_g_gb_align

        # Compose the linear-quadratic game matrices.
        self.compose_lq_game(nt, dt)
    
    @staticmethod
    def compose_bandit_costs(Q, q, R, r, tidx, params, cfg):
        """
        Sum the quadratic cost matrices for Bandit

        # Args:
        - tidx (int): discrete time node to be encoded
        """

        idx_b = params.BANDIT_PLAYER_IDX

        # Set: minimize alignment error between bandit velocity and lady position relative to bandit
        Qc, qc = LadyBanditGuardLQ.cost_bandit_lady_alignment_proxy(cfg.w_b_bl_align, params)
        Q = Q.at[tidx,idx_b].set(Qc)
        q = q.at[tidx,idx_b].set(qc)

        # Add: minimize distance between bandit and lady
        Qc, qc = LadyBanditGuardLQ.cost_bandit_lady_distance(cfg.w_b_bl_dist, params)
        Q = Q.at[tidx,idx_b].add(Qc)
        q = q.at[tidx,idx_b].add(qc)

        # Add: maximize alignment error between guard velocity and bandit position relative to guard
        Qc, qc = LadyBanditGuardLQ.cost_guard_bandit_alignment_proxy(cfg.w_b_gb_align, params)
        Q = Q.at[tidx,idx_b].add(Qc)
        q = q.at[tidx,idx_b].add(qc)

        # Add: maximize distance between guard and bandit
        Qc, qc = LadyBanditGuardLQ.cost_guard_bandit_distance(cfg.w_b_gb_dist, params)
        Q = Q.at[tidx,idx_b].add(Qc)
        q = q.at[tidx,idx_b].add(qc)

        # Add: maximize distance between lady and target
        Qc, qc = LadyBanditGuardLQ.cost_lady_target_distance(cfg.w_b_lt_dist, cfg.px_target, cfg.py_target, params)
        Q = Q.at[tidx,idx_b].add(-Qc)
        q = q.at[tidx,idx_b].add(-qc)

        # Set: minimize control effort of accelerations along both axes equally
        i_ax_b = params.GAME_AUX_CTRL.I_BANDIT_AX
        i_ay_b = params.GAME_AUX_CTRL.I_BANDIT_AY
        R = R.at[tidx, idx_b, i_ax_b, i_ax_b].set(cfg.w_b_acc)
        R = R.at[tidx, idx_b, i_ay_b, i_ay_b].set(cfg.w_b_acc)

        return Q, q, R, r

    @staticmethod
    def compose_lady_costs(Q, q, R, r, tidx, params, cfg):
        """
        Sum the quadratic cost matrices for Lady

        # Args:
        - tidx (int): discrete time node to be encoded
        """

        idx_l = params.LADY_PLAYER_IDX
        
        # Set: maximize alignment error between bandit velocity and lady position relative to bandit
        Qc, qc = LadyBanditGuardLQ.cost_bandit_lady_alignment_proxy(cfg.w_l_bl_align, params)
        Q = Q.at[tidx,idx_l].set(-Qc)
        q = q.at[tidx,idx_l].set(-qc)

        # Add: maximize distance between bandit and lady
        Qc, qc = LadyBanditGuardLQ.cost_bandit_lady_distance(cfg.w_l_bl_dist, params)
        Q = Q.at[tidx,idx_l].add(-Qc)
        q = q.at[tidx,idx_l].add(-qc)

        # Add: minimize alignment error between guard velocity and bandit position relative to guard
        Qc, qc = LadyBanditGuardLQ.cost_guard_bandit_alignment_proxy(cfg.w_l_gb_align, params)
        Q = Q.at[tidx,idx_l].add(-Qc)
        q = q.at[tidx,idx_l].add(-qc)

        # Add: minimize distance between guard and bandit
        Qc, qc = LadyBanditGuardLQ.cost_guard_bandit_distance(cfg.w_l_gb_dist, params)
        Q = Q.at[tidx,idx_l].add(-Qc)
        q = q.at[tidx,idx_l].add(-qc)

        # Add: minimize distance between lady and target
        Qc, qc = LadyBanditGuardLQ.cost_lady_target_distance(cfg.w_l_lt_dist, cfg.px_target, cfg.py_target, params)
        Q = Q.at[tidx,idx_l].add(Qc)
        q = q.at[tidx,idx_l].add(qc)

        # Set: minimize control effort of accelerations along both axes equally
        i_ax_l = params.GAME_AUX_CTRL.I_LADY_AX
        i_ay_l = params.GAME_AUX_CTRL.I_LADY_AY
        R = R.at[tidx, idx_l, i_ax_l, i_ax_l].set(cfg.w_l_acc)
        R = R.at[tidx, idx_l, i_ay_l, i_ay_l].set(cfg.w_l_acc)

        return Q, q, R, r

    @staticmethod
    def compose_guard_costs(Q, q, R, r, tidx, params, cfg):
        """
        Sum the quadratic cost matrices for Guard

        # Args:
        - tidx (int): discrete time node to be encoded
        """

        idx_g = params.GUARD_PLAYER_IDX

        # Set: maximize alignment error between bandit velocity and lady position relative to bandit
        Qc, qc = LadyBanditGuardLQ.cost_bandit_lady_alignment_proxy(cfg.w_g_bl_align, params)
        Q = Q.at[tidx,idx_g].set(-Qc)
        q = q.at[tidx,idx_g].set(-qc)

        # Add: maximize distance between bandit and lady
        Qc, qc = LadyBanditGuardLQ.cost_bandit_lady_distance(cfg.w_g_bl_dist, params)
        Q = Q.at[tidx,idx_g].add(-Qc)
        q = q.at[tidx,idx_g].add(-qc)

        # Add: minimize alignment error between guard velocity and bandit position relative to guard
        Qc, qc = LadyBanditGuardLQ.cost_guard_bandit_alignment_proxy(cfg.w_g_gb_align, params)
        Q = Q.at[tidx,idx_g].add(-Qc)
        q = q.at[tidx,idx_g].add(-qc)

        # Add: minimize distance between guard and bandit
        Qc, qc = LadyBanditGuardLQ.cost_guard_bandit_distance(cfg.w_g_gb_dist, params)
        Q = Q.at[tidx,idx_g].add(-Qc)
        q = q.at[tidx,idx_g].add(-qc)

        # Add: minimize distance between lady and target
        Qc, qc = LadyBanditGuardLQ.cost_lady_target_distance(cfg.w_g_lt_dist, cfg.px_target, cfg.py_target, params)
        Q = Q.at[tidx,idx_g].add(Qc)
        q = q.at[tidx,idx_g].add(qc)

        # Set: minimize control effort of accelerations along both axes equally
        i_ax_g = params.GAME_AUX_CTRL.I_GUARD_AX
        i_ay_g = params.GAME_AUX_CTRL.I_GUARD_AY
        R = R.at[tidx, idx_g, i_ax_g, i_ax_g].set(cfg.w_g_acc)
        R = R.at[tidx, idx_g, i_ay_g, i_ay_g].set(cfg.w_g_acc)

        return Q, q, R, r

    @staticmethod
    def cost_bandit_lady_alignment_proxy(c_gammaBL: float, params) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Constructs the quadratic cost matrix Q and vector q for the Bandit-Lady alignment proxy cost,
        the cost is from the perspective of the bandits objective to minimize the angle
        between the bandits velocity vector and the position vector of the lady relative
        to the bandit, thus minimizing -dot(v_B, p_{L/B}) in the form 
        J = 1/2 xi^T Q xi + q^T xi.

        Args:
            c_gammaBL (float): Weighting coefficient for the alignment proxy cost.

        Returns:
            Q (jax.numpy.ndarray): 12x12 symmetric matrix encoding the quadratic cost.
            q (jax.numpy.ndarray): 12-vector  encoding linear cost(zeros).
        """
        assert c_gammaBL > 0, f"weight c_gammaBL must be positive, got {c_gammaBL}"

        nx = params.GAME_AUX_STATE.NX
        Q = jnp.zeros((nx,nx))

        # xi2*xi0 term (+c): Q[0,2] and Q[2,0]
        i_px_b = params.GAME_AUX_STATE.I_BANDIT_PX
        i_vx_b = params.GAME_AUX_STATE.I_BANDIT_VX
        Q = Q.at[i_px_b, i_vx_b].set(c_gammaBL)
        Q = Q.at[i_vx_b, i_px_b].set(c_gammaBL)

        # xi3*xi1 term (+c): Q[1,3] and Q[3,1]
        i_py_b = params.GAME_AUX_STATE.I_BANDIT_PY
        i_vy_b = params.GAME_AUX_STATE.I_BANDIT_VY
        Q = Q.at[i_py_b, i_vy_b].set(c_gammaBL)
        Q = Q.at[i_vy_b, i_py_b].set(c_gammaBL)

        # -xi2*xi4 term (-c): Q[2,4] and Q[4,2]
        i_px_l = params.GAME_AUX_STATE.I_LADY_PX
        Q = Q.at[i_vx_b, i_px_l].set(-c_gammaBL)
        Q = Q.at[i_px_l, i_vx_b].set(-c_gammaBL)

        # -xi3*xi5 term (-c): Q[3,5] and Q[5,3]
        i_py_l = params.GAME_AUX_STATE.I_LADY_PY
        Q = Q.at[i_vy_b, i_py_l].set(-c_gammaBL)
        Q = Q.at[i_py_l, i_vy_b].set(-c_gammaBL)

        q = jnp.zeros(nx)

        return Q, q

    @staticmethod
    def cost_guard_bandit_alignment_proxy(c_gammaGB: float, params) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Construct the quadratic cost matrices Q and q for the Guard-Bandit alignment proxy cost.

        The cost is from the perspective of the Bandit's objective to maximize the misalignment
        (i.e. minimize the alignment) of the Guard's velocity vector v_G with the Bandit's
        position vector relative to the guard: p_{B/G} = p_B - p_G. This is accomplished
        by minimizing dot(v_G, p_{B/G})

        Parameters
        ----------
        c_gammaGB : float
            Cost weight for the Guard-Bandit alignment proxy cost.

        Returns
        -------
        Q : (12, 12) jnp.ndarray
            Symmetric matrix for the quadratic cost term (1/2) * xi^T Q xi.

        q : (12,) jnp.ndarray
            Linear term for the cost: q^T xi.
        """
        assert c_gammaGB > 0, f"weight c_gammaGB must be positive, got {c_gammaGB}"
        nx = params.GAME_AUX_STATE.NX
        Q = jnp.zeros((nx, nx))
        q = jnp.zeros(nx)

        # unpack aux state indices for ease of bookkeeping
        i_px_b = params.GAME_AUX_STATE.I_BANDIT_PX
        i_py_b = params.GAME_AUX_STATE.I_BANDIT_PY
        i_px_g = params.GAME_AUX_STATE.I_GUARD_PX
        i_py_g = params.GAME_AUX_STATE.I_GUARD_PY
        i_vx_g = params.GAME_AUX_STATE.I_GUARD_VX
        i_vy_g = params.GAME_AUX_STATE.I_GUARD_VY

        # Set symmetric off-diagonal entries for Q
        Q = Q.at[i_px_b, i_vx_g].set(+c_gammaGB)
        Q = Q.at[i_vx_g, i_px_b].set(+c_gammaGB)
        Q = Q.at[i_px_g, i_vx_g].set(-c_gammaGB)
        Q = Q.at[i_vx_g, i_px_g].set(-c_gammaGB)

        Q = Q.at[i_py_b, i_vy_g].set(+c_gammaGB)
        Q = Q.at[i_vy_g, i_py_b].set(+c_gammaGB)
        Q = Q.at[i_py_g, i_vy_g].set(-c_gammaGB)
        Q = Q.at[i_vy_g, i_py_g].set(-c_gammaGB)

        return Q, q


def solve_example():
    """
    Build, solve, and propagate the Lady-Bandit-Guard LQ example.

    Returns
    -------
    tuple
        ``(lbg, strategy, trajectory)`` where ``lbg`` is the IR example
        wrapper, ``strategy`` is the feedback Nash strategy returned by the LQ
        solver, and ``trajectory`` is the propagated auxiliary state/control
        trajectory.
    """
    lbg = LadyBanditGuardLQ()
    strategy = solve_lqgame_feedback(lbg.game)
    trajectory = propagate_system_trajectory(
        lbg.game.cs,
        x0=DEFAULT_INITIAL_STATE,
        strategy=strategy,
    )

    return lbg, strategy, trajectory


def main() -> None:
    lbg, strategy, trajectory = solve_example()

    xs = trajectory.xs
    us = trajectory.us
    p = lbg.PARAMS
    bandit_pos_idx = jnp.asarray([
        p.GAME_AUX_STATE.I_BANDIT_PX,
        p.GAME_AUX_STATE.I_BANDIT_PY,
    ])
    lady_pos_idx = jnp.asarray([
        p.GAME_AUX_STATE.I_LADY_PX,
        p.GAME_AUX_STATE.I_LADY_PY,
    ])
    guard_pos_idx = jnp.asarray([
        p.GAME_AUX_STATE.I_GUARD_PX,
        p.GAME_AUX_STATE.I_GUARD_PY,
    ])

    print("Lady-Bandit-Guard LQ IR example solved.")
    print(f"Strategy shape: P={strategy.P.shape}, alpha={strategy.alpha.shape}")
    print(f"Trajectory shape: xs={xs.shape}, us={us.shape}")
    print("Initial positions:")
    print(
        "  "
        f"bandit={xs[0, bandit_pos_idx]}, "
        f"lady={xs[0, lady_pos_idx]}, "
        f"guard={xs[0, guard_pos_idx]}"
    )
    print("Final positions:")
    print(
        "  "
        f"bandit={xs[-1, bandit_pos_idx]}, "
        f"lady={xs[-1, lady_pos_idx]}, "
        f"guard={xs[-1, guard_pos_idx]}"
    )
    print(f"First joint control: {us[0] if us.size else jnp.array([])}")


if __name__ == "__main__":
    main()
