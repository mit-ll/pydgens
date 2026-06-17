# Copyright 2026 MIT Lincoln Laboratory
# SPDX-License-Identifier: MIT

"""
Advanced example: a nonlinear Lady-Bandit-Guard game in IR form.

This is the nonlinear/iLQ counterpart to ``ir_lady_bandit_guard.py``. The game
logic is the same target-guarding story, but each player uses unicycle dynamics

    x_player = [px, py, theta, speed]
    u_player = [theta_dot, speed_dot]

and the costs are generic nonlinear callables rather than LQ matrices.

Use this example after reading ``unicycle.py``, ``ir_unicycle.py``, and
``ir_lady_bandit_guard.py``. It shows how those ideas combine in a larger
three-player IR game:

    1. define a time grid
    2. define joint continuous-time unicycle dynamics
    3. define one nonlinear running cost per player
    4. encode each player's control dimension with ``u_splits``
    5. build the nonlinear IR game
    6. seed iLQ with an initial operating point
    7. solve with the iLQ solver
"""

from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp

from pydgens.examples._ir_reporting import format_ir_feedback_summary
from pydgens.ir.costtypes import PlayerCostSpecContinuous
from pydgens.ir.gametypes import NonlinearGameType1
from pydgens.ir.strategytypes import FixedStepAffineStrategies
from pydgens.ir.systemtypes import (
    SampledContinuousSystemType1,
    propagate_system_trajectory,
)
from pydgens.ir.timetypes import TimeGrid
from pydgens.solvers.ilqsolver import solve_ilqgame_feedback


_EPS = 1e-8 # small number used to avoid divide by zero


DEFAULT_INITIAL_STATE = jnp.array([
    -10.0, 0.0, 0.0, 1.0,
    10.0, 0.0, 3.14159, 1.0,
    0.0, 10.0, -1.570795, 1.0,
])


class LadyBanditGuardNonlinear:
    '''
    Build the nonlinear IR representation of a 3-player Lady-Bandit-Guard game.

    Each player is a 2D unicycle. The dynamics are based on the "4D unicycle"
    described in Sec V of the paper:
    > Fridovich-Keil, David, Vicenc Rubies-Royo, and Claire J. Tomlin. 
    > "An iterative quadratic method for general-sum differential games with 
    > feedback linearizable dynamics." 2020 IEEE International Conference on 
    > Robotics and Automation (ICRA). IEEE, 2020.

    Nomenclature comes from paper: 
    > Rusnak, Ilan. "The lady, the bandits and the body guards–a two team dynamic game." 
    > IFAC Proceedings Volumes 38, no. 1 (2005): 441-446.

    The joint game state and control are parameterized by:
    - n (int) = 12: dimension of joint game state vector
    - m (int) = 6: dimension of joint game control vector 
    - x_t (jnp.ndarray size (n,)): is the joint state vector,
        which is a concatenation of Bandit, Lady, and Guard (each size (4,)) states in that order
        - x_t[0] = px_B : x-position of bandit at time t [m]
        - x_t[1] = py_B : y-position of bandit at time t [m]
        - x_t[2] = th_B : heading angle (theta) of bandit at time t [rad]
        - x_t[3] = vt_B : total velocity (linear speed) of bandit at time t [m/s]
        - x_t[4] = px_L : x-position of lady at time t [m]
        - x_t[5] = py_L : y-position of lady at time t [m]
        - x_t[6] = th_L : heading angle (theta) of lady at time t [rad]
        - x_t[7] = vt_L : total velocity (linear speed) of lady at time t [m/s]
        - x_t[8] = px_G : x-position of guard at time t [m]
        - x_t[9] = py_G : y-position of guard at time t [m]
        - x_t[10] = th_G : heading angle (theta) of guard at time t [rad]
        - x_t[11] = vt_G : total velocity (linear speed) of guard at time t [m/s]
    - u_t (jnp.ndarray size (m,)): is the joint control vector,
        which is a concatenation of Bandit, Lady, and Guard (each size (2,)) controls in that order
        - u_t[0] = dth_B : rate of change heading (theta dot) of bandit at time t [rad/s]
        - u_t[1] = dvt_B : linear acceleration (vt dot) of bandit at time t [m/s/s]
        - u_t[2] = dth_L : rate of change heading (theta dot) of lady at time t [rad/s]
        - u_t[3] = dvt_L : linear acceleration (vt dot) of lady at time t [m/s/s]
        - u_t[4] = dth_G : rate of change heading (theta dot) of guard at time t [rad/s]
        - u_t[5] = dvt_G : linear acceleration (vt dot) of guard at time t [m/s/s]
    '''

    # Hard-coded, static parameters of game
    PARAMS = SimpleNamespace()
    PARAMS.N_PLAYERS = 3
    PARAMS.BANDIT_PLAYER_IDX = 0
    PARAMS.LADY_PLAYER_IDX = 1
    PARAMS.GUARD_PLAYER_IDX = 2

    # Joint state space parameterization
    PARAMS.GAME_STATE = SimpleNamespace()
    PARAMS.GAME_STATE.NX = 12    # dimension of joint state space
    PARAMS.GAME_STATE.NX_BANDIT = 4
    PARAMS.GAME_STATE.NX_LADY = 4
    PARAMS.GAME_STATE.NX_GUARD = 4
    PARAMS.GAME_STATE.I_BANDIT_PX = 0
    PARAMS.GAME_STATE.I_BANDIT_PY = 1
    PARAMS.GAME_STATE.I_BANDIT_TH = 2
    PARAMS.GAME_STATE.I_BANDIT_VT = 3
    PARAMS.GAME_STATE.I_LADY_PX = 4
    PARAMS.GAME_STATE.I_LADY_PY = 5
    PARAMS.GAME_STATE.I_LADY_TH = 6
    PARAMS.GAME_STATE.I_LADY_VT = 7
    PARAMS.GAME_STATE.I_GUARD_PX = 8
    PARAMS.GAME_STATE.I_GUARD_PY = 9
    PARAMS.GAME_STATE.I_GUARD_TH = 10
    PARAMS.GAME_STATE.I_GUARD_VT = 11

    # Joint control space parameterization
    PARAMS.GAME_CTRL = SimpleNamespace()
    PARAMS.GAME_CTRL.NU = 6    # dimension of joint control space
    PARAMS.GAME_CTRL.NU_BANDIT = 2
    PARAMS.GAME_CTRL.NU_LADY = 2
    PARAMS.GAME_CTRL.NU_GUARD = 2
    PARAMS.GAME_CTRL.I_BANDIT_DTH = 0
    PARAMS.GAME_CTRL.I_BANDIT_DVT = 1
    PARAMS.GAME_CTRL.I_LADY_DTH = 2
    PARAMS.GAME_CTRL.I_LADY_DVT = 3
    PARAMS.GAME_CTRL.I_GUARD_DTH = 4
    PARAMS.GAME_CTRL.I_GUARD_DVT = 5

    # default time components
    DEFAULT_N_TIMENODES = 10
    DEFAULT_TIMESTEP_SIZE = 1.0

    # default bandit cost weights
    DEFAULT_B_BL_ALIGN_WEIGHT = 1.0
    DEFAULT_B_BL_DIST_WEIGHT = 1.0
    DEFAULT_B_GB_ALIGN_WEIGHT = 1.0
    DEFAULT_B_GB_DIST_WEIGHT = 1.0
    DEFAULT_B_LT_DIST_WEIGHT = 1.0
    DEFAULT_B_SPEED_WEIGHT = 1.0
    DEFAULT_B_ACCEL_WEIGHT = 1.0
    DEFAULT_B_OMEGA_WEIGHT = 1.0
    DEFAULT_B_CRUISE_SPEED = 1.0

    # default lady cost weights
    DEFAULT_L_BL_ALIGN_WEIGHT = 1.0
    DEFAULT_L_BL_DIST_WEIGHT = 1.0
    DEFAULT_L_GB_ALIGN_WEIGHT = 1.0
    DEFAULT_L_GB_DIST_WEIGHT = 1.0
    DEFAULT_L_LT_DIST_WEIGHT = 1.0
    DEFAULT_L_SPEED_WEIGHT = 1.0
    DEFAULT_L_ACCEL_WEIGHT = 1.0
    DEFAULT_L_OMEGA_WEIGHT = 1.0
    DEFAULT_L_CRUISE_SPEED = 1.0
    DEFAULT_TARGET_PX = 0.0
    DEFAULT_TARGET_PY = 0.0

    # default guard cost weights
    DEFAULT_G_BL_ALIGN_WEIGHT = 1.0
    DEFAULT_G_BL_DIST_WEIGHT = 1.0
    DEFAULT_G_GB_ALIGN_WEIGHT = 1.0
    DEFAULT_G_GB_DIST_WEIGHT = 1.0
    DEFAULT_G_LT_DIST_WEIGHT = 1.0
    DEFAULT_G_SPEED_WEIGHT = 1.0
    DEFAULT_G_ACCEL_WEIGHT = 1.0
    DEFAULT_G_OMEGA_WEIGHT = 1.0
    DEFAULT_G_CRUISE_SPEED = 1.0

    def __init__(self, 
        nt: int=DEFAULT_N_TIMENODES,
        dt: float=DEFAULT_TIMESTEP_SIZE,
        px_target: float=DEFAULT_TARGET_PX,
        py_target: float=DEFAULT_TARGET_PY,
        v_b_cruise: float=DEFAULT_B_CRUISE_SPEED,
        v_l_cruise: float=DEFAULT_L_CRUISE_SPEED,
        v_g_cruise: float=DEFAULT_G_CRUISE_SPEED,
        w_b_bl_align: float=DEFAULT_B_BL_ALIGN_WEIGHT,
        w_b_bl_dist: float=DEFAULT_B_BL_DIST_WEIGHT,
        w_b_gb_align: float=DEFAULT_B_GB_ALIGN_WEIGHT,
        w_b_gb_dist: float=DEFAULT_B_GB_DIST_WEIGHT,
        w_b_lt_dist: float=DEFAULT_B_LT_DIST_WEIGHT,
        w_b_spd: float=DEFAULT_B_SPEED_WEIGHT,
        w_b_omg: float=DEFAULT_B_OMEGA_WEIGHT,
        w_b_acc: float=DEFAULT_B_ACCEL_WEIGHT,
        w_l_bl_align: float=DEFAULT_L_BL_ALIGN_WEIGHT,
        w_l_bl_dist: float=DEFAULT_L_BL_DIST_WEIGHT,
        w_l_gb_align: float=DEFAULT_L_GB_ALIGN_WEIGHT,
        w_l_gb_dist: float=DEFAULT_L_GB_DIST_WEIGHT,
        w_l_lt_dist: float=DEFAULT_L_LT_DIST_WEIGHT,
        w_l_spd: float=DEFAULT_L_SPEED_WEIGHT,
        w_l_omg: float=DEFAULT_L_OMEGA_WEIGHT,
        w_l_acc: float=DEFAULT_L_ACCEL_WEIGHT,
        w_g_bl_align: float=DEFAULT_G_BL_ALIGN_WEIGHT,
        w_g_bl_dist: float=DEFAULT_G_BL_DIST_WEIGHT,
        w_g_gb_align: float=DEFAULT_G_GB_ALIGN_WEIGHT,
        w_g_gb_dist: float=DEFAULT_G_GB_DIST_WEIGHT,
        w_g_lt_dist: float=DEFAULT_G_LT_DIST_WEIGHT,
        w_g_spd: float=DEFAULT_G_SPEED_WEIGHT,
        w_g_omg: float=DEFAULT_G_OMEGA_WEIGHT,
        w_g_acc: float=DEFAULT_G_ACCEL_WEIGHT,
    ):
        """
        # Args:
        - tg (TimeGrid): time characteristics (nt, dt, t0)
        - px_target (float): x-position of lady's target [m]
        - py_target (float): y-position of lady's target [m]
        - v_b_cruise (float): bandit's target cruise speed [m/s]
        - v_l_cruise (float): lady's target cruise speed [m/s]
        - v_g_cruise (float): guard's target cruise speed [m/s]
        - w_b_bl_align (float): bandit's cost weight for aligning bandit heading to lady
        - w_b_bl_dist (float): bandit's cost weight for minimizing bandit distance to lady
        - w_b_gb_align (float): bandit's cost weight for misaligning guard heading to bandit 
        - w_b_gb_dist (float): bandit's cost weight for maximizing guard distance to bandit
        - w_b_lt_dist (float): bandit's cost weight for maximizing lady distance to target
        - w_b_spd (float): bandit's cost weight on cruise speed deviation
        - w_b_omg (float): bandit's cost weight on control effort of turnrate
        - w_b_acc (float): bandit's cost weight on control effort of linear acceleration
        - w_l_bl_align (float): lady's cost weight for misaligning bandit heading to lady
        - w_l_bl_dist (float): lady's cost weight for maximizing bandit distance to lady
        - w_l_gb_align (float): lady's cost weight for aligning guard heading to bandit 
        - w_l_gb_dist (float): lady's cost weight for minimizing guard distance to bandit
        - w_l_lt_dist (float): lady's cost weight for minimizing lady distance to target
        - w_l_spd (float): lady's cost weight on cruise speed deviation
        - w_l_omg (float): lady's cost weight on control effort of turnrate
        - w_l_acc (float): lady's cost weight on control effort of linear acceleration
        - w_g_bl_align (float): guard's cost weight for misaligning bandit heading to lady
        - w_g_bl_dist (float): guard's cost weight for maximizing bandit distance to lady
        - w_g_gb_align (float): guard's cost weight for aligning guard heading to bandit 
        - w_g_gb_dist (float): guard's cost weight for minimizing guard distance to bandit
        - w_g_lt_dist (float): guard's cost weight for minimizing lady distance to target
        - w_g_spd (float): guard's cost weight on cruise speed deviation
        - w_g_omg (float): guard's cost weight on control effort of turnrate
        - w_g_acc (float): guard's cost weight on control effort of linear acceleration
        """

        assert v_b_cruise >= 0
        assert v_l_cruise >= 0
        assert v_g_cruise >= 0
        assert w_b_bl_align >= 0
        assert w_b_bl_dist >= 0
        assert w_b_gb_align >= 0
        assert w_b_gb_dist >= 0
        assert w_b_lt_dist >= 0
        assert w_b_spd >= 0
        assert w_b_omg >= 0
        assert w_b_acc >= 0
        assert w_l_bl_align >= 0
        assert w_l_bl_dist >= 0
        assert w_l_gb_align >= 0
        assert w_l_gb_dist >= 0
        assert w_l_lt_dist >= 0
        assert w_l_spd >= 0
        assert w_l_omg >= 0
        assert w_l_acc >= 0
        assert w_g_bl_align >= 0
        assert w_g_bl_dist >= 0
        assert w_g_gb_align >= 0
        assert w_g_gb_dist >= 0
        assert w_g_lt_dist >= 0
        assert w_g_spd >= 0
        assert w_g_omg >= 0
        assert w_g_acc >= 0

        # Step 1: define the time grid.
        tg = TimeGrid(nt=nt, dt=dt)

        # Store scalar game parameters used inside the running-cost callables.
        self.px_target = px_target
        self.py_target = py_target
        self.v_b_cruise = v_b_cruise
        self.v_l_cruise = v_l_cruise
        self.v_g_cruise = v_g_cruise

        self.w_b_bl_align = w_b_bl_align
        self.w_b_bl_dist = w_b_bl_dist
        self.w_b_gb_align = w_b_gb_align
        self.w_b_gb_dist = w_b_gb_dist
        self.w_b_lt_dist = w_b_lt_dist
        self.w_b_spd = w_b_spd
        self.w_b_omg = w_b_omg
        self.w_b_acc = w_b_acc

        self.w_l_bl_align = w_l_bl_align
        self.w_l_bl_dist = w_l_bl_dist
        self.w_l_gb_align = w_l_gb_align
        self.w_l_gb_dist = w_l_gb_dist
        self.w_l_lt_dist = w_l_lt_dist
        self.w_l_spd = w_l_spd
        self.w_l_omg = w_l_omg
        self.w_l_acc = w_l_acc

        self.w_g_bl_align = w_g_bl_align
        self.w_g_bl_dist = w_g_bl_dist
        self.w_g_gb_align = w_g_gb_align
        self.w_g_gb_dist = w_g_gb_dist
        self.w_g_lt_dist = w_g_lt_dist
        self.w_g_spd = w_g_spd
        self.w_g_omg = w_g_omg
        self.w_g_acc = w_g_acc

        # Step 4: encode player ownership of the joint control vector.
        N = self.PARAMS.N_PLAYERS

        u_splits = N*[None]
        u_splits[self.PARAMS.BANDIT_PLAYER_IDX] = self.PARAMS.GAME_CTRL.NU_BANDIT
        u_splits[self.PARAMS.LADY_PLAYER_IDX] = self.PARAMS.GAME_CTRL.NU_LADY
        u_splits[self.PARAMS.GUARD_PLAYER_IDX] = self.PARAMS.GAME_CTRL.NU_GUARD

        # Step 3: define one running cost per player.
        #
        # The lambdas close over scalar weights, then call static methods so
        # JAX does not trace the whole Python object.
        costs = N*[None]
        costs[self.PARAMS.BANDIT_PLAYER_IDX] = PlayerCostSpecContinuous(
            running = lambda t, x, u: LadyBanditGuardNonlinear.bandit_cost(t, x, u, 
                params=self.PARAMS, 
                px_target=self.px_target,
                py_target=self.py_target,
                v_b_cruise=self.v_b_cruise,
                w_b_bl_align = self.w_b_bl_align,
                w_b_bl_dist = self.w_b_bl_dist,
                w_b_gb_align = self.w_b_gb_align,
                w_b_gb_dist = self.w_b_gb_dist,
                w_b_lt_dist = self.w_b_lt_dist,
                w_b_spd = self.w_b_spd,
                w_b_omg = self.w_b_omg,
                w_b_acc = self.w_b_acc
            )
        )
        
        costs[self.PARAMS.LADY_PLAYER_IDX] = PlayerCostSpecContinuous(
            running = lambda t, x, u: LadyBanditGuardNonlinear.lady_cost(t, x, u, 
                params=self.PARAMS, 
                px_target=self.px_target,
                py_target=self.py_target,
                v_l_cruise = self.v_l_cruise,
                w_l_bl_align = self.w_l_bl_align,
                w_l_bl_dist = self.w_l_bl_dist,
                w_l_gb_align = self.w_l_gb_align,
                w_l_gb_dist = self.w_l_gb_dist,
                w_l_lt_dist = self.w_l_lt_dist,
                w_l_spd = self.w_l_spd,
                w_l_omg = self.w_l_omg,
                w_l_acc = self.w_l_acc
            )
        )
        
        costs[self.PARAMS.GUARD_PLAYER_IDX] = PlayerCostSpecContinuous(
            running = lambda t, x, u: LadyBanditGuardNonlinear.guard_cost(t, x, u, 
                params=self.PARAMS, 
                px_target=self.px_target,
                py_target=self.py_target,
                v_g_cruise = self.v_g_cruise,
                w_g_bl_align = self.w_g_bl_align,
                w_g_bl_dist = self.w_g_bl_dist,
                w_g_gb_align = self.w_g_gb_align,
                w_g_gb_dist = self.w_g_gb_dist,
                w_g_lt_dist = self.w_g_lt_dist,
                w_g_spd = self.w_g_spd,
                w_g_omg = self.w_g_omg,
                w_g_acc = self.w_g_acc
            )
        )
        
        # Step 2: define the joint nonlinear dynamics.
        cs = SampledContinuousSystemType1(
            tg = tg,
            nx = self.PARAMS.GAME_STATE.NX,
            nu = self.PARAMS.GAME_CTRL.NU,
            dynamics = lambda t, x, u: LadyBanditGuardNonlinear.dynamics(t, x, u, self.PARAMS)
        )
        
        # Step 5: build the nonlinear IR game.
        self.game = NonlinearGameType1(
            cs = cs,
            N = N,
            costs = costs,
            u_splits = jnp.asarray(u_splits)
        )

    @staticmethod
    def dynamics(t:float, x:jnp.ndarray, u:jnp.ndarray, params:SimpleNamespace) -> jnp.ndarray:
        """
        equations of motion for 3x 2D aircrafts (i.e. 4D unicycle)

        Arguments:
        - t : float
            timestamp at which joint state derivative is computed
        - x : jnp.ndarray of shape (n,)
            joint state vector at which point joint state derivative is computed
        - u : jnp.ndarray of shape (m,)
            joint control vector at which point joint state derivative is computed

        Returns
        - dxdt : jnp.ndarray of shape (n,)
            joint state time derivative at point (t, x, u)
        """

        dxdt = jnp.zeros((params.GAME_STATE.NX,))

        # Bandit Dynamics
        i_px_b = params.GAME_STATE.I_BANDIT_PX
        i_py_b = params.GAME_STATE.I_BANDIT_PY
        i_th_b = params.GAME_STATE.I_BANDIT_TH
        i_vt_b = params.GAME_STATE.I_BANDIT_VT 
        i_dth_b = params.GAME_CTRL.I_BANDIT_DTH
        i_dvt_b = params.GAME_CTRL.I_BANDIT_DVT
        dxdt = dxdt.at[i_px_b].set(x[i_vt_b] * jnp.cos(x[i_th_b]))
        dxdt = dxdt.at[i_py_b].set(x[i_vt_b] * jnp.sin(x[i_th_b]))
        dxdt = dxdt.at[i_th_b].set(u[i_dth_b])
        dxdt = dxdt.at[i_vt_b].set(u[i_dvt_b])

        # Lady Dynamics
        i_px_l = params.GAME_STATE.I_LADY_PX
        i_py_l = params.GAME_STATE.I_LADY_PY
        i_th_l = params.GAME_STATE.I_LADY_TH
        i_vt_l = params.GAME_STATE.I_LADY_VT 
        i_dth_l = params.GAME_CTRL.I_LADY_DTH
        i_dvt_l = params.GAME_CTRL.I_LADY_DVT
        dxdt = dxdt.at[i_px_l].set(x[i_vt_l] * jnp.cos(x[i_th_l]))
        dxdt = dxdt.at[i_py_l].set(x[i_vt_l] * jnp.sin(x[i_th_l]))
        dxdt = dxdt.at[i_th_l].set(u[i_dth_l])
        dxdt = dxdt.at[i_vt_l].set(u[i_dvt_l])

        # Guard Dynamics
        i_px_g = params.GAME_STATE.I_GUARD_PX
        i_py_g = params.GAME_STATE.I_GUARD_PY
        i_th_g = params.GAME_STATE.I_GUARD_TH
        i_vt_g = params.GAME_STATE.I_GUARD_VT 
        i_dth_g = params.GAME_CTRL.I_GUARD_DTH
        i_dvt_g = params.GAME_CTRL.I_GUARD_DVT
        dxdt = dxdt.at[i_px_g].set(x[i_vt_g] * jnp.cos(x[i_th_g]))
        dxdt = dxdt.at[i_py_g].set(x[i_vt_g] * jnp.sin(x[i_th_g]))
        dxdt = dxdt.at[i_th_g].set(u[i_dth_g])
        dxdt = dxdt.at[i_vt_g].set(u[i_dvt_g])

        return dxdt
    
    @staticmethod
    def bandit_cost(
        t:float, x:jnp.ndarray, u:jnp.ndarray, 
        params:SimpleNamespace,
        px_target : float,
        py_target : float,
        v_b_cruise : float,
        w_b_bl_align : float,
        w_b_bl_dist : float,
        w_b_gb_align : float,
        w_b_gb_dist : float,
        w_b_lt_dist : float,
        w_b_spd : float,
        w_b_omg : float,
        w_b_acc : float,
    ) -> float:
        """
        Bandit's cost function

        Note that this is a staticmethod so that JAX doesn't try to trace object state in self

        Arguments:
        - t : float
            timestamp at which cost is computed
        - x : jnp.ndarray of shape (n,)
            joint state vector at which point cost is computed
        - u : jnp.ndarray of shape (m,)
            joint control vector at which point cost is computed
        - params : SimpleNamespace
            parameters that describe the structure of the game object
            e.g. indices of variables in the game state vector
        - v_b_cruise (float): bandit's target cruise speed [m/s]
        - px_target (float): x-position of lady's target [m]
        - py_target (float): y-position of lady's target [m]
        - w_b_bl_align (float): bandit's cost weight for aligning bandit heading to lady
        - w_b_bl_dist (float): bandit's cost weight for minimizing bandit distance to lady
        - w_b_gb_align (float): bandit's cost weight for misaligning guard heading to bandit 
        - w_b_gb_dist (float): bandit's cost weight for maximizing guard distance to bandit
        - w_b_lt_dist (float): bandit's cost weight for maximizing lady distance to target
        - w_b_spd (float): bandit's cost weight on deviation from cruise speed
        - w_b_omg (float): bandit's cost weight on control effort of turnrate
        - w_b_acc (float): bandit's cost weight on control effort of acceleration

        Returns
        - cost : float
        """

        cost = 0.0

        # Minimize alignment error (minimize negative cosine) between bandit heading and lady position relative to bandit
        assert w_b_bl_align >= 0, f"weight of bandit-lady alignment cost must be non-negativee, got {w_b_bl_align}"
        cost += - w_b_bl_align * LadyBanditGuardNonlinear.bandit_lady_alignment_cosine(t=t, x=x, u=u, params=params)

        # Minimize distance-squared between bandit and lady
        assert w_b_bl_dist >= 0, f"weight of bandit-lady proximity cost must be non-negative, got {w_b_bl_dist}"
        cost += w_b_bl_dist * LadyBanditGuardNonlinear.bandit_lady_proximity(t=t, x=x, u=u, params=params)

        # Maximize alignment error (minimize cosine) between guard heading and bandit position relative to guard
        assert w_b_gb_align >= 0, f"weight of guard-bandit alignment cost must be non-negative, got {w_b_gb_align}"
        cost += w_b_gb_align * LadyBanditGuardNonlinear.guard_bandit_alignment_cosine(t=t, x=x, u=u, params=params)

        # Maximize distance-squared (minimize negative distance-squared) between guard and bandit
        assert w_b_gb_dist >= 0, f"weight of guard-bandit proximity cost must be non-negative, got {w_b_gb_dist}"
        cost += - w_b_gb_dist * LadyBanditGuardNonlinear.guard_bandit_proximity(t=t, x=x, u=u, params=params)

        # Maximize distance-squard (minimize negative distance-squared) between lady and target
        assert w_b_lt_dist >= 0, f"weight of lady-target proximity cost must be non-negative, got {w_b_lt_dist}"
        cost += - w_b_lt_dist * LadyBanditGuardNonlinear.lady_target_deviation(
            t=t, x=x, u=u, params=params, 
            px_target=px_target, py_target=py_target)

        # Minimize bandit speed deviation from cruise speed
        assert v_b_cruise > 0, f"bandit cruise velocity must be positive, got {v_b_cruise}"
        assert w_b_spd >= 0, f"weight on bandit cruise speed deviation must be non-negative, got {w_b_spd}"
        cost += w_b_spd * LadyBanditGuardNonlinear.bandit_speed_deviation(t=t, x=x, u=u, params=params, v_target=v_b_cruise)

        # Minimize control effort on turnrate and linear acceleration
        assert w_b_omg >= 0, f"weight on bandit turnrate control effort must be non-negative, got {w_b_omg}"
        cost += w_b_omg * LadyBanditGuardNonlinear.bandit_turnrate_effort(t=t, x=x, u=u, params=params)

        assert w_b_acc >= 0, f"weight on bandit linear acceleration control effort must be non-negative, got {w_b_acc}"
        cost += w_b_acc * LadyBanditGuardNonlinear.bandit_accel_effort(t=t, x=x, u=u, params=params)

        return cost
    
    @staticmethod
    def lady_cost(
        t:float, x:jnp.ndarray, u:jnp.ndarray, 
        params:SimpleNamespace,
        px_target : float,
        py_target : float,
        v_l_cruise : float,
        w_l_bl_align : float,
        w_l_bl_dist : float,
        w_l_gb_align : float,
        w_l_gb_dist : float,
        w_l_lt_dist : float,
        w_l_spd : float,
        w_l_omg : float,
        w_l_acc : float,
    ) -> float:
        """
        Lady's cost function

        Note that this is a staticmethod so that JAX doesn't try to trace object state in self

        Arguments:
        - t : float
            timestamp at which cost is computed
        - x : jnp.ndarray of shape (n,)
            joint state vector at which point cost is computed
        - u : jnp.ndarray of shape (m,)
            joint control vector at which point cost is computed
        - params : SimpleNamespace
            parameters that describe the structure of the game object
            e.g. indices of variables in the game state vector
        - px_target (float): x-position of lady's target [m]
        - py_target (float): y-position of lady's target [m]
        - v_l_cruise (float): lady's target cruise speed [m/s]
        - w_l_bl_align (float): lady's cost weight for misaligning bandit heading to lady
        - w_l_bl_dist (float): lady's cost weight for maximizing bandit distance to lady
        - w_l_gb_align (float): lady's cost weight for aligning guard heading to bandit 
        - w_l_gb_dist (float): lady's cost weight for minimizing guard distance to bandit
        - w_l_lt_dist (float): lady's cost weight for minimizing lady distance to target
        - w_l_spd (float): lady's cost weight on deviation from cruise speed
        - w_l_omg (float): lady's cost weight on control effort of turnrate
        - w_l_acc (float): lady's cost weight on control effort of acceleration

        Returns
        - cost : float
        """

        cost = 0.0

        # Maximize alignment error (minimize cosine) between bandit heading and lady position relative to bandit
        assert w_l_bl_align >= 0, f"weight of bandit-lady alignment cost must be non-negativee, got {w_l_bl_align}"
        cost += w_l_bl_align * LadyBanditGuardNonlinear.bandit_lady_alignment_cosine(t=t, x=x, u=u, params=params)

        # Maximize distance-squared between bandit and lady
        assert w_l_bl_dist >= 0, f"weight of bandit-lady proximity cost must be non-negative, got {w_l_bl_dist}"
        cost += - w_l_bl_dist * LadyBanditGuardNonlinear.bandit_lady_proximity(t=t, x=x, u=u, params=params)

        # Minimize alignment error (minimize negative cosine) between guard heading and bandit position relative to guard
        assert w_l_gb_align >= 0, f"weight of guard-bandit alignment cost must be non-negative, got {w_l_gb_align}"
        cost += - w_l_gb_align * LadyBanditGuardNonlinear.guard_bandit_alignment_cosine(t=t, x=x, u=u, params=params)

        # Minimize distance-squared (minimize distance-squared) between guard and bandit
        assert w_l_gb_dist >= 0, f"weight of guard-bandit proximity cost must be non-negative, got {w_l_gb_dist}"
        cost += w_l_gb_dist * LadyBanditGuardNonlinear.guard_bandit_proximity(t=t, x=x, u=u, params=params)

        # Minimize distance-squared between lady and target
        assert w_l_lt_dist >= 0, f"weight of lady-target proximity cost must be non-negative, got {w_l_lt_dist}"
        cost += w_l_lt_dist * LadyBanditGuardNonlinear.lady_target_deviation(
            t=t, x=x, u=u, params=params, 
            px_target=px_target, py_target=py_target)

        # Minimize lady speed deviation from cruise speed
        assert v_l_cruise > 0, f"lady cruise velocity must be positive, got {v_l_cruise}"
        assert w_l_spd >= 0, f"weight on lady cruise speed deviation must be non-negative, got {w_l_spd}"
        cost += w_l_spd * LadyBanditGuardNonlinear.lady_speed_deviation(t=t, x=x, u=u, params=params, v_target=v_l_cruise)

        # Minimize control effort on turnrate and linear acceleration
        assert w_l_omg >= 0, f"weight on bandit turnrate control effort must be non-negative, got {w_l_omg}"
        cost += w_l_omg * LadyBanditGuardNonlinear.lady_turnrate_effort(t=t, x=x, u=u, params=params)

        assert w_l_acc >= 0, f"weight on lady linear acceleration control effort must be non-negative, got {w_l_acc}"
        cost += w_l_acc * LadyBanditGuardNonlinear.lady_accel_effort(t=t, x=x, u=u, params=params)

        return cost
    
    @staticmethod
    def guard_cost(
        t:float, x:jnp.ndarray, u:jnp.ndarray, 
        params:SimpleNamespace,
        px_target : float,
        py_target : float,
        v_g_cruise : float,
        w_g_bl_align : float,
        w_g_bl_dist : float,
        w_g_gb_align : float,
        w_g_gb_dist : float,
        w_g_lt_dist : float,
        w_g_spd : float,
        w_g_omg : float,
        w_g_acc : float,
    ) -> float:
        """
        Guard's cost function

        Note that this is a staticmethod so that JAX doesn't try to trace object state in self

        Arguments:
        - t : float
            timestamp at which cost is computed
        - x : jnp.ndarray of shape (n,)
            joint state vector at which point cost is computed
        - u : jnp.ndarray of shape (m,)
            joint control vector at which point cost is computed
        - params : SimpleNamespace
            parameters that describe the structure of the game object
            e.g. indices of variables in the game state vector
        - px_target (float): x-position of lady's target [m]
        - py_target (float): y-position of lady's target [m]
        - v_g_cruise (float): guard's target cruise speed [m/s]
        - w_g_bl_align (float): guard's cost weight for misaligning bandit heading to lady
        - w_g_bl_dist (float): guard's cost weight for maximizing bandit distance to lady
        - w_g_gb_align (float): guard's cost weight for aligning guard heading to bandit 
        - w_g_gb_dist (float): guard's cost weight for minimizing guard distance to bandit
        - w_g_lt_dist (float): guard's cost weight for minimizing lady distance to target
        - w_g_spd (float): guard's cost weight on deviation from cruise speed
        - w_g_omg (float): guard's cost weight on control effort of turnrate
        - w_g_acc (float): guard's cost weight on control effort of acceleration

        Returns
        - cost : float
        """

        cost = 0.0

        # Maximize alignment error (minimize cosine) between bandit heading and lady position relative to bandit
        assert w_g_bl_align >= 0, f"weight of bandit-lady alignment cost must be non-negativee, got {w_g_bl_align}"
        cost += w_g_bl_align * LadyBanditGuardNonlinear.bandit_lady_alignment_cosine(t=t, x=x, u=u, params=params)

        # Maximize distance-squared between bandit and lady
        assert w_g_bl_dist >= 0, f"weight of bandit-lady proximity cost must be non-negative, got {w_g_bl_dist}"
        cost += - w_g_bl_dist * LadyBanditGuardNonlinear.bandit_lady_proximity(t=t, x=x, u=u, params=params)

        # Minimize alignment error (minimize negative cosine) between guard heading and bandit position relative to guard
        assert w_g_gb_align >= 0, f"weight of guard-bandit alignment cost must be non-negative, got {w_g_gb_align}"
        cost += - w_g_gb_align * LadyBanditGuardNonlinear.guard_bandit_alignment_cosine(t=t, x=x, u=u, params=params)

        # Minimize distance-squared (minimize distance-squared) between guard and bandit
        assert w_g_gb_dist >= 0, f"weight of guard-bandit proximity cost must be non-negative, got {w_g_gb_dist}"
        cost += w_g_gb_dist * LadyBanditGuardNonlinear.guard_bandit_proximity(t=t, x=x, u=u, params=params)

        # Minimize distance-squared between lady and target
        assert w_g_lt_dist >= 0, f"weight of lady-target proximity cost must be non-negative, got {w_g_lt_dist}"
        cost += w_g_lt_dist * LadyBanditGuardNonlinear.lady_target_deviation(
            t=t, x=x, u=u, params=params, 
            px_target=px_target, py_target=py_target)

        # Minimize guard speed deviation from cruise speed
        assert v_g_cruise > 0, f"guard cruise velocity must be positive, got {v_g_cruise}"
        assert w_g_spd >= 0, f"weight on guard cruise speed deviation must be non-negative, got {w_g_spd}"
        cost += w_g_spd * LadyBanditGuardNonlinear.guard_speed_deviation(t=t, x=x, u=u, params=params, v_target=v_g_cruise)

        # Minimize control effort on turnrate and linear acceleration
        assert w_g_omg >= 0, f"weight on guard turnrate control effort must be non-negative, got {w_g_omg}"
        cost += w_g_omg * LadyBanditGuardNonlinear.guard_turnrate_effort(t=t, x=x, u=u, params=params)

        assert w_g_acc >= 0, f"weight on guard linear acceleration control effort must be non-negative, got {w_g_acc}"
        cost += w_g_acc * LadyBanditGuardNonlinear.guard_accel_effort(t=t, x=x, u=u, params=params)

        return cost
    
    @staticmethod
    def bandit_lady_alignment_cosine(t:float, x:jnp.ndarray, u:jnp.ndarray, params:SimpleNamespace) -> float:
        """
        Compute Bandit-Lady alignment error for later use in cost computations

        Alignment error is computed as the cosine of the angle between the Guard's heading
        vector (i.e. velocity vector) and the Bandit's position relative to the guard. 
        A cosine of 1.0 means that these vectors are aligned, while a cosine of -1 means
        they are oppositely aligned.

        Note that this is a staticmethod so that JAX doesn't try to trace object state in self

        Args:
        - t : float
            timestamp at which cost is computed
        - x : jnp.ndarray of shape (n,)
            joint state vector at which point cost is computed
        - u : jnp.ndarray of shape (m,)
            joint control vector at which point cost is computed
        - params : SimpleNamespace
            parameters that describe the structure of the game object
            e.g. indices of variables in the game state vector

        Returns:
        - c : float
            cosine of angle between bandit heading (vel vector) and lady position relative to bandit
        """

        # unpack indices for ease of use
        i_px_b = params.GAME_STATE.I_BANDIT_PX
        i_py_b = params.GAME_STATE.I_BANDIT_PY
        i_th_b = params.GAME_STATE.I_BANDIT_TH
        i_vt_b = params.GAME_STATE.I_BANDIT_VT
        i_px_l = params.GAME_STATE.I_LADY_PX
        i_py_l = params.GAME_STATE.I_LADY_PY

        # position of lady (l) wrt bandit (b), expressed in world frame (w)
        px_l_b__w = x[i_px_l] - x[i_px_b]
        py_l_b__w = x[i_py_l] - x[i_py_b]
        p_l_b__w = jnp.array([px_l_b__w, py_l_b__w])
        p_l_b = jnp.linalg.norm(p_l_b__w)

        # heading of bandit (b) wrt world frame (w), expressed in world frame (w)
        vx_b_w__w = x[i_vt_b] * jnp.cos(x[i_th_b])
        vy_b_w__w = x[i_vt_b] * jnp.sin(x[i_th_b])
        v_b_w__w = jnp.array([vx_b_w__w, vy_b_w__w])
        v_b_w = jnp.linalg.norm(v_b_w__w)

        # # Note: conditionals cause problems with JAX tracing with batching functions (e.g. vmap)
        # if jnp.isclose(p_l_b, 0.0):
        #     # if zero-distance between lady and bandit, set max cosine (min alignment error)
        #     # to avoid divide by zero
        #     return 1.0
        # elif jnp.isclose(v_b_w, 0.0):
        #     # if bandit is not moving, set min cosine (max alignment error)
        #     # to avoid divide by zero
        #     return -1.0
        # else:
        #     return jnp.dot(v_b_w__w, p_l_b__w) / (p_l_b * v_b_w)
        
        # add small number to denominator to avoid divide by zero
        return jnp.dot(v_b_w__w, p_l_b__w) / (p_l_b * v_b_w + _EPS)

    @staticmethod    
    def guard_bandit_alignment_cosine(t:float, x:jnp.ndarray, u:jnp.ndarray, params:SimpleNamespace) -> float:
        """
        Compute Guard-Bandit alignment error for later use in cost computations

        Alignment error is computed as the cosine of the angle between the Guard's heading
        vector (i.e. velocity vector) and the Bandit's position relative to the guard. 
        A cosine of 1.0 means that these vectors are aligned, while a cosine of -1 means
        they are oppositely aligned. 

        Note that this is a staticmethod so that JAX doesn't try to trace object state in self

        Args:
        - t : float
            timestamp at which cost is computed
        - x : jnp.ndarray of shape (n,)
            joint state vector at which point cost is computed
        - u : jnp.ndarray of shape (m,)
            joint control vector at which point cost is computed
        - params : SimpleNamespace
            parameters that describe the structure of the game object
            e.g. indices of variables in the game state vector

        Returns:
        - c : float
            cosine of angle between guard heading (vel vector) and bandit position relative to guard
        """

        # unpack indices for ease of use
        i_px_g = params.GAME_STATE.I_GUARD_PX
        i_py_g = params.GAME_STATE.I_GUARD_PY
        i_th_g = params.GAME_STATE.I_GUARD_TH
        i_vt_g = params.GAME_STATE.I_GUARD_VT
        i_px_b = params.GAME_STATE.I_BANDIT_PX
        i_py_b = params.GAME_STATE.I_BANDIT_PY

        # position of bandit (b) wrt guard (b), expressed in world frame (w)
        px_b_g__w = x[i_px_b] - x[i_px_g]
        py_b_g__w = x[i_py_b] - x[i_py_g]
        p_b_g__w = jnp.array([px_b_g__w, py_b_g__w])
        p_b_g = jnp.linalg.norm(p_b_g__w)

        # heading of guard (b) wrt world frame (w), expressed in world frame (w)
        vx_g_w__w = x[i_vt_g] * jnp.cos(x[i_th_g])
        vy_g_w__w = x[i_vt_g] * jnp.sin(x[i_th_g])
        v_g_w__w = jnp.array([vx_g_w__w, vy_g_w__w])
        v_g_w = jnp.linalg.norm(v_g_w__w)

        # # Note: conditionals cause problems with JAX tracing with batching functions (e.g. vmap)
        # if jnp.isclose(p_b_g, 0.0):
        #     # if zero-distance between bandit and guard, set max cosine (min alignment error)
        #     # to avoid divide by zero
        #     return 1.0
        # elif jnp.isclose(v_g_w, 0.0):
        #     # if guard is not moving, set min cosine (max alignment error)
        #     # to avoid divide by zero
        #     return -1.0
        # else:
        #     return jnp.dot(v_g_w__w, p_b_g__w) / (p_b_g * v_g_w)
        
        # add small number to denominator to avoid divide by zero
        return jnp.dot(v_g_w__w, p_b_g__w) / (p_b_g * v_g_w + _EPS)
        
    @staticmethod
    def bandit_lady_proximity(t:float, x:jnp.ndarray, u:jnp.ndarray, params:SimpleNamespace) -> float:
        """
        Compute Bandit-Lady square distance to later be used in cost computations

        Note that this is a staticmethod so that JAX doesn't try to trace object state in self

        Args:
        - t : float
            timestamp at which cost is computed
        - x : jnp.ndarray of shape (n,)
            joint state vector at which point cost is computed
        - u : jnp.ndarray of shape (m,)
            joint control vector at which point cost is computed
        - params : SimpleNamespace
            parameters that describe the structure of the game object
            e.g. indices of variables in the game state vector

        Returns:
        - c : float
            square distance between bandit and lady
        """

        # unpack indices for ease of use
        i_px_b = params.GAME_STATE.I_BANDIT_PX
        i_py_b = params.GAME_STATE.I_BANDIT_PY
        i_px_l = params.GAME_STATE.I_LADY_PX
        i_py_l = params.GAME_STATE.I_LADY_PY

        # position of lady (l) wrt bandit (b), expressed in world frame (w)
        px_l_b__w = x[i_px_l] - x[i_px_b]
        py_l_b__w = x[i_py_l] - x[i_py_b]
        
        return px_l_b__w * px_l_b__w + py_l_b__w * py_l_b__w
    
    @staticmethod
    def guard_bandit_proximity(t:float, x:jnp.ndarray, u:jnp.ndarray, params:SimpleNamespace) -> float:
        """
        Compute Guard-Bandit square distance to later be used in cost computations

        Note that this is a staticmethod so that JAX doesn't try to trace object state in self

        Args:
        - t : float
            timestamp at which cost is computed
        - x : jnp.ndarray of shape (n,)
            joint state vector at which point cost is computed
        - u : jnp.ndarray of shape (m,)
            joint control vector at which point cost is computed
        - params : SimpleNamespace
            parameters that describe the structure of the game object
            e.g. indices of variables in the game state vector

        Returns:
        - c : float
            square distance between guard and bandit
        """

        # unpack indices for ease of use
        i_px_g = params.GAME_STATE.I_GUARD_PX
        i_py_g = params.GAME_STATE.I_GUARD_PY
        i_px_b = params.GAME_STATE.I_BANDIT_PX
        i_py_b = params.GAME_STATE.I_BANDIT_PY

        # position of bandit (b) wrt guard (g), expressed in world frame (w)
        px_b_g__w = x[i_px_b] - x[i_px_g]
        py_b_g__w = x[i_py_b] - x[i_py_g]
        
        return px_b_g__w * px_b_g__w + py_b_g__w * py_b_g__w
    
    @staticmethod
    def bandit_speed_deviation(t:float, x:jnp.ndarray, u:jnp.ndarray, 
        params:SimpleNamespace, 
        v_target:float) -> float:
        """
        Compute Bandit's speed deviation from a target cruise speed to later be used in cost computations

        Note that this is a staticmethod so that JAX doesn't try to trace object state in self

        Args:
        - t : float
            timestamp at which cost is computed
        - x : jnp.ndarray of shape (n,)
            joint state vector at which point cost is computed
        - u : jnp.ndarray of shape (m,)
            joint control vector at which point cost is computed
        - params : SimpleNamespace
            parameters that describe the structure of the game object
            e.g. indices of variables in the game state vector
        - v_target : float [m/s]
            target cruise speed (inertial) bandit is trying to match

        Returns:
        - c : float
            square speed deviation from target speed
        """

        # get speed of bandit
        v_b_w = x[params.GAME_STATE.I_BANDIT_VT]
        
        return (v_b_w - v_target)**2
    
    @staticmethod
    def lady_speed_deviation(t:float, x:jnp.ndarray, u:jnp.ndarray, 
        params:SimpleNamespace, 
        v_target:float) -> float:
        """
        Compute Lady's speed deviation from a target cruise speed to later be used in cost computations

        Note that this is a staticmethod so that JAX doesn't try to trace object state in self

        Args:
        - t : float
            timestamp at which cost is computed
        - x : jnp.ndarray of shape (n,)
            joint state vector at which point cost is computed
        - u : jnp.ndarray of shape (m,)
            joint control vector at which point cost is computed
        - params : SimpleNamespace
            parameters that describe the structure of the game object
            e.g. indices of variables in the game state vector
        - v_target : float [m/s]
            target cruise speed (inertial) lady is trying to match

        Returns:
        - c : float
            square speed deviation from target speed
        """

        # get speed of lady
        v_l_w = x[params.GAME_STATE.I_LADY_VT]
        
        return (v_l_w - v_target)**2
    
    @staticmethod
    def guard_speed_deviation(t:float, x:jnp.ndarray, u:jnp.ndarray, 
        params:SimpleNamespace, 
        v_target:float) -> float:
        """
        Compute Guard's speed deviation from a target cruise speed to later be used in cost computations

        Note that this is a staticmethod so that JAX doesn't try to trace object state in self

        Args:
        - t : float
            timestamp at which cost is computed
        - x : jnp.ndarray of shape (n,)
            joint state vector at which point cost is computed
        - u : jnp.ndarray of shape (m,)
            joint control vector at which point cost is computed
        - params : SimpleNamespace
            parameters that describe the structure of the game object
            e.g. indices of variables in the game state vector
        - v_target : float [m/s]
            target cruise speed (inertial) guard is trying to match

        Returns:
        - c : float
            square speed deviation from target speed
        """

        # get speed of guard
        v_g_w = x[params.GAME_STATE.I_GUARD_VT]
        
        return (v_g_w - v_target)**2
    
    @staticmethod
    def bandit_turnrate_effort(t:float, x:jnp.ndarray, u:jnp.ndarray, 
        params:SimpleNamespace) -> float:
        """
        Compute Bandit's control effort on turn rate to later be used in cost computations

        Note that this is a staticmethod so that JAX doesn't try to trace object state in self

        Args:
        - t : float
            timestamp at which cost is computed
        - x : jnp.ndarray of shape (n,)
            joint state vector at which point cost is computed
        - u : jnp.ndarray of shape (m,)
            joint control vector at which point cost is computed
        - params : SimpleNamespace
            parameters that describe the structure of the game object
            e.g. indices of variables in the game state vector

        Returns:
        - c : float
            turnrate squared
        """

        # get turnrate of bandit with respect to world frame
        omg_b_w = u[params.GAME_CTRL.I_BANDIT_DTH]
        
        return omg_b_w * omg_b_w
    
    @staticmethod
    def lady_turnrate_effort(t:float, x:jnp.ndarray, u:jnp.ndarray, 
        params:SimpleNamespace) -> float:
        """
        Compute Lady's control effort on turn rate to later be used in cost computations

        Note that this is a staticmethod so that JAX doesn't try to trace object state in self

        Args:
        - t : float
            timestamp at which cost is computed
        - x : jnp.ndarray of shape (n,)
            joint state vector at which point cost is computed
        - u : jnp.ndarray of shape (m,)
            joint control vector at which point cost is computed
        - params : SimpleNamespace
            parameters that describe the structure of the game object
            e.g. indices of variables in the game state vector

        Returns:
        - c : float
            turnrate squared
        """

        # get turnrate of lady with respect to world frame
        omg_l_w = u[params.GAME_CTRL.I_LADY_DTH]
        
        return omg_l_w * omg_l_w
    
    @staticmethod
    def guard_turnrate_effort(t:float, x:jnp.ndarray, u:jnp.ndarray, 
        params:SimpleNamespace) -> float:
        """
        Compute Guard's control effort on turn rate to later be used in cost computations

        Note that this is a staticmethod so that JAX doesn't try to trace object state in self

        Args:
        - t : float
            timestamp at which cost is computed
        - x : jnp.ndarray of shape (n,)
            joint state vector at which point cost is computed
        - u : jnp.ndarray of shape (m,)
            joint control vector at which point cost is computed
        - params : SimpleNamespace
            parameters that describe the structure of the game object
            e.g. indices of variables in the game state vector

        Returns:
        - c : float
            turnrate squared
        """

        # get turnrate of guard with respect to world frame
        omg_g_w = u[params.GAME_CTRL.I_GUARD_DTH]
        
        return omg_g_w * omg_g_w
    
    @staticmethod
    def bandit_accel_effort(t:float, x:jnp.ndarray, u:jnp.ndarray, 
        params:SimpleNamespace) -> float:
        """
        Compute Bandit's control effort on linear acceleration to later be used in cost computations

        Note that this is a staticmethod so that JAX doesn't try to trace object state in self

        Args:
        - t : float
            timestamp at which cost is computed
        - x : jnp.ndarray of shape (n,)
            joint state vector at which point cost is computed
        - u : jnp.ndarray of shape (m,)
            joint control vector at which point cost is computed
        - params : SimpleNamespace
            parameters that describe the structure of the game object
            e.g. indices of variables in the game state vector

        Returns:
        - c : float
            acceleration squared
        """

        # get linear acceleration of bandit with respect to world frame
        acc_b_w = u[params.GAME_CTRL.I_BANDIT_DVT]
        
        return acc_b_w * acc_b_w
    
    @staticmethod
    def lady_accel_effort(t:float, x:jnp.ndarray, u:jnp.ndarray, 
        params:SimpleNamespace) -> float:
        """
        Compute Lady's control effort on linear acceleration to later be used in cost computations

        Note that this is a staticmethod so that JAX doesn't try to trace object state in self

        Args:
        - t : float
            timestamp at which cost is computed
        - x : jnp.ndarray of shape (n,)
            joint state vector at which point cost is computed
        - u : jnp.ndarray of shape (m,)
            joint control vector at which point cost is computed
        - params : SimpleNamespace
            parameters that describe the structure of the game object
            e.g. indices of variables in the game state vector

        Returns:
        - c : float
            acceleration squared
        """

        # get linear acceleration of lady with respect to world frame
        acc_l_w = u[params.GAME_CTRL.I_LADY_DVT]
        
        return acc_l_w * acc_l_w
    
    @staticmethod
    def guard_accel_effort(t:float, x:jnp.ndarray, u:jnp.ndarray, 
        params:SimpleNamespace) -> float:
        """
        Compute Guard's control effort on linear acceleration to later be used in cost computations

        Note that this is a staticmethod so that JAX doesn't try to trace object state in self

        Args:
        - t : float
            timestamp at which cost is computed
        - x : jnp.ndarray of shape (n,)
            joint state vector at which point cost is computed
        - u : jnp.ndarray of shape (m,)
            joint control vector at which point cost is computed
        - params : SimpleNamespace
            parameters that describe the structure of the game object
            e.g. indices of variables in the game state vector

        Returns:
        - c : float
            acceleration squared
        """

        # get linear acceleration of guard with respect to world frame
        acc_g_w = u[params.GAME_CTRL.I_GUARD_DVT]
        
        return acc_g_w * acc_g_w


    @staticmethod
    def lady_target_deviation(
        t:float, x:jnp.ndarray, u:jnp.ndarray, 
        params:SimpleNamespace, 
        px_target:float, 
        py_target:float) -> float:
        """
        Compute Lady's position deviation from a target position to later be used in cost computations

        Note that this is a staticmethod so that JAX doesn't try to trace object state in self

        Args:
        - t : float
            timestamp at which cost is computed
        - x : jnp.ndarray of shape (n,)
            joint state vector at which point cost is computed
        - u : jnp.ndarray of shape (m,)
            joint control vector at which point cost is computed
        - params : SimpleNamespace
            parameters that describe the structure of the game object
            e.g. indices of variables in the game state vector
        - px_target : float [m]
            x-position of target in world frame
        - py_target : float [m]
            y-position of target in world frame

        Returns:
        - c : float
            squared distance between lady and target position
        """

        # unpack indices for ease of use
        i_px_l = params.GAME_STATE.I_LADY_PX
        i_py_l = params.GAME_STATE.I_LADY_PY

        # get position of target (l) relative to the target (t) in world frame (w)
        px_l_t__w = x[i_px_l] - px_target
        py_l_t__w = x[i_py_l] - py_target
        
        return px_l_t__w * px_l_t__w + py_l_t__w * py_l_t__w


def initial_strategy(game: NonlinearGameType1) -> FixedStepAffineStrategies:
    """
    Build the initial affine strategy used to seed iLQ.

    The iLQ solver improves a local operating point. For an example, a zero
    strategy is a readable starting point: all players initially apply zero
    turn-rate and acceleration commands, then the solver iteratively refines
    both the trajectory and feedback strategy.
    """
    P = jnp.zeros((game.nsteps, game.nu, game.nx))
    alpha = jnp.zeros((game.nsteps, game.nu))
    return FixedStepAffineStrategies(tg=game.tg, P=P, alpha=alpha)


def solve_example():
    """
    Build, initialize, and solve the nonlinear Lady-Bandit-Guard IR example.

    Returns
    -------
    tuple
        ``(lbg, converged, trajectory, strategy)`` where ``lbg`` is the
        example wrapper, ``trajectory`` is the final operating trajectory, and
        ``strategy`` is the feedback strategy returned by the iLQ solver.
    """
    lbg = LadyBanditGuardNonlinear()
    init_strat = initial_strategy(lbg.game)
    init_traj = propagate_system_trajectory(
        lbg.game.cs,
        x0=DEFAULT_INITIAL_STATE,
        strategy=init_strat,
    )

    converged, trajectory, strategy = solve_ilqgame_feedback(
        lbg.game,
        DEFAULT_INITIAL_STATE,
        init_traj=init_traj,
        init_strat=init_strat,
        backtrack_max_iters=10,
    )

    return lbg, converged, trajectory, strategy


def main() -> None:
    lbg, converged, trajectory, strategy = solve_example()

    xs = trajectory.xs
    us = trajectory.us
    p = lbg.PARAMS
    bandit_pos_idx = jnp.asarray([
        p.GAME_STATE.I_BANDIT_PX,
        p.GAME_STATE.I_BANDIT_PY,
    ])
    lady_pos_idx = jnp.asarray([
        p.GAME_STATE.I_LADY_PX,
        p.GAME_STATE.I_LADY_PY,
    ])
    guard_pos_idx = jnp.asarray([
        p.GAME_STATE.I_GUARD_PX,
        p.GAME_STATE.I_GUARD_PY,
    ])

    print(
        format_ir_feedback_summary(
            "IR Solve Summary",
            solver="ilq",
            converged=converged,
            trajectory=trajectory,
            strategy=strategy,
        )
    )
    print("\n=== example-specific checks ===")
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


if __name__ == "__main__":
    main()
