# Copyright 2026 MIT Lincoln Laboratory
# SPDX-License-Identifier: MIT

# Two-player, 1D double integrators seeking terminal states while maintaining
# a separation requirement

from __future__ import annotations

import logging

from typing import Tuple

import jax.numpy as jnp

import pydgens.ir.systemtypes as systypes
import pydgens.ir.gametypes as gametypes
import pydgens.ir.trajectorytypes as trajtypes
import pydgens.ir.constrainttypes as contypes
import pydgens.ir.costtypes as costtypes
import pydgens.ir.altypes as altypes
import pydgens.solvers.alsolver as alsolver

# ---------------------------------------------------------------------
# Small indexing helpers (so constraints/costs stay readable)
# Joint state: x = [p1, v1, p2, v2]
# Joint control: u = [a1, a2]
# ---------------------------------------------------------------------

def x_p(i: int) -> int:
    return 2 * i + 0

def x_v(i: int) -> int:
    return 2 * i + 1

def u_a(i: int) -> int:
    return i

# ---------------------------------------------------------------------
# Problem construction
# ---------------------------------------------------------------------

def make_game_two_player_merge(
    *,
    nt: int = 21,
    dt: float = 0.1,
    a_max: float = 2.0,
    v_max: float = 6.0,
    d_min: float = 1.0,
    p1_0: float = 0.0,
    dtype=jnp.float32,
) -> Tuple[gametypes.NonlinearGameType2, trajtypes.FixedStepPrimalDualTrajectory, altypes.JointAugmentedLagrangianState]:
    """
    Two-player 1D merge-like toy:
      - each player: (p_i, v_i), control a_i
      - objective: reach terminal goals while using small effort
      - constraints: |a_i|<=a_max, 0<=v_i<=v_max, p2-p1>=d_min, terminal equality goals
    """
    # time grid (time step size and horizon) over which problem is defined
    tg = systypes.TimeGrid(nt=nt, dt=dt, t0=0.0)
    N = 2   # number of players
    nx = 4  # dimension of joint state space
    nu = 2  # dimension of joint control space
    u_splits = jnp.array([1, 1], dtype=jnp.int32)   # which player controls which slice of joint control space

    # initial conditions that ensure feasibility
    p1_0 = 0.0  # player 1 initial position
    v1_0 = 0.0  # player 1 initial velocity
    p2_0 = d_min + 0.2  # player 2 initial position
    v2_0 = 0.0  # player 2 initial velocity

    # target/goal conditions (may not be exactly feasible,
    # but evaluated as costs, not constraints)
    p1_f = 4.0
    v1_f = 4.0
    p2_f = p1_f + d_min + 0.2
    v2_f = 4.0

    # --------------------------
    # Continuous-time dynamics
    # xdot = f(t,x,u)
    # Joint state & control
    # --------------------------
    def f_cont(t, x, u):
        # x = [p1, v1, p2, v2], joint state vector
        # u = [a1, a2],         joint control vector
        p1, v1, p2, v2 = x
        a1, a2 = u
        return jnp.array([v1, a1, v2, a2], dtype=x.dtype)

    cs = systypes.SampledContinuousSystemType1(tg=tg, dynamics=f_cont, nx=nx, nu=nu)

    # --------------------------
    # Costs (LOCAL control, separable)
    # running_i(t, x, u_i) where u_i is shape (nu_i,)
    # terminal_i(t, x)
    # --------------------------
    # Targets: player 1 slightly behind player 2; both want moderate speed
    p_star = jnp.array([p1_f, p2_f], dtype=dtype)
    v_star = jnp.array([v1_f, v2_f], dtype=dtype)

    q_p = 1.0
    q_v = 0.2
    r_a = 0.05
    qf_p = 20.0
    qf_v = 2.0

    def make_player_cost(i: int) -> costtypes.PlayerCostSpecContinuous:
        def running_i(t, x, u_i):
            # u_i is (1,), local accel for player i
            pi = x[x_p(i)]
            vi = x[x_v(i)]
            ai = u_i[0]
            return 0.5 * (
                q_p * (pi - p_star[i]) ** 2 +
                q_v * (vi - v_star[i]) ** 2 +
                r_a * (ai) ** 2
            )

        def terminal_i(t, x):
            pi = x[x_p(i)]
            vi = x[x_v(i)]
            return 0.5 * (
                qf_p * (pi - p_star[i]) ** 2 +
                qf_v * (vi - v_star[i]) ** 2
            )

        # return costtypes.PlayerCostSpecContinuous(
        #     running=running_i,
        #     terminal=terminal_i,
        #     control_domain=costtypes.ControlDomain.LOCAL,
        #     control_coupling=costtypes.ControlCoupling.SEPARABLE,
        # )
        return costtypes.PlayerCostSpecContinuous(
            running=lambda t, x, u_i: 0.0,
            terminal=terminal_i,
            control_domain=costtypes.ControlDomain.LOCAL,
            control_coupling=costtypes.ControlStructure.LOCAL_ONLY,
        )

    costs = [make_player_cost(0), make_player_cost(1)]

    # --------------------------
    # Constraints via blocks
    # --------------------------
    # Helper: "all stage steps" are k=0..nt-2
    active_all = tuple(range(nt - 1))

    # 1) Control bounds |a_i| <= a_max for both players, every step.
    # c(t,x,u) <= 0
    def u_box(t, x, u):
        a1 = u[u_a(0)]
        a2 = u[u_a(1)]
        return jnp.array([
            a1 - a_max,
            -a1 - a_max,
            a2 - a_max,
            -a2 - a_max,
        ], dtype=u.dtype)

    b_u = contypes.ConstraintBlockGridMap(
        tg=tg, func=u_box, cdim_out_step=4, active_steps=active_all, iseq=False, terminal=False
    )

    # 2) Speed bounds 0 <= v_i <= v_max, every step.
    def v_box(t, x, u):
        v1 = x[x_v(0)]
        v2 = x[x_v(1)]
        return jnp.array([
            v1 - v_max,
            -v1,        # v1 >= 0  =>  -v1 <= 0
            v2 - v_max,
            -v2,
        ], dtype=x.dtype)

    b_v = contypes.ConstraintBlockGridMap(
        tg=tg, func=v_box, cdim_out_step=4, active_steps=active_all, iseq=False, terminal=False
    )

    # 3) Separation: p2 - p1 >= d_min  =>  d_min - (p2 - p1) <= 0
    def separation(t, x, u):
        p1 = x[x_p(0)]
        p2 = x[x_p(1)]
        return jnp.array([d_min - (p2 - p1)], dtype=x.dtype)

    b_sep = contypes.ConstraintBlockGridMap(
        tg=tg, func=separation, cdim_out_step=1, active_steps=active_all, iseq=False, terminal=False
    )

    # # 4) Terminal equalities: hit target positions/speeds at k=nt-1
    # # c(t,x) == 0
    # def terminal_goal(t, x):
    #     p1 = x[x_p(0)]
    #     v1 = x[x_v(0)]
    #     p2 = x[x_p(1)]
    #     v2 = x[x_v(1)]
    #     return jnp.array([
    #         p1 - p_star[0],
    #         v1 - v_star[0],
    #         p2 - p_star[1],
    #         v2 - v_star[1],
    #     ], dtype=x.dtype)

    # b_goal = contypes.ConstraintBlockGridMap(
    #     tg=tg, func=terminal_goal, cdim_out_step=4, active_steps=None, iseq=True, terminal=True
    # )

    constraints = contypes.GameConstraintGridMap(
        ineq_blocks=(b_u, b_v, b_sep),
        # eq_blocks=(b_goal,),
    )

    # --------------------------
    # Game
    # --------------------------
    nlgame = gametypes.NonlinearGameType2(
        cs=cs,
        N=N,
        costs=costs,
        constraints=constraints,
        u_splits=u_splits,
    )

    # --------------------------
    # Initial guess trajectory
    # --------------------------
    x0 = jnp.array([p1_0, v1_0, p2_0, v2_0], dtype=dtype)

    # crude guess: linearly ramp positions to targets, constant velocity guess
    ts = jnp.linspace(0.0, 1.0, nt, dtype=dtype)
    p1_guess = x0[x_p(0)] + ts * (p_star[0] - x0[x_p(0)])
    p2_guess = x0[x_p(1)] + ts * (p_star[1] - x0[x_p(1)])
    v1_guess = jnp.ones((nt,), dtype=dtype) * v_star[0]
    v2_guess = jnp.ones((nt,), dtype=dtype) * v_star[1]

    xs0 = jnp.stack([p1_guess, v1_guess, p2_guess, v2_guess], axis=1)  # (nt,nx)
    xs0 = xs0.at[0].set(x0)

    us0 = jnp.zeros((nt - 1, nu), dtype=dtype)         # (K,nu)
    ls0 = jnp.zeros((nt - 1, N, cs.nx), dtype=dtype)   # (K,N,nx)

    op0 = trajtypes.FixedStepPrimalDualTrajectory(tg=tg, xs=xs0, us=us0, ls=ls0)

    # --------------------------
    # Initial AL state (λ, ρ)
    # --------------------------
    # Initialize AL params with λ=0 and ρ=1: first outer iter behaves like a penalty method (esp. for violated ineqs);
    # then λ is learned via dual ascent and ρ is increased to enforce feasibility if needed.
    alstate0 = altypes.JointAugmentedLagrangianState(
        lam_ineq=jnp.zeros((constraints.nc_ineq,), dtype=dtype),
        rho_ineq=jnp.ones((constraints.nc_ineq,), dtype=dtype),
        lam_eq=jnp.zeros((constraints.nc_eq,), dtype=dtype),
        rho_eq=jnp.ones((constraints.nc_eq,), dtype=dtype),
    )

    return nlgame, op0, alstate0


def main():
    nlgame, op0, alstate0 = make_game_two_player_merge()

    # create logger to pass to solver
    logging.basicConfig(level=logging.INFO) # global config of root logger, to be run once
    logger = logging.getLogger("al_solver")
    logger.setLevel(logging.DEBUG)

    op_out, al_out, diag = alsolver.al_solve_autodiff(
        nlgame,
        op0,
        alstate0,
        discretize_method="rk2",
        ineq_activation="altro",
        max_iters=10,
        rho_increase=10.0,
        rho_max=1e6,
        # ALGAMES convergence tolerances (tune as needed)
        opt_tol=1e-3,
        dyn_tol=1e-4,
        ineq_tol=1e-4,
        eq_tol=1e-4,
        # inner-loop newton root finding controls
        newton_max_iters=20,
        newton_max_rejects=6,
        newton_step_rtol=1e-7,
        newton_step_atol=1e-8,
        # Tikhonov regularization parameters for linear system solver
        reg_init=0.0,
        reg_min_on_fail=1e-6,
        reg_increase=10.0,
        reg_max=1e6,
        reg_max_iters=32,
        # backtracking linesearch parameters
        ls_alpha0=1.0,
        ls_tau=0.5,
        ls_beta=0.25,
        ls_max_iters=25,
        # norm used for merit function and linesearching convergence checks
        normkind="l1_mean",
        # logger=logger
    )

    print("\n=== AL solve summary ===")
    print(f"converged: {diag.converged}  reason: {diag.reason}  iters: {diag.iters}")
    if diag.history:
        print(f"last outer diag:\n  {diag.history[-1]}")

    # Show a few trajectory rows (start/mid/end)
    xs = op_out.xs
    us = op_out.us
    nt = xs.shape[0]
    mid = nt // 2

    def row(k: int) -> str:
        p1, v1, p2, v2 = map(float, xs[k])
        if k < nt - 1:
            a1, a2 = map(float, us[k])
        else:
            a1, a2 = float("nan"), float("nan")
        return f"k={k:02d}  p1={p1:+.3f} v1={v1:+.3f}  p2={p2:+.3f} v2={v2:+.3f}  a1={a1:+.3f} a2={a2:+.3f}"

    print("\n=== trajectory sample ===")
    print(row(0))
    print(row(mid))
    print(row(nt - 2))
    print(row(nt - 1))

    # Quick feasibility sanity checks (not exhaustive)
    p_sep_min = float(jnp.min(xs[:-1, x_p(1)] - xs[:-1, x_p(0)]))
    v_min = float(jnp.min(jnp.array([xs[:-1, x_v(0)], xs[:-1, x_v(1)]])))
    print("\n=== quick checks ===")
    print(f"min separation over stages: {p_sep_min:.3f}")
    print(f"min speed over stages:      {v_min:.3f}")


if __name__ == "__main__":
    main()