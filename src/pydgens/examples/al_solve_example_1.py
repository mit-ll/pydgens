# Copyright 2026 MIT Lincoln Laboratory
# SPDX-License-Identifier: MIT

"""
Example 1: Two-player single-integrator, running-cost-only, control bounds.

This is meant to be the "hello world" for the AL solver stack:
- multiple players (N=2)
- non-trivial horizon (nt > 2)
- dynamics feasibility (handled via μ in the primal-dual trajectory)
- inequality constraints (control bounds) handled via AL (λ, ρ)
- no state bounds, no coupling constraints (keep it vanilla)

Dynamics
--------
Joint state:  x = [p1, p2]
Joint control u = [u1, u2]
Continuous-time dynamics:  ṗ_i = u_i
Discretized internally (e.g. RK2):  p_{k+1} ≈ p_k + dt * u_k

Costs (per player, LOCAL control)
---------------------------------
Running only (terminal cost = 0):
  l_i(t, x, u_i) = 0.5*q*(p_i - p_i_goal)^2 + 0.5*r*u_i^2

Constraints
-----------
Control bounds per stage:
  |u_i| <= u_max   (encoded as 2 inequalities per player)

Intuition / expected behavior
-----------------------------
- Each player independently pushes toward its goal.
- If u_max is high enough, you'll see a smooth (roughly proportional) policy.
- If u_max is tight, you'll see saturation at ±u_max early on.

Note on AL init (λ=0, ρ=1)
--------------------------
A common starting point: λ=0 and modest ρ (often 1) makes the first outer iteration behave
like a simple penalty method; λ is then “learned” via dual ascent, and ρ increases if needed.
"""

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


# Joint state indices: x = [p1, p2]
def x_p(i: int) -> int:
    return i


# Joint control indices: u = [u1, u2]
def u_i(i: int) -> int:
    return i


def make_game_two_player_single_integrator(
    *,
    nt: int = 31,
    dt: float = 0.1,
    u_max: float = 2.0,
    q: float = 2.0,
    r: float = 0.05,
    p0: Tuple[float, float] = (0.0, 1.0),
    p_goal: Tuple[float, float] = (4.0, -2.0),
    dtype=jnp.float32,
) -> Tuple[
    gametypes.NonlinearGameType2,
    trajtypes.FixedStepPrimalDualTrajectory,
    altypes.JointAugmentedLagrangianState,
]:
    tg = systypes.TimeGrid(nt=nt, dt=dt, t0=0.0)
    N = 2
    nx = 2
    nu = 2
    u_splits = jnp.array([1, 1], dtype=jnp.int32)
    K = nt - 1

    # Continuous-time dynamics: p_dot = u
    def f_cont(t, x, u):
        # x: (2,), u: (2,)
        return jnp.array([u[u_i(0)], u[u_i(1)]], dtype=x.dtype)

    cs = systypes.SampledContinuousSystemType1(tg=tg, dynamics=f_cont, nx=nx, nu=nu)

    p_star = jnp.array([p_goal[0], p_goal[1]], dtype=dtype)

    # Running cost must be LOCAL control domain per your NonlinearGameType2 rules
    def make_player_cost(i: int) -> costtypes.PlayerCostSpecContinuous:
        def running_i(t, x, u_local):
            # u_local shape (1,)
            pi = x[x_p(i)]
            ui = u_local[0]
            return 0.5 * (q * (pi - p_star[i]) ** 2 + r * (ui ** 2))

        def terminal_zero(t, x):
            return jnp.array(0.0, dtype=x.dtype)

        return costtypes.PlayerCostSpecContinuous(
            running=running_i,
            terminal=terminal_zero,
            control_domain=costtypes.ControlDomain.LOCAL,
            control_coupling=costtypes.ControlStructure.LOCAL_ONLY,
        )

    costs = [make_player_cost(0), make_player_cost(1)]

    # --- constraints: |u_i| <= u_max at all stages ---
    active_all = tuple(range(K))

    def u_box(t, x, u):
        u1 = u[u_i(0)]
        u2 = u[u_i(1)]
        return jnp.array(
            [u1 - u_max, -u1 - u_max, u2 - u_max, -u2 - u_max],
            dtype=u.dtype,
        )

    b_u = contypes.ConstraintBlockGridMap(
        tg=tg,
        func=u_box,
        cdim_out_step=4,
        active_steps=active_all,
        iseq=False,
        terminal=False,
    )

    constraints = contypes.GameConstraintGridMap(ineq_blocks=(b_u,), eq_blocks=())

    nlgame = gametypes.NonlinearGameType2(
        cs=cs,
        N=N,
        costs=costs,
        constraints=constraints,
        u_splits=u_splits,
    )

    # --- initial guess: straight-line position interpolation + constant control guess ---
    x0 = jnp.array([p0[0], p0[1]], dtype=dtype)
    T = float((nt - 1) * dt)

    # “constant velocity to goal” guess (then clip to bounds)
    u_guess = (p_star - x0) / T
    u_guess = jnp.clip(u_guess, -u_max, u_max)

    # Build xs0 by integrating the guessed constant controls (discrete single-integrator)
    # p_{k+1} = p_k + dt*u_guess
    ps = [x0]
    xk = x0
    for _ in range(K):
        xk = xk + dt * u_guess
        ps.append(xk)
    xs0 = jnp.stack(ps, axis=0)  # (nt,2)

    # control guess is constant each stage
    us0 = jnp.tile(u_guess[None, :], (K, 1))  # (K,2)

    # dynamics multipliers μ init to zero
    ls0 = jnp.zeros((K, N, nx), dtype=dtype)

    op0 = trajtypes.FixedStepPrimalDualTrajectory(tg=tg, xs=xs0, us=us0, ls=ls0)

    # AL init: λ=0, ρ=1
    alstate0 = altypes.JointAugmentedLagrangianState(
        lam_ineq=jnp.zeros((constraints.nc_ineq,), dtype=dtype),
        rho_ineq=jnp.ones((constraints.nc_ineq,), dtype=dtype),
        lam_eq=jnp.zeros((constraints.nc_eq,), dtype=dtype),
        rho_eq=jnp.ones((constraints.nc_eq,), dtype=dtype),
    )

    return nlgame, op0, alstate0


def main():
    nlgame, op0, alstate0 = make_game_two_player_single_integrator(
        nt=31,
        dt=0.1,
        u_max=2.0,
        q=2.0,
        r=0.05,
        p0=(0.0, 1.0),
        p_goal=(4.0, -2.0),
        dtype=jnp.float32,
    )

    # Optional debug logging
    logging.basicConfig(level=logging.INFO)  # configure handlers once
    logging.getLogger("pydgens.alsolver").setLevel(logging.DEBUG)   # enable debug logging for alsolve module
    alsolver.logger.debug("debug enabled")  # (optional) verify

    op_out, al_out, diag = alsolver.al_solve_autodiff(
        nlgame,
        op0,
        alstate0,
        discretize_method="rk2",
        ineq_activation="altro",
        max_iters=10,
        rho_increase=10.0,
        rho_max=1e6,
        # ALGAMES-like tolerances (tune later)
        opt_tol=1e-3,
        dyn_tol=1e-4,
        ineq_tol=1e-4,
        eq_tol=1e-4,
        # inner
        newton_max_iters=20,
        newton_max_rejects=6,
        newton_step_rtol=1e-7,
        newton_step_atol=1e-8,
        reg_init=0.0,
        reg_min_on_fail=1e-6,
        reg_increase=10.0,
        reg_max=1e6,
        reg_max_iters=32,
        ls_alpha0=1.0,
        ls_tau=0.5,
        ls_beta=0.25,
        ls_max_iters=25,
        normkind="l1_mean",
    )

    print("\n=== AL solve summary ===")
    print(f"converged: {diag.converged}  reason: {diag.reason}  iters: {diag.iters}")
    if diag.history:
        print(f"last outer diag:\n  {diag.history[-1]}")

    xs = op_out.xs
    us = op_out.us
    nt = xs.shape[0]
    K = nt - 1
    mid = nt // 2

    def row(k: int) -> str:
        p1, p2 = map(float, xs[k])
        if k < K:
            u1, u2 = map(float, us[k])
        else:
            u1, u2 = float("nan"), float("nan")
        return f"k={k:02d}  p1={p1:+.3f} p2={p2:+.3f}  u1={u1:+.3f} u2={u2:+.3f}"

    print("\n=== trajectory sample ===")
    print(row(0))
    print(row(mid))
    print(row(nt - 2))
    print(row(nt - 1))

    # Quick checks
    u1_max = float(jnp.max(jnp.abs(us[:, 0]))) if us.size else 0.0
    u2_max = float(jnp.max(jnp.abs(us[:, 1]))) if us.size else 0.0
    print("\n=== quick checks ===")
    print(f"max |u1| over stages: {u1_max:.3f}")
    print(f"max |u2| over stages: {u2_max:.3f}")

    # Compare to “constant to goal” intuition
    dt = float(op_out.tg.dt)
    T = float((nt - 1) * dt)
    p0 = xs[0]
    pT = xs[-1]
    u_const = (pT - p0) / T
    print("\n=== intuition check ===")
    print(f"implied constant u from achieved endpoints: u1={float(u_const[0]):+.3f}, u2={float(u_const[1]):+.3f}")
    print("(If bounds are active, you may see saturation; otherwise u should be moderate and smooth.)")


if __name__ == "__main__":
    main()