# Copyright 2026 MIT Lincoln Laboratory
# SPDX-License-Identifier: MIT

"""
Advanced example: a constrained two-player integrator game in IR form.

This is the lower-level companion to ``constrained_integrators.py``. It skips
the frontend factories and constructs the augmented-Lagrangian (AL) solver's
intermediate representation directly:

    1. define a time grid
    2. define joint continuous-time dynamics
    3. define one local-control running cost per player
    4. define stagewise inequality constraints
    5. build the constrained nonlinear IR game
    6. seed the primal-dual trajectory and AL state
    7. solve with the AL solver

Problem
-------
Joint state:

    x = [p1, p2]

Joint control:

    u = [u1, u2]

Dynamics:

    p_dot_i = u_i

Each player independently wants to move toward its own goal while paying
control effort. The only constraints are simple control bounds:

    |u_i| <= u_max

This keeps the example focused on the AL-specific IR objects:
``FixedStepPrimalDualTrajectory``, ``JointAugmentedLagrangianState``, and
``GameConstraintGridMap``.
"""

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


def build_ir_constrained_integrator_game(
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
    # Step 1: define the time grid and dimensions.
    tg = systypes.TimeGrid(nt=nt, dt=dt, t0=0.0)
    N = 2
    nx = 2
    nu = 2
    u_splits = jnp.array([1, 1], dtype=jnp.int32)
    K = nt - 1

    # Step 2: define the joint continuous-time dynamics.
    #
    # The AL IR uses a joint dynamics callable, just like the nonlinear iLQ
    # examples. Player ownership of control entries is encoded separately by
    # ``u_splits``.
    def f_cont(t, x, u):
        # x: (2,), u: (2,)
        return jnp.array([u[u_i(0)], u[u_i(1)]], dtype=x.dtype)

    cs = systypes.SampledContinuousSystemType1(tg=tg, dynamics=f_cont, nx=nx, nu=nu)

    p_star = jnp.array([p_goal[0], p_goal[1]], dtype=dtype)

    # Step 3: define one local-control running cost per player.
    #
    # ``NonlinearGameType2`` expects AL costs to state whether they consume
    # local or joint controls. Here each player cost sees only ``u_i``.
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

    # Step 4: define stagewise inequality constraints.
    #
    # Inequalities are represented in standard form c(t, x, u) <= 0. The box
    # bounds |u_i| <= u_max therefore become two scalar inequalities per
    # player at each control interval.
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

    # Step 5: build the constrained nonlinear IR game.
    nlgame = gametypes.NonlinearGameType2(
        cs=cs,
        N=N,
        costs=costs,
        constraints=constraints,
        u_splits=u_splits,
    )

    # Step 6a: seed the primal trajectory with a straight-line control guess.
    x0 = jnp.array([p0[0], p0[1]], dtype=dtype)
    T = float((nt - 1) * dt)

    # Constant velocity to each goal, clipped to satisfy the control bounds.
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

    # Dynamics multipliers start at zero. The AL/Newton iterations update them.
    ls0 = jnp.zeros((K, N, nx), dtype=dtype)

    op0 = trajtypes.FixedStepPrimalDualTrajectory(tg=tg, xs=xs0, us=us0, ls=ls0)

    # Step 6b: seed the AL state with lambda=0 and rho=1.
    #
    # This makes the first outer iteration behave like a modest penalty
    # method. Later outer iterations update lambda and may increase rho.
    alstate0 = altypes.JointAugmentedLagrangianState(
        lam_ineq=jnp.zeros((constraints.nc_ineq,), dtype=dtype),
        rho_ineq=jnp.ones((constraints.nc_ineq,), dtype=dtype),
        lam_eq=jnp.zeros((constraints.nc_eq,), dtype=dtype),
        rho_eq=jnp.ones((constraints.nc_eq,), dtype=dtype),
    )

    return nlgame, op0, alstate0


def solve_example():
    """
    Build and solve the constrained integrator AL IR example.
    """
    nlgame, op0, alstate0 = build_ir_constrained_integrator_game(
        nt=31,
        dt=0.1,
        u_max=2.0,
        q=2.0,
        r=0.05,
        p0=(0.0, 1.0),
        p_goal=(4.0, -2.0),
        dtype=jnp.float32,
    )

    # Step 7: solve with the augmented-Lagrangian solver.
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

    return nlgame, op_out, al_out, diag


def main():
    nlgame, op_out, al_out, diag = solve_example()

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
