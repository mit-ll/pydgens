# Copyright 2026 MIT Lincoln Laboratory
# SPDX-License-Identifier: MIT

"""
Showcase example: orbital Lady-Bandit-Guard as a linear-quadratic game.

The lady satellite is treated as the passive reference orbit. The bandit and
guard are modeled as chaser spacecraft in a right-handed NTW/Hill frame
relative to the lady:

    x = [
        p_bandit, v_bandit,
        p_guard,  v_guard,
    ]

where each position/velocity block is 3D. The controls are relative
accelerations in the same frame:

    u = [a_bandit, a_guard]

The game remains linear-quadratic:

    - CW/Hill dynamics are linear.
    - The bandit is rewarded for approaching the lady and moving the guard
      away from its target point on the bandit-lady line.
    - The guard is rewarded for placing itself near that same target point.
    - Both players pay quadratic maneuver effort.

The coupled distance terms require full state cost matrices, so this example
uses ``pdg.matrix_quadratic_cost(...)`` instead of the simpler diagonal
``pdg.quadratic_cost(...)`` helper.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

import pydgens as pdg


NX = 12
NU = 6

I_BANDIT_POS = slice(0, 3)
I_BANDIT_VEL = slice(3, 6)
I_GUARD_POS = slice(6, 9)
I_GUARD_VEL = slice(9, 12)

I_BANDIT_ACC = slice(0, 3)
I_GUARD_ACC = slice(3, 6)


@dataclass(frozen=True)
class OrbitalLBGWeights:
    """
    Cost parameters for the orbital LBG example.

    State weights may be negative because this is a general-sum LQ game. For
    example, the bandit's negative guard-line-target weight rewards increasing
    the guard's miss distance from its desired point on the bandit-lady line.
    """

    alpha_guard_target: float = 0.5
    bandit_lady_distance: float = 0.1
    bandit_guard_line_target: float = -0.25
    guard_line_target: float = 1.0
    guard_bandit_near_lady: float = -0.25
    bandit_control: float = 1.0e10
    guard_control: float = 1.0e10
    terminal_scale: float = 2.0


def cw_continuous_matrices(mean_motion: float):
    """
    Continuous-time Clohessy-Wiltshire matrices for one 3D chaser.

    State:
        [x, y, z, xdot, ydot, zdot]

    Control:
        [ax, ay, az]
    """
    n = float(mean_motion)

    A = jnp.array([
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [3.0 * n**2, 0.0, 0.0, 0.0, 2.0 * n, 0.0],
        [0.0, 0.0, 0.0, -2.0 * n, 0.0, 0.0],
        [0.0, 0.0, -(n**2), 0.0, 0.0, 0.0],
    ])

    B = jnp.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])

    return A, B


def orbital_lbg_continuous_matrices(mean_motion: float):
    """
    Stack two CW chasers: bandit and guard relative to the lady.
    """
    A_cw, B_cw = cw_continuous_matrices(mean_motion)

    A = jnp.zeros((NX, NX))
    A = A.at[0:6, 0:6].set(A_cw)
    A = A.at[6:12, 6:12].set(A_cw)

    B = jnp.zeros((NX, NU))
    B = B.at[0:6, 0:3].set(B_cw)
    B = B.at[6:12, 3:6].set(B_cw)

    return A, B


def _position_selection(rows: slice):
    S = jnp.zeros((3, NX))
    indices = jnp.arange(rows.start, rows.stop)
    return S.at[jnp.arange(3), indices].set(1.0)


def orbital_lbg_cost_matrices(weights: OrbitalLBGWeights):
    """
    Build coupled LQ state/control matrices for bandit and guard.

    The guard target term is

        ||p_guard - alpha p_bandit||^2

    where ``alpha=1`` means "meet the bandit" and ``alpha=0`` means "sit at
    the lady/reference origin."
    """
    S_pb = _position_selection(I_BANDIT_POS)
    S_pg = _position_selection(I_GUARD_POS)

    Q_bandit_lady = S_pb.T @ S_pb

    S_guard_line_target = S_pg - weights.alpha_guard_target * S_pb
    Q_guard_line_target = S_guard_line_target.T @ S_guard_line_target

    Q_bandit = (
        weights.bandit_lady_distance * Q_bandit_lady
        +
        weights.bandit_guard_line_target * Q_guard_line_target
    )

    Q_guard = (
        weights.guard_line_target * Q_guard_line_target
        +
        weights.guard_bandit_near_lady * Q_bandit_lady
    )

    R_bandit = jnp.zeros((NU, NU))
    R_bandit = R_bandit.at[jnp.arange(3), jnp.arange(3)].set(weights.bandit_control)

    R_guard = jnp.zeros((NU, NU))
    guard_acc_indices = jnp.arange(3, 6)
    R_guard = R_guard.at[guard_acc_indices, guard_acc_indices].set(weights.guard_control)

    return Q_bandit, R_bandit, Q_guard, R_guard


def build_satellite_lady_bandit_guard_game(
    *,
    nt: int = 61,
    dt: float = 30.0,
    mean_motion: float = 0.0011,
    weights: OrbitalLBGWeights = OrbitalLBGWeights(),
):
    """
    Build the frontend LQ orbital Lady-Bandit-Guard game.

    Returns
    -------
    tuple
        ``(game, x0, weights, player_bandit, player_guard)``.
    """
    # Positions are kilometers and velocities are kilometers/second in the
    # local CW frame. The scenario starts with the bandit closer to the lady
    # and the guard offset on the other side of the reference orbit.
    x0 = jnp.array([
        -3.0, -4.0, 0.2, 0.000, 0.002, 0.000,
        4.5, 3.0, -0.4, 0.000, -0.001, 0.000,
    ])
    # x0 = jnp.zeros((12,))

    tg = pdg.time_grid(
        nt=nt,
        dt=dt,
    )

    A, B = orbital_lbg_continuous_matrices(mean_motion)
    dynamics = pdg.linear_dynamics(
        A=A,
        B=B,
    )

    Q_bandit, R_bandit, Q_guard, R_guard = orbital_lbg_cost_matrices(weights)

    bandit_cost = pdg.matrix_quadratic_cost(
        nx=NX,
        nu=NU,
        state_matrix=Q_bandit,
        terminal_state_matrix=weights.terminal_scale * Q_bandit,
        control_matrix=R_bandit,
    )

    guard_cost = pdg.matrix_quadratic_cost(
        nx=NX,
        nu=NU,
        state_matrix=Q_guard,
        terminal_state_matrix=weights.terminal_scale * Q_guard,
        control_matrix=R_guard,
    )

    player_bandit = pdg.player(
        name="bandit",
        cost=bandit_cost,
        joint_ctrl_slice=I_BANDIT_ACC,
        state_view=range(I_BANDIT_POS.start, I_BANDIT_VEL.stop),
    )

    player_guard = pdg.player(
        name="guard",
        cost=guard_cost,
        joint_ctrl_slice=I_GUARD_ACC,
        state_view=range(I_GUARD_POS.start, I_GUARD_VEL.stop),
    )

    game = pdg.game(
        tg=tg,
        dynamics=dynamics,
        players=[player_bandit, player_guard],
    )

    return game, x0, weights, player_bandit, player_guard


def solve_example():
    """
    Build and solve the orbital LBG example.
    """
    game, x0, weights, player_bandit, player_guard = build_satellite_lady_bandit_guard_game()
    solution = pdg.solve(
        game,
        x0=x0,
        method="lq",
    )
    return game, x0, weights, player_bandit, player_guard, solution


def main() -> None:
    game, x0, weights, player_bandit, player_guard, solution = solve_example()

    states = solution.states
    controls = solution.joint_controls

    p_bandit = states[:, I_BANDIT_POS]
    p_guard = states[:, I_GUARD_POS]
    guard_target_error = p_guard - weights.alpha_guard_target * p_bandit

    bandit_lady_dist = jnp.linalg.norm(p_bandit, axis=1)
    bandit_guard_dist = jnp.linalg.norm(p_guard - p_bandit, axis=1)
    guard_target_dist = jnp.linalg.norm(guard_target_error, axis=1)

    bandit_acc = controls[:, player_bandit.joint_ctrl_slice]
    guard_acc = controls[:, player_guard.joint_ctrl_slice]

    print()
    print(solution.format_summary("Satellite Lady-Bandit-Guard"))

    print("\nADDED CHECKS:")
    print(f"initial bandit-lady distance: {float(bandit_lady_dist[0]):.3f} km")
    print(f"final bandit-lady distance:   {float(bandit_lady_dist[-1]):.3f} km")
    print(f"initial bandit-guard distance: {float(bandit_guard_dist[0]):.3f} km")
    print(f"final bandit-guard distance:   {float(bandit_guard_dist[-1]):.3f} km")
    print(f"final guard-line-target error: {float(guard_target_dist[-1]):.3f} km")
    print(f"max bandit accel: {float(jnp.max(jnp.linalg.norm(bandit_acc, axis=1))):.3e} km/s^2")
    print(f"max guard accel:  {float(jnp.max(jnp.linalg.norm(guard_acc, axis=1))):.3e} km/s^2")


if __name__ == "__main__":
    main()
