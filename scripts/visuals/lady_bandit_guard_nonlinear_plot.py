# Copyright 2026 MIT Lincoln Laboratory
# SPDX-License-Identifier: MIT

"""
Generate plots for the nonlinear Lady-Bandit-Guard IR example.

This script intentionally lives outside ``src/pydgens`` because plotting is a
documentation/visualization concern, not a core package dependency.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt

from pydgens.examples.ir_lady_bandit_guard_nonlinear import solve_example


def heading_alignment_angle(px1, py1, vx1, vy1, px2, py2):
    """
    Compute the wrapped angle between velocity and relative-position vectors.
    """
    angle_v = jnp.arctan2(vy1, vx1)
    angle_r = jnp.arctan2(py2 - py1, px2 - px1)
    gamma = angle_r - angle_v
    return (gamma + jnp.pi) % (2 * jnp.pi) - jnp.pi


def make_figure(lbg, trajectory):
    """
    Build the summary figure for a solved nonlinear LBG trajectory.
    """
    x = trajectory.xs
    u = trajectory.us
    p = lbg.PARAMS

    px_b = x[:, p.GAME_STATE.I_BANDIT_PX]
    py_b = x[:, p.GAME_STATE.I_BANDIT_PY]
    th_b = x[:, p.GAME_STATE.I_BANDIT_TH]
    vt_b = x[:, p.GAME_STATE.I_BANDIT_VT]
    dth_b = u[:, p.GAME_CTRL.I_BANDIT_DTH]
    dvt_b = u[:, p.GAME_CTRL.I_BANDIT_DVT]

    px_l = x[:, p.GAME_STATE.I_LADY_PX]
    py_l = x[:, p.GAME_STATE.I_LADY_PY]
    th_l = x[:, p.GAME_STATE.I_LADY_TH]
    vt_l = x[:, p.GAME_STATE.I_LADY_VT]
    dth_l = u[:, p.GAME_CTRL.I_LADY_DTH]
    dvt_l = u[:, p.GAME_CTRL.I_LADY_DVT]

    px_g = x[:, p.GAME_STATE.I_GUARD_PX]
    py_g = x[:, p.GAME_STATE.I_GUARD_PY]
    th_g = x[:, p.GAME_STATE.I_GUARD_TH]
    vt_g = x[:, p.GAME_STATE.I_GUARD_VT]
    dth_g = u[:, p.GAME_CTRL.I_GUARD_DTH]
    dvt_g = u[:, p.GAME_CTRL.I_GUARD_DVT]

    state_time = jnp.arange(x.shape[0])
    control_time = jnp.arange(u.shape[0])

    vx_b = vt_b * jnp.cos(th_b)
    vy_b = vt_b * jnp.sin(th_b)
    vx_l = vt_l * jnp.cos(th_l)
    vy_l = vt_l * jnp.sin(th_l)
    vx_g = vt_g * jnp.cos(th_g)
    vy_g = vt_g * jnp.sin(th_g)

    ax_b = dvt_b * jnp.cos(th_b[:-1]) - vt_b[:-1] * dth_b * jnp.sin(th_b[:-1])
    ay_b = dvt_b * jnp.sin(th_b[:-1]) + vt_b[:-1] * dth_b * jnp.cos(th_b[:-1])
    ax_l = dvt_l * jnp.cos(th_l[:-1]) - vt_l[:-1] * dth_l * jnp.sin(th_l[:-1])
    ay_l = dvt_l * jnp.sin(th_l[:-1]) + vt_l[:-1] * dth_l * jnp.cos(th_l[:-1])
    ax_g = dvt_g * jnp.cos(th_g[:-1]) - vt_g[:-1] * dth_g * jnp.sin(th_g[:-1])
    ay_g = dvt_g * jnp.sin(th_g[:-1]) + vt_g[:-1] * dth_g * jnp.cos(th_g[:-1])

    gamma_bl = heading_alignment_angle(
        px1=px_b,
        py1=py_b,
        vx1=vx_b,
        vy1=vy_b,
        px2=px_l,
        py2=py_l,
    )
    gamma_gb = heading_alignment_angle(
        px1=px_g,
        py1=py_g,
        vx1=vx_g,
        vy1=vy_g,
        px2=px_b,
        py2=py_b,
    )

    dist_bl = jnp.sqrt((px_l - px_b) ** 2 + (py_l - py_b) ** 2)
    dist_gb = jnp.sqrt((px_b - px_g) ** 2 + (py_b - py_g) ** 2)
    dist_lt = jnp.sqrt((lbg.px_target - px_l) ** 2 + (lbg.py_target - py_l) ** 2)

    speed_b = jnp.abs(vt_b)
    speed_l = jnp.abs(vt_l)
    speed_g = jnp.abs(vt_g)

    acc_b = jnp.sqrt(ax_b ** 2 + ay_b ** 2)
    acc_l = jnp.sqrt(ax_l ** 2 + ay_l ** 2)
    acc_g = jnp.sqrt(ax_g ** 2 + ay_g ** 2)

    fig, axs = plt.subplots(2, 3, figsize=(12, 8))

    ax = axs[0, 0]
    ax.plot(px_b, py_b, label="Bandit", marker="o")
    ax.plot(px_l, py_l, label="Lady", marker="x")
    ax.plot(px_g, py_g, label="Guard", marker="^")
    ax.scatter(px_b[0], py_b[0], color="C0", marker="s")
    ax.scatter(px_l[0], py_l[0], color="C1", marker="s")
    ax.scatter(px_g[0], py_g[0], color="C2", marker="s")
    ax.set_title("Player Trajectories")
    ax.set_xlabel("x position")
    ax.set_ylabel("y position")
    ax.grid(True)
    ax.legend()

    ax = axs[0, 1]
    ax.plot(state_time, gamma_bl * 180 / jnp.pi, label="Bandit-Lady")
    ax.plot(state_time, gamma_gb * 180 / jnp.pi, label="Guard-Bandit")
    ax.set_title("Heading Alignment")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Angle (deg)")
    ax.grid(True)
    ax.legend()

    ax = axs[1, 0]
    ax.plot(state_time, dist_bl, label="Bandit-Lady")
    ax.plot(state_time, dist_gb, label="Guard-Bandit")
    ax.plot(state_time, dist_lt, label="Lady-Target")
    ax.set_title("Distances")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Distance")
    ax.grid(True)
    ax.legend()

    ax = axs[1, 1]
    ax.plot(state_time, speed_b, label="Bandit")
    ax.plot(state_time, speed_l, label="Lady")
    ax.plot(state_time, speed_g, label="Guard")
    ax.set_title("Speeds")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Speed")
    ax.grid(True)
    ax.legend()

    ax = axs[1, 2]
    ax.plot(control_time, acc_b, label="Bandit")
    ax.plot(control_time, acc_l, label="Lady")
    ax.plot(control_time, acc_g, label="Guard")
    ax.set_title("Acceleration Magnitudes")
    ax.set_xlabel("Control step")
    ax.set_ylabel("Acceleration")
    ax.grid(True)
    ax.legend()

    axs[0, 2].axis("off")
    fig.tight_layout()
    return fig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a nonlinear Lady-Bandit-Guard example plot.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/assets/lady_bandit_guard_nonlinear.png"),
        help="Output image path.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the figure interactively after saving.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lbg, _, trajectory, _ = solve_example()
    fig = make_figure(lbg, trajectory)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    print(f"Wrote {args.output}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
