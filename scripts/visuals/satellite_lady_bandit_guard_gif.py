# Copyright 2026 MIT Lincoln Laboratory
# SPDX-License-Identifier: MIT

"""
Generate an animated GIF for the satellite Lady-Bandit-Guard example.

This script intentionally lives outside ``src/pydgens`` because animation is a
documentation/visualization concern, not a core package dependency.

The visual emphasizes the feedback nature of the LQ solution: the game is
solved once, then the same equilibrium strategy is rolled out from a Monte
Carlo family of nearby initial conditions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

import pydgens as pdg
from pydgens.examples.satellite_lady_bandit_guard import (
    I_BANDIT_POS,
    I_BANDIT_VEL,
    I_GUARD_POS,
    I_GUARD_VEL,
    build_satellite_lady_bandit_guard_game,
)
from pydgens.ir.systemtypes import propagate_system_trajectory


BANDIT_COLOR = "#dc2626"
GUARD_COLOR = "#2563eb"
LADY_COLOR = "#111827"
TARGET_COLOR = "#16a34a"


def sample_initial_conditions(
    x0,
    *,
    num_samples: int,
    seed: int,
    position_sigma: float,
    z_position_sigma: float,
    velocity_sigma: float,
):
    """
    Sample nearby initial states around the example's nominal condition.

    The nominal state is always the first sample so the animation has one
    reproducible highlighted rollout.
    """
    if num_samples < 1:
        raise ValueError("`num_samples` must be at least 1.")

    x0 = np.asarray(x0, dtype=float)
    samples = [x0.copy()]
    rng = np.random.default_rng(seed)

    pos_scale = np.array([position_sigma, position_sigma, z_position_sigma])
    vel_scale = np.array([velocity_sigma, velocity_sigma, velocity_sigma])

    for _ in range(num_samples - 1):
        perturb = np.zeros_like(x0)
        perturb[I_BANDIT_POS] = rng.normal(scale=pos_scale)
        perturb[I_GUARD_POS] = rng.normal(scale=pos_scale)
        perturb[I_BANDIT_VEL] = rng.normal(scale=vel_scale)
        perturb[I_GUARD_VEL] = rng.normal(scale=vel_scale)
        samples.append(x0 + perturb)

    return jnp.asarray(np.stack(samples, axis=0))


def solve_feedback_rollouts(
    *,
    num_samples: int,
    seed: int,
    position_sigma: float,
    z_position_sigma: float,
    velocity_sigma: float,
    nt: int,
    dt: float,
):
    """
    Solve the LQ game once, then roll out that strategy from sampled states.
    """
    game, x0, weights, _, _ = build_satellite_lady_bandit_guard_game(
        nt=nt,
        dt=dt,
    )
    solution = pdg.solve(
        game,
        x0=x0,
        method="lq",
    )

    sampled_x0s = sample_initial_conditions(
        x0,
        num_samples=num_samples,
        seed=seed,
        position_sigma=position_sigma,
        z_position_sigma=z_position_sigma,
        velocity_sigma=velocity_sigma,
    )

    lqgame = game.to_ir()
    trajectories = []
    for sampled_x0 in sampled_x0s:
        trajectory = propagate_system_trajectory(
            lqgame.cs,
            x0=sampled_x0,
            strategy=solution.strategy,
        )
        trajectories.append(np.asarray(trajectory.xs))

    return np.stack(trajectories, axis=0), weights


def _positions(states, rows: slice):
    return states[:, :, rows]


def _axis_limits(*position_clouds, margin: float = -2.0):
    points = np.concatenate([
        np.reshape(cloud, (-1, 3))
        for cloud in position_clouds
    ], axis=0)
    points = np.concatenate([points, np.zeros((1, 3))], axis=0)

    center = 0.5 * (points.min(axis=0) + points.max(axis=0))
    radius = 0.5 * np.max(points.max(axis=0) - points.min(axis=0)) + margin

    # return [
    #     (center[0] - radius, center[0] + radius),
    #     (center[1] - radius, center[1] + radius),
    #     (center[2] - radius, center[2] + radius),
    # ]
    # return [
    #     (center[0] - 1.0, center[0] + 1.0),
    #     (center[1] - 1.0, center[1] + 1.0),
    #     (center[2] - 0.5, center[2] + 0.5),
    # ]
    return [
        (-2.0, 2.0),
        (-2.0, 2.0),
        (-0.5, 0.5),
    ]



def _set_3d_point(scatter, point):
    scatter._offsets3d = (
        np.asarray([point[0]]),
        np.asarray([point[1]]),
        np.asarray([point[2]]),
    )


def _plot_static_context(ax, bandit_pos, guard_pos, guard_target_pos):
    """
    Plot faded full trajectories so the moving markers read as rollouts from a
    sampled feedback strategy, not a single hand-picked path.
    """
    for i in range(bandit_pos.shape[0]):
        is_nominal = i == 0
        ax.plot(
            bandit_pos[i, :, 0],
            bandit_pos[i, :, 1],
            bandit_pos[i, :, 2],
            color=BANDIT_COLOR,
            linewidth=2.2 if is_nominal else 0.9,
            alpha=0.95 if is_nominal else 0.1,
        )
        ax.plot(
            guard_pos[i, :, 0],
            guard_pos[i, :, 1],
            guard_pos[i, :, 2],
            color=GUARD_COLOR,
            linewidth=2.2 if is_nominal else 0.9,
            alpha=0.95 if is_nominal else 0.1,
        )

    ax.plot(
        guard_target_pos[0, :, 0],
        guard_target_pos[0, :, 1],
        guard_target_pos[0, :, 2],
        color=TARGET_COLOR,
        linestyle="--",
        linewidth=1.4,
        alpha=0.8,
        label="nominal guard target",
    )


def make_animation(
    *,
    states,
    weights,
    output: Path,
    fps: int,
    dpi: int,
    figure_width: float,
    figure_height: float,
    z_box_aspect: float,
):
    """
    Render a Monte Carlo feedback trajectory bundle.
    """
    if figure_width <= 0.0:
        raise ValueError("`figure_width` must be positive.")
    if figure_height <= 0.0:
        raise ValueError("`figure_height` must be positive.")
    if z_box_aspect <= 0.0:
        raise ValueError("`z_box_aspect` must be positive.")

    bandit_pos = _positions(states, I_BANDIT_POS)
    guard_pos = _positions(states, I_GUARD_POS)
    guard_target_pos = weights.alpha_guard_target * bandit_pos

    fig = plt.figure(figsize=(figure_width, figure_height))
    ax = fig.add_subplot(111, projection="3d")
    _plot_static_context(ax, bandit_pos, guard_pos, guard_target_pos)

    lady = ax.scatter(
        [0.0],
        [0.0],
        [0.0],
        color=LADY_COLOR,
        s=95,
        marker="*",
        label="lady / reference",
        depthshade=False,
        zorder=20,
    )
    del lady

    bandit_markers = []
    guard_markers = []
    target_markers = []
    for i in range(states.shape[0]):
        is_nominal = i == 0
        marker_size = 52 if is_nominal else 18
        marker_alpha = 1.0 if is_nominal else 0.15
        bandit_markers.append(ax.scatter(
            [],
            [],
            [],
            color=BANDIT_COLOR,
            s=marker_size,
            alpha=marker_alpha,
            depthshade=False,
            label="bandit" if is_nominal else None,
        ))
        guard_markers.append(ax.scatter(
            [],
            [],
            [],
            color=GUARD_COLOR,
            s=marker_size,
            alpha=marker_alpha,
            depthshade=False,
            label="guard" if is_nominal else None,
        ))
        target_markers.append(ax.scatter(
            [],
            [],
            [],
            color=TARGET_COLOR,
            s=24 if is_nominal else 10,
            alpha=0.8 if is_nominal else 0.25,
            marker="x",
            depthshade=False,
            label="guard target point" if is_nominal else None,
        ))

    xlim, ylim, zlim = _axis_limits(bandit_pos, guard_pos)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.set_box_aspect((1, 1, z_box_aspect))

    ax.set_xlabel("radial x [km]", labelpad=4)
    ax.set_ylabel("along-track y [km]", labelpad=4)
    ax.set_zlabel("cross-track z [km]", labelpad=4)
    ax.set_title("Orbital Lady-Bandit-Guard: LQ Feedback Rollouts", pad=2)
    ax.grid(True, linewidth=0.1, alpha=0.05)
    # ax.grid(False)
    ax.view_init(elev=24, azim=-55)
    ax.legend(loc="upper left", fontsize=8)

    frame_text = ax.text2D(
        0.5,
        0.02,
        "",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
    )
    ax.set_position([0.0, 0.0, 1.0, 0.94])

    num_frames = states.shape[1]

    def update(frame):
        for i in range(states.shape[0]):
            _set_3d_point(bandit_markers[i], bandit_pos[i, frame])
            _set_3d_point(guard_markers[i], guard_pos[i, frame])
            _set_3d_point(target_markers[i], guard_target_pos[i, frame])

        frame_text.set_text(
            f"one feedback Nash strategy rolled out from {states.shape[0]} initial states "
            f"| step {frame:02d}/{num_frames - 1:02d}"
        )
        return []

    animation = FuncAnimation(
        fig,
        update,
        frames=num_frames,
        interval=1000 / fps,
        blit=False,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a satellite Lady-Bandit-Guard Monte Carlo GIF.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/assets/satellite_lady_bandit_guard.gif"),
        help="Output GIF path.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=36,
        help="Number of sampled initial states, including the nominal example state.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for sampled initial states.",
    )
    parser.add_argument(
        "--position-sigma",
        type=float,
        default=1.2,
        help="Initial in-plane position perturbation standard deviation [km].",
    )
    parser.add_argument(
        "--z-position-sigma",
        type=float,
        default=0.35,
        help="Initial cross-track position perturbation standard deviation [km].",
    )
    parser.add_argument(
        "--velocity-sigma",
        type=float,
        default=4.0e-4,
        help="Initial velocity perturbation standard deviation [km/s].",
    )
    parser.add_argument(
        "--nt",
        type=int,
        default=61,
        help="Number of time nodes used by the example game.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=30.0,
        help="Time step used by the example game [s].",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second for the GIF.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=130,
        help="Output DPI.",
    )
    parser.add_argument(
        "--figure-width",
        type=float,
        default=8.0,
        help="Figure width in inches.",
    )
    parser.add_argument(
        "--figure-height",
        type=float,
        default=4.8,
        help="Figure height in inches. Lower values produce a shorter GIF.",
    )
    parser.add_argument(
        "--z-box-aspect",
        type=float,
        default=0.35,
        help=(
            "3D axes z-aspect relative to x/y. Lower values flatten the "
            "cross-track axis without post-processing image distortion."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    states, weights = solve_feedback_rollouts(
        num_samples=args.samples,
        seed=args.seed,
        position_sigma=args.position_sigma,
        z_position_sigma=args.z_position_sigma,
        velocity_sigma=args.velocity_sigma,
        nt=args.nt,
        dt=args.dt,
    )
    make_animation(
        states=states,
        weights=weights,
        output=args.output,
        fps=args.fps,
        dpi=args.dpi,
        figure_width=args.figure_width,
        figure_height=args.figure_height,
        z_box_aspect=args.z_box_aspect,
    )

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
