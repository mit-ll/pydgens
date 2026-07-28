# Copyright 2026 MIT Lincoln Laboratory
# SPDX-License-Identifier: MIT

"""Generate an animated GIF for the eight-player terminal-cost arm tutorial.

This script is intentionally separate from ``pydgens.examples.robot_arm``:
the example defines and solves the game, while this documentation utility owns
the optional Matplotlib/Pillow dependencies and presentation choices.

The left panel shows the planar serial arm, its initial pose, the shared
terminal target, and the end-effector path. The right panel shows the eight
player-owned joint-rate trajectories; every color corresponds to one player
and one arm joint.

Run from the repository root:

    uv run --extra visuals python scripts/visuals/robot_arm_gif.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

import pydgens as pdg
from pydgens.examples.robot_arm import (
    build_robot_arm_game,
    make_terminal_target_initial_strategy,
)


INITIAL_COLOR = "#94a3b8"
TARGET_COLOR = "#16a34a"
PATH_COLOR = "#dc2626"
PLAYER_COLORS = tuple(plt.get_cmap("tab10")(i) for i in range(8))


def arm_points(joint_angles, link_lengths) -> np.ndarray:
    """Return base and link-endpoint coordinates for one arm configuration."""
    angles = np.asarray(joint_angles, dtype=float)
    lengths = np.asarray(link_lengths, dtype=float)
    orientations = np.cumsum(angles)
    endpoints = np.cumsum(
        np.column_stack((lengths * np.cos(orientations), lengths * np.sin(orientations))),
        axis=0,
    )
    return np.vstack((np.zeros((1, 2)), endpoints))


def make_animation(
    *,
    states,
    controls,
    link_lengths,
    target_midpoint,
    target_position,
    dt: float,
    output: Path,
    fps: int = 10,
    dpi: int = 140,
    close: bool = True,
) -> None:
    """Render the arm and the eight player control histories to a GIF."""
    states = np.asarray(states, dtype=float)
    controls = np.asarray(controls, dtype=float)
    link_lengths = np.asarray(link_lengths, dtype=float)
    target_midpoint = np.asarray(target_midpoint, dtype=float)
    target_position = np.asarray(target_position, dtype=float)

    points_by_frame = np.stack(
        [arm_points(joint_angles, link_lengths) for joint_angles in states],
        axis=0,
    )
    end_effector_path = points_by_frame[:, -1, :]
    midpoint_index = len(link_lengths) // 2
    midpoint_path = points_by_frame[:, midpoint_index, :]
    initial_points = points_by_frame[0]
    maximum_reach = float(np.sum(link_lengths))

    fig, (ax_arm, ax_rates) = plt.subplots(
        1,
        2,
        figsize=(12, 5.5),
        gridspec_kw={"width_ratios": (1.0, 1.15)},
    )

    # -----------------------------------------------------------------
    # Left panel: arm geometry in task space
    # -----------------------------------------------------------------
    ax_arm.set_aspect("equal", adjustable="box")
    ax_arm.set_xlim(-maximum_reach - 0.15, maximum_reach + 0.15)
    ax_arm.set_ylim(-maximum_reach - 0.15, maximum_reach + 0.15)
    ax_arm.set_xlabel("end-effector x")
    ax_arm.set_ylabel("end-effector y")
    ax_arm.grid(True, linewidth=0.5, alpha=0.4)
    ax_arm.set_title("Eight-Link Cooperative Arm")

    ax_arm.plot(
        initial_points[:, 0],
        initial_points[:, 1],
        linestyle="--",
        color=INITIAL_COLOR,
        linewidth=2.0,
        label="initial arm",
        zorder=1,
    )
    ax_arm.scatter(
        target_midpoint[0],
        target_midpoint[1],
        marker="X",
        s=105,
        color=TARGET_COLOR,
        edgecolor="white",
        linewidth=0.9,
        label="midpoint target",
        zorder=5,
    )
    ax_arm.scatter(
        target_position[0],
        target_position[1],
        marker="*",
        s=190,
        color=TARGET_COLOR,
        edgecolor="white",
        linewidth=0.9,
        label="terminal target",
        zorder=5,
    )
    path_line, = ax_arm.plot(
        [], [], color=PATH_COLOR, linewidth=1.7, alpha=0.75,
        label="end-effector path", zorder=2,
    )
    midpoint_path_line, = ax_arm.plot(
        [], [], color="#7c3aed", linewidth=1.7, alpha=0.75,
        linestyle="--", label="midpoint path", zorder=2,
    )
    # Link i uses Player i's color, matching the rate trace in the right panel.
    link_lines = []
    for i in range(link_lengths.shape[0]):
        link_line, = ax_arm.plot(
            [], [],
            color=PLAYER_COLORS[i % len(PLAYER_COLORS)],
            linewidth=3.5,
            solid_capstyle="round",
            zorder=4,
        )
        link_lines.append(link_line)
    joint_markers, = ax_arm.plot(
        [], [], "o",
        color="#111827",
        markersize=4.5,
        markerfacecolor="white",
        markeredgewidth=1.0,
        zorder=5,
    )
    terminal_text = ax_arm.text(
        0.03,
        0.97,
        "",
        transform=ax_arm.transAxes,
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85},
    )
    ax_arm.legend(loc="lower left", fontsize=8)

    # -----------------------------------------------------------------
    # Right panel: one control history for each player / joint
    # -----------------------------------------------------------------
    control_times = np.arange(controls.shape[0]) * dt
    state_times = np.arange(states.shape[0]) * dt
    for i in range(controls.shape[1]):
        ax_rates.plot(
            control_times,
            controls[:, i],
            color=PLAYER_COLORS[i % len(PLAYER_COLORS)],
            linewidth=1.8,
            label=f"Player {i + 1} / joint {i + 1}",
        )
    time_cursor = ax_rates.axvline(0.0, color="#111827", linestyle="--", linewidth=1.1)
    ax_rates.axhline(0.0, color="0.5", linewidth=0.7)
    ax_rates.set_title("Player-Owned Joint-Rate Controls")
    ax_rates.set_xlabel("time [s]")
    ax_rates.set_ylabel("joint rate [rad/s]")
    ax_rates.grid(True, linewidth=0.5, alpha=0.4)
    ax_rates.legend(loc="best", fontsize=7, ncol=2)

    fig.tight_layout()

    def update(frame: int):
        points = points_by_frame[frame]
        for i, link_line in enumerate(link_lines):
            link_line.set_data(points[i:i + 2, 0], points[i:i + 2, 1])
        joint_markers.set_data(points[:, 0], points[:, 1])
        path_line.set_data(
            end_effector_path[:frame + 1, 0],
            end_effector_path[:frame + 1, 1],
        )
        midpoint_path_line.set_data(
            midpoint_path[:frame + 1, 0],
            midpoint_path[:frame + 1, 1],
        )
        time_cursor.set_xdata([state_times[frame], state_times[frame]])

        midpoint_error_norm = np.linalg.norm(midpoint_path[frame] - target_midpoint)
        end_error_norm = np.linalg.norm(end_effector_path[frame] - target_position)
        terminal_text.set_text(
            f"time: {state_times[frame]:.2f} s\n"
            f"midpoint error: {midpoint_error_norm:.4f} m\n"
            f"end error: {end_error_norm:.4f} m"
        )
        return (
            *link_lines,
            joint_markers,
            path_line,
            midpoint_path_line,
            time_cursor,
            terminal_text,
        )

    animation = FuncAnimation(
        fig,
        update,
        frames=states.shape[0],
        interval=1000 / fps,
        blit=False,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output, writer=PillowWriter(fps=fps), dpi=dpi)
    if close:
        plt.close(fig)


def parse_args():
    """Parse command-line options for GIF generation."""
    parser = argparse.ArgumentParser(
        description="Generate an eight-player robot-arm terminal-cost GIF.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/assets/robot_arm.gif"),
        help="Output GIF path.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=140,
        help="Output GIF DPI.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the final frame after saving the GIF.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("`--fps` must be positive.")
    if args.dpi <= 0:
        raise ValueError("`--dpi` must be positive.")

    (
        game,
        x0,
        link_lengths,
        target_midpoint,
        target_position,
        warm_start_joint_angles,
        players,
    ) = build_robot_arm_game()
    init_strat = make_terminal_target_initial_strategy(
        tg=game.tg,
        x0=x0,
        warm_start_joint_angles=warm_start_joint_angles,
    )
    solution = pdg.solve(
        game,
        x0=x0,
        method="ilq",
        max_iters=50,
        converged_max_diff=5e-2,
        init_strat=init_strat,
    )
    if not solution.converged:
        raise RuntimeError("iLQ did not converge; refusing to render a misleading GIF.")

    make_animation(
        states=solution.states,
        controls=solution.joint_controls,
        link_lengths=link_lengths,
        target_midpoint=target_midpoint,
        target_position=target_position,
        dt=game.tg.dt,
        output=args.output,
        fps=args.fps,
        dpi=args.dpi,
        close=not args.show,
    )
    print(f"Rendered {len(players)}-player arm GIF to {args.output}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
