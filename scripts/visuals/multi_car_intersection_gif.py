# Copyright 2026 MIT Lincoln Laboratory
# SPDX-License-Identifier: MIT

"""
Generate an animated GIF for the multi-car intersection example.

This script intentionally lives outside ``src/pydgens`` because animation is a
documentation/visualization concern, not a core package dependency.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Polygon

from pydgens.examples.multi_car_intersection import (
    car_state_slice,
    solve_example,
)


CAR_COLORS = ("#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2")


def simulate_naive_rollout(game, x0, car_specs):
    """
    Simulate the zero-control rollout for visual comparison.
    """
    x = jnp.asarray(x0)
    u = jnp.zeros((2 * len(car_specs),), dtype=x.dtype)
    xs = [x]

    for k in range(game.tg.nsteps):
        t = game.tg.t0 + k * game.tg.dt
        x = x + game.tg.dt * game.dynamics.evaluate(t, x, u)
        xs.append(x)

    return jnp.stack(xs, axis=0)


def _xy_trajectory(states, car_index: int):
    s = car_state_slice(car_index)
    return np.asarray(states[:, s.start:s.start + 2])


def _heading(states, car_index: int, frame: int) -> float:
    s = car_state_slice(car_index)
    return float(np.asarray(states[frame, s.start + 2]))


def _car_polygon(center, heading, *, length=0.9, width=0.42):
    """
    Return oriented rectangle vertices for a simple car footprint.
    """
    cx, cy = center
    forward = np.array([np.cos(heading), np.sin(heading)])
    lateral = np.array([-np.sin(heading), np.cos(heading)])

    half_l = 0.5 * length
    half_w = 0.5 * width
    return np.array([
        [cx, cy] + half_l * forward + half_w * lateral,
        [cx, cy] + half_l * forward - half_w * lateral,
        [cx, cy] - half_l * forward - half_w * lateral,
        [cx, cy] - half_l * forward + half_w * lateral,
    ])


def _plot_lane(ax, spec, *, extent=10.0):
    center = np.asarray(spec.lane_point, dtype=float)
    tangent = np.array([np.cos(spec.lane_heading), np.sin(spec.lane_heading)])
    normal = np.array([-np.sin(spec.lane_heading), np.cos(spec.lane_heading)])

    p0 = center - extent * tangent
    p1 = center + extent * tangent
    half_width = 0.5 * spec.lane_width

    ax.plot(
        [p0[0], p1[0]],
        [p0[1], p1[1]],
        color="0.45",
        linestyle="--",
        linewidth=1.0,
        alpha=0.7,
    )

    for sign in (-1.0, 1.0):
        q0 = p0 + sign * half_width * normal
        q1 = p1 + sign * half_width * normal
        ax.plot(
            [q0[0], q1[0]],
            [q0[1], q1[1]],
            color="0.78",
            linewidth=1.0,
        )


def _setup_axis(ax, car_specs, title: str):
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, linewidth=0.4, alpha=0.45)

    for spec in car_specs:
        _plot_lane(ax, spec)
        ax.scatter(
            spec.goal_xy[0],
            spec.goal_xy[1],
            marker="*",
            color="black",
            s=80,
            zorder=5,
        )

    points = []
    for spec in car_specs:
        points.append(spec.initial_state[:2])
        points.append(spec.goal_xy)
    points = np.asarray(points, dtype=float)
    margin = 2.0
    ax.set_xlim(float(points[:, 0].min() - margin), float(points[:, 0].max() + margin))
    ax.set_ylim(float(points[:, 1].min() - margin), float(points[:, 1].max() + margin))


def _init_artists(ax, states, car_specs):
    lines = []
    bodies = []

    for i, spec in enumerate(car_specs):
        color = CAR_COLORS[i % len(CAR_COLORS)]
        xy = _xy_trajectory(states, i)

        line, = ax.plot([], [], color=color, linewidth=2.0, label=spec.name)
        body = Polygon(
            _car_polygon(xy[0], _heading(states, i, 0)),
            closed=True,
            facecolor=color,
            edgecolor="white",
            linewidth=0.8,
            alpha=0.95,
            zorder=10,
        )
        ax.add_patch(body)
        lines.append(line)
        bodies.append(body)

    ax.legend(loc="upper right", fontsize=8)
    return lines, bodies


def make_animation(
    *,
    solution_states,
    car_specs,
    output: Path,
    naive_states=None,
    fps: int = 10,
    dpi: int = 130,
):
    """
    Render the solved trajectory, optionally beside a zero-control rollout.
    """
    solution_states = np.asarray(solution_states)
    naive_states = None if naive_states is None else np.asarray(naive_states)

    if naive_states is None:
        fig, axes = plt.subplots(1, 1, figsize=(7, 7))
        axes = [axes]
        panel_states = [solution_states]
        titles = ["iLQ Feedback Nash Rollout"]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
        panel_states = [naive_states, solution_states]
        titles = ["Naive Zero-Control Rollout", "iLQ Feedback Nash Rollout"]

    artists = []
    for ax, states, title in zip(axes, panel_states, titles):
        _setup_axis(ax, car_specs, title)
        artists.append(_init_artists(ax, states, car_specs))

    frame_text = fig.text(0.5, 0.02, "", ha="center")
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))

    num_frames = solution_states.shape[0]

    def update(frame):
        for states, (lines, bodies) in zip(panel_states, artists):
            for i, (line, body) in enumerate(zip(lines, bodies)):
                xy = _xy_trajectory(states, i)
                line.set_data(xy[:frame + 1, 0], xy[:frame + 1, 1])
                body.set_xy(_car_polygon(xy[frame], _heading(states, i, frame)))

        frame_text.set_text(f"step {frame:02d} / {num_frames - 1:02d}")
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
        description="Generate a multi-car intersection GIF.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/assets/multi_car_intersection.gif"),
        help="Output GIF path.",
    )
    parser.add_argument(
        "--compare-naive",
        action="store_true",
        help="Render the zero-control rollout beside the iLQ rollout.",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    game, x0, car_specs, _, solution = solve_example()
    naive_states = (
        simulate_naive_rollout(game, x0, car_specs)
        if args.compare_naive
        else None
    )

    make_animation(
        solution_states=solution.states,
        naive_states=naive_states,
        car_specs=car_specs,
        output=args.output,
        fps=args.fps,
        dpi=args.dpi,
    )

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
