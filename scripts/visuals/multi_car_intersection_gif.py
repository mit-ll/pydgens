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


def _xy_at(x, car_index: int):
    s = car_state_slice(car_index)
    return np.asarray(x[s.start:s.start + 2])


def _set_car_state(x, car_index: int, car_state):
    s = car_state_slice(car_index)
    return x.at[s].set(car_state)


def _collision_events(
    x,
    *,
    num_cars: int,
    collision_radius: float,
    frozen: set[int],
    recorded_pairs: set[tuple[int, int]],
    frame: int,
):
    """
    Return all newly detected collisions at a frame.
    """
    events = []

    for i in range(num_cars):
        xy_i = _xy_at(x, i)
        for j in range(i + 1, num_cars):
            pair = (i, j)
            if pair in recorded_pairs:
                continue

            # If both cars were already stopped by earlier collisions, do not
            # keep rediscovering a static contact pair.
            if i in frozen and j in frozen:
                continue

            xy_j = _xy_at(x, j)
            dist = float(np.linalg.norm(xy_i - xy_j))
            if dist <= collision_radius:
                events.append({
                    "frame": frame,
                    "pair": (i, j),
                    "position": 0.5 * (xy_i + xy_j),
                    "distance": dist,
                })

    return events


def simulate_naive_rollout(game, x0, car_specs, *, collision_radius: float = 0.2):
    """
    Simulate the zero-control rollout for visual comparison.

    If a pair enters ``collision_radius``, only the colliding cars freeze. The
    remaining cars continue forward, so the animation can show multiple
    collisions in one naive rollout.
    """
    x = jnp.asarray(x0)
    u = jnp.zeros((2 * len(car_specs),), dtype=x.dtype)
    xs = [x]
    collisions = []
    frozen = set()
    recorded_pairs = set()

    for k in range(game.tg.nsteps):
        t = game.tg.t0 + k * game.tg.dt
        x_prev = x
        x_next = x + game.tg.dt * game.dynamics.evaluate(t, x, u)

        for car_index in frozen:
            x_next = _set_car_state(
                x_next,
                car_index,
                x_prev[car_state_slice(car_index)],
            )

        new_collisions = _collision_events(
            x_next,
            num_cars=len(car_specs),
            collision_radius=collision_radius,
            frozen=frozen,
            recorded_pairs=recorded_pairs,
            frame=k + 1,
        )

        for event in new_collisions:
            collisions.append(event)
            recorded_pairs.add(event["pair"])
            frozen.update(event["pair"])

        x = x_next
        xs.append(x)

    return jnp.stack(xs, axis=0), collisions


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


def _init_collision_artist(ax, collisions):
    """
    Create hidden collision marker/text artists for a panel.
    """
    if not collisions:
        return None

    marker = ax.scatter(
        [],
        [],
        marker="x",
        color="#ef4444",
        s=180,
        linewidths=3,
        zorder=20,
    )
    text = ax.text(
        0.03,
        0.95,
        "",
        transform=ax.transAxes,
        color="#ef4444",
        fontsize=10,
        fontweight="bold",
        va="top",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "#ef4444",
            "alpha": 0.9,
        },
    )
    return marker, text


def make_animation(
    *,
    solution_states,
    car_specs,
    output: Path,
    naive_states=None,
    naive_collisions=None,
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
        panel_collisions = [None]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
        panel_states = [naive_states, solution_states]
        titles = ["Naive Zero-Control Rollout", "iLQ Feedback Nash Rollout"]
        panel_collisions = [naive_collisions, None]

    artists = []
    collision_artists = []
    for ax, states, title, collisions in zip(axes, panel_states, titles, panel_collisions):
        _setup_axis(ax, car_specs, title)
        artists.append(_init_artists(ax, states, car_specs))
        collision_artists.append(_init_collision_artist(ax, collisions))

    frame_text = fig.text(0.5, 0.02, "", ha="center")
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))

    num_frames = solution_states.shape[0]

    def update(frame):
        for states, (lines, bodies), collisions, collision_artist in zip(
            panel_states,
            artists,
            panel_collisions,
            collision_artists,
        ):
            for i, (line, body) in enumerate(zip(lines, bodies)):
                xy = _xy_trajectory(states, i)
                line.set_data(xy[:frame + 1, 0], xy[:frame + 1, 1])
                body.set_xy(_car_polygon(xy[frame], _heading(states, i, frame)))

            if collision_artist is not None:
                marker, text = collision_artist
                visible_collisions = [
                    collision
                    for collision in collisions
                    if frame >= collision["frame"]
                ]
                if visible_collisions:
                    marker.set_offsets(np.asarray([
                        collision["position"]
                        for collision in visible_collisions
                    ]))
                    labels = []
                    for collision in visible_collisions:
                        i, j = collision["pair"]
                        labels.append(f"{car_specs[i].name} + {car_specs[j].name}")
                    text.set_text("collisions:\n" + "\n".join(labels))
                else:
                    marker.set_offsets(np.empty((0, 2)))
                    text.set_text("")

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
        "--collision-radius",
        type=float,
        default=0.2,
        help="Naive-rollout distance threshold used to mark and freeze collisions.",
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
    naive_states = None
    naive_collisions = None
    if args.compare_naive:
        naive_states, naive_collisions = simulate_naive_rollout(
            game,
            x0,
            car_specs,
            collision_radius=args.collision_radius,
        )

    make_animation(
        solution_states=solution.states,
        naive_states=naive_states,
        naive_collisions=naive_collisions,
        car_specs=car_specs,
        output=args.output,
        fps=args.fps,
        dpi=args.dpi,
    )

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
