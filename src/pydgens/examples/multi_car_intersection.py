# Copyright 2026 MIT Lincoln Laboratory
# SPDX-License-Identifier: MIT

"""
Multiple cars negotiate an intersection, while avoiding collisions.

A showcase nonlinear game built with the frontend API and solved with ILQ. 

Each player controls one bicycle-like car:

    x_car = [px, py, theta, v]
    u_car = [phi, a]

with continuous-time dynamics

    px_dot    = v cos(theta)
    py_dot    = v sin(theta)
    theta_dot = v tan(phi) / L
    v_dot     = a

Each car wants to move toward its own goal, stay near its lane centerline,
avoid leaving the lane, keep a nominal speed, respect soft speed bounds, avoid
other cars, and avoid excessive steering/acceleration.

The important modeling point is that this remains an unconstrained nonlinear
game. Collision avoidance, lane boundaries, and speed bounds are all 
represented as soft penalties in each player's cost. 
True state/control constraints currently belong to the augmented-Lagrangian 
solver path, which returns local open-loop trajectories rather than iLQ feedback
strategies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax.numpy as jnp

import pydgens as pdg

NX_CAR = 4  # state dimensions per car (player)
NU_CAR = 2  # control dimensions per car (player)

@dataclass(frozen=True)
class CarSpec:
    """
    Semantic parameters for one car/player in the intersection game.

    ``lane_heading`` defines the straight lane centerline direction. The lane
    itself is the infinite line through ``lane_point`` with that heading. This
    keeps the first prototype compact while still supporting cars entering the
    intersection from different angles.
    """

    name: str
    initial_state: tuple[float, float, float, float]
    goal_xy: tuple[float, float]
    lane_point: tuple[float, float]
    lane_heading: float
    lane_width: float = 1.0
    wheelbase: float = 2.5
    nominal_speed: float = 3.0
    min_speed: float = 0.0
    max_speed: float = 6.0


@dataclass(frozen=True)
class CarCostWeights:
    """
    Cost weights shared by the default cars.
    """

    proximity: float = 20.0
    goal: float = 0.8
    steering: float = 0.25
    acceleration: float = 0.2
    lane_centerline: float = 0.8
    lane_boundary: float = 10.0
    nominal_speed: float = 0.6
    speed_bounds: float = 12.0
    proximity_threshold: float = 2.5


def car_state_slice(car_index: int) -> slice:
    start = NX_CAR * car_index
    return slice(start, start + 4)


def car_control_slice(car_index: int) -> slice:
    start = NU_CAR * car_index
    return slice(start, start + 2)


def default_car_specs() -> tuple[CarSpec, ...]:
    """
    Return a compact four-car intersection scenario.

    Cars enter from the west, east, south, and north. With the zero steering
    and zero acceleration rollout, all cars continue toward the center of the
    intersection, which makes this a useful starting point for later visual
    comparisons between naive and game-aware behavior.
    """
    return (
        CarSpec(
            name="eastbound",
            initial_state=(-8.0, -1.0, 0.0, 3.0),
            goal_xy=(8.0, -1.0),
            lane_point=(0.0, -1.0),
            lane_heading=0.0,
        ),
        CarSpec(
            name="westbound",
            initial_state=(8.0, 1.0, jnp.pi, 3.0),
            goal_xy=(-8.0, 1.0),
            lane_point=(0.0, 1.0),
            lane_heading=jnp.pi,
        ),
        CarSpec(
            name="northbound",
            initial_state=(1.0, -8.0, 0.5 * jnp.pi, 3.0),
            goal_xy=(1.0, 8.0),
            lane_point=(1.0, 0.0),
            lane_heading=0.5 * jnp.pi,
        ),
        CarSpec(
            name="southbound",
            initial_state=(-1.0, 8.0, -0.5 * jnp.pi, 3.0),
            goal_xy=(-1.0, -8.0),
            lane_point=(-1.0, 0.0),
            lane_heading=-0.5 * jnp.pi,
        ),
    )


def pack_initial_state(car_specs: Sequence[CarSpec]):
    return jnp.asarray([value for spec in car_specs for value in spec.initial_state])


def car_intersection_dynamics(car_specs: Sequence[CarSpec]):
    """
    Build the joint car dynamics callable for the supplied cars.
    """
    wheelbases = jnp.asarray([spec.wheelbase for spec in car_specs])
    num_cars = len(car_specs)

    def dynamics(t, x, u):
        del t
        dxdt = []
        for i in range(num_cars):
            xi = x[car_state_slice(i)]
            ui = u[car_control_slice(i)]

            px, py, theta, speed = xi
            phi, accel = ui
            wheelbase = wheelbases[i]

            dxdt.extend([
                speed * jnp.cos(theta),
                speed * jnp.sin(theta),
                speed * jnp.tan(phi) / wheelbase,
                accel,
            ])

        return jnp.asarray(dxdt)

    return dynamics


def _lane_lateral_error(px, py, spec: CarSpec):
    """
    Signed distance from a point to a straight lane centerline.
    """
    lane_px, lane_py = spec.lane_point
    dx = px - lane_px
    dy = py - lane_py
    return -jnp.sin(spec.lane_heading) * dx + jnp.cos(spec.lane_heading) * dy


def make_car_cost(
    *,
    car_index: int,
    car_specs: Sequence[CarSpec],
    weights: CarCostWeights,
):
    """
    Build one player's soft-penalty cost.

    The running cost is written in joint coordinates, which is the frontend
    convention for nonlinear games. The player object later declares which
    two entries of the joint control vector this car actually owns.
    """
    spec = car_specs[car_index]
    other_indices = tuple(i for i in range(len(car_specs)) if i != car_index)
    goal_xy = jnp.asarray(spec.goal_xy)
    lane_half_width = 0.5 * spec.lane_width

    def running(t, x, u):
        del t
        xi = x[car_state_slice(car_index)]
        ui = u[car_control_slice(car_index)]

        px, py, theta, speed = xi
        phi, accel = ui

        # Soft collision-avoidance penalty. This is intentionally symmetric:
        # each nearby pair appears in both players' objectives.
        proximity_cost = 0.0
        for j in other_indices:
            xj = x[car_state_slice(j)]
            dist = jnp.sqrt((px - xj[0]) ** 2 + (py - xj[1]) ** 2 + 1e-8)
            overlap = jnp.maximum(weights.proximity_threshold - dist, 0.0)
            proximity_cost = proximity_cost + overlap**2

        lateral_error = _lane_lateral_error(px, py, spec)
        lane_centerline_cost = lateral_error**2
        lane_boundary_violation = jnp.maximum(jnp.abs(lateral_error) - lane_half_width, 0.0)

        low_speed_violation = jnp.maximum(spec.min_speed - speed, 0.0)
        high_speed_violation = jnp.maximum(speed - spec.max_speed, 0.0)

        return (
            weights.proximity * proximity_cost
            + weights.goal * jnp.sum((jnp.asarray([px, py]) - goal_xy) ** 2)
            + weights.steering * phi**2
            + weights.acceleration * accel**2
            + weights.lane_centerline * lane_centerline_cost
            + weights.lane_boundary * lane_boundary_violation**2
            + weights.nominal_speed * (speed - spec.nominal_speed) ** 2
            + weights.speed_bounds * (low_speed_violation**2 + high_speed_violation**2)
        )

    return pdg.player_cost(running=running)


def build_multi_car_intersection_game(
    *,
    car_specs: Sequence[CarSpec] | None = None,
    weights: CarCostWeights = CarCostWeights(),
    nt: int = 41,
    dt: float = 0.1,
):
    """
    Build a parameterized frontend nonlinear intersection game.

    Returns
    -------
    tuple
        ``(game, x0, car_specs, players)``.
    """
    if car_specs is None:
        car_specs = default_car_specs()

    car_specs = tuple(car_specs)
    num_cars = len(car_specs)
    nx = NX_CAR * num_cars
    nu = NU_CAR * num_cars

    tg = pdg.time_grid(nt=nt, dt=dt)
    x0 = pack_initial_state(car_specs)

    dynamics = pdg.nonlinear_dynamics(
        nx=nx,
        nu=nu,
        dynamics=car_intersection_dynamics(car_specs),
    )

    players = []
    for i, spec in enumerate(car_specs):
        players.append(
            pdg.player(
                name=spec.name,
                cost=make_car_cost(
                    car_index=i,
                    car_specs=car_specs,
                    weights=weights,
                ),
                joint_ctrl_slice=car_control_slice(i),
                state_view=range(car_state_slice(i).start, car_state_slice(i).stop),
            )
        )

    game = pdg.game(
        tg=tg,
        dynamics=dynamics,
        players=players,
    )

    return game, x0, car_specs, tuple(players)


def min_pairwise_distance(states, num_cars: int) -> float:
    min_dist = jnp.inf
    for i in range(num_cars):
        xy_i = states[:, car_state_slice(i).start:car_state_slice(i).start + 2]
        for j in range(i + 1, num_cars):
            xy_j = states[:, car_state_slice(j).start:car_state_slice(j).start + 2]
            dist = jnp.linalg.norm(xy_i - xy_j, axis=1)
            min_dist = jnp.minimum(min_dist, jnp.min(dist))

    return float(min_dist)


def main() -> None:
    game, x0, car_specs, players = build_multi_car_intersection_game()

    solution = pdg.solve(
        game,
        x0=x0,
        method="ilq",
    )

    states = solution.states
    joint_controls = solution.joint_controls

    print()
    print(solution.format_summary("Multi-Car Intersection"))

    print("\nADDED CHECKS:")
    print(f"cars: {len(car_specs)}")
    print(f"minimum pairwise distance: {min_pairwise_distance(states, len(car_specs)):.3f}")

    for i, (spec, player) in enumerate(zip(car_specs, players)):
        s = car_state_slice(i)
        c = player.joint_ctrl_slice
        final_xy = states[-1, s][:2]
        final_speed = states[-1, s][3]
        first_control = joint_controls[0, c] if joint_controls.size else jnp.array([])
        print(
            f"{spec.name}: final_xy={final_xy}, "
            f"goal={jnp.asarray(spec.goal_xy)}, "
            f"final_speed={float(final_speed):+.3f}, "
            f"u[0]={first_control}"
        )

    print("\nNOTE:")
    print(
        "Collision avoidance, lane keeping, and speed bounds are soft costs in "
        "this iLQ example. A later visuals script can compare this rollout "
        "against a naive zero-control rollout."
    )


if __name__ == "__main__":
    main()
