# Copyright 2026 MIT Lincoln Laboratory
# SPDX-License-Identifier: MIT

import pytest
import jax.numpy as jnp

import pydgens as pdg

from pydgens.examples.constrained_integrators import main


@pytest.mark.slow
def test_constrained_integrators_smoketest():
    main()


def _make_constrained_integrators_problem():
    """Mirror the beginner constrained-integrators example without printing."""
    nt = 31
    dt = 0.1
    u_max = 2.0

    q = 2.0
    r = 0.05

    x0 = jnp.array([0.0, 1.0], dtype=jnp.float32)
    x_goal = jnp.array([4.0, -2.0], dtype=jnp.float32)

    tg = pdg.time_grid(nt=nt, dt=dt)
    dynamics = pdg.nonlinear_dynamics(
        nx=2,
        nu=2,
        dynamics=lambda t, x, u: jnp.array([u[0], u[1]], dtype=x.dtype),
    )

    player_1_cost = pdg.player_cost(
        running=lambda t, x, u: 0.5 * (q * (x[0] - x_goal[0]) ** 2 + r * u[0] ** 2),
        terminal=lambda t, x: jnp.asarray(0.0, dtype=x.dtype),
    )
    player_2_cost = pdg.player_cost(
        running=lambda t, x, u: 0.5 * (q * (x[1] - x_goal[1]) ** 2 + r * u[1] ** 2),
        terminal=lambda t, x: jnp.asarray(0.0, dtype=x.dtype),
    )

    players = [
        pdg.player(name="player_1", cost=player_1_cost, joint_ctrl_slice=slice(0, 1)),
        pdg.player(name="player_2", cost=player_2_cost, joint_ctrl_slice=slice(1, 2)),
    ]
    constraints = pdg.constraint_set(
        pdg.control_bounds(lower=-u_max, upper=u_max),
    )

    game = pdg.game(
        tg=tg,
        dynamics=dynamics,
        players=players,
        constraints=constraints,
    )
    return game, x0


def _benchmark_constrained_integrators_backend(benchmark, *, jacobian_backend):
    game, x0 = _make_constrained_integrators_problem()

    def run():
        result = pdg.solve(
            game,
            x0=x0,
            method="al",
            jacobian_backend=jacobian_backend,
            max_iters=1,
            newton_max_iters=1,
            newton_max_rejects=1,
            ls_max_iters=4,
        )
        assert result.method == "al"
        assert result.primal_dual_trajectory is not None
        return result.primal_dual_trajectory.xs.block_until_ready()

    benchmark(run)


@pytest.mark.benchmark(group="example-constrained-integrators-al-backend-001")
def test_constrained_integrators_autodiff_backend_warm_perf(benchmark):
    _benchmark_constrained_integrators_backend(benchmark, jacobian_backend="autodiff")


@pytest.mark.benchmark(group="example-constrained-integrators-al-backend-001")
def test_constrained_integrators_structured_backend_warm_perf(benchmark):
    _benchmark_constrained_integrators_backend(benchmark, jacobian_backend="structured")
