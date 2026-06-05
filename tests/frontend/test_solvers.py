# Copyright 2026 MIT Lincoln Laboratory
# SPDX-License-Identifier: MIT

# tests/frontend/test_solvers.py

from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import pytest

import pydgens as pdg
import pydgens.frontend.solvers as fsolvers


def _make_lq_game():

    tg = pdg.time_grid(
        nt=5,
        dt=0.1,
    )

    dynamics = pdg.linear_dynamics(
        A=jnp.eye(2),
        B=jnp.eye(2),
    )

    p1 = pdg.player(
        cost=pdg.quadratic_cost(
            nx=2,
            nu=2,
            control_weights=[1.0],
            control_indices=[0],
        ),
        joint_ctrl_slice=slice(0, 1),
        name="p1",
    )

    p2 = pdg.player(
        cost=pdg.quadratic_cost(
            nx=2,
            nu=2,
            control_weights=[1.0],
            control_indices=[1],
        ),
        joint_ctrl_slice=slice(1, 2),
        name="p2",
    )

    return pdg.game(
        tg=tg,
        dynamics=dynamics,
        players=[p1, p2],
    )


def _make_nonlinear_game(*, nt=5):

    tg = pdg.time_grid(
        nt=nt,
        dt=0.1,
    )

    dynamics = pdg.nonlinear_dynamics(
        nx=2,
        nu=2,
        dynamics=lambda t, x, u: jnp.array([
            x[1] + u[0],
            -x[0] + u[1],
        ]),
    )

    p1 = pdg.player(
        cost=pdg.player_cost(
            running=lambda t, x, u: x[0] ** 2 + u[0] ** 2,
        ),
        joint_ctrl_slice=slice(0, 1),
        name="p1",
    )

    p2 = pdg.player(
        cost=pdg.player_cost(
            running=lambda t, x, u: x[1] ** 2 + u[1] ** 2,
        ),
        joint_ctrl_slice=slice(1, 2),
        name="p2",
    )

    return pdg.game(
        tg=tg,
        dynamics=dynamics,
        players=[p1, p2],
    )


def _make_constrained_nonlinear_game(*, nt=5):

    tg = pdg.time_grid(
        nt=nt,
        dt=0.1,
    )

    dynamics = pdg.nonlinear_dynamics(
        nx=2,
        nu=2,
        dynamics=lambda t, x, u: jnp.array([
            x[1] + u[0],
            -x[0] + u[1],
        ]),
    )

    p1 = pdg.player(
        cost=pdg.player_cost(
            running=lambda t, x, u: x[0] ** 2 + u[0] ** 2,
        ),
        joint_ctrl_slice=slice(0, 1),
        name="p1",
    )

    p2 = pdg.player(
        cost=pdg.player_cost(
            running=lambda t, x, u: x[1] ** 2 + u[1] ** 2,
        ),
        joint_ctrl_slice=slice(1, 2),
        name="p2",
    )

    cons = pdg.constraint_set(
        pdg.control_bounds(
            lower=-1.0,
            upper=1.0,
            indices=[0],
        ),
    )

    return pdg.game(
        tg=tg,
        dynamics=dynamics,
        players=[p1, p2],
        constraints=cons,
    )


def test_solve_is_exposed_at_top_level():

    assert pdg.solve is fsolvers.solve


def test_solve_lqgame_auto_dispatches_to_lq_solver(monkeypatch):

    game = _make_lq_game()

    strategy = object()
    trajectory = object()

    def fake_solve_lqgame_feedback(lqgame, **kwargs):
        assert isinstance(
            lqgame,
            fsolvers.LinearQuadraticGameType1,
        )
        return strategy

    def fake_propagate_system_trajectory(cs, x0, strategy):
        assert strategy is strategy_obj
        assert jnp.allclose(x0, jnp.array([1.0, -1.0]))
        return trajectory

    strategy_obj = strategy

    monkeypatch.setattr(
        fsolvers,
        "solve_lqgame_feedback",
        fake_solve_lqgame_feedback,
    )
    monkeypatch.setattr(
        fsolvers,
        "propagate_system_trajectory",
        fake_propagate_system_trajectory,
    )

    result = pdg.solve(
        game,
        x0=jnp.array([1.0, -1.0]),
    )

    assert result.method == "lq"
    assert result.converged is True
    assert result.strategy is strategy
    assert result.trajectory is trajectory
    assert result.raw is strategy


def test_solve_lqgame_without_x0_does_not_propagate(monkeypatch):

    game = _make_lq_game()

    strategy = object()

    monkeypatch.setattr(
        fsolvers,
        "solve_lqgame_feedback",
        lambda lqgame, **kwargs: strategy,
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError(
            "propagate_system_trajectory should not be called"
        )

    monkeypatch.setattr(
        fsolvers,
        "propagate_system_trajectory",
        fail_if_called,
    )

    result = pdg.solve(game)

    assert result.method == "lq"
    assert result.strategy is strategy
    assert result.trajectory is None


def test_solve_lqgame_zero_step_with_x0_propagates_single_node_trajectory(monkeypatch):

    tg = pdg.time_grid(
        nt=1,
        dt=0.1,
    )

    dynamics = pdg.linear_dynamics(
        A=jnp.eye(2),
        B=jnp.eye(2),
    )

    game = pdg.game(
        tg=tg,
        dynamics=dynamics,
        players=[
            pdg.player(
                cost=pdg.quadratic_cost(
                    nx=2,
                    nu=2,
                    control_weights=[1.0],
                    control_indices=[0],
                ),
                joint_ctrl_slice=slice(0, 1),
                name="p1",
            ),
            pdg.player(
                cost=pdg.quadratic_cost(
                    nx=2,
                    nu=2,
                    control_weights=[1.0],
                    control_indices=[1],
                ),
                joint_ctrl_slice=slice(1, 2),
                name="p2",
            ),
        ],
    )

    def fake_solve_lqgame_feedback(lqgame, **kwargs):
        return fsolvers.FixedStepAffineStrategies(
            tg=lqgame.tg,
            P=jnp.zeros((0, lqgame.nu, lqgame.nx)),
            alpha=jnp.zeros((0, lqgame.nu)),
        )

    monkeypatch.setattr(
        fsolvers,
        "solve_lqgame_feedback",
        fake_solve_lqgame_feedback,
    )

    x0 = jnp.array([1.5, -2.5])

    result = pdg.solve(
        game,
        x0=x0,
    )

    assert result.method == "lq"
    assert result.converged is True
    assert result.trajectory is not None
    assert result.trajectory.xs.shape == (1, 2)
    assert result.trajectory.us.shape == (0, 2)
    assert jnp.allclose(result.trajectory.xs[0], x0)


def test_solve_rejects_incompatible_explicit_method():

    game = _make_lq_game()

    with pytest.raises(ValueError, match="incompatible"):

        pdg.solve(
            game,
            method="ilq",
        )


def test_solve_ilq_auto_dispatches(monkeypatch):

    class FakeNLGame1:

        tg = SimpleNamespace(nt=3)

    monkeypatch.setattr(
        fsolvers,
        "NonlinearGameType1",
        FakeNLGame1,
    )

    game = FakeNLGame1()

    trajectory = object()
    strategy = object()

    def fake_solve_ilqgame_feedback(nlgame, x0, **kwargs):
        assert nlgame is game
        assert jnp.allclose(x0, jnp.array([2.0]))
        return False, trajectory, strategy

    monkeypatch.setattr(
        fsolvers,
        "solve_ilqgame_feedback",
        fake_solve_ilqgame_feedback,
    )

    result = pdg.solve(
        game,
        x0=jnp.array([2.0]),
    )

    assert result.method == "ilq"
    assert result.converged is False
    assert result.trajectory is trajectory
    assert result.strategy is strategy


def test_solve_ilq_frontend_game_auto_dispatches(monkeypatch):

    game = _make_nonlinear_game()

    trajectory = object()
    strategy = object()

    def fake_solve_ilqgame_feedback(nlgame, x0, **kwargs):
        assert isinstance(
            nlgame,
            fsolvers.NonlinearGameType1,
        )
        assert jnp.allclose(x0, jnp.array([2.0, -1.0]))
        return True, trajectory, strategy

    monkeypatch.setattr(
        fsolvers,
        "solve_ilqgame_feedback",
        fake_solve_ilqgame_feedback,
    )

    result = pdg.solve(
        game,
        x0=jnp.array([2.0, -1.0]),
    )

    assert result.method == "ilq"
    assert result.converged is True
    assert result.trajectory is trajectory
    assert result.strategy is strategy


def test_solve_ilq_requires_x0(monkeypatch):

    class FakeNLGame1:
        pass

    monkeypatch.setattr(
        fsolvers,
        "NonlinearGameType1",
        FakeNLGame1,
    )

    with pytest.raises(ValueError, match="`x0` is required"):

        pdg.solve(FakeNLGame1())


def test_solve_ilq_frontend_game_requires_x0(monkeypatch):

    game = _make_nonlinear_game()

    with pytest.raises(ValueError, match="`x0` is required"):

        pdg.solve(game)


def test_solve_al_frontend_game_auto_dispatches(monkeypatch):

    game = _make_constrained_nonlinear_game()

    class FakeALState:
        pass

    captured = {}

    def fake_init_joint_augmented_lagrangian_state(*, nc_ineq, nc_eq):
        captured["nc_ineq"] = nc_ineq
        captured["nc_eq"] = nc_eq
        return FakeALState()

    def fake_al_solve_autodiff(nlgame, op0, alstate0, **kwargs):
        captured["game"] = nlgame
        captured["op0"] = op0
        captured["alstate0"] = alstate0
        return "pdtraj", "alstate", "diag"

    monkeypatch.setattr(
        fsolvers,
        "init_joint_augmented_lagrangian_state",
        fake_init_joint_augmented_lagrangian_state,
    )
    monkeypatch.setattr(
        fsolvers,
        "al_solve_autodiff",
        fake_al_solve_autodiff,
    )

    result = pdg.solve(
        game,
        x0=jnp.array([1.0, -2.0]),
    )

    assert result.method == "al"
    assert result.primal_dual_trajectory == "pdtraj"
    assert result.al_state == "alstate"
    assert result.diagnostics == "diag"
    assert isinstance(captured["game"], fsolvers.NonlinearGameType2)
    assert isinstance(
        captured["op0"],
        fsolvers.FixedStepPrimalDualTrajectory,
    )
    assert captured["alstate0"].__class__ is FakeALState


def test_solve_al_auto_dispatches_with_default_initialization(monkeypatch):

    class FakeNLGame2:

        tg = pdg.time_grid(nt=4, dt=0.5)
        nt = 4
        nx = 2
        nu = 3
        N = 2
        constraints = SimpleNamespace(
            nc_ineq=5,
            nc_eq=2,
        )

    class FakeALState:
        pass

    monkeypatch.setattr(
        fsolvers,
        "NonlinearGameType2",
        FakeNLGame2,
    )

    captured = {}

    def fake_init_joint_augmented_lagrangian_state(*, nc_ineq, nc_eq):
        captured["nc_ineq"] = nc_ineq
        captured["nc_eq"] = nc_eq
        return FakeALState()

    def fake_al_solve_autodiff(nlgame, op0, alstate0, **kwargs):
        captured["game"] = nlgame
        captured["op0"] = op0
        captured["alstate0"] = alstate0
        return "pdtraj", "alstate", "diag"

    monkeypatch.setattr(
        fsolvers,
        "init_joint_augmented_lagrangian_state",
        fake_init_joint_augmented_lagrangian_state,
    )
    monkeypatch.setattr(
        fsolvers,
        "al_solve_autodiff",
        fake_al_solve_autodiff,
    )

    game = FakeNLGame2()

    result = pdg.solve(
        game,
        x0=jnp.array([3.0, -1.0]),
    )

    assert result.method == "al"
    assert result.primal_dual_trajectory == "pdtraj"
    assert result.al_state == "alstate"
    assert result.diagnostics == "diag"

    assert captured["game"] is game
    assert captured["nc_ineq"] == 5
    assert captured["nc_eq"] == 2
    assert isinstance(
        captured["op0"],
        fsolvers.FixedStepPrimalDualTrajectory,
    )
    assert jnp.allclose(
        captured["op0"].xs[0],
        jnp.array([3.0, -1.0]),
    )
    assert captured["alstate0"].__class__ is FakeALState


def test_solve_al_requires_x0_or_op0(monkeypatch):

    class FakeNLGame2:

        tg = pdg.time_grid(nt=3, dt=1.0)
        nt = 3
        nx = 1
        nu = 1
        N = 1
        constraints = SimpleNamespace(
            nc_ineq=0,
            nc_eq=0,
        )

    monkeypatch.setattr(
        fsolvers,
        "NonlinearGameType2",
        FakeNLGame2,
    )

    with pytest.raises(ValueError, match="`x0` or `op0`"):

        pdg.solve(FakeNLGame2())


def test_solve_al_zero_step_default_initialization(monkeypatch):

    class FakeNLGame2:

        tg = pdg.time_grid(nt=1, dt=0.5)
        nt = 1
        nx = 2
        nu = 3
        N = 2
        constraints = SimpleNamespace(
            nc_ineq=1,
            nc_eq=0,
        )

    class FakeALState:
        pass

    monkeypatch.setattr(
        fsolvers,
        "NonlinearGameType2",
        FakeNLGame2,
    )

    captured = {}

    def fake_init_joint_augmented_lagrangian_state(*, nc_ineq, nc_eq):
        captured["nc_ineq"] = nc_ineq
        captured["nc_eq"] = nc_eq
        return FakeALState()

    def fake_al_solve_autodiff(nlgame, op0, alstate0, **kwargs):
        captured["game"] = nlgame
        captured["op0"] = op0
        captured["alstate0"] = alstate0
        return "pdtraj", "alstate", "diag"

    monkeypatch.setattr(
        fsolvers,
        "init_joint_augmented_lagrangian_state",
        fake_init_joint_augmented_lagrangian_state,
    )
    monkeypatch.setattr(
        fsolvers,
        "al_solve_autodiff",
        fake_al_solve_autodiff,
    )

    game = FakeNLGame2()
    x0 = jnp.array([3.0, -1.0])

    result = pdg.solve(
        game,
        x0=x0,
    )

    assert result.method == "al"
    assert result.primal_dual_trajectory == "pdtraj"
    assert result.al_state == "alstate"
    assert result.diagnostics == "diag"
    assert captured["game"] is game
    assert captured["nc_ineq"] == 1
    assert captured["nc_eq"] == 0
    assert isinstance(captured["op0"], fsolvers.FixedStepPrimalDualTrajectory)
    assert captured["op0"].xs.shape == (1, game.nx)
    assert captured["op0"].us.shape == (0, game.nu)
    assert captured["op0"].ls.shape == (0, game.N, game.nx)
    assert jnp.allclose(captured["op0"].xs[0], x0)
    assert captured["alstate0"].__class__ is FakeALState
