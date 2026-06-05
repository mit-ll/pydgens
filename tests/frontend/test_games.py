# tests/frontend/test_games.py

import jax
import pytest
import jax.numpy as jnp

# direct import classes/functions that support tests
from pydgens.ir.timetypes import (
    time_grid,
)
from pydgens.frontend.costs import (
    ContinuousPlayerCost,
    QuadraticPlayerCost,
)
from pydgens.frontend.players import (
    Player,
    LQPlayer,
)
from pydgens.frontend.dynamics import (
    LTIContinuousSystem,
    NonlinearContinuousSystem,
)
from pydgens.ir.gametypes import (
    LinearQuadraticGameType1,
    NonlinearGameType1,
    NonlinearGameType2,
)
from pydgens.ir.costtypes import (
    ControlDomain,
    ControlStructure,
)

# module under test (via public api)
import pydgens as pdg

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def make_system():

    A = jnp.eye(2)

    B = jnp.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    return LTIContinuousSystem(
        A=A,
        B=B,
    )

def make_system_nx5_nu7():

    A = jnp.eye(5)

    B = jnp.ones((5,7))

    return LTIContinuousSystem(
        A=A,
        B=B,
    )


def make_nonlinear_system():

    return NonlinearContinuousSystem(
        nx=2,
        nu=2,
        dynamics=lambda t, x, u: jnp.array([
            x[1] + u[0],
            -x[0] + u[1],
        ]),
    )


def make_timegrid():

    return time_grid(
        nt=11,
        dt=0.1
    )


def make_player(
    *,
    ctrl_slice,
    nx=2,
    nu=2,
    name=None,
):

    cost = QuadraticPlayerCost(
        nx=nx,
        nu=nu,
    )

    start = ctrl_slice.start
    stop = ctrl_slice.stop

    cost.add_control_cost(
        weights=[1.0] * (stop - start),
        indices=list(range(start, stop)),
    )

    return LQPlayer(
        cost=cost,
        joint_ctrl_slice=ctrl_slice,
        name=name,
    )


def make_nonlinear_player(
    *,
    ctrl_slice,
    name=None,
):
    cost = ContinuousPlayerCost(
        running=lambda t, x, u: x[0] ** 2 + u[0] ** 2 + u[1] ** 2,
    )

    return Player(
        cost=cost,
        joint_ctrl_slice=ctrl_slice,
        name=name,
    )


def make_constraints():

    return pdg.constraint_set(
        pdg.control_bounds(
            lower=-1.0,
            upper=1.0,
            indices=[0],
        ),
        pdg.state_bounds(
            lower=-5.0,
            upper=5.0,
            indices=[1],
            include_terminal=True,
        ),
    )


# ---------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------


def test_lqgame_constructs():

    game = pdg.games.LQGame(
        tg=make_timegrid(),
        dynamics=make_system(),
        players=(
            make_player(ctrl_slice=slice(0, 1)),
            make_player(ctrl_slice=slice(1, 2)),
        ),
    )

    assert game.nx == 2
    assert game.nu == 2
    assert game.num_players == 2


def test_game_factory_returns_lqgame_for_lti_dynamics_and_lq_players():

    game = pdg.game(
        tg=make_timegrid(),
        dynamics=make_system(),
        players=[
            make_player(ctrl_slice=slice(0, 1)),
            make_player(ctrl_slice=slice(1, 2)),
        ],
    )

    assert isinstance(game, pdg.games.LQGame)


def test_game_factory_returns_nonlineargame_for_nonlinear_dynamics_and_generic_players():

    game = pdg.game(
        tg=make_timegrid(),
        dynamics=make_nonlinear_system(),
        players=[
            make_nonlinear_player(ctrl_slice=slice(0, 1)),
            make_nonlinear_player(ctrl_slice=slice(1, 2)),
        ],
    )

    assert isinstance(game, pdg.games.NonlinearGame)


def test_game_factory_returns_constrained_nonlineargame_when_constraints_supplied():

    game = pdg.game(
        tg=make_timegrid(),
        dynamics=make_nonlinear_system(),
        players=[
            make_nonlinear_player(ctrl_slice=slice(0, 1)),
            make_nonlinear_player(ctrl_slice=slice(1, 2)),
        ],
        constraints=make_constraints(),
    )

    assert isinstance(game, pdg.games.ConstrainedNonlinearGame)


def test_game_factory_forwards_discretization():

    game = pdg.game(
        tg=make_timegrid(),
        dynamics=make_system(),
        players=[
            make_player(ctrl_slice=slice(0, 1)),
            make_player(ctrl_slice=slice(1, 2)),
        ],
        discretization="euler",
    )

    assert isinstance(game, pdg.games.LQGame)
    assert game.discretization == "euler"


def test_game_factory_rejects_unsupported_player_types():

    cost = QuadraticPlayerCost(
        nx=2,
        nu=2,
    )

    unsupported_player = pdg.players.Player(
        cost=cost,
        joint_ctrl_slice=slice(0, 1),
    )

    with pytest.raises(NotImplementedError, match="Currently supported"):

        pdg.game(
            tg=make_timegrid(),
            dynamics=make_system(),
            players=[unsupported_player],
        )


def test_game_factory_rejects_frontend_constrained_lq_games_for_now():

    with pytest.raises(NotImplementedError, match="constrained LQ"):
        pdg.game(
            tg=make_timegrid(),
            dynamics=make_system(),
            players=[
                make_player(ctrl_slice=slice(0, 1)),
                make_player(ctrl_slice=slice(1, 2)),
            ],
            constraints=make_constraints(),
        )


def test_players_stored_as_tuple():

    game = pdg.games.LQGame(
        tg=make_timegrid(),
        dynamics=make_system(),
        players=[
            make_player(ctrl_slice=slice(0, 1)),
            make_player(ctrl_slice=slice(1, 2)),
        ],
    )

    assert isinstance(game.players, tuple)


def test_nonlineargame_players_stored_as_tuple():

    game = pdg.games.NonlinearGame(
        tg=make_timegrid(),
        dynamics=make_nonlinear_system(),
        players=[
            make_nonlinear_player(ctrl_slice=slice(0, 1)),
            make_nonlinear_player(ctrl_slice=slice(1, 2)),
        ],
    )

    assert isinstance(game.players, tuple)


def test_constrained_nonlineargame_players_stored_as_tuple():

    game = pdg.games.ConstrainedNonlinearGame(
        tg=make_timegrid(),
        dynamics=make_nonlinear_system(),
        players=[
            make_nonlinear_player(ctrl_slice=slice(0, 1)),
            make_nonlinear_player(ctrl_slice=slice(1, 2)),
        ],
        constraints=make_constraints(),
    )

    assert isinstance(game.players, tuple)


# ---------------------------------------------------------------------
# Validation: players
# ---------------------------------------------------------------------


def test_players_must_not_be_empty():

    with pytest.raises(ValueError, match="at least one"):

        pdg.games.LQGame(
            tg=make_timegrid(),
            dynamics=make_system(),
            players=[],
        )

    with pytest.raises(ValueError, match="at least one"):

        pdg.games.NonlinearGame(
            tg=make_timegrid(),
            dynamics=make_nonlinear_system(),
            players=[],
        )

    with pytest.raises(ValueError, match="at least one"):

        pdg.games.ConstrainedNonlinearGame(
            tg=make_timegrid(),
            dynamics=make_nonlinear_system(),
            players=[],
            constraints=make_constraints(),
        )


def test_players_must_be_lqplayers():

    with pytest.raises(TypeError, match="LQPlayer"):

        pdg.games.LQGame(
            tg=make_timegrid(),
            dynamics=make_system(),
            players=[object()],
        )


def test_nonlineargame_players_must_be_generic_players():

    with pytest.raises(TypeError, match="generic nonlinear `Player`"):

        pdg.games.NonlinearGame(
            tg=make_timegrid(),
            dynamics=make_nonlinear_system(),
            players=[object()],
        )


def test_nonlineargame_players_must_use_continuous_player_cost():

    bad_player = Player(
        cost=QuadraticPlayerCost(nx=2, nu=2),
        joint_ctrl_slice=slice(0, 2),
    )

    with pytest.raises(TypeError, match="ContinuousPlayerCost"):

        pdg.games.NonlinearGame(
            tg=make_timegrid(),
            dynamics=make_nonlinear_system(),
            players=[bad_player],
        )


# ---------------------------------------------------------------------
# Validation: discretization
# ---------------------------------------------------------------------


def test_discretization_must_be_valid():

    with pytest.raises(ValueError, match="discretization"):

        pdg.games.LQGame(
            tg=make_timegrid(),
            dynamics=make_system(),
            players=[
                make_player(ctrl_slice=slice(0, 1)),
                make_player(ctrl_slice=slice(1, 2)),
            ],
            discretization="invalid",
        )


# ---------------------------------------------------------------------
# Validation: player ordering
# ---------------------------------------------------------------------


def test_player_control_slices_must_match_list_order():

    p1 = make_player(
        ctrl_slice=slice(1, 2),
        name="p1",
    )

    p2 = make_player(
        ctrl_slice=slice(0, 1),
        name="p2",
    )

    with pytest.raises(ValueError, match="Inconsistent control ordering"):

        pdg.games.LQGame(
            tg=make_timegrid(),
            dynamics=make_system(),
            players=[p1, p2],
        )


def test_player_control_slices_must_be_contiguous():

    p1 = make_player(
        ctrl_slice=slice(0, 1),
        nu=3,
    )

    p2 = make_player(
        ctrl_slice=slice(2, 3),
        nu=3,
    )

    with pytest.raises(ValueError, match="Inconsistent control ordering"):

        pdg.games.LQGame(
            tg=make_timegrid(),
            dynamics=LTIContinuousSystem(
                A=jnp.eye(2),
                B=jnp.ones((2, 3)),
            ),
            players=[p1, p2],
        )


def test_player_control_slices_must_cover_full_joint_control():

    system = make_system()

    player = make_player(
        ctrl_slice=slice(0, 1),
    )

    with pytest.raises(ValueError, match="full joint control vector"):

        pdg.games.LQGame(
            tg=make_timegrid(),
            dynamics=system,
            players=[player],
        )


# ---------------------------------------------------------------------
# Validation: dimension consistency
# ---------------------------------------------------------------------


def test_player_cost_nx_must_match_game():

    system = make_system()

    bad_player = make_player(
        ctrl_slice=slice(0, 1),
        nx=3,
        nu=2,
    )

    good_player = make_player(
        ctrl_slice=slice(1, 2),
        nx=2,
        nu=2,
    )

    with pytest.raises(ValueError, match="cost nx"):

        pdg.games.LQGame(
            tg=make_timegrid(),
            dynamics=system,
            players=[bad_player, good_player],
        )


def test_player_cost_nu_must_match_game():

    system = make_system()

    bad_player = make_player(
        ctrl_slice=slice(0, 1),
        nx=2,
        nu=3,
    )

    good_player = make_player(
        ctrl_slice=slice(1, 2),
        nx=2,
        nu=2,
    )

    with pytest.raises(ValueError, match="cost nu"):

        pdg.games.LQGame(
            tg=make_timegrid(),
            dynamics=system,
            players=[bad_player, good_player],
        )


# ---------------------------------------------------------------------
# IR lowering
# ---------------------------------------------------------------------


def test_to_ir_returns_ir_game():

    game = pdg.games.LQGame(
        tg=make_timegrid(),
        dynamics=make_system(),
        players=[
            make_player(ctrl_slice=slice(0, 1)),
            make_player(ctrl_slice=slice(1, 2)),
        ],
    )

    ir_game = game.to_ir()

    assert isinstance(
        ir_game,
        LinearQuadraticGameType1,
    )


def test_to_ir_sets_correct_num_players():

    game = pdg.games.LQGame(
        tg=make_timegrid(),
        dynamics=make_system(),
        players=[
            make_player(ctrl_slice=slice(0, 1)),
            make_player(ctrl_slice=slice(1, 2)),
        ],
    )

    ir_game = game.to_ir()

    assert ir_game.N == 2


def test_to_ir_sets_correct_u_splits():

    game = pdg.games.LQGame(
        tg=make_timegrid(),
        dynamics=make_system(),
        players=[
            make_player(ctrl_slice=slice(0, 1)),
            make_player(ctrl_slice=slice(1, 2)),
        ],
    )

    ir_game = game.to_ir()

    assert jnp.allclose(
        ir_game.u_splits,
        jnp.array([1, 1]),
    )


def test_to_ir_sets_nontrivial_u_splits():

    dynamics = LTIContinuousSystem(
        A=jnp.eye(2),
        B=jnp.ones((2, 5)),
    )

    game = pdg.games.LQGame(
        tg=make_timegrid(),
        dynamics=dynamics,
        players=[
            make_player(ctrl_slice=slice(0, 2), nu=5),
            make_player(ctrl_slice=slice(2, 5), nu=5),
        ],
    )

    ir_game = game.to_ir()

    assert jnp.array_equal(
        ir_game.u_splits,
        jnp.array([2, 3]),
    )


def test_to_ir_sets_full_tensor_shapes():

    tg = make_timegrid()

    game = pdg.games.LQGame(
        tg=tg,
        dynamics=make_system(),
        players=[
            make_player(ctrl_slice=slice(0, 1)),
            make_player(ctrl_slice=slice(1, 2)),
        ],
    )

    ir_game = game.to_ir()

    assert ir_game.Q.shape == (tg.nt-1, 2, 2, 2)
    assert ir_game.q.shape == (tg.nt-1, 2, 2)
    assert ir_game.R.shape == (tg.nt-1, 2, 2, 2)
    assert ir_game.r.shape == (tg.nt-1, 2, 2)

def test_to_ir_sets_full_tensor_shapes_2():

    tg = make_timegrid()

    game = pdg.games.LQGame(
        tg=tg,
        dynamics=make_system_nx5_nu7(),
        players=[
            make_player(nx=5, nu=7, ctrl_slice=slice(0, 2)),
            make_player(nx=5, nu=7, ctrl_slice=slice(2, 6)),
            make_player(nx=5, nu=7, ctrl_slice=slice(6, 7)),
        ],
    )

    ir_game = game.to_ir()

    assert ir_game.Q.shape == (tg.nt-1, 3, 5, 5)
    assert ir_game.q.shape == (tg.nt-1, 3, 5)
    assert ir_game.R.shape == (tg.nt-1, 3, 7, 7)
    assert ir_game.r.shape == (tg.nt-1, 3, 7)


def test_to_ir_zero_step_time_grid_produces_empty_stage_arrays():

    tg = pdg.time_grid(
        nt=1,
        dt=0.1,
    )

    game = pdg.games.LQGame(
        tg=tg,
        dynamics=make_system(),
        players=[
            make_player(ctrl_slice=slice(0, 1)),
            make_player(ctrl_slice=slice(1, 2)),
        ],
    )

    ir_game = game.to_ir()

    assert ir_game.cs.nsteps == 0
    assert ir_game.cs.A.shape == (0, game.nx, game.nx)
    assert ir_game.cs.B.shape == (0, game.nx, game.nu)
    assert ir_game.Q.shape == (0, game.num_players, game.nx, game.nx)
    assert ir_game.q.shape == (0, game.num_players, game.nx)
    assert ir_game.R.shape == (0, game.num_players, game.nu, game.nu)
    assert ir_game.r.shape == (0, game.num_players, game.nu)
    assert jnp.array_equal(ir_game.u_splits, jnp.array([1, 1]))


# ---------------------------------------------------------------------
# IR cost lowering
# ---------------------------------------------------------------------


def test_to_ir_converts_Qp_to_Q():

    cost = QuadraticPlayerCost(
        nx=2,
        nu=2,
    )

    cost.Qp = jnp.diag(jnp.array([1.0, 2.0]))

    cost.add_control_cost(
        weights=[1.0],
        indices=[0],
    )

    player = LQPlayer(
        cost=cost,
        joint_ctrl_slice=slice(0, 1),
    )

    other_player = make_player(
        ctrl_slice=slice(1, 2),
    )

    game = pdg.games.LQGame(
        tg=make_timegrid(),
        dynamics=make_system(),
        players=[player, other_player],
    )

    ir_game = game.to_ir()

    expected_Q = 2.0 * cost.Qp

    assert jnp.allclose(
        ir_game.Q[0, 0],
        expected_Q,
    )


def test_to_ir_converts_x_ref_to_q():

    cost = QuadraticPlayerCost(
        nx=2,
        nu=2,
    )

    cost.Qp = jnp.diag(jnp.array([2.0, 3.0]))

    cost.x_ref = jnp.array([10.0, 20.0])

    cost.add_control_cost(
        weights=[1.0],
        indices=[0],
    )

    player = LQPlayer(
        cost=cost,
        joint_ctrl_slice=slice(0, 1),
    )

    other_player = make_player(
        ctrl_slice=slice(1, 2),
    )

    game = pdg.games.LQGame(
        tg=make_timegrid(),
        dynamics=make_system(),
        players=[player, other_player],
    )

    ir_game = game.to_ir()

    expected_q = -2.0 * cost.Qp @ cost.x_ref

    assert jnp.allclose(
        ir_game.q[0, 0],
        expected_q,
    )


def test_to_ir_converts_Rp_to_R():

    cost = QuadraticPlayerCost(
        nx=2,
        nu=2,
    )

    cost.Rp = jnp.diag(jnp.array([4.0, 5.0]))

    player = LQPlayer(
        cost=cost,
        joint_ctrl_slice=slice(0, 2),
    )

    game = pdg.games.LQGame(
        tg=make_timegrid(),
        dynamics=make_system(),
        players=[player],
    )

    ir_game = game.to_ir()

    expected_R = 2.0 * cost.Rp

    assert jnp.allclose(
        ir_game.R[0, 0],
        expected_R,
    )


def test_to_ir_converts_u_ref_to_r():

    cost = QuadraticPlayerCost(
        nx=2,
        nu=2,
    )

    cost.Rp = jnp.diag(jnp.array([2.0, 3.0]))

    cost.u_ref = jnp.array([10.0, 20.0])

    player = LQPlayer(
        cost=cost,
        joint_ctrl_slice=slice(0, 2),
    )

    game = pdg.games.LQGame(
        tg=make_timegrid(),
        dynamics=make_system(),
        players=[player],
    )

    ir_game = game.to_ir()

    expected_r = -2.0 * cost.Rp @ cost.u_ref

    assert jnp.allclose(
        ir_game.r[0, 0],
        expected_r,
    )


def test_to_ir_converts_terminal_state_cost_to_Qf_and_qf():

    cost = QuadraticPlayerCost(
        nx=2,
        nu=2,
    )

    cost.Rp = jnp.eye(2)

    cost.Qp_terminal = jnp.diag(jnp.array([2.0, 3.0]))
    cost.x_ref_terminal = jnp.array([10.0, 20.0])

    player = LQPlayer(
        cost=cost,
        joint_ctrl_slice=slice(0, 2),
    )

    game = pdg.games.LQGame(
        tg=make_timegrid(),
        dynamics=make_system(),
        players=[player],
    )

    ir_game = game.to_ir()

    assert jnp.allclose(ir_game.Qf[0], 2.0 * cost.Qp_terminal)
    assert jnp.allclose(ir_game.qf[0], -2.0 * cost.Qp_terminal @ cost.x_ref_terminal)


def test_to_ir_preserves_player_axis_order():

    cost1 = QuadraticPlayerCost(
        nx=2,
        nu=2,
    )
    cost1.add_control_cost(
        weights=[1.0],
        indices=[0],
    )
    cost1.Qp = jnp.diag(jnp.array([1.0, 3.0]))
    cost1.Rp = jnp.diag(jnp.array([4.0, 5.0]))

    cost2 = QuadraticPlayerCost(
        nx=2,
        nu=2,
    )
    cost2.add_control_cost(
        weights=[1.0],
        indices=[1],
    )
    cost2.Qp = jnp.diag(jnp.array([7.0, 11.0]))
    cost2.Rp = jnp.diag(jnp.array([13.0, 17.0]))

    game = pdg.games.LQGame(
        tg=make_timegrid(),
        dynamics=make_system(),
        players=[
            LQPlayer(
                cost=cost1,
                joint_ctrl_slice=slice(0, 1),
                name="p1",
            ),
            LQPlayer(
                cost=cost2,
                joint_ctrl_slice=slice(1, 2),
                name="p2",
            ),
        ],
    )

    ir_game = game.to_ir()

    expected_Q1 = jnp.broadcast_to(
        2.0 * cost1.Qp,
        (game.tg.nt-1, cost1.nx, cost1.nx),
    )
    expected_Q2 = jnp.broadcast_to(
        2.0 * cost2.Qp,
        (game.tg.nt-1, cost2.nx, cost2.nx),
    )
    expected_R1 = jnp.broadcast_to(
        2.0 * cost1.Rp,
        (game.tg.nt-1, cost1.nu, cost1.nu),
    )
    expected_R2 = jnp.broadcast_to(
        2.0 * cost2.Rp,
        (game.tg.nt-1, cost2.nu, cost2.nu),
    )

    assert jnp.allclose(ir_game.Q[:, 0], expected_Q1)
    assert jnp.allclose(ir_game.Q[:, 1], expected_Q2)
    assert jnp.allclose(ir_game.R[:, 0], expected_R1)
    assert jnp.allclose(ir_game.R[:, 1], expected_R2)


# ---------------------------------------------------------------------
# Time broadcasting
# ---------------------------------------------------------------------


def test_to_ir_broadcasts_over_time_grid():

    tg = make_timegrid()

    game = pdg.games.LQGame(
        tg=tg,
        dynamics=make_system(),
        players=[
            make_player(ctrl_slice=slice(0, 1)),
            make_player(ctrl_slice=slice(1, 2)),
        ],
    )

    ir_game = game.to_ir()

    # horizon = tg.nt - 1

    assert ir_game.Q.shape[0] == tg.nt-1
    assert ir_game.q.shape[0] == tg.nt-1
    assert ir_game.R.shape[0] == tg.nt-1
    assert ir_game.r.shape[0] == tg.nt-1


def test_to_ir_broadcasts_constant_costs_over_all_time_indices():

    cost = QuadraticPlayerCost(
        nx=2,
        nu=2,
    )

    cost.Qp = jnp.diag(jnp.array([2.0, 5.0]))
    cost.x_ref = jnp.array([10.0, -3.0])
    cost.Rp = jnp.diag(jnp.array([7.0, 13.0]))
    cost.u_ref = jnp.array([4.0, -2.0])

    game = pdg.games.LQGame(
        tg=make_timegrid(),
        dynamics=make_system(),
        players=[
            LQPlayer(
                cost=cost,
                joint_ctrl_slice=slice(0, 2),
            )
        ],
    )

    ir_game = game.to_ir()

    assert jnp.allclose(ir_game.Q[0, 0], ir_game.Q[-1, 0])
    assert jnp.allclose(ir_game.q[0, 0], ir_game.q[-1, 0])
    assert jnp.allclose(ir_game.R[0, 0], ir_game.R[-1, 0])
    assert jnp.allclose(ir_game.r[0, 0], ir_game.r[-1, 0])


def test_named_player_is_used_in_validation_error():

    bad_player = make_player(
        ctrl_slice=slice(1, 2),
        name="alice",
    )
    good_player = make_player(
        ctrl_slice=slice(0, 1),
        name="bob",
    )

    with pytest.raises(ValueError, match="alice"):

        pdg.games.LQGame(
            tg=make_timegrid(),
            dynamics=make_system(),
            players=[bad_player, good_player],
        )


def test_nonlineargame_to_ir_returns_ir_game():

    game = pdg.games.NonlinearGame(
        tg=make_timegrid(),
        dynamics=make_nonlinear_system(),
        players=[
            make_nonlinear_player(ctrl_slice=slice(0, 1)),
            make_nonlinear_player(ctrl_slice=slice(1, 2)),
        ],
    )

    ir_game = game.to_ir()

    assert isinstance(ir_game, NonlinearGameType1)
    assert ir_game.tg == game.tg
    assert ir_game.nx == game.nx
    assert ir_game.nu == game.nu
    assert jnp.array_equal(ir_game.u_splits, jnp.array([1, 1], dtype=jnp.int32))


def test_nonlineargame_to_ir_zero_step_time_grid_produces_valid_ir():

    tg = time_grid(
        nt=1,
        dt=0.1,
    )

    game = pdg.games.NonlinearGame(
        tg=tg,
        dynamics=make_nonlinear_system(),
        players=[
            make_nonlinear_player(ctrl_slice=slice(0, 1)),
            make_nonlinear_player(ctrl_slice=slice(1, 2)),
        ],
    )

    ir_game = game.to_ir()

    assert isinstance(ir_game, NonlinearGameType1)
    assert ir_game.cs.nt == 1
    assert ir_game.cs.nsteps == 0
    assert jnp.array_equal(ir_game.u_splits, jnp.array([1, 1], dtype=jnp.int32))


def test_constrained_nonlineargame_requires_constraint_set():

    with pytest.raises(TypeError, match="ConstraintSet"):
        pdg.games.ConstrainedNonlinearGame(
            tg=make_timegrid(),
            dynamics=make_nonlinear_system(),
            players=[
                make_nonlinear_player(ctrl_slice=slice(0, 1)),
                make_nonlinear_player(ctrl_slice=slice(1, 2)),
            ],
            constraints="not constraints",
        )


def test_constrained_nonlineargame_to_ir_returns_ir_game_type_2():

    game = pdg.games.ConstrainedNonlinearGame(
        tg=make_timegrid(),
        dynamics=make_nonlinear_system(),
        players=[
            make_nonlinear_player(ctrl_slice=slice(0, 1)),
            make_nonlinear_player(ctrl_slice=slice(1, 2)),
        ],
        constraints=make_constraints(),
    )

    ir_game = game.to_ir()

    assert isinstance(ir_game, NonlinearGameType2)
    assert ir_game.tg == game.tg
    assert ir_game.nx == game.nx
    assert ir_game.nu == game.nu
    assert jnp.array_equal(ir_game.u_splits, jnp.array([1, 1], dtype=jnp.int32))
    assert ir_game.constraints.nc_ineq > 0


def test_constrained_nonlineargame_to_ir_wraps_joint_costs_as_local_costs():

    running_1 = lambda t, x, u: 2.0 * u[0] + 7.0
    running_2 = lambda t, x, u: -3.0 * u[1] + x[0]

    players = [
        Player(
            cost=ContinuousPlayerCost(running=running_1),
            joint_ctrl_slice=slice(0, 1),
            name="p1",
        ),
        Player(
            cost=ContinuousPlayerCost(running=running_2),
            joint_ctrl_slice=slice(1, 2),
            name="p2",
        ),
    ]

    game = pdg.games.ConstrainedNonlinearGame(
        tg=make_timegrid(),
        dynamics=make_nonlinear_system(),
        players=players,
        constraints=make_constraints(),
    )

    ir_game = game.to_ir()

    x = jnp.array([5.0, -2.0])
    t = 0.3

    assert ir_game.costs[0].control_domain is ControlDomain.LOCAL
    assert ir_game.costs[1].control_domain is ControlDomain.LOCAL
    assert ir_game.costs[0].control_structure is ControlStructure.LOCAL_ONLY
    assert ir_game.costs[1].control_structure is ControlStructure.LOCAL_ONLY

    assert jnp.isclose(
        ir_game.costs[0].running(t, x, jnp.array([4.0])),
        running_1(t, x, jnp.array([4.0, 0.0])),
    )
    assert jnp.isclose(
        ir_game.costs[1].running(t, x, jnp.array([-3.0])),
        running_2(t, x, jnp.array([0.0, -3.0])),
    )


def test_constrained_nonlineargame_local_lowering_matches_joint_cost_over_samples():

    running_1 = lambda t, x, u: jnp.sin(t) + x[0] ** 2 + 3.0 * u[0] + u[0] ** 2
    running_2 = lambda t, x, u: x[1] * u[1] + 0.5 * u[1] ** 2

    players = [
        Player(
            cost=ContinuousPlayerCost(running=running_1),
            joint_ctrl_slice=slice(0, 1),
            name="p1",
        ),
        Player(
            cost=ContinuousPlayerCost(running=running_2),
            joint_ctrl_slice=slice(1, 2),
            name="p2",
        ),
    ]

    game = pdg.games.ConstrainedNonlinearGame(
        tg=make_timegrid(),
        dynamics=make_nonlinear_system(),
        players=players,
        constraints=make_constraints(),
    )
    ir_game = game.to_ir()

    samples = [
        (0.0, jnp.array([1.0, -2.0]), jnp.array([3.0]), jnp.array([-4.0])),
        (0.3, jnp.array([-1.5, 0.25]), jnp.array([-2.0]), jnp.array([1.5])),
        (1.2, jnp.array([0.2, 4.0]), jnp.array([0.0]), jnp.array([2.0])),
    ]

    for t, x, u1, u2 in samples:
        assert jnp.isclose(
            ir_game.costs[0].running(t, x, u1),
            running_1(t, x, jnp.array([u1[0], 0.0])),
        )
        assert jnp.isclose(
            ir_game.costs[1].running(t, x, u2),
            running_2(t, x, jnp.array([0.0, u2[0]])),
        )


def test_constrained_nonlineargame_local_lowering_preserves_owned_block_gradient():

    running_1 = lambda t, x, u: 1.5 * u[0] ** 2 + x[0] * u[0] + jnp.cos(t)
    running_2 = lambda t, x, u: 0.25 * u[1] ** 2 - x[1] * u[1]

    players = [
        Player(
            cost=ContinuousPlayerCost(running=running_1),
            joint_ctrl_slice=slice(0, 1),
            name="p1",
        ),
        Player(
            cost=ContinuousPlayerCost(running=running_2),
            joint_ctrl_slice=slice(1, 2),
            name="p2",
        ),
    ]

    game = pdg.games.ConstrainedNonlinearGame(
        tg=make_timegrid(),
        dynamics=make_nonlinear_system(),
        players=players,
        constraints=make_constraints(),
    )
    ir_game = game.to_ir()

    t = 0.4
    x = jnp.array([2.0, -3.0])
    u1 = jnp.array([1.25])
    u2 = jnp.array([-0.75])

    grad_local_1 = jax.grad(lambda u: ir_game.costs[0].running(t, x, u))(u1)
    grad_joint_1 = jax.grad(
        lambda u_joint: running_1(t, x, u_joint)
    )(jnp.array([u1[0], 0.0]))

    grad_local_2 = jax.grad(lambda u: ir_game.costs[1].running(t, x, u))(u2)
    grad_joint_2 = jax.grad(
        lambda u_joint: running_2(t, x, u_joint)
    )(jnp.array([0.0, u2[0]]))

    assert jnp.allclose(grad_local_1, grad_joint_1[0:1])
    assert jnp.allclose(grad_local_2, grad_joint_2[1:2])


def test_constrained_nonlineargame_to_ir_preserves_explicit_terminal_costs():

    terminal_1 = lambda t, x: x[0] ** 2 + t
    terminal_2 = lambda t, x: -2.0 * x[1] + 3.0

    players = [
        Player(
            cost=ContinuousPlayerCost(
                running=lambda t, x, u: u[0] ** 2,
                terminal=terminal_1,
            ),
            joint_ctrl_slice=slice(0, 1),
            name="p1",
        ),
        Player(
            cost=ContinuousPlayerCost(
                running=lambda t, x, u: u[1] ** 2,
                terminal=terminal_2,
            ),
            joint_ctrl_slice=slice(1, 2),
            name="p2",
        ),
    ]

    game = pdg.games.ConstrainedNonlinearGame(
        tg=make_timegrid(),
        dynamics=make_nonlinear_system(),
        players=players,
        constraints=make_constraints(),
    )

    ir_game = game.to_ir()

    t = 0.7
    x = jnp.array([4.0, -1.0])
    assert jnp.isclose(ir_game.costs[0].terminal(t, x), terminal_1(t, x))
    assert jnp.isclose(ir_game.costs[1].terminal(t, x), terminal_2(t, x))


def test_constrained_nonlineargame_local_lowering_erases_cross_player_control_dependence():

    running_1 = lambda t, x, u: u[0] ** 2 + 5.0 * u[1]
    running_2 = lambda t, x, u: u[1] ** 2

    players = [
        Player(
            cost=ContinuousPlayerCost(running=running_1),
            joint_ctrl_slice=slice(0, 1),
            name="p1",
        ),
        Player(
            cost=ContinuousPlayerCost(running=running_2),
            joint_ctrl_slice=slice(1, 2),
            name="p2",
        ),
    ]

    game = pdg.games.ConstrainedNonlinearGame(
        tg=make_timegrid(),
        dynamics=make_nonlinear_system(),
        players=players,
        constraints=make_constraints(),
    )

    ir_game = game.to_ir()

    t = 0.2
    x = jnp.array([0.0, 0.0])
    u1 = jnp.array([2.0])

    # This documents the current frontend limitation: if a supposed AL-
    # compatible frontend cost actually depends on another player's control,
    # lowering to local coordinates will silently erase that dependence.
    lowered_value = ir_game.costs[0].running(t, x, u1)
    original_value_with_other_player = running_1(t, x, jnp.array([2.0, 3.0]))

    assert not jnp.isclose(lowered_value, original_value_with_other_player)


def test_constrained_nonlineargame_to_ir_supplies_zero_terminal_when_missing():

    game = pdg.games.ConstrainedNonlinearGame(
        tg=make_timegrid(),
        dynamics=make_nonlinear_system(),
        players=[
            make_nonlinear_player(ctrl_slice=slice(0, 1)),
            make_nonlinear_player(ctrl_slice=slice(1, 2)),
        ],
        constraints=make_constraints(),
    )

    ir_game = game.to_ir()

    x = jnp.array([1.0, 2.0])
    assert jnp.isclose(ir_game.costs[0].terminal(0.0, x), 0.0)
    assert jnp.isclose(ir_game.costs[1].terminal(0.0, x), 0.0)
