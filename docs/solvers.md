---
icon: material/chart-timeline-variant
---

# Solvers

This page is a sparse map of the solver families exposed by PYDGENS. It is not
yet a full theory guide.

## Linear-Quadratic Games

The `LQ` path solves finite-horizon linear-quadratic dynamic games for feedback
Nash strategies.

Used by:

- [`tug_o_war.py`](../src/pydgens/examples/tug_o_war.py)
- [`satellite_lady_bandit_guard.py`](../src/pydgens/examples/satellite_lady_bandit_guard.py)

Theory notes:

- TODO: summarize finite-horizon coupled Riccati/backward-recursion structure.
- TODO: document sign conventions and frontend-to-IR quadratic scaling.
- References: [Dynamic Noncooperative Game Theory](references.md#dynamic-noncooperative-game-theory), [Feedback LQ Nash Derivation](references.md#feedback-lq-nash-derivation).

## Iterative Linear-Quadratic Games

The `iLQ` path repeatedly builds local linear-quadratic approximations of a
nonlinear game and solves those approximations for local feedback Nash updates.

Used by:

- [`unicycle.py`](../src/pydgens/examples/unicycle.py)
- [`multi_car_intersection.py`](../src/pydgens/examples/multi_car_intersection.py)

Theory notes:

- TODO: summarize local game approximation, feedback update, and line-search behavior.
- TODO: clarify convergence diagnostics and failure modes.
- References: [iLQGames](references.md#ilqgames), [Smooth Game Theory](references.md#smooth-game-theory).

## Augmented-Lagrangian Games

The `AL` path targets constrained nonlinear games with local open-loop
trajectories. This solver path is currently beta/pre-release.

Used by:

- [`constrained_integrators.py`](../src/pydgens/examples/constrained_integrators.py)

Theory notes:

- TODO: document the augmented-Lagrangian state, multiplier updates, and regularization strategy.
- TODO: explain which constraints are currently supported by the frontend.
- Reference: [ALGAMES](references.md#algames).
