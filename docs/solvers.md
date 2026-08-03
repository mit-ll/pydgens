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

- [`tug_o_war.py`](https://github.com/mit-ll/pydgens/blob/main/src/pydgens/examples/tug_o_war.py)
- [`satellite_lady_bandit_guard.py`](https://github.com/mit-ll/pydgens/blob/main/src/pydgens/examples/satellite_lady_bandit_guard.py)

Theory notes:

- TODO: summarize finite-horizon coupled Riccati/backward-recursion structure.
- TODO: document sign conventions and frontend-to-IR quadratic scaling.
- References: [Dynamic Noncooperative Game Theory](references.md#dynamic-noncooperative-game-theory), [Feedback LQ Nash Derivation](references.md#feedback-lq-nash-derivation).

## Iterative Linear-Quadratic Games

The `iLQ` path repeatedly builds local linear-quadratic approximations of a
nonlinear game and solves those approximations for local feedback Nash updates.

The iLQ solver accepts scalar or per-state-component absolute bounds for both
its convergence check (`converged_max_diff`) and backtracking rollout check
(`backtrack_scale_max_diff`). Supplying a vector of length `nx` is useful for
mixed-unit states, for example separate position and heading tolerances.

Iterative solvers accept `diagnostics_level="off"`, `"basic"`, or
`"detailed"`. This controls retained diagnostics rather than logger
configuration: solver loggers emit compact renderings of retained records only
when the caller has independently enabled the corresponding logging level.
For iLQ, each basic iteration record includes the raw state-update infinity
norm, its componentwise convergence-tolerance-normalized infinity norm, and
the numeric time-node and state-index location of that normalized maximum.

Used by:

- [`unicycle.py`](https://github.com/mit-ll/pydgens/blob/main/src/pydgens/examples/unicycle.py)
- [`multi_car_intersection.py`](https://github.com/mit-ll/pydgens/blob/main/src/pydgens/examples/multi_car_intersection.py)

Theory notes:

- TODO: summarize local game approximation, feedback update, and line-search behavior.
- TODO: clarify convergence diagnostics and failure modes.
- References: [iLQGames](references.md#ilqgames), [Smooth Game Theory](references.md#smooth-game-theory).

## Augmented-Lagrangian Games

The `AL` path targets constrained nonlinear games with local open-loop
trajectories. This solver path is currently beta/pre-release.

Used by:

- [`constrained_integrators.py`](https://github.com/mit-ll/pydgens/blob/main/src/pydgens/examples/constrained_integrators.py)

Theory notes:

- TODO: document the augmented-Lagrangian state, multiplier updates, and regularization strategy.
- TODO: explain which constraints are currently supported by the frontend.
- Reference: [ALGAMES](references.md#algames).
