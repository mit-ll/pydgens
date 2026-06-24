# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [UNRELEASED] - XXXX.XX.XX

### Added

- `test_alsolver.py` Added regression and benchmark coverage for constraint-heavy AL residual and stationarity-metrics paths.
- `test_alsolver.py` Added correctness and benchmark coverage comparing the experimental structured dynamics Jacobian slice against the autodiff Jacobian backend.
- `alsolver.py` Added logger-oriented diagnostics for tracing AL solver progress and identifying expensive residual/Jacobian evaluation paths.
- `alsolver.py` Added an experimental structured Jacobian backend for AL residual Jacobians, including dynamics feasibility blocks, dynamics-multiplier stationarity blocks, nonlinear-dynamics curvature blocks, player-local cost Hessian blocks, and auxiliary-constraint curvature blocks.
- `alsolver.py` Added a backend-dispatching `jacobian_al_residual_flat` entry point for selecting autodiff or structured AL residual Jacobian assembly.
- `test_alsolver.py` Added correctness and benchmark coverage comparing the experimental structured Jacobian backend against the autodiff backend across dynamics-only, nonlinear-dynamics, quadratic-cost, linear-constraint, and nonlinear-constraint cases.


### Changed

- `pyproject.toml` reorganized optional dependencies into `test`, `profile`, `docs`, `visuals`, `dev` (all prior dependecies), and `full` (pass-through wrapper of `dev`) for clarity of the purpose of each dependency
- `alsolver.py` Refactored augmented-Lagrangian residual assembly to reuse shared constraint linearizations and residual ingredients across stationarity-gradient computation.
- `alsolver.py` Reduced duplicated residual work in the stationarity Newton metrics path by deriving optimality, dynamics violation, and merit metrics from a single structured AL residual evaluation.
- `alsolver.py` Added a `jacobian_backend` option to the Newton-step, stationarity-solve, and AL-solve paths, with the structured AL residual Jacobian as the default and the brute-force autodiff backend available by explicit opt-in.
- `alsolver.py` Introduced `newton_step` as the preferred neutral Newton-step entry point while retaining `newton_step_autodiff` as a backward-compatible wrapper.
- `alsolver.py` Introduced `newton_solve_stationarity` and `al_solve` as preferred neutral solver entry points while retaining the `_autodiff` names as backward-compatible wrappers.
- `test_alsolver.py` Added Newton-step dispatch and benchmark coverage comparing autodiff and structured Jacobian backends end-to-end.
- `constrainttypes.py` Extended constraint linearizations to retain the originating constraint callable for structured Jacobian Hessian assembly.

## [v1.0.0] - 2026.06.23

### Added

- `README.md` adding example visuals
- `README.md` and `examples.md` linking the satellite LBG example to the related `spacegym-kspdg` competition environment
- `README.md` adding a minimal frontend API snippet for defining and solving an iLQ game
- `README.md` replacing the standalone documentation section with compact top-level navigation links
- `docs/solvers.md` sparse solver theory/reference scaffold
- `docs/references.md` expanded into a citeable reference index for solver documentation
- `README.md` switching image/source links to absolute URLs for PyPI rendering
- `examples/multi_car_intersection.py` frontend iLQ showcase example with bicycle-like car dynamics and soft intersection costs
- `examples/satellite_lady_bandit_guard.py` frontend LQ showcase example with Clohessy-Wiltshire orbital dynamics
- `scripts/visuals/multi_car_intersection_gif.py` for generating animated GIFs of the multi-car intersection example
- `scripts/visuals/satellite_lady_bandit_guard_gif.py` for generating Monte Carlo feedback rollout GIFs of the orbital LBG example
- `frontend/costs.py` matrix-based quadratic cost helper for advanced LQ games with coupled state costs
- `frontend/__init__.py` curated frontend namespace matching the top-level constructor API
- `docs/api.md` generated API reference scaffold using `mkdocstrings-python`
- Example smoke tests for the multi-car intersection and satellite Lady-Bandit-Guard examples
- `pillow` to optional dependencies for GIF rendering from `scripts/visuals`

### Changed

- `examples.md` adding links to source code and reformatting table
- `examples/multi_car_intersection.py` now warm-starts iLQ with a staggered lane-following initial strategy
- `QuadraticPlayerCost` matrix property setters now provide the canonical validation path for full state/control matrices
- Frontend public API docstrings clarified for generated API documentation
- `quadratic_cost(...)` now allows negative diagonal state and terminal-state weights, matching full state matrix behavior

## [v0.6.2] - 2026.06.17

### Added

- `test_tug_o_war.py` smoketests of basic LQ example
- `test_unicycle.py` smoketest of basic ILQ example
- `test_constrained_integrators.py` smoketest of basic AL example
- `frontend/solvers.py` a `SolveResult.__str__` method to unify result printing 
- `examples/_ir_reporting.py` helper functions for unifying result printing for IR examples
- `scripts/visuals/lady_bandit_guard_plot.py` for generating optional documentation plots outside the package
- `scripts/visuals/lady_bandit_guard_nonlinear_plot.py` for generating optional plots for the nonlinear LBG example
- `matplotlib` to optional dependencies to generate plots from `scripts/visuals`

### Changed

- `examples/constrained_integrators.py` removing specific args for solver to make example simpler, more friendly for a first-time user
- renamed `examples/unicycle1.py` to `examples/ir_unicycle.py` for clarity of its purpose
- `examples/ir_unicycle.py` now mirrors `examples/unicycle.py` as an advanced IR companion example
- renamed the legacy double-integrator LBG example to `examples/ir_lady_bandit_guard.py` and folded the non-plotting runner path into the example
- renamed the legacy aerial LBG example to `examples/ir_lady_bandit_guard_nonlinear.py` and folded the non-plotting runner path into the example
- renamed `examples/al_solve_example_1.py` to `examples/ir_constrained_integrators.py` and reframed it as an advanced AL IR tutorial
- renamed `examples/al_solve_example_xxx.py` to `examples/ir_constrained_double_integrator_diagnostic.py` and marked it as a solver diagnostic reference

### Removed 

- `examples/run_unicycle1.py` to clarify/simplify examples
- `examples/run_doubleint_lqlbg.py` and `examples/doubleint_lqlbg.json` to keep plotting/configuration helpers out of the package examples
- `examples/run_aeriallbg1.py` and `examples/aeriallbg1_cfg.json` to keep plotting/configuration helpers out of the package examples

## [v0.6.1] - 2026.06.05

### Changed

- `README.md` and `docs/`: cleaning up for first publication

### Fixed

- `workflows/`: specified uv setup hash
- `workflows/docs.yml`: testing push deploy

## [v0.6.0] - 2026.06.05

## Public Code Release

This version marks the start of open-sourcing.

The git history starts fresh at this point but the semantic versioning and CHANGELOG from closed-source development are kept for posterity

## [v0.5.1] - 2026.06.04

### Added

- `.github/workflows/publish.yml`: added PyPI release workflow using GitHub Actions trusted publishing.

### Changed

- `.github/workflows/`: replaced private-runner/private-container CI configuration with public GitHub-hosted workflows for tests, docs, packaging checks, and manual benchmarks.
- `zensical.toml` and `docs/`: simplified public documentation configuration, set the public GitHub Pages URL, and refreshed the initial docs shell for installation, examples, testing, references, and project layout.
- `README.md` and `pyproject.toml`: updated public-facing project metadata, PyPI packaging metadata, repository URLs, and installation/release notes for the public `mit-ll/pydgens` migration.
- `.gitignore`: added local environment and tool-cache ignores for public contributor workflows.
- `uv.lock`: Updating, removes diffrax references

### Removed

- `archives/`: removing old code for migration to public
- `tests/deprecated_tests.py`: removed tests that depended on the deleted `archives/` package.

## [v0.5.0] - 2026.06.02

### Added

- __init__.py: public API for `time_grid`, `linear_dynamics`, `player`, (many more)
- timetypes.py: public factory function `time_grid`
- systemtypes.py: public factor function `linear_dynamics` and new system type `LinearContinuousSystem`
- playertypes.py: new module for public API for clearly defining player objects in games
- frontend/players.py: public-facing interface for defining players within a game (note that, currently there is no explicit IR for players since this information is encoded in the systems and game definitions; however, it is much more intuitive for users to define player objects than whole systems/game objects)
- frontend/{dynamics,games,costs,constraints}.py: pythonic (OOP), public-facing interfaces for the jax-friendly (functional paradigm), intermediate representations (IR) of {systemtypes,gametypes,costtypes}.py
- frontend/solvers.py: pythonic (OOP), public-facing interface for jax-based solvers in `ir/solvers/`

### Changed

- Separated codebase into `frontend/`, `ir/` (intermediate representations), and `solvers/`.
    - `frontend/` is meant to hold user-facing, object oriented, pythonic functions and classes and the functions necessary to convert these into intermediate representations
    - `ir/` holds intermediate representations of objects using in game solving algorithms. These representations are characterized by highly-structured, functional-paradigm, traceable, jax-friendly dataclasses.
    - `solvers/` contain the algorithms for computing equilibria of various games. These solvers work on intermediate representations of game definitions  Solvers includ LQ, ILQ, and AL
    - To maintain JIT/JAX-friendly code hygeine: `frontend/` can import `ir/` and `solvers/`; but `ir/` and `solvers/` should NEVER IMPORT `frontend/`. This way we know that the objects used as inputs to game solvers are all traceble for JAX autodiff

- Refactored TimeGrid to make the distinction between time nodes (nt) and steps between nodes (nsteps) explicit
    - Making FixedStepSystemTrajectory have xs (nt, nx) and us shape (nt-1, nu)
    - Making FixedStepAffineStrategy have length nt-1
    - Making LinearDiscreteSystemTyp1 have nt-1 transition matrices
    - Making quadratic running costs have nt-1 time conventions
    - Making LinearQuadraticGameType1 follow nt-1 time conventions for dynamics and costs
    - Updated lqsolver and ilqsolver, and respective tests, to strictly adhere to new convention

- Updating approved-value regression tests
    - exampes/test_unicycle.py
    - approvals/example_doubleint/
    - approvals/lqsolver

- Refactored LQ games to include terminal state costs Qf, qf
    - gametypes.py: LinearQuadraticGameType1 now accepts Qf and qf matrices
    - lqsolver.py: Solvers now take Qf and qf to initialize Z0 and zeta0

- Renamed ControlCoupling to ControlStructure and added for enums to more clearly delineate between control cost types


### Fixed

- Initialization of Z and zeta in solve_lqgame_feedback to make consistent with a no-terminal-state-cost convention for LQ Games that is implied in https://github.com/HJReachability/ilqgames/blob/master/derivations/feedback_lq_nash.pdf

## [v0.4.2] - 2026.03.26

### Fixed

- Making CI scripts executable again after copyright header update

## [v0.4.1] - 2026.03.26

### Added

- LICENSE.txt
- SPDX.spdx
- README.md: logo and disclaimer distro A markings
- All .py files: adding copyright header

### Removed

- pyproject.toml: diffrax dependency

## [v0.4.0] - 2026.03.24

### Added
- alsolver.py: collection of dataclasses and functions to execute the augmented lagrangian solver
    - JointAugmentedLagrangianState dataclass to hold constraint multipliers and penalty weights
    - ALResidualStruct dataclass to hold the gradients of Lagrangians and discrete dynamics residual
    - many more, not all listed here
    - al_solve_autodiff: the top-level function for calling the ALGAMES-like augmented lagrangian solver. 
- scripts/: Scripts that, for example, help with CI testing but are not meant to be tested themselves (all tested code should be in src/)
- trajectorytypes.py: FixedStepPrimalDualTrajectory to encode state, control, and langrange trajectories from Algames.jl
- trajectorytypes.py: get_player_control_trajectory function for parsing a single player's control trajectory from a joint control trajectory
- costtypes.py: gradient functions for playerwise cost and terminal cost functions
- test_costtypes.py: unit tests for cost gradient functions
- utils.py: euler_step, rk2_step, rk3_step integrator
- test_utils.py: unit tests for euler_step, rk2_step, rk3_step, rk4_step
- systemtypes.py: `make_discrete_dynamics_step_map` for computing discrete dynamics function of continuous system at a single step which is a partial-generalization of the now-renamed `discretize_dynamics`, which only was for linear systems extended across full time horizon using euler integration.
- systemtypes.py: `jacobian_discrete_dynamics_step` for computing partial derivatives of dynamics with respect to joint state and joint control at a particular (time, state, control) tuple
- systemtypes.py: `jacobian_discrete_dynamics_trajectory` the trajectory-extended version of jacobian_discrete_dynamics_step
- constrainttypes.py: starting new dataclasses to define constraint functions for augmented lagrangian solver
- test_constrainttypes.py: pytests for new constraints module
- ~~constrainttypes.py: BasicConstraint dataclass and evaluate and jacobian and cost_expansion functions for constraints~~
- constrainttypes.py: ~~JointConstraintMap~~ ConstraintBlockGridMap and GameConstraintGridMap dataclasses to hold inequality and equality constraints (except dynamic constraints), including state and control constraints, across all players 
- constrainttypes.py: Helper class ConstraintStepLinearization and functions _normalize_constraint_output_1d, _linearize_step_constraint_kernel, _linearize_terminal_constraint_kernel, build_constraint_step_linearizations, accumulate_Jt_weighted_vector to better manage gradient/jacobian/linearization of constraints for lagrangian gradient computation
- gametypes.py: NonlinearGameType2 to hold constrained nonlinear games
- alsolver.py: gradient_aug_lagrangian_trajectory function for computing gradient of each player's augmented lagrangian function with respect to state and controls, along a trajectory
- alsolver.py: gradient_aug_lagrangian_playerwise_trajectory_dynamics function for computing the dynamics term of the augmented lagrangian
- alsolver.py: gradient_aug_lagrangian_trajectory_constraints function for computing the constraints term of the augmented lagrangian
- alsolver.py: gradient_aug_lagrangian_trajectory_penalty function for computing the constraints penalty term of the augmented lagrangian
- costtypes.py: PlayerCostSpecContinuous a light wrapper of cost callables to better imply/enforce cost API contract
- costtypes.py: Helper classes and functions the help in enforcing API contract on costs within downstream functions (e.g. games and solvers). Classes include PlayerCostFnCtrlJoint, PlayerCostFnCtrlLocal, PlayerCostFnTerminal, ControlDomain, ControlCoupling, validate_player_cost_spec_continuous, detect_control_coupling

### Changed
- workflows/benchmark.yml: updating to run more sophisticated benchmark automatically using most recent release as baseline and enabling manually specificed baseline and target and manual run. Removing failure-on-regression since benchmarking hardware is not consistent enough and failures are almost certainly due to processor load variability, rather than actual regressions. Adding readable summary for inspection
- systemtypes.py: renamed `discretize_dynamics` to `discretize_extended_linear_dynamics_euler` to clarify that it is a special case of discretization
- systemtypes.py and utils.py: moving rk4_step to utils to declutter systemtypes.py and prep for other integrator functions
- systemtypes.py, trajectorytypes.py, gametypes.py: renamed single-dispatch functions to make them easier to stacktrace
- gametypes.py: converting LinearQuadraticGameType1 and NonlinearGameType1 to stdlib frozen dataclasses instead of flax.struct.dataclasses because they hold non-array data like python callables
- gametypes.py: enforcing u_splits be integer array for all GameTypes
- costtypes.py: renamed quadraticize_no_checks to quadraticize_cost_joint_ctrl_no_checks, quadraticize_cost_playerwise to quadraticize_cost_joint_ctrl_playerwise, quadraticize_cost_playerwise_trajectory to quadraticize_cost_joint_ctrl_playerwise_trajectory
- gametypes.py: enforcing player cost functions to be PlayerCostSpecContinuous
- test_gametypes.py: importing gametypes as module, instead of individual classes and functions
- test_lqsolver.py, test_example_doubleint.py: tuning test tolerances to pass on other systems


## [v0.3.1] - 2026.02.18

### Changed:

- github/workflows: removed uv setup in github action workflows to enable performance benchmarking back to v0.2.0 (which did not have uv or a uv.lock)

## [v0.3.0] - 2026.02.18

### Added:

- .github/workflows: yml files for running github actions to be run via AWS GovCloud including benchmark.yml (performance benchmarks), python-builds.yml (fast unit tests), and documentation.yml (documentation builds) 
- docs/: documentation served by zensical
- pyproject.toml: dependencies for document creation (zensical) and expanded profiling tests
- uv.lock: explicit dependency version info needed for consistent github actions testing

## [v0.2.0] - 2025.11.13

### Added
- test_example_doubleint.py: approved values regression tests for lq solver
- tests/: benchmark tests for lqsolver

### Changed
- lqsolver.py: converting python loop to `lax.scan` for performance boost

## [v0.1.1] - 2025.11.10

### Added
- costtypes.py: add function for computing scalar cost from quadratic cost matrices and moved goal cost function from lqsolver

### Changed
- examples/doubleint.py: modified param passing so that they should be able to be overridden/modified more completely in child classes

### Removed
- lqsolver.py: commented code and unused imports

## [v0.1.0] - 2025.11.06

Arbitrarily bumping to v0.1.0 now that code has reached a moderate level of maturity and testing (probably should have bumped at last version).

### Added
- examples/run_* scripts: separating the example source code from a runner script to enable easy importing of example classes by external libraries without requiring bloated dependencies like matplotlib which are only needed for analysis in the example runner scripts

### Fixed
- examples/aeriallbg1.py: small error comment fixes

## [v0.0.5] - 2025.10.14

### Added
- examples/unicycle1.py: added jax profiler to example script for finding bottlenecks in the solver execution
- pyproject.toml: pytest marks for regression and slow tests
- test_example_unicycle1.py: basic regression and performance tests
- costtypes.py: quadraticize_no_checks is the core functionality of quadraticize_cost_playerwise_trajectory and quadraticize_cost_playerwise, but without any input checking to avoid jit errors
- pyproject.toml, test_example_unicycle1.py: adding pytest-benchmark and benchmark tests for Unicycle1 for regression testing computational performance
- test_costtypes.py: adding benchmark test for quadraticize_cost_playerwise_trajectory
- utils/generators.py: module for functions that create randomized functions or data to be used in (for example) testing
- test_gametypes.py: adding benchmark test for approx_linear_quadratic_game
- test_systemtypes.py: adding benchmark tests for propagate_system_trajectory and next_x

### Changed
- timetypes.py: removed conditionals from `disc2cont` to speed up execution. Note that out-of-bounds discrete time steps will no longer cause NaNs, which may lead to unexpected, silent behavior
- lqsolver.py: made is_block_diagonal check possible to be turned off to speed up computation. Turned off in ilqsolver.py, but added some unit tests to check that approx_linear_quadratic_game produces block-diagonal R matrices
- lqsolver.py: refactored solve_lqgame_feedback for better, more efficient indexing (cut solve_lqgame_feedback avg execution time in half)
- costtypes.py: optimizing quadraticize_cost_playerwise_trajectory using a mask on cross terms instead of iterating through each other player's effect on player i's R_i. Reduces time to first execution of quadraticize_cost_playerwise_trajectory significantly
- systemtypes.py: optimizing linearize_dynamics to produce ~5x speedup in first execution, and 400x speedup in warm execution of the function
- systemtypes.py: optimizing propogate_system_trajectory
- systemtypes.py: deprecating `next_x` and adding warning

## [v0.0.4] - 2025.10.02

### Added
- examples/unicycle1.py: very simple example game for debugging purposes
- gametypes.approx_linear_quadratic_game: function for approximating a linear quadratic game from a nonlinear one. This packages into one function a multistep process that used to be contained in ilqsolver.solve_approx_lqgame_feedback
- costtypes.py: to isolate quadraticize_cost functions to avoid circular imports between ilqsolver.py and gametypes.py

### Changed
- ilqsolver.py: replacing multi-step linearization-discretization with single call to approx_linear_discrete_system

### Fixed
- lqsolver.py: fixed indexing problem with backwards iteration through time that was missing the final index
- ilqsolver.py: backtrack_scale_strategy move the mapping from strategy from (delx, delu)->(x,u) into backtrack_scale_strategy, where it should have always been, in order the rescale the appropriate portion of the alpha term

### Removed
- ilqsolver.py: solve_approx_lqgame_feedback function, its functionality has been redistributed to functions like gametypes.py approx_linear_quadratic_game and backtrack_scale (where mapping )

## [v0.0.3] - 2025.09.29

### Added
- pyproject.toml: flax dependency
- timetypes.py: dataclasses for defining time characteristics used as core component composed into other classes (systemtypes, gametypes, strategytypes, trajectorytypes) to ensure compatibility
- systemtypes.py: dataclasses for defining control systems to disambiguate concepts like continuous control systems from linear discrete systems and make them not context-dependent. Using immutable flax.struct.dataclass to make these more JAX-compatible
- archives/deprecated.py: module for storing deprecated functions for reference in new functions while cleaning up existing modules
- ilqsolver.py: adding convergence boolean and operating point trajectory to output of solve_ilqgame_feedback
- ilqsolver.py: adding logger for better debugging
- examples/target_guard_3N_nlgame.py: example of running ilq solver on a 3-player target guarding problem under simplified 3DOF-aircraft dynamics (i.e. 4D unicyle)

### Fixed
- ilqsolver.py: Reworked solve_ilqgame_feedback function so as not to trivially return initial strategy when it was used to compute the initial trajectory. Also, fixed how the LQGame solution is interpretted to account for the fact that the LQGame is a Taylor series expansion around the current operating point, and thus the strategy from solving the LQGame is inherently in the (delx, delu) space, rather than the (x, u) space

### Changed
- get_game_trajectory -> propagate_system_trajectory that takes as input newly defined system type (e.g. SampledContinuousSystem) to be more explicit about how propagation should work
- gametypes.py: Refactoring game type classes to immutable flax.struct.dataclass to make them more JAX-compatible. Added NonlinearGameType1 dataclass
- trajectorytypes.py: Refactoring SystemTrajector to immutable flax.struct.dataclass to make them more JAX-compatible
- strategytypes.py: Refactoring AffineStrategy to immutable flax.struct.dataclass to make them more JAX-compatible
- lqsolver.py: Updating to use new control system and gametype framework
- ilqsolver.py: changed inputs to solve_ilqgame_feedback to use initial state used to compute first trajectory
- ilqsolver.solve_approx_lqgame_feedback: discretizing the linearized dynamics to align with problem formulation in iLQGames and theory: https://github.com/HJReachability/ilqgames/blob/master/derivations/feedback_lq_nash.pdf
- ilqsolver.py: re-interpretted game.T parameter to be the length of the time vector `ts` in system trajectory. This differs from the previous interpretation as T being the number of time steps (which can ambiguously mean that one time step may imply time vector of length 1 or length 2 because you may record the times at both ends of the single step or not). This new interpretation is in line with ilqgames.jl game horizon definition. It also means that the max time in the time vector is (T-1)*dt, rather than T*dt.
- ilqsolver.py, systemtypes.py: reorganizing linearize_dynamics and discretize_dynamics into systemtypes.py
- trajectorytypes.py: refactoring ts vector into dt and to (and implicit nt) in order to enable compatibility checking between trajectories, strategies, and control systems
- trajectorytypes.py: renaming SystemTrajectory dataclass to FixedStepSystemTrajectory to be more explicit
- strategytypes.py: renaming AffineStrategy to FixedStepAffineStrategies and composing TimeGrid to enforce consistency with other objects
- examples: renamed target_guard_3N_fblin_lqgame -> aeriallbg2, and target_guard_3N_nlgame -> aeriallbg1


## [v0.0.2] - 2025.06.30

First implementation of iLQGames algorithms for solving nonlinear/nonquadratic games 

### Added 
- ilqsolver.py: `solve_ilqgame_feedback` function the runs the iterative linear-quadratic algorithm to solve for local feedback Nash equilibrium
- ilqsolver.py: `backtrack_scale_strategy` function for scaling a new AffineStrategy such that trajectories of such strategy do not deviate too much from trajectories of an existing strategy
- gametypes.py to define classes of games: AbstractBaseGame, AbstractFiniteHorizonGame, LinearQuadraticGame, NonlinearGame
- strategytypes.py to define classes of strategies (e.g. AffineStrategy)
- trajectorytypes.py to define classes of system trajectories
- examples/dubins_flat_lq: example problem sandboxing the usage of LQ games to develop strategies of nonlinear, yet differentially-flat systems (INCOMPLETE)
- examples/unicycle_fblin_lq: example problem sandboxing the usage of LQ games to develop strategies of nonlinear, yet feedback linearizable system
- examples/plane_fblin_lq.py: example problem in 5D airplane dynamics with pointing and pursuit cost functions
- examples/target_guard_3N_fblin_lqgame.py: example of 3-player target guarding game under feedback linearizable 4D unicycle dynamics
- tests/runtime_analysis.py: cursory analysis of TTFE (time-to-first-execution) and SSET (steady-state-execution-time) for comparison with julia. Run with `python tests/runtime_analysis.py`

### Changed
- renamed ilqgames.py to ilqsolver.py to better distinguish game definitions vs game solvers
- update ilqsolver.py to take as input a LinearQuadraticGame object instead of the individual matrices
- quadraticize_cost_playerwise now outputs block diagonal matrices for hessian of cost with respect to control (i.e. R_i) instead of a list of jnp.ndarrays in order to better align with LinearQuadraticGame definitions


## [v0.0.1] - 2025.05.20

### Added

- pyproject.toml: jax and diffrax dependencies
- src/ilqgames.py: jax implementation of ilqgames functions
- src/lqsolver.py: jax implementation of linear-quadratic feedback Nash solver
- src/utils/utils.py: general-purpose utility functions
- utils.py: is_block_diagonal to test a if matrix is block diagonal
- utils.py: is_positive_semidefinite to test if a matrix is positive semidefinite
- ilqgames.py: get_game_trajectory function and tests for propagating game trajectory from affine control policy
- ilqgames.py: linearize_dynamics function and tests for computing jacobians of dynamics at operating points
- ilqgames.py: quadraticize_cost_playerwise and tests for computing quadraticization of cost functions at operating points
