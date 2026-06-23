---
icon: material/file-document-check
---

# Examples

Examples live in [`src/pydgens/examples/`](../src/pydgens/examples/). These scripts are useful both as smoke tests and as starting points for new game formulations.

## At A Glance

| Example | Command | Solve Path | Interface | Notes |
| --- | --- | --- | --- | --- |
| [Tug-of-War](../src/pydgens/examples/tug_o_war.py) | `python -m pydgens.examples.tug_o_war` | `LQ` -> feedback Nash equilibrium | Top-level API | Smallest LQ walkthrough; includes analytical comparison. |
| [Satellite Lady-Bandit-Guard](../src/pydgens/examples/satellite_lady_bandit_guard.py) | `python -m pydgens.examples.satellite_lady_bandit_guard` | `LQ` -> feedback Nash equilibrium | Top-level API | Showcase orbital LQ example with Clohessy-Wiltshire dynamics and coupled quadratic costs; related to [`spacegym-kspdg`](https://github.com/mit-ll/spacegym-kspdg). |
| [Two-Player Unicycle](../src/pydgens/examples/unicycle.py) | `python -m pydgens.examples.unicycle` | `iLQ` -> local feedback Nash equilibrium | Top-level API | Beginner nonlinear example using the semantic frontend. |
| [Multi-Car Intersection](../src/pydgens/examples/multi_car_intersection.py) | `python -m pydgens.examples.multi_car_intersection` | `iLQ` -> local feedback Nash equilibrium | Top-level API | Showcase nonlinear example with soft collision, lane, and speed penalties. |
| [Constrained Integrators](../src/pydgens/examples/constrained_integrators.py) | `python -m pydgens.examples.constrained_integrators` | `AL` -> local open-loop Nash equilibrium | Top-level API | Beginner constrained example using the frontend API. |
| [IR Two-Player Unicycle](../src/pydgens/examples/ir_unicycle.py) | `python -m pydgens.examples.ir_unicycle` | `iLQ` -> local feedback Nash equilibrium | IR | Direct IR companion to `unicycle.py`. |
| [IR Lady-Bandit-Guard](../src/pydgens/examples/ir_lady_bandit_guard.py) | `python -m pydgens.examples.ir_lady_bandit_guard` | `LQ` -> feedback Nash equilibrium | IR | Advanced three-player LQ example built from cost matrices. |
| [IR Constrained Integrators](../src/pydgens/examples/ir_constrained_integrators.py) | `python -m pydgens.examples.ir_constrained_integrators` | `AL` -> local open-loop Nash equilibrium | IR | Advanced AL companion to `constrained_integrators.py`. |
| [IR Nonlinear Lady-Bandit-Guard (Diagnostic)](../src/pydgens/examples/ir_lady_bandit_guard_nonlinear.py) | `python -m pydgens.examples.ir_lady_bandit_guard_nonlinear` | `iLQ` -> local feedback Nash equilibrium attempt | IR | Runs to completion but may report non-convergence. |
| [IR Constrained Double-Integrator (Diagnostic)](../src/pydgens/examples/ir_constrained_double_integrator_diagnostic.py) | `python -m pydgens.examples.ir_constrained_double_integrator_diagnostic` | `AL` -> local open-loop Nash equilibrium attempt | IR | Prints a problem summary by default; use `--run-solver` to exercise the known-problematic solve path. |

The `Interface` column distinguishes the beginner-facing semantic frontend from the lower-level solver IR. Diagnostic examples are marked directly in the `Example` column.

## Tug-of-War

Source: [`src/pydgens/examples/tug_o_war.py`](../src/pydgens/examples/tug_o_war.py)

Compute a feedback Nash equilibrium for a simple linear-quadratic game with the `LQ` solver:

```bash
python -m pydgens.examples.tug_o_war
```

## Satellite Lady-Bandit-Guard

Source: [`src/pydgens/examples/satellite_lady_bandit_guard.py`](../src/pydgens/examples/satellite_lady_bandit_guard.py)

Run a linear-quadratic orbital Lady-Bandit-Guard game where bandit and guard satellites move under Clohessy-Wiltshire dynamics relative to a passive lady/reference orbit:

```bash
python -m pydgens.examples.satellite_lady_bandit_guard
```

This simplified LQ example is closely related to the orbital pursuit-evasion-protection setting in [`mit-ll/spacegym-kspdg`](https://github.com/mit-ll/spacegym-kspdg), a separate public competition environment.

Optional visualization script:

```bash
python scripts/visuals/satellite_lady_bandit_guard_gif.py  --samples 32  --position-sigma 4.0 --z-position-sigma 0.1 --nt 101 --output docs/assets/satellite_lady_bandit_guard.gif
```

## Two-Player Unicycle

Source: [`src/pydgens/examples/unicycle.py`](../src/pydgens/examples/unicycle.py)

Run the iterative linear-quadratic solver (`iLQ`) on a nonlinear game where two players control a shared unicycle system:

```bash
python -m pydgens.examples.unicycle
```

## Multi-Car Intersection

Source: [`src/pydgens/examples/multi_car_intersection.py`](../src/pydgens/examples/multi_car_intersection.py)

Run a four-car nonlinear intersection game with bicycle-like vehicle dynamics, soft collision avoidance, lane keeping, speed penalties, and iLQ feedback solving:

```bash
python -m pydgens.examples.multi_car_intersection
```

Optional visualization script:

```bash
python scripts/visuals/multi_car_intersection_gif.py --compare-naive
```

## Constrained Integrators

Source: [`src/pydgens/examples/constrained_integrators.py`](../src/pydgens/examples/constrained_integrators.py)

Run the augmented-Lagrangian (`AL`) solver on a constrained nonlinear game with bounded controls:

```bash
python -m pydgens.examples.constrained_integrators
```

## IR Two-Player Unicycle

Source: [`src/pydgens/examples/ir_unicycle.py`](../src/pydgens/examples/ir_unicycle.py)

Build the same unicycle game directly with the intermediate representation (`IR`) objects used by the iLQ solver:

```bash
python -m pydgens.examples.ir_unicycle
```

## IR Lady-Bandit-Guard

Source: [`src/pydgens/examples/ir_lady_bandit_guard.py`](../src/pydgens/examples/ir_lady_bandit_guard.py)

Build and solve a richer linear-quadratic Lady-Bandit-Guard game directly with the LQ solver's IR objects:

```bash
python -m pydgens.examples.ir_lady_bandit_guard
```

## IR Constrained Integrators

Source: [`src/pydgens/examples/ir_constrained_integrators.py`](../src/pydgens/examples/ir_constrained_integrators.py)

Build and solve the constrained integrator game directly with the augmented-Lagrangian (`AL`) solver's IR objects:

```bash
python -m pydgens.examples.ir_constrained_integrators
```

## Solver Diagnostic Examples

These examples are useful reference problems for solver development, but they are not polished smoke tests. They may fail to converge, take a long time, or require explicit opt-in flags before running the solver.

### IR Nonlinear Lady-Bandit-Guard

Source: [`src/pydgens/examples/ir_lady_bandit_guard_nonlinear.py`](../src/pydgens/examples/ir_lady_bandit_guard_nonlinear.py)

Run the nonlinear unicycle-dynamics Lady-Bandit-Guard counterpart directly with the iLQ solver's IR objects. This example currently executes to completion but may report non-convergence.

```bash
python -m pydgens.examples.ir_lady_bandit_guard_nonlinear
```

### IR Constrained Double-Integrator Diagnostic

Source: [`src/pydgens/examples/ir_constrained_double_integrator_diagnostic.py`](../src/pydgens/examples/ir_constrained_double_integrator_diagnostic.py)

Inspect a constrained double-integrator merge problem retained for future AL solver improvements. By default this command prints the problem summary and does not run the currently problematic solver path.

```bash
python -m pydgens.examples.ir_constrained_double_integrator_diagnostic
```

To run the diagnostic solver path explicitly:

```bash
python -m pydgens.examples.ir_constrained_double_integrator_diagnostic --run-solver
```
