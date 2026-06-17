---
icon: material/file-document-check
---

# Examples

Examples live in [`src/pydgens/examples/`](../src/pydgens/examples/). These scripts are useful both as smoke tests and as starting points for new game formulations.

## Tug-of-War

Compute a feedback Nash equilibrium for a simple linear-quadratic game with the `LQ` solver:

```bash
python -m pydgens.examples.tug_o_war
```

## Two-Player Unicycle

Run the iterative linear-quadratic solver (`iLQ`) on a nonlinear game where two players control a shared unicycle system:

```bash
python -m pydgens.examples.unicycle
```

## IR Two-Player Unicycle

Build the same unicycle game directly with the intermediate representation (`IR`) objects used by the iLQ solver:

```bash
python -m pydgens.examples.ir_unicycle
```

## IR Lady-Bandit-Guard

Build and solve a richer linear-quadratic Lady-Bandit-Guard game directly with the LQ solver's IR objects:

```bash
python -m pydgens.examples.ir_lady_bandit_guard
```

## Constrained Integrators

Run the augmented-Lagrangian (`AL`) solver on a constrained nonlinear game with bounded controls:

```bash
python -m pydgens.examples.constrained_integrators
```

## IR Constrained Integrators

Build and solve the constrained integrator game directly with the augmented-Lagrangian (`AL`) solver's IR objects:

```bash
python -m pydgens.examples.ir_constrained_integrators
```

## Solver Diagnostic Examples

These examples are useful reference problems for solver development, but they are not polished smoke tests. They may fail to converge, take a long time, or require explicit opt-in flags before running the solver.

### IR Nonlinear Lady-Bandit-Guard

Run the nonlinear unicycle-dynamics Lady-Bandit-Guard counterpart directly with the iLQ solver's IR objects. This example currently executes to completion but may report non-convergence.

```bash
python -m pydgens.examples.ir_lady_bandit_guard_nonlinear
```

### IR Constrained Double-Integrator Diagnostic

Inspect a constrained double-integrator merge problem retained for future AL solver improvements. By default this command prints the problem summary and does not run the currently problematic solver path.

```bash
python -m pydgens.examples.ir_constrained_double_integrator_diagnostic
```

To run the diagnostic solver path explicitly:

```bash
python -m pydgens.examples.ir_constrained_double_integrator_diagnostic --run-solver
```
