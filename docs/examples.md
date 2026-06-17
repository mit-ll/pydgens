---
icon: material/file-document-check
---

# Examples

Examples live in [`src/pydgens/examples/`](../src/pydgens/examples/). These scripts are useful both as smoke tests and as starting points for new game formulations.

## Tug-of-War

Compute a feedback Nash equilibrium for a simple linear-quadratic game with the `LQ` solver:

```bash
python src/pydgens/examples/tug_o_war.py
```

## Two-Player Unicycle

Run the iterative linear-quadratic solver (`iLQ`) on a nonlinear game where two players control a shared unicycle system:

```bash
python src/pydgens/examples/unicycle.py
```

## IR Two-Player Unicycle

Build the same unicycle game directly with the intermediate representation (`IR`) objects used by the iLQ solver:

```bash
python src/pydgens/examples/ir_unicycle.py
```

## Constrained Integrators

Run the augmented-Lagrangian (`AL`) solver on a constrained nonlinear game with bounded controls:

```bash
python src/pydgens/examples/constrained_integrators.py
```
