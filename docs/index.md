---
icon: lucide/rocket
---

# Get started

PYDGENS is a Python/JAX package for approximating equilibrium solutions in multi-player dynamic and differential games.

PYDGENS is a pre-`1.0` release. The package is ready for early adopters, but the public API may continue to evolve as the modeling frontend, examples, and solver interfaces mature.

## Installation

```bash
pip install pydgens
```

PYDGENS requires Python `3.12` or newer.

## Solvers

PYDGENS currently supports three main solver paths:

- `LQ`: linear-quadratic, unconstrained games solved for feedback Nash strategies
- `iLQ`: nonlinear, unconstrained games solved for local feedback Nash strategies
- `AL`: constrained nonlinear games solved with an augmented-Lagrangian workflow for local open-loop trajectories

See [Solvers](solvers.md) for a sparse theory map and references.

## Quick Start

Run a minimal linear-quadratic example:

```bash
python src/pydgens/examples/tug_o_war.py
```

Run the quick test suite:

```bash
pytest tests/ -v -s -m "not slow"
```

## Development

For local development:

```bash
pip install -e .[full]
```

For a locked contributor environment:

```bash
uv sync --extra dev
source .venv/bin/activate
```

## Project Layout

- `src/pydgens/frontend/`: user-facing modeling helpers.
- `src/pydgens/ir/`: JAX-friendly intermediate representations used by solvers.
- `src/pydgens/solvers/`: numerical algorithms for equilibria, trajectories, and constrained solves.
- `src/pydgens/examples/`: runnable game examples.
- `tests/`: unit, regression, integration, and benchmark tests.
