---
icon: lucide/rocket
---

# Get started

PYDGENS is a Python/JAX package for approximating equilibrium solutions in multi-player dynamic and differential games.

The package is under active development. The public API is intended for early adopters and may change before a stable `1.0` release.

## Installation

After the first PyPI release:

```bash
pip install pydgens
```

For local development:

```bash
pip install -e .[full]
```

For a locked contributor environment:

```bash
uv sync --extra dev
source .venv/bin/activate
```

## Quick Check

Run the quick test suite:

```bash
pytest tests/ -v -s -m "not slow"
```

Run a small example:

```bash
python src/pydgens/examples/tug_o_war.py
```

## Project Layout

- `src/pydgens/frontend/`: user-facing modeling helpers.
- `src/pydgens/ir/`: JAX-friendly intermediate representations used by solvers.
- `src/pydgens/solvers/`: numerical algorithms for equilibria, trajectories, and constrained solves.
- `src/pydgens/examples/`: runnable game examples.
- `tests/`: unit, regression, integration, and benchmark tests.

## Release Notes

Public releases are built from the public GitHub repository and published to PyPI through GitHub Actions.
