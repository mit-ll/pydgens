# PYDGENS: Python/JAX Differential Game Equilibria Numerical Solvers

<p align="center">
  <img src="docs/assets/pydgens-logo.png" alt="PYDGENS logo" width="300">
</p>

PYDGENS provides numerical solvers for approximating equilibrium solutions in multi-player, general-sum dynamic and differential games. The package currently focuses on linear-quadratic feedback Nash games, iterative linear-quadratic methods for nonlinear games, and augmented-Lagrangian workflows for constrained games.

PYDGENS is under active development. The public API is usable for early adopters, but it may change before a stable `1.0` release.

## Installation

After the first PyPI release:

```bash
pip install pydgens
```

For development from a local clone:

```bash
pip install -e .[full]
```

Contributors can also use `uv` for a reproducible environment:

```bash
uv sync --extra dev
source .venv/bin/activate
```

## Examples

Run a minimal linear-quadratic tug-of-war game:

```bash
python src/pydgens/examples/tug_o_war.py
```

Run a nonlinear two-player unicycle example:

```bash
python src/pydgens/examples/run_unicycle1.py
```

Run the double-integrator lady-bandit-guard example:

```bash
python src/pydgens/examples/run_doubleint_lqlbg.py --cfg C1_001
```

More examples live in [`src/pydgens/examples/`](src/pydgens/examples/).

## Testing

Quick tests:

```bash
pytest tests/ -v -s -m "not slow"
```

Slow and benchmark-oriented tests:

```bash
pytest tests/ -v -m "slow" --benchmark-columns='mean, min, max, stddev, rounds'
```

## Documentation

The documentation site is built with Zensical:

```bash
zensical serve
```

Open <http://localhost:8000> to preview the local site.

## Release Status

The public repository is intended to start with a clean git history. Release artifacts will be built from the public repository and published to PyPI through GitHub Actions after the PyPI trusted publisher is configured.

## Disclaimer

DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Under Secretary of War for Research and
Engineering under Air Force Contract No. FA8702-15-D-0001 or FA8702-25-D-B002. Any
opinions, findings, conclusions or recommendations expressed in this material are those of
the author(s) and do not necessarily reflect the views of the Under Secretary of War for
Research and Engineering.

© 2026 Massachusetts Institute of Technology.

Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)

SPDX-License-Identifier: MIT

The software/firmware is provided to you on an As-Is basis.

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part
252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government
rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed
above. Use of this work other than as specifically authorized by the U.S. Government may
violate any copyrights that exist in this work.
