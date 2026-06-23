# PYDGENS: Python/JAX Differential Game Equilibria Numerical Solvers

<p align="center">
  <img src="https://raw.githubusercontent.com/mit-ll/pydgens/main/docs/assets/pydgens-logo.png" alt="PYDGENS logo" width="300">
</p>

<h3 align="center">
  <a href="https://mit-ll.github.io/pydgens/">Documentation</a>
  ·
  <a href="https://mit-ll.github.io/pydgens/examples/">Examples</a>
  ·
  <a href="https://mit-ll.github.io/pydgens/solvers/">Solvers</a>
  ·
  <a href="https://github.com/mit-ll/pydgens">GitHub</a>
</h3>

PYDGENS provides numerical solvers for approximating equilibrium solutions in multi-player, general-sum dynamic and differential games. The package currently focuses on linear-quadratic feedback Nash games, iterative linear-quadratic methods for nonlinear games, and augmented-Lagrangian workflows for constrained games.

PYDGENS is a pre-`1.0` release. The package is ready for early adopters, but the public API may continue to evolve as the modeling frontend, examples, and solver interfaces mature.

<table>
  <tr>
    <td width="72%">
      <img src="https://raw.githubusercontent.com/mit-ll/pydgens/main/docs/assets/multi_car_intersection.gif" alt="Multi-car intersection game solved with PYDGENS" width="100%">
    </td>
    <td width="28%">
      <strong>Multi-car intersection</strong>
      <br><br>
      Naive collisions compared to an iLQ feedback solution.
      <br><br>
      <a href="https://github.com/mit-ll/pydgens/blob/main/src/pydgens/examples/multi_car_intersection.py">Source</a> ·
      <a href="https://mit-ll.github.io/pydgens/examples/#multi-car-intersection">Docs</a>
    </td>
  </tr>
  <tr>
    <td width="72%">
      <img src="https://raw.githubusercontent.com/mit-ll/pydgens/main/docs/assets/satellite_lady_bandit_guard.gif" alt="Satellite Lady-Bandit-Guard game solved with PYDGENS" width="100%">
    </td>
    <td width="28%">
      <strong>Satellite Lady-Bandit-Guard</strong>
      <br><br>
      One LQ feedback Nash strategy rolled out from many initial states.
      <br><br>
      <a href="https://github.com/mit-ll/pydgens/blob/main/src/pydgens/examples/satellite_lady_bandit_guard.py">Source</a> ·
      <a href="https://mit-ll.github.io/pydgens/examples/#satellite-lady-bandit-guard">Docs</a> ·
      <a href="https://github.com/mit-ll/spacegym-kspdg">spacegym-kspdg</a>
    </td>
  </tr>
</table>

## Installation

```bash
pip install pydgens
```

PYDGENS requires Python `3.12` or newer.

## Solvers

PYDGENS currently supports three main solver paths. See the [solver notes](https://mit-ll.github.io/pydgens/solvers/) for a sparse theory map and references.

| Solver | Use case | Equilibrium style |
| --- | --- | --- |
| `LQ` | Linear dynamics with quadratic costs | feedback Nash |
| `iLQ` | Nonlinear unconstrained games | local feedback Nash |
| `AL` | Constrained nonlinear games | local open-loop Nash (_pre-release, beta_) |

## Usage Example

Define and solve for the local Nash equilibrium of a nonlinear game by combining a time grid, dynamics, player costs, and player-owned control slices:

```python
import jax.numpy as jnp
import pydgens as pdg

x0 = jnp.array([4.0, 4.0, 0.0, 0.0])  # px, py, heading, speed

game = pdg.game(
    tg=pdg.time_grid(nt=34, dt=0.1),
    dynamics=pdg.nonlinear_dynamics(
        nx=4,
        nu=2,
        dynamics=lambda t, x, u: jnp.array([
            x[3] * jnp.cos(x[2]),
            x[3] * jnp.sin(x[2]),
            u[0],
            u[1],
        ]),
    ),
    players=[
        pdg.player(
            name="turn",
            joint_ctrl_slice=slice(0, 1),
            cost=pdg.player_cost(
                running=lambda t, x, u: x[0] ** 2 + x[1] ** 2 + u[0] ** 2,
            ),
        ),
        pdg.player(
            name="speed",
            joint_ctrl_slice=slice(1, 2),
            cost=pdg.player_cost(
                running=lambda t, x, u: (x[3] - 1.0) ** 2 + u[1] ** 2,
            ),
        ),
    ],
)

solution = pdg.solve(game, x0=x0, method="ilq")
print(solution)
```

Further examples of solving for equilibria in differential games can be run directly with:

```bash
python -m pydgens.examples.tug_o_war  # minimal linear-quadratic (LQ) game
python -m pydgens.examples.unicycle   # nonlinear game solved with iterative method
python -m pydgens.examples.constrained_integrators  # constrained game solved Lagrangian method
```

A comprehensive list of examples is included in the [examples documentation](https://mit-ll.github.io/pydgens/examples/).


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
