"""Frontend semantic modeling API for PYDGENS.

The :mod:`pydgens.frontend` namespace collects the higher-level modeling
objects and convenience constructors used to define and solve games before
they are lowered into solver-facing IR objects.

Most users can import these same constructors directly from ``pydgens``.
This namespace exists for users who want the frontend API grouped under its
own subpackage.
"""

# Public frontend submodules.
from pydgens.frontend import constraints
from pydgens.frontend import costs
from pydgens.frontend import dynamics
from pydgens.frontend import games
from pydgens.frontend import players
from pydgens.frontend import solvers

# Supporting constructor used by frontend game definitions.
from pydgens.ir.timetypes import time_grid

# Beginner-facing semantic modeling factories.
from pydgens.frontend.constraints import constraint_set
from pydgens.frontend.constraints import control_bounds
from pydgens.frontend.constraints import state_bounds
from pydgens.frontend.costs import matrix_quadratic_cost
from pydgens.frontend.costs import player_cost
from pydgens.frontend.costs import quadratic_cost
from pydgens.frontend.dynamics import linear_dynamics
from pydgens.frontend.dynamics import nonlinear_dynamics
from pydgens.frontend.games import game
from pydgens.frontend.players import player
from pydgens.frontend.solvers import solve

__all__ = [
    "constraints",
    "costs",
    "dynamics",
    "games",
    "players",
    "solvers",
    "time_grid",
    "constraint_set",
    "control_bounds",
    "state_bounds",
    "matrix_quadratic_cost",
    "player_cost",
    "quadratic_cost",
    "linear_dynamics",
    "nonlinear_dynamics",
    "game",
    "player",
    "solve",
]
