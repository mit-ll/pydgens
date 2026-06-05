# Copyright 2026 MIT Lincoln Laboratory
# SPDX-License-Identifier: MIT

# Single-sourcing package version
# https://packaging.python.org/guides/single-sourcing-package-version/

__version__ = "0.5.1"

# User-facing semantic modeling APIs
from pydgens.frontend import dynamics
from pydgens.frontend import costs
from pydgens.frontend import players
from pydgens.frontend import games
from pydgens.frontend import constraints

# Convenience constructors for common workflows
from pydgens.ir.timetypes import time_grid
from pydgens.frontend.dynamics import linear_dynamics
from pydgens.frontend.dynamics import nonlinear_dynamics
from pydgens.frontend.costs import player_cost
from pydgens.frontend.costs import quadratic_cost
from pydgens.frontend.constraints import control_bounds
from pydgens.frontend.constraints import state_bounds
from pydgens.frontend.constraints import constraint_set
from pydgens.frontend.players import player
from pydgens.frontend.games import game
from pydgens.frontend.solvers import solve

__all__ = [
    "dynamics",
    "costs",
    "players",
    "games",
    "constraints",
    "time_grid",
    "linear_dynamics",
    "nonlinear_dynamics",
    "player_cost",
    "quadratic_cost",
    "control_bounds",
    "state_bounds",
    "constraint_set",
    "player",
    "game",
    "solve",
]
