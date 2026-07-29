# Copyright 2026 MIT Lincoln Laboratory
# SPDX-License-Identifier: MIT

"""Common, lightweight solver-diagnostic types.

Solver-specific diagnostics may add fields, but every iterative solver should
expose this small termination contract to frontend callers.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SolverDiag:
    """Common termination summary for an iterative solver.

    Attributes
    ----------
    converged
        Whether the solver met its convergence criterion.
    iters
        Number of outer iterations executed.
    reason
        Stable, solver-specific termination reason.
    """

    converged: bool
    iters: int
    reason: str
