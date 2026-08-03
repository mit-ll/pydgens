# Copyright 2026 MIT Lincoln Laboratory
# SPDX-License-Identifier: MIT

"""Common, lightweight solver-diagnostic types.

Solver-specific diagnostics may add fields, but every iterative solver should
expose this small termination contract to frontend callers.
"""

from dataclasses import dataclass
from typing import Literal, get_args


DiagnosticsLevel = Literal["off", "basic", "detailed"]


def validate_diagnostics_level(level: DiagnosticsLevel) -> DiagnosticsLevel:
    """Validate a solver diagnostics collection level."""
    if level not in get_args(DiagnosticsLevel):
        raise ValueError(
            "diagnostics_level must be one of "
            f"{get_args(DiagnosticsLevel)}, got {level!r}."
        )
    return level


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
