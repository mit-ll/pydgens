# Copyright 2026 MIT Lincoln Laboratory
# SPDX-License-Identifier: MIT

"""
Small reporting helpers for IR example scripts.

These functions intentionally keep the examples explicit about the fact that
they are working with lower-level, solver-native objects rather than the
frontend ``SolveResult`` wrapper.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp


def _append_common_lines(
    lines: list[str],
    *,
    converged: bool | None = None,
    diagnostics: Any = None,
    states=None,
    controls=None,
) -> None:
    if converged is not None:
        lines.append(f"converged: {converged}")

    reason = getattr(diagnostics, "reason", None)
    iters = getattr(diagnostics, "iters", None)
    if reason is not None:
        lines.append(f"reason:    {reason}")
    if iters is not None:
        lines.append(f"iters:     {iters}")

    if states is not None:
        lines.append(f"states:    shape={states.shape}")
        lines.append(f"x0:        {jnp.asarray(states[0])}")
        lines.append(f"xT:        {jnp.asarray(states[-1])}")

    if controls is not None:
        lines.append(f"controls:  shape={controls.shape}")
        if controls.size:
            lines.append(f"u[0]:      {jnp.asarray(controls[0])}")
            lines.append(f"u[-1]:     {jnp.asarray(controls[-1])}")
        else:
            lines.append("u[0]:      []")


def format_ir_feedback_summary(
    title: str,
    *,
    solver: str,
    trajectory,
    strategy=None,
    converged: bool | None = None,
    diagnostics: Any = None,
) -> str:
    """
    Format a compact summary for IR LQ/iLQ feedback-solver examples.
    """
    lines = [
        f"=== {title} ===",
        "interface: IR",
        f"solver:    {solver}",
        "result:    raw solver outputs",
    ]

    _append_common_lines(
        lines,
        converged=converged,
        diagnostics=diagnostics,
        states=trajectory.xs,
        controls=trajectory.us,
    )

    if strategy is not None:
        lines.append(f"strategy:  {type(strategy).__name__}")
        p = getattr(strategy, "P", None)
        alpha = getattr(strategy, "alpha", None)
        if p is not None:
            lines.append(f"P:         shape={p.shape}")
        if alpha is not None:
            lines.append(f"alpha:     shape={alpha.shape}")

    return "\n".join(lines)


def format_ir_al_summary(
    title: str,
    *,
    primal_dual_trajectory,
    al_state=None,
    diagnostics: Any = None,
) -> str:
    """
    Format a compact summary for IR augmented-Lagrangian examples.
    """
    lines = [
        f"=== {title} ===",
        "interface: IR",
        "solver:    al",
        "result:    raw solver outputs",
    ]

    _append_common_lines(
        lines,
        converged=getattr(diagnostics, "converged", None),
        diagnostics=diagnostics,
        states=primal_dual_trajectory.xs,
        controls=primal_dual_trajectory.us,
    )

    if al_state is not None:
        lam_ineq = getattr(al_state, "lam_ineq", None)
        lam_eq = getattr(al_state, "lam_eq", None)
        if lam_ineq is not None:
            lines.append(f"lam_ineq:  shape={lam_ineq.shape}")
        if lam_eq is not None:
            lines.append(f"lam_eq:    shape={lam_eq.shape}")

    return "\n".join(lines)
