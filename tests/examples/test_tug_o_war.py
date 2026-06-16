# Copyright 2026 MIT Lincoln Laboratory
# SPDX-License-Identifier: MIT

import pytest

from pydgens.examples.tug_o_war import analytic_solution, main


def test_analytic_solution_default_problem():
    u1_star, u2_star, x1_star = analytic_solution(
        x0=0.0,
        target_1=1.0,
        target_2=-1.0,
        r1=0.5,
        r2=2.0,
    )

    assert u1_star == pytest.approx(8.0 / 7.0)
    assert u2_star == pytest.approx(-5.0 / 7.0)
    assert x1_star == pytest.approx(3.0 / 7.0)


def test_tug_o_war_smoketest_matches_analytic_solution():
    main()
