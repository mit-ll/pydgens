# Copyright 2026 MIT Lincoln Laboratory
# SPDX-License-Identifier: MIT

import pytest

from pydgens.examples.multi_car_intersection import solve_example


@pytest.mark.slow
def test_multi_car_intersection_smoketest():
    solve_example()
