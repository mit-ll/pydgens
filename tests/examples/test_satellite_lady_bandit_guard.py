# Copyright 2026 MIT Lincoln Laboratory
# SPDX-License-Identifier: MIT

import pytest

from pydgens.examples.satellite_lady_bandit_guard import solve_example

@pytest.mark.slow
def test_satellite_lady_bandit_guard_smoketest():
    solve_example()
