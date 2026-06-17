# Copyright 2026 MIT Lincoln Laboratory
# SPDX-License-Identifier: MIT

import pytest

from pydgens.examples.constrained_integrators import main

@pytest.mark.slow
def test_constrained_integrators_smoketest():
    main()