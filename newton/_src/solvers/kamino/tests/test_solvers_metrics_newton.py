# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for `solvers/metrics.py`."""

import unittest

import numpy as np
import warp as wp

# from newton.tests.utils.basics import (
#     build_box_on_plane,
#     build_boxes_hinged,
#     build_boxes_nunchaku_vertical,
# )
from newton._src.solvers.kamino._src.solvers.metrics import SolutionMetricsNewton
from newton._src.solvers.kamino._src.utils import logger as msg
from newton._src.solvers.kamino.tests import setup_tests, test_context
from newton._src.solvers.kamino.tests.test_solvers_padmm import TestSetup
from newton._src.solvers.kamino.tests.utils.extract import (
    extract_cts_jacobians,
    extract_delassus,
    extract_info_vectors,
    extract_problem_vector,
)

###
# Helpers
###


###
# Tests
###


class TestSolverMetricsNewton(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        self.verbose = test_context.verbose  # Set to True for detailed output
        self.seed = 42

        # Set debug-level logging to print verbose test output to console
        if self.verbose:
            print("\n")  # Add newline before test output for better readability
            msg.set_log_level(msg.LogLevel.INFO)
        else:
            msg.reset_log_level()

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()

    def test_00_make_default(self):
        """
        Test creating a SolutionMetrics instance with default initialization.
        """
        pass

    def test_01_finalize_default(self):
        """
        Test creating a SolutionMetrics instance with default initialization and then finalizing all memory allocations.
        """
        pass

    def test_02_evaluate_on_box_on_plane(self):
        """
        TODO
        """

    def test_03_evaluate_on_boxes_hinged(self):
        """
        TODO
        """

    def test_04_evaluate_on_boxes_nunchaku_vertical(self):
        """
        TODO
        """
        # Create the test problem
        test = TestSetup(
            builder_fn=None,
            max_world_contacts=8,
            gravity=True,
            perturb=True,
            device=self.default_device,
            sparse=False,
        )

        # Creating a default solver metrics evaluator from the test model
        metrics = SolutionMetricsNewton(model=test.model, data=test.data)


###
# Test execution
###

if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
