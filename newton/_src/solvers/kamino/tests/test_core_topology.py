# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for core geometry containers and operations"""

import unittest

import warp as wp

from newton._src.solvers.kamino._src.core.topology import draw_graph, parse_graph
from newton._src.solvers.kamino._src.utils import logger as msg
from newton._src.solvers.kamino.tests import setup_tests, test_context

###
# Tests
###


class TestTopologyUtils(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        self.verbose = test_context.verbose  # Set to True for detailed output

        # Set debug-level logging to print verbose test output to console
        if self.verbose:
            print("\n")  # Add newline before test output for better readability
            msg.set_log_level(msg.LogLevel.DEBUG)
        else:
            msg.reset_log_level()

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()

    def test_00_find_islands_and_orphans(self):
        nodes = [-1, 0, 1, 2, 3, 4, 5, 6, 7]
        edges = [
            (-1, 0),
            (0, 1),
            (1, 2),  # island: [0, 1, 2]
            (-1, 3),  # orphan: 3
            (4, 5),  # island: [4, 5]
            (-1, 6),  # orphan: 6
            # 7 has no edges -> isolated
        ]

        result = parse_graph(nodes, edges)

        # TERMINOLOGY:
        #
        #
        #
        #
        #

        print("\n")
        print("Islands :", result["islands"])
        print("Orphans :", result["orphans"])
        print("Isolated:", result["isolated"])
        print("Components:", result["components"])

        # TODO
        draw_graph(nodes, edges, root=-1, figsize=(6, 4))

        # self.assertEqual(result["islands"], expected_islands)
        # self.assertEqual(result["orphans"], expected_orphans)
        # self.assertEqual(result["isolated"], expected_isolated)

    def test_01_parse_graph(self):
        nodes = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        edges = [
            (-1, 0),
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 1),
            (4, 0),  # island: [0, 1, 2, 3, 4]
            (-1, 8),  # orphan: 8
            (6, 7),  # island: [6, 7]
            (-1, 9),  # orphan: 9
            # 10-15 have no edges -> isolated
        ]

        result = parse_graph(nodes, edges)

        print("\n")
        print("Islands :", result["islands"])
        print("Orphans :", result["orphans"])
        print("Isolated:", result["isolated"])
        print("Components:", result["components"])

        # TODO
        draw_graph(nodes, edges, root=-1, figsize=(10, 10))

        # self.assertEqual(result["islands"], expected_islands)
        # self.assertEqual(result["orphans"], expected_orphans)
        # self.assertEqual(result["isolated"], expected_isolated)


###
# Test execution
###

if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
