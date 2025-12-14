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

"""Unit tests for the :class:`SolverKamino` class"""

import unittest

import warp as wp

from newton._src.solvers.kamino.core.control import Control
from newton._src.solvers.kamino.core.state import State
from newton._src.solvers.kamino.core.types import float32
from newton._src.solvers.kamino.examples import print_progress_bar
from newton._src.solvers.kamino.geometry.contacts import Contacts
from newton._src.solvers.kamino.models.builders.basics import build_cartpole
from newton._src.solvers.kamino.models.builders.utils import make_homogeneous_builder
from newton._src.solvers.kamino.solver_kamino import SolverKamino
from newton._src.solvers.kamino.tests import setup_tests, test_context
from newton._src.solvers.kamino.utils import logger as msg

###
# Kernels
###


@wp.kernel
def _test_control_callback(
    model_dt: wp.array(dtype=float32),
    state_t: wp.array(dtype=float32),
    control_tau_j: wp.array(dtype=float32),
):
    """
    An example control callback kernel.
    """
    # Retrieve the world index from the thread ID
    wid = wp.tid()

    # Get the fixed time-step and current time
    dt = model_dt[wid]
    t = state_t[wid]

    # Define the time window for the active external force profile
    t_start = float32(0.0)
    t_end = 10.0 * dt

    # Compute the first actuated joint index for the current world
    aid = wid * 2 + 0

    # Apply a time-dependent external force
    if t > t_start and t < t_end:
        control_tau_j[aid] = 0.1
    else:
        control_tau_j[aid] = 0.0


###
# Launchers
###


def test_control_callback(
    solver: SolverKamino,
    state_in: State,
    state_out: State,
    control: Control,
    contacts: Contacts
):
    """
    A control callback function
    """
    wp.launch(
        _test_control_callback,
        dim=solver._model.size.num_worlds,
        inputs=[
            solver._model.time.dt,
            solver._data.time.time,
            control.tau_j,
        ],
    )


###
# Tests
###


class TestSolverKaminoSettings(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        self.verbose = test_context.verbose  # Set to True to enable verbose output

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

    def test_00_TODO(self):
        pass


class TestSolverKamino(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        self.verbose = test_context.verbose  # Set to True to enable verbose output

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

    def test_00_TODO(self):
        pass


###
# Test execution
###

if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
