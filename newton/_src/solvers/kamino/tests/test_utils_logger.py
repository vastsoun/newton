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

"""Kamino: Tests for logging utilities"""

import unittest

import numpy as np

import newton._src.solvers.kamino.utils.logger as msg
from newton._src.solvers.kamino.utils.logger import Logger

###
# Tests
###


class TestUtilsLogger(unittest.TestCase):
    def setUp(self):
        self.verbose = True  # Set to True for verbose output

    def test_new_logger(self):
        """Test use of the custom logger."""
        print("")  # Print a newline for better readability in the output
        logger = Logger()
        log = logger.get()
        log.info("This is an info message.")
        log.debug("This is a debug message.")
        log.warning("This is a warning message.")
        log.error("This is an error message.")
        log.critical("This is a critical message.")

    def test_default_logger(self):
        """Test use of the custom logger."""
        print("")  # Print a newline for better readability in the output
        msg.info("This is an info message.")
        msg.debug("This is a debug message.")
        msg.warning("This is a warning message.")
        msg.error("This is an error message.")
        msg.critical("This is a critical message.")


###
# Test execution
###

if __name__ == "__main__":
    # Global numpy configurations
    np.set_printoptions(linewidth=500, precision=10, suppress=True)  # Suppress scientific notation

    # Run all tests
    unittest.main(verbosity=2)
