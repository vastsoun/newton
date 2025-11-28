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

import os
import unittest

from newton._src.solvers.kamino.tests.utils.setup import setup_tests

###
# Utilities
###


# Overload of TextTestResult printing a header for each new test module
class ModuleHeaderTestResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._current_module = None

    def startTest(self, test):
        module = test.__class__.__module__
        if module != self._current_module:
            self._current_module = module
            filename = module.replace(".", "/") + ".py"

            # Print spacing + header
            self.stream.write("\n\n")
            self.stream.write(f"=== Running tests in: {filename} ===\n")
            self.stream.write("\n")
            self.stream.flush()

        super().startTest(test)


# Overload of TextTestRunner printing a header for each new test module
class ModuleHeaderTestRunner(unittest.TextTestRunner):
    resultclass = ModuleHeaderTestResult


###
# Test execution
###

if __name__ == "__main__":
    # Perform global setup
    setup_tests(verbose=False, device="cuda", clear_cache=False)

    # Detect all unit tests
    test_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    tests = unittest.defaultTestLoader.discover(test_folder, pattern="test_*.py")

    # Run tests
    ModuleHeaderTestRunner(verbosity=2).run(tests)
