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

"""Unit tests for linear algebra matrix analysis utilities"""

import unittest

import numpy as np

# Module to be tested
import newton._src.solvers.kamino.utils.linalg as linalg

###
# Tests
###


class TestUtilsLinAlgMatrix(unittest.TestCase):
    def setUp(self):
        self.verbose = False  # Set to True for verbose output

    def test_01_spd_matrix_properties(self):
        A = linalg.random_spd_matrix(dim=10, dtype=np.float32, scale=4.0, seed=42)
        A_props = linalg.SquareSymmetricMatrixProperties(A)
        print(f"A (shape: {A.shape}, dtype: {A.dtype}):\n{A}\n") if self.verbose else None
        print(f"A properties:\n{A_props}\n") if self.verbose else None
        print(f"cond(A): {np.linalg.cond(A)}\n") if self.verbose else None
        print(f"det(A): {np.linalg.det(A)}\n") if self.verbose else None
        self.assertAlmostEqual(A_props.cond, np.linalg.cond(A), places=6)


###
# Test execution
###

if __name__ == "__main__":
    # Global numpy configurations
    np.set_printoptions(linewidth=10000, precision=10, suppress=True)  # Suppress scientific notation

    # Run all tests
    unittest.main(verbosity=2)
