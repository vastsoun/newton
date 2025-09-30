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

import unittest

import numpy as np

import newton._src.solvers.kamino.utils.linalg as linalg
import newton._src.solvers.kamino.utils.logger as msg

###
# Tests
###


class TestUtilsLinAlgEigval(unittest.TestCase):
    def setUp(self):
        self.verbose = False  # Set to True for verbose output
        if self.verbose:
            msg.set_log_level(msg.LogLevel.DEBUG)

        # Define test parameters
        self.dtype = np.float32
        self.epsilon = np.finfo(self.dtype).eps
        self.atol = np.finfo(self.dtype).eps
        self.rtol = np.finfo(self.dtype).eps
        self.max_iterations: int = 1000

        # Print test configuration
        msg.debug(
            "\nTest Setup:"
            f"\n  dtype: {self.dtype}"
            f"\n  epsilon: {self.epsilon}"
            f"\n  atol: {self.atol}"
            f"\n  rtol: {self.rtol}"
            f"\n  max_iterations: {self.max_iterations}\n"
        )

        # Define a random number generator
        self.rng = np.random.default_rng(seed=0)

        # Construct a random square symmetric matrix
        self.scale = 10.0
        self.shape = (6, 6)
        A = self.scale * self.rng.standard_normal(size=self.shape)
        A = A @ A.T
        self.A = self.dtype(0.5) * (A + A.T)
        msg.debug(f"\nA {self.A.shape}[{self.A.dtype}]:\n{self.A}\n")

        # Compute matrix properties
        self.norm_A = np.linalg.norm(self.A)
        self.rank_A = np.linalg.matrix_rank(self.A)
        self.cond_A = np.linalg.cond(self.A)
        msg.debug(
            f"\nmatrix properties:\n  norm(A): {self.norm_A}\n  rank(A): {self.rank_A}\n  cond(A): {self.cond_A}\n"
        )

        # Compute reference eigenvalues using NumPy
        self.lambdas_A = np.linalg.eigvals(self.A)
        msg.debug(
            "reference eigenvalues:"
            f"\n  lambda(A): {self.lambdas_A}"
            f"\n  lambda_max(A): {self.lambdas_A.max()}"
            f"\n  lambda_min(A): {self.lambdas_A.min()}\n"
        )

    def test_01_power_iteration(self):
        pi = linalg.PowerIteration(atol=self.atol, rtol=self.rtol, max_iterations=1000)
        lambdas_pi_max = pi.largest(self.A)
        lambdas_pi_min = pi.smallest(self.A)
        msg.debug(f"PowerIteration: max_eigval: {pi.max_eigenvalue}")
        msg.debug(f"PowerIteration: min_eigval: {pi.min_eigenvalue}")
        msg.debug(f"PowerIteration: converged: {pi.converged}")
        msg.debug(f"PowerIteration: iterations: {pi.iterations}")
        msg.debug(f"PowerIteration: max_residual: {pi.max_residual}")
        msg.debug(f"PowerIteration: min_residual: {pi.min_residual}\n")
        self.assertTrue(pi.converged)
        self.assertAlmostEqual(lambdas_pi_max, self.lambdas_A.max(), places=3)
        self.assertAlmostEqual(lambdas_pi_min, self.lambdas_A.min(), places=2)

    def test_02_gram_iteration(self):
        gi = linalg.GramIteration(atol=self.atol, rtol=self.rtol, max_iterations=1000)
        lambdas_pi_max = gi.largest(self.A)
        msg.debug(f"PowerIteration: max_eigval: {gi.max_eigenvalue}")
        msg.debug(f"PowerIteration: converged: {gi.converged}")
        msg.debug(f"PowerIteration: iterations: {gi.iterations}")
        msg.debug(f"PowerIteration: max_residual: {gi.max_residual}")
        self.assertTrue(gi.converged)
        self.assertAlmostEqual(lambdas_pi_max, self.lambdas_A.max(), places=6)


###
# Test execution
###

if __name__ == "__main__":
    # Global numpy configurations
    np.set_printoptions(linewidth=2000, precision=10, threshold=10000, suppress=True)  # Suppress scientific notation

    # Run all tests
    unittest.main(verbosity=2)
