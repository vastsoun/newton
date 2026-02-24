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

"""
Unit tests for `dynamics/wrenches.py`.
"""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino.core.model import Model, ModelData
from newton._src.solvers.kamino.core.types import float32
from newton._src.solvers.kamino.dynamics.wrenches import (
    compute_constraint_body_wrenches_dense,
    compute_constraint_body_wrenches_sparse,
    compute_joint_dof_body_wrenches_dense,
    compute_joint_dof_body_wrenches_sparse,
)
from newton._src.solvers.kamino.geometry.contacts import Contacts
from newton._src.solvers.kamino.kinematics.jacobians import DenseSystemJacobians, SparseSystemJacobians
from newton._src.solvers.kamino.kinematics.limits import Limits
from newton._src.solvers.kamino.tests import setup_tests, test_context
from newton._src.solvers.kamino.tests.test_kinematics_jacobians import (
    make_test_problem_fourbar,
)
from newton._src.solvers.kamino.utils import logger as msg

###
# Constants
###


test_wrench_rtol = 1e-7
test_wrench_atol = 1e-8


###
# Helper Functions
###


def make_constraint_multiplier_arrays(model: Model) -> tuple[wp.array, wp.array]:
    with wp.ScopedDevice(model.device):
        lambdas = wp.zeros(model.size.sum_of_max_total_cts, dtype=float32)
    return model.info.total_cts_offset, lambdas


def compute_and_compare_dense_sparse_jacobian_wrenches(
    model: Model,
    data: ModelData,
    limits: Limits,
    contacts: Contacts,
    rtol: float = 1e-7,
    atol: float = 0.0,
):
    # Create the Jacobians container
    jacobians_dense = DenseSystemJacobians(model=model, limits=limits, contacts=contacts)
    jacobians_sparse = SparseSystemJacobians(model=model, limits=limits, contacts=contacts)
    wp.synchronize()

    # Build the system Jacobians
    jacobians_dense.build(model=model, data=data, limits=limits.data, contacts=contacts.data)
    jacobians_sparse.build(model=model, data=data, limits=limits.data, contacts=contacts.data)
    wp.synchronize()

    # Create arrays for the constraint multipliers and initialize them
    lambdas_start, lambdas = make_constraint_multiplier_arrays(model)
    lambdas.fill_(1.0)

    # Initialize the generalized joint actuation forces
    data.joints.tau_j.fill_(1.0)

    # Compute the wrenches using the dense Jacobians
    compute_joint_dof_body_wrenches_dense(
        model=model,
        data=data,
        jacobians=jacobians_dense,
        reset_to_zero=True,
    )
    compute_constraint_body_wrenches_dense(
        model=model,
        data=data,
        jacobians=jacobians_dense,
        lambdas_offsets=lambdas_start,
        lambdas_data=lambdas,
        limits=limits.data,
        contacts=contacts.data,
        reset_to_zero=True,
    )
    w_a_i_dense_np = data.bodies.w_a_i.numpy().copy()
    w_j_i_dense_np = data.bodies.w_j_i.numpy().copy()
    w_l_i_dense_np = data.bodies.w_l_i.numpy().copy()
    w_c_i_dense_np = data.bodies.w_c_i.numpy().copy()

    # Compute the wrenches using the sparse Jacobians
    compute_joint_dof_body_wrenches_sparse(
        model=model,
        data=data,
        jacobians=jacobians_sparse,
        reset_to_zero=True,
    )
    compute_constraint_body_wrenches_sparse(
        model=model,
        data=data,
        jacobians=jacobians_sparse,
        lambdas_offsets=lambdas_start,
        lambdas_data=lambdas,
        reset_to_zero=True,
    )
    w_a_i_sparse_np = data.bodies.w_a_i.numpy().copy()
    w_j_i_sparse_np = data.bodies.w_j_i.numpy().copy()
    w_l_i_sparse_np = data.bodies.w_l_i.numpy().copy()
    w_c_i_sparse_np = data.bodies.w_c_i.numpy().copy()

    # Check that the wrenches computed using the dense and sparse Jacobians are close
    np.testing.assert_allclose(w_a_i_dense_np, w_a_i_sparse_np, rtol=rtol, atol=atol)
    np.testing.assert_allclose(w_j_i_dense_np, w_j_i_sparse_np, rtol=rtol, atol=atol)
    np.testing.assert_allclose(w_l_i_dense_np, w_l_i_sparse_np, rtol=rtol, atol=atol)
    np.testing.assert_allclose(w_c_i_dense_np, w_c_i_sparse_np, rtol=rtol, atol=atol)


###
# Tests
###


class TestDynamicsWrenches(unittest.TestCase):
    def setUp(self):
        # Configs
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        # self.verbose = test_context.verbose  # Set to True for verbose output
        self.verbose = True  # Set to True for verbose output

        # Set info-level logging to print verbose test output to console
        if self.verbose:
            print("\n")  # Add newline before test output for better readability
            msg.set_log_level(msg.LogLevel.INFO)
        else:
            msg.reset_log_level()

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()

    def test_01_compute_wrenches_for_single_fourbar_with_limits_and_contacts(self):
        # Construct the test problem
        model, data, limits, contacts = make_test_problem_fourbar(
            device=self.default_device,
            max_world_contacts=12,
            num_worlds=1,
            with_limits=True,
            with_contacts=True,
            verbose=self.verbose,
        )

        # Compute and compare the wrenches using the dense and sparse Jacobians
        compute_and_compare_dense_sparse_jacobian_wrenches(
            model=model,
            data=data,
            limits=limits,
            contacts=contacts,
            rtol=test_wrench_rtol,
            atol=test_wrench_atol,
        )

    def test_02_compute_wrenches_for_multiple_fourbars_with_limits_and_contacts(self):
        # Construct the test problem
        model, data, limits, contacts = make_test_problem_fourbar(
            device=self.default_device,
            max_world_contacts=12,
            num_worlds=3,
            with_limits=True,
            with_contacts=True,
            verbose=self.verbose,
        )

        # Compute and compare the wrenches using the dense and sparse Jacobians
        compute_and_compare_dense_sparse_jacobian_wrenches(
            model=model,
            data=data,
            limits=limits,
            contacts=contacts,
            rtol=test_wrench_rtol,
            atol=test_wrench_atol,
        )

    def test_03_compute_wrenches_heterogeneous_model_with_limits_and_contacts(self):
        pass  # TODO


###
# Test execution
###

if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
