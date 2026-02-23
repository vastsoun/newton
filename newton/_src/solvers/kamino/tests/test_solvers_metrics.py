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

"""Unit tests for `solvers/metrics.py`."""

import os
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino.dynamics.dual import DualProblem
from newton._src.solvers.kamino.integrators.euler import integrate_euler_semi_implicit
from newton._src.solvers.kamino.kinematics.jacobians import SparseSystemJacobians
from newton._src.solvers.kamino.models.builders.basics import build_box_on_plane, build_boxes_hinged
from newton._src.solvers.kamino.solvers.metrics import SolutionMetrics
from newton._src.solvers.kamino.solvers.padmm import PADMMSolver
from newton._src.solvers.kamino.solvers.padmm.types import PADMMData
from newton._src.solvers.kamino.tests import setup_tests, test_context
from newton._src.solvers.kamino.tests.test_solvers_padmm import TestSetup
from newton._src.solvers.kamino.tests.utils.extract import (
    extract_cts_jacobians,
    extract_delassus,
    extract_info_vectors,
    extract_problem_vector,
)
from newton._src.solvers.kamino.utils import logger as msg

###
# Helpers
###


def compute_metrics_numpy(problem: DualProblem, solver_data: PADMMData) -> dict[np.ndarray]:
    """Compute numpy solution to metrics."""
    output = {}
    output["r_v_plus"] = []
    output["s"] = []
    output["f_ccp"] = []
    output["f_ncp"] = []
    output["v_aug"] = []
    output["r_ncp_p"] = []
    output["r_ncp_d"] = []
    output["r_ncp_c"] = []
    output["r_ncp_natmap"] = []

    D = extract_delassus(problem.delassus, only_active_dims=True)
    num_matrices = len(D)

    lambdas = extract_problem_vector(problem.delassus, solver_data.solution.lambdas.numpy().astype(np.float64), True)
    v_plus_est = extract_problem_vector(problem.delassus, solver_data.solution.v_plus.numpy().astype(np.float64), True)
    v_f = extract_problem_vector(problem.delassus, problem.data.v_f.numpy().astype(np.float64), True)
    P = extract_problem_vector(problem.delassus, problem.data.P.numpy().astype(np.float64), True)
    sigma = solver_data.state.sigma.numpy().astype(np.float64)

    mu = extract_info_vectors(
        problem.data.cio.numpy(), problem.data.mu.numpy().astype(np.float64), problem.delassus.info.dim.numpy()
    )

    num_joint_cts = problem.data.njc.numpy()
    num_contacts = problem.data.nc.numpy()
    num_limits = problem.data.nl.numpy()
    contact_group_offset = problem.data.ccgo.numpy()
    limit_group_offset = problem.data.lcgo.numpy()

    for mat_id in range(num_matrices):
        D_i = D[mat_id]
        lambdas_i = lambdas[mat_id]
        v_plus_est_i = v_plus_est[mat_id]
        v_f_i = v_f[mat_id]
        mu_i = mu[mat_id]
        P_inv_i = np.reciprocal(P[mat_id])
        sigma_i = sigma[mat_id, 0]

        # Compute the post-event constraint-space velocity from the current solution: v_plus = v_f + D @ lambda
        v_plus_true_i = np.diag(P_inv_i) @ (
            v_f_i + ((D_i - sigma_i * np.identity(len(P_inv_i))) @ (np.diag(P_inv_i) @ lambdas_i))
        )
        # Compute the post-event constraint-space velocity error as: r_v_plus = || v_plus_est - v_plus_true ||_inf
        r_v_plus_i = np.max(np.abs(v_plus_est_i - v_plus_true_i))
        output["r_v_plus"].append(r_v_plus_i)

        # Compute the De Saxce correction for each contact as: s = G(v_plus)
        s_i = np.zeros_like(v_plus_true_i)
        for contact_id in range(num_contacts[mat_id]):
            v_idx = contact_group_offset[mat_id] + 3 * contact_id
            s_i[v_idx + 2] = mu_i[contact_id] * np.linalg.norm(v_plus_true_i[v_idx : v_idx + 2])
        output["s"].append(s_i)

        # Compute the CCP optimization objective as: f_ccp = 0.5 * lambda.dot(v_plus + v_f)
        f_ccp_i = 0.5 * lambdas_i.dot(v_f_i + v_plus_true_i)
        output["f_ccp"].append(f_ccp_i)

        # Compute the NCP optimization objective as:  f_ncp = f_ccp + lambda.dot(s)
        f_ncp_i = f_ccp_i + lambdas_i.dot(s_i)
        output["f_ncp"].append(f_ncp_i)

        # Compute the augmented post-event constraint-space velocity as: v_aug = v_plus + s
        v_aug_i = v_plus_true_i + s_i
        output["v_aug"].append(v_aug_i)

        # Compute the NCP primal residual as: r_p := || lambda - proj_K(lambda) ||_inf
        r_ncp_p_i = 0.0
        for limit_id in range(num_limits[mat_id]):
            lcio = limit_group_offset[mat_id] + limit_id
            r_ncp_p_i = np.max(r_ncp_p_i, np.abs(lambdas_i[lcio] - np.max(0.0, lambdas_i[lcio])))

        def project_to_coulomb_cone(x, mu):
            xt_norm = np.linalg.norm(x[:2])
            if mu * xt_norm > -x[2]:
                if xt_norm <= mu * x[2]:
                    return x
                else:
                    ys = (mu * xt_norm + x[2]) / (mu * mu + 1.0)
                    yts = mu * ys / xt_norm
                    return np.array([yts * x[0], yts * x[1], ys])
            return np.zeros(3)

        for contact_id in range(num_contacts[mat_id]):
            ccio = contact_group_offset[mat_id] + 3 * contact_id
            lambda_c = lambdas_i[ccio : ccio + 3] - project_to_coulomb_cone(
                lambdas_i[ccio : ccio + 3], mu_i[contact_id]
            )
            r_ncp_p_i = np.max([r_ncp_p_i, np.max(np.abs(lambda_c))])

        output["r_ncp_p"].append(r_ncp_p_i)

        # Compute the NCP dual residual as: r_d := || v_plus + s - proj_dual_K(v_plus + s)  ||_inf
        r_ncp_d_i = 0.0
        for jid in range(num_joint_cts[mat_id]):
            v_j = v_aug_i[jid]
            r_j = np.abs(v_j)
            r_ncp_d_i = max(r_ncp_d_i, r_j)

        for lid in range(num_limits[mat_id]):
            v_l = float(v_aug_i[limit_group_offset[mat_id] + lid])
            v_l -= np.max(0.0, v_l)
            r_l = np.abs(v_l)
            r_ncp_d_i = max(r_ncp_d_i, r_l)

        def project_to_coulomb_dual_cone(x: np.ndarray, mu: float) -> np.ndarray:
            xn = x[2]
            xt_norm = np.linalg.norm(x[:2])
            y = np.zeros(3)
            if xt_norm > -mu * xn:
                if mu * xt_norm <= xn:
                    y = x
                else:
                    ys = (xt_norm + mu * xn) / (mu * mu + 1.0)
                    yts = ys / xt_norm
                    y[0] = yts * x[0]
                    y[1] = yts * x[1]
                    y[2] = mu * ys
            return y

        for cid in range(num_contacts[mat_id]):
            ccio_c = contact_group_offset[mat_id] + 3 * cid
            mu_c = mu_i[cid]
            v_c = v_aug_i[ccio_c : ccio_c + 3].copy()
            v_c -= project_to_coulomb_dual_cone(v_c, mu_c)
            r_c = np.max(np.abs(v_c))
            r_ncp_d_i = max(r_ncp_d_i, r_c)

        output["r_ncp_d"].append(r_ncp_d_i)

        # Compute the NCP complementarity (lambda _|_ (v_plus + s)) residual as r_c := || lambda.dot(v_plus + s) ||_inf
        r_ncp_c_i = 0.0
        for lid in range(num_limits[mat_id]):
            lcio = limit_group_offset[mat_id] + lid
            v_l = v_aug_i[lcio]
            lambda_l = lambdas_i[lcio]
            r_l = np.abs(v_l * lambda_l)
            r_ncp_c_i = max(r_ncp_c_i, r_l)

        for cid in range(num_contacts[mat_id]):
            ccio = contact_group_offset[mat_id] + 3 * cid
            v_c = v_aug_i[ccio : ccio + 3]
            lambda_c = lambdas_i[ccio : ccio + 3]
            r_c = np.abs(np.dot(v_c, lambda_c))
            r_ncp_c_i = max(r_ncp_c_i, r_c)
        output["r_ncp_c"].append(r_ncp_c_i)

        # Compute the natural-map residuals as: r_natmap = || lambda - proj_K(lambda - (v + s)) ||_inf
        r_ncp_natmap_i = 0.0
        for lid in range(num_limits[mat_id]):
            lcio = limit_group_offset[mat_id] + lid
            v_l = v_aug_i[lcio]
            lambda_l = lambdas_i[lcio]
            lambda_l -= np.max(0.0, lambda_l - v_l)
            lambda_l = np.abs(lambda_l)
            r_ncp_natmap_i = max(r_ncp_natmap_i, lambda_l)

        for cid in range(num_contacts[mat_id]):
            ccio = contact_group_offset[mat_id] + 3 * cid
            mu_c = mu_i[cid]
            v_c = v_aug_i[ccio : ccio + 3]
            lambda_c = lambdas_i[ccio : ccio + 3]
            lambda_c -= project_to_coulomb_cone(lambda_c - v_c, mu_c)
            lambda_c = np.abs(lambda_c)
            lambda_c_max = np.max(lambda_c)
            r_ncp_natmap_i = max(r_ncp_natmap_i, lambda_c_max)

        output["r_ncp_natmap"].append(r_ncp_natmap_i)

    return output


###
# Tests
###


class TestSolverMetrics(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        self.verbose = test_context.verbose  # Set to True for detailed output
        self.savefig = False  # Set to True to generate solver info plots
        self.output_dir = os.path.dirname(os.path.realpath(__file__)) + "/output/test_solvers_padmm"

        # Create output directory if saving figures
        if self.savefig:
            os.makedirs(self.output_dir, exist_ok=True)

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
        # Creating a default solver metrics evaluator without any model
        # should result in an instance without any memory allocation.
        metrics = SolutionMetrics()
        self.assertIsNone(metrics._device)
        self.assertIsNone(metrics._data)
        self.assertIsNone(metrics._buffer_s)
        self.assertIsNone(metrics._buffer_v)

        # Requesting the solver data container when the
        # solver has not been finalized should raise an
        # error since no allocations have been made.
        self.assertRaises(RuntimeError, lambda: metrics.data)

    def test_01_finalize_default(self):
        """
        Test creating a SolutionMetrics instance with default initialization and then finalizing all memory allocations.
        """
        # Create a test setup
        test = TestSetup(builder_fn=build_box_on_plane, max_world_contacts=8, device=self.default_device)

        # Creating a default solver metrics evaluator without any model
        # should result in an instance without any memory allocation.
        metrics = SolutionMetrics()

        # Finalize the solver with a model
        metrics.finalize(test.model)

        # Check that the solver has been properly allocated
        self.assertIsNotNone(metrics._data)
        self.assertIsNotNone(metrics._device)
        self.assertIs(metrics._device, test.model.device)
        self.assertIsNotNone(metrics._buffer_s)
        self.assertIsNotNone(metrics._buffer_v)

        # Check allocation sizes
        msg.info("num_worlds: %s", test.model.size.num_worlds)
        msg.info("sum_of_max_total_cts: %s", test.model.size.sum_of_max_total_cts)
        msg.info("buffer_s size: %s", metrics._buffer_s.size)
        msg.info("buffer_v size: %s", metrics._buffer_v.size)
        self.assertEqual(metrics._buffer_s.size, test.model.size.sum_of_max_total_cts)
        self.assertEqual(metrics._buffer_v.size, test.model.size.sum_of_max_total_cts)
        self.assertEqual(metrics.data.r_eom.size, test.model.size.num_worlds)
        self.assertEqual(metrics.data.r_eom_argmax.size, test.model.size.num_worlds)
        self.assertEqual(metrics.data.r_kinematics.size, test.model.size.num_worlds)
        self.assertEqual(metrics.data.r_kinematics_argmax.size, test.model.size.num_worlds)
        self.assertEqual(metrics.data.r_cts_joints.size, test.model.size.num_worlds)
        self.assertEqual(metrics.data.r_cts_joints_argmax.size, test.model.size.num_worlds)
        self.assertEqual(metrics.data.r_cts_limits.size, test.model.size.num_worlds)
        self.assertEqual(metrics.data.r_cts_limits_argmax.size, test.model.size.num_worlds)
        self.assertEqual(metrics.data.r_cts_contacts.size, test.model.size.num_worlds)
        self.assertEqual(metrics.data.r_cts_contacts_argmax.size, test.model.size.num_worlds)
        self.assertEqual(metrics.data.r_v_plus.size, test.model.size.num_worlds)
        self.assertEqual(metrics.data.r_v_plus_argmax.size, test.model.size.num_worlds)
        self.assertEqual(metrics.data.r_ncp_primal.size, test.model.size.num_worlds)
        self.assertEqual(metrics.data.r_ncp_primal_argmax.size, test.model.size.num_worlds)
        self.assertEqual(metrics.data.r_ncp_dual.size, test.model.size.num_worlds)
        self.assertEqual(metrics.data.r_ncp_dual_argmax.size, test.model.size.num_worlds)
        self.assertEqual(metrics.data.r_ncp_compl.size, test.model.size.num_worlds)
        self.assertEqual(metrics.data.r_ncp_compl_argmax.size, test.model.size.num_worlds)
        self.assertEqual(metrics.data.r_vi_natmap.size, test.model.size.num_worlds)
        self.assertEqual(metrics.data.r_vi_natmap_argmax.size, test.model.size.num_worlds)

    def test_02_evaluate_trivial_solution(self):
        """
        Tests evaluating metrics on an all-zeros trivial solution.
        """
        # Create the test problem
        test = TestSetup(
            builder_fn=build_box_on_plane,
            max_world_contacts=4,
            gravity=False,
            perturb=False,
            device=self.default_device,
        )

        # Creating a default solver metrics evaluator from the test model
        metrics = SolutionMetrics(model=test.model)

        # Define a trivial solution (all zeros)
        with wp.ScopedDevice(test.model.device):
            sigma = wp.zeros(test.model.size.num_worlds, dtype=wp.vec2f)
            lambdas = wp.zeros(test.model.size.sum_of_max_total_cts, dtype=wp.float32)
            v_plus = wp.zeros(test.model.size.sum_of_max_total_cts, dtype=wp.float32)

        # Build the test problem and integrate the state over a single time-step
        test.build()
        integrate_euler_semi_implicit(model=test.model, data=test.data)

        nl = test.limits.model_active_limits.numpy()[0] if test.limits.model_max_limits_host > 0 else 0
        nc = test.contacts.model_active_contacts.numpy()[0] if test.contacts.model_max_contacts_host > 0 else 0
        msg.info("num active limits: %s", nl)
        msg.info("num active contacts: %s\n", nc)
        self.assertEqual(nl, 0)
        self.assertEqual(nc, 4)

        # Compute the metrics on the trivial solution
        metrics.reset()
        metrics.evaluate(
            sigma=sigma,
            lambdas=lambdas,
            v_plus=v_plus,
            model=test.model,
            data=test.data,
            state_p=test.state_p,
            problem=test.problem,
            jacobians=test.jacobians,
            limits=test.limits,
            contacts=test.contacts,
        )

        # Optional verbose output
        msg.info("metrics.r_eom: %s", metrics.data.r_eom)
        msg.info("metrics.r_kinematics: %s", metrics.data.r_kinematics)
        msg.info("metrics.r_cts_joints: %s", metrics.data.r_cts_joints)
        msg.info("metrics.r_cts_limits: %s", metrics.data.r_cts_limits)
        msg.info("metrics.r_cts_contacts: %s", metrics.data.r_cts_contacts)
        msg.info("metrics.r_v_plus: %s", metrics.data.r_v_plus)
        msg.info("metrics.r_ncp_primal: %s", metrics.data.r_ncp_primal)
        msg.info("metrics.r_ncp_dual: %s", metrics.data.r_ncp_dual)
        msg.info("metrics.r_ncp_compl: %s", metrics.data.r_ncp_compl)
        msg.info("metrics.r_vi_natmap: %s\n", metrics.data.r_vi_natmap)

        # Extract the maximum contact penetration to use for validation
        nc = test.contacts.model_active_contacts.numpy()[0]
        max_contact_penetration = 0.0
        for cid in range(nc):
            pen = test.contacts.gapfunc.numpy()[cid][3]
            max_contact_penetration = max(max_contact_penetration, pen)

        # Check that all metrics are zero
        np.testing.assert_allclose(metrics.data.r_eom.numpy()[0], 0.0)
        np.testing.assert_allclose(metrics.data.r_kinematics.numpy()[0], 0.0)
        np.testing.assert_allclose(metrics.data.r_cts_joints.numpy()[0], 0.0)
        np.testing.assert_allclose(metrics.data.r_cts_limits.numpy()[0], 0.0)
        np.testing.assert_allclose(metrics.data.r_cts_contacts.numpy()[0], max_contact_penetration)
        np.testing.assert_allclose(metrics.data.r_ncp_primal.numpy()[0], 0.0)
        np.testing.assert_allclose(metrics.data.r_ncp_dual.numpy()[0], 0.0)
        np.testing.assert_allclose(metrics.data.r_ncp_compl.numpy()[0], 0.0)
        np.testing.assert_allclose(metrics.data.r_vi_natmap.numpy()[0], 0.0)

        # Optional verbose output
        msg.info("metrics.r_eom_argmax: %s", metrics.data.r_eom_argmax)
        msg.info("metrics.r_kinematics_argmax: %s", metrics.data.r_kinematics_argmax)
        msg.info("metrics.r_cts_joints_argmax: %s", metrics.data.r_cts_joints_argmax)
        msg.info("metrics.r_cts_limits_argmax: %s", metrics.data.r_cts_limits_argmax)
        msg.info("metrics.r_cts_contacts_argmax: %s", metrics.data.r_cts_contacts_argmax)
        msg.info("metrics.r_v_plus_argmax: %s", metrics.data.r_v_plus_argmax)
        msg.info("metrics.r_ncp_primal_argmax: %s", metrics.data.r_ncp_primal_argmax)
        msg.info("metrics.r_ncp_dual_argmax: %s", metrics.data.r_ncp_dual_argmax)
        msg.info("metrics.r_ncp_compl_argmax: %s", metrics.data.r_ncp_compl_argmax)
        msg.info("metrics.r_vi_natmap_argmax: %s\n", metrics.data.r_vi_natmap_argmax)

        # Check that all argmax indices are correct
        np.testing.assert_allclose(metrics.data.r_eom_argmax.numpy()[0], 0)  # only one body
        np.testing.assert_allclose(metrics.data.r_kinematics_argmax.numpy()[0], -1)  # no joints
        np.testing.assert_allclose(metrics.data.r_cts_joints_argmax.numpy()[0], -1)  # no joints
        np.testing.assert_allclose(metrics.data.r_cts_limits_argmax.numpy()[0], -1)  # no limits
        # NOTE: all contacts will have the same residual,
        # so the argmax will evaluate to the last constraint
        np.testing.assert_allclose(metrics.data.r_v_plus_argmax.numpy()[0], 11)
        # NOTE: all contacts will have the same penetration,
        # so the argmax will evaluate to the last contact
        np.testing.assert_allclose(metrics.data.r_cts_contacts_argmax.numpy()[0], 3)
        np.testing.assert_allclose(metrics.data.r_ncp_primal_argmax.numpy()[0], 3)
        np.testing.assert_allclose(metrics.data.r_ncp_dual_argmax.numpy()[0], 3)
        np.testing.assert_allclose(metrics.data.r_ncp_compl_argmax.numpy()[0], 3)
        np.testing.assert_allclose(metrics.data.r_vi_natmap_argmax.numpy()[0], 3)

    def test_03_evaluate_padmm_solution_box_on_plane(self):
        """
        Tests evaluating metrics on a solution computed with the Proximal-ADMM (PADMM) solver.
        """
        # Create the test problem
        test = TestSetup(
            builder_fn=build_box_on_plane,
            max_world_contacts=4,
            gravity=True,
            perturb=True,
            device=self.default_device,
        )

        # Create the PADMM solver
        solver = PADMMSolver(model=test.model, use_acceleration=False, collect_info=True)

        # Creating a default solver metrics evaluator from the test model
        metrics = SolutionMetrics(model=test.model)

        # Solve the test problem
        test.build()
        solver.reset()
        solver.coldstart()
        solver.solve(problem=test.problem)
        integrate_euler_semi_implicit(model=test.model, data=test.data)

        # Compute the metrics on the trivial solution
        metrics.reset()
        metrics.evaluate(
            sigma=solver.data.state.sigma,
            lambdas=solver.data.solution.lambdas,
            v_plus=solver.data.solution.v_plus,
            model=test.model,
            data=test.data,
            state_p=test.state_p,
            problem=test.problem,
            jacobians=test.jacobians,
            limits=test.limits,
            contacts=test.contacts,
        )

        nl = test.limits.model_active_limits.numpy()[0] if test.limits.model_max_limits_host > 0 else 0
        nc = test.contacts.model_active_contacts.numpy()[0] if test.contacts.model_max_contacts_host > 0 else 0
        msg.info("num active limits: %s", nl)
        msg.info("num active contacts: %s\n", nc)

        # Optional verbose output
        msg.info("metrics.r_eom: %s", metrics.data.r_eom)
        msg.info("metrics.r_kinematics: %s", metrics.data.r_kinematics)
        msg.info("metrics.r_cts_joints: %s", metrics.data.r_cts_joints)
        msg.info("metrics.r_cts_limits: %s", metrics.data.r_cts_limits)
        msg.info("metrics.r_cts_contacts: %s", metrics.data.r_cts_contacts)
        msg.info("metrics.r_v_plus: %s", metrics.data.r_v_plus)
        msg.info("metrics.r_ncp_primal: %s", metrics.data.r_ncp_primal)
        msg.info("metrics.r_ncp_dual: %s", metrics.data.r_ncp_dual)
        msg.info("metrics.r_ncp_compl: %s", metrics.data.r_ncp_compl)
        msg.info("metrics.r_vi_natmap: %s\n", metrics.data.r_vi_natmap)

        # Extract the maximum contact penetration to use for validation
        nc = test.contacts.model_active_contacts.numpy()[0]
        max_contact_penetration = 0.0
        for cid in range(nc):
            pen = test.contacts.gapfunc.numpy()[cid][3]
            max_contact_penetration = max(max_contact_penetration, pen)

        # Check that all metrics are zero
        accuracy = 5  # number of decimal places for accuracy
        self.assertAlmostEqual(metrics.data.r_eom.numpy()[0], 0.0, places=accuracy)
        self.assertAlmostEqual(metrics.data.r_kinematics.numpy()[0], 0.0, places=accuracy)
        self.assertAlmostEqual(metrics.data.r_cts_joints.numpy()[0], 0.0, places=accuracy)
        self.assertAlmostEqual(metrics.data.r_cts_limits.numpy()[0], 0.0, places=accuracy)
        self.assertAlmostEqual(metrics.data.r_cts_contacts.numpy()[0], max_contact_penetration, places=accuracy)
        self.assertAlmostEqual(metrics.data.r_ncp_primal.numpy()[0], 0.0, places=accuracy)
        self.assertAlmostEqual(metrics.data.r_ncp_dual.numpy()[0], 0.0, places=accuracy)
        self.assertAlmostEqual(metrics.data.r_ncp_compl.numpy()[0], 0.0, places=accuracy)
        self.assertAlmostEqual(metrics.data.r_vi_natmap.numpy()[0], 0.0, places=accuracy)

        # Optional verbose output
        msg.info("metrics.r_eom_argmax: %s", metrics.data.r_eom_argmax)
        msg.info("metrics.r_kinematics_argmax: %s", metrics.data.r_kinematics_argmax)
        msg.info("metrics.r_cts_joints_argmax: %s", metrics.data.r_cts_joints_argmax)
        msg.info("metrics.r_cts_limits_argmax: %s", metrics.data.r_cts_limits_argmax)
        msg.info("metrics.r_cts_contacts_argmax: %s", metrics.data.r_cts_contacts_argmax)
        msg.info("metrics.r_v_plus_argmax: %s", metrics.data.r_v_plus_argmax)
        msg.info("metrics.r_ncp_primal_argmax: %s", metrics.data.r_ncp_primal_argmax)
        msg.info("metrics.r_ncp_dual_argmax: %s", metrics.data.r_ncp_dual_argmax)
        msg.info("metrics.r_ncp_compl_argmax: %s", metrics.data.r_ncp_compl_argmax)
        msg.info("metrics.r_vi_natmap_argmax: %s\n", metrics.data.r_vi_natmap_argmax)

    def test_04_evaluate_padmm_solution_boxes_hinged(self):
        """
        Tests evaluating metrics on a solution computed with the Proximal-ADMM (PADMM) solver.
        """
        # Create the test problem
        test = TestSetup(
            builder_fn=build_boxes_hinged,
            max_world_contacts=8,
            gravity=True,
            perturb=True,
            device=self.default_device,
        )

        # Create the PADMM solver
        solver = PADMMSolver(model=test.model, use_acceleration=False, collect_info=True)

        # Creating a default solver metrics evaluator from the test model
        metrics = SolutionMetrics(model=test.model)

        # Solve the test problem
        test.build()
        solver.reset()
        solver.coldstart()
        solver.solve(problem=test.problem)
        integrate_euler_semi_implicit(model=test.model, data=test.data)

        # Compute the metrics on the trivial solution
        metrics.evaluate(
            sigma=solver.data.state.sigma,
            lambdas=solver.data.solution.lambdas,
            v_plus=solver.data.solution.v_plus,
            model=test.model,
            data=test.data,
            state_p=test.state_p,
            problem=test.problem,
            jacobians=test.jacobians,
            limits=test.limits,
            contacts=test.contacts,
        )

        nl = test.limits.model_active_limits.numpy()[0] if test.limits.model_max_limits_host > 0 else 0
        nc = test.contacts.model_active_contacts.numpy()[0] if test.contacts.model_max_contacts_host > 0 else 0
        msg.info("num active limits: %s", nl)
        msg.info("num active contacts: %s\n", nc)

        # Optional verbose output
        msg.info("metrics.r_eom: %s", metrics.data.r_eom)
        msg.info("metrics.r_kinematics: %s", metrics.data.r_kinematics)
        msg.info("metrics.r_cts_joints: %s", metrics.data.r_cts_joints)
        msg.info("metrics.r_cts_limits: %s", metrics.data.r_cts_limits)
        msg.info("metrics.r_cts_contacts: %s", metrics.data.r_cts_contacts)
        msg.info("metrics.r_v_plus: %s", metrics.data.r_v_plus)
        msg.info("metrics.r_ncp_primal: %s", metrics.data.r_ncp_primal)
        msg.info("metrics.r_ncp_dual: %s", metrics.data.r_ncp_dual)
        msg.info("metrics.r_ncp_compl: %s", metrics.data.r_ncp_compl)
        msg.info("metrics.r_vi_natmap: %s\n", metrics.data.r_vi_natmap)

        # Extract the maximum contact penetration to use for validation
        max_contact_penetration = 0.0
        for cid in range(nc):
            pen = test.contacts.gapfunc.numpy()[cid][3]
            max_contact_penetration = max(max_contact_penetration, pen)

        # Check that all metrics are zero
        accuracy = 5  # number of decimal places for accuracy
        self.assertAlmostEqual(metrics.data.r_eom.numpy()[0], 0.0, places=accuracy)
        self.assertAlmostEqual(metrics.data.r_kinematics.numpy()[0], 0.0, places=accuracy)
        self.assertAlmostEqual(metrics.data.r_cts_joints.numpy()[0], 0.0, places=accuracy)
        self.assertAlmostEqual(metrics.data.r_cts_limits.numpy()[0], 0.0, places=accuracy)
        self.assertAlmostEqual(metrics.data.r_cts_contacts.numpy()[0], max_contact_penetration, places=accuracy)
        self.assertAlmostEqual(metrics.data.r_ncp_primal.numpy()[0], 0.0, places=accuracy)
        self.assertAlmostEqual(metrics.data.r_ncp_dual.numpy()[0], 0.0, places=4)  # less accurate, but still correct
        self.assertAlmostEqual(metrics.data.r_ncp_compl.numpy()[0], 0.0, places=accuracy)
        self.assertAlmostEqual(metrics.data.r_vi_natmap.numpy()[0], 0.0, places=accuracy)

        # Optional verbose output
        msg.info("metrics.r_eom_argmax: %s", metrics.data.r_eom_argmax)
        msg.info("metrics.r_kinematics_argmax: %s", metrics.data.r_kinematics_argmax)
        msg.info("metrics.r_cts_joints_argmax: %s", metrics.data.r_cts_joints_argmax)
        msg.info("metrics.r_cts_limits_argmax: %s", metrics.data.r_cts_limits_argmax)
        msg.info("metrics.r_cts_contacts_argmax: %s", metrics.data.r_cts_contacts_argmax)
        msg.info("metrics.r_v_plus_argmax: %s", metrics.data.r_v_plus_argmax)
        msg.info("metrics.r_ncp_primal_argmax: %s", metrics.data.r_ncp_primal_argmax)
        msg.info("metrics.r_ncp_dual_argmax: %s", metrics.data.r_ncp_dual_argmax)
        msg.info("metrics.r_ncp_compl_argmax: %s", metrics.data.r_ncp_compl_argmax)
        msg.info("metrics.r_vi_natmap_argmax: %s\n", metrics.data.r_vi_natmap_argmax)

    def test_05_compare_dense_sparse_padmm_solution_boxes_hinged(self):
        """
        Compares metrics evaluated on dense and sparse problems on a solution computed with the
        Proximal-ADMM (PADMM) solver.
        """
        # Create the test problem
        test = TestSetup(
            builder_fn=build_boxes_hinged,
            max_world_contacts=8,
            gravity=True,
            perturb=True,
            device=self.default_device,
            sparse=False,
        )

        # Create the PADMM solver
        solver = PADMMSolver(model=test.model, use_acceleration=False, collect_info=True)

        # Creating a default solver metrics evaluator from the test model
        metrics_dense = SolutionMetrics(model=test.model)
        metrics_sparse = SolutionMetrics(model=test.model)

        # Create sparse version of the Jacobians
        jacobians_sparse = SparseSystemJacobians(
            model=test.model,
            limits=test.limits,
            contacts=test.detector.contacts,
            device=self.default_device,
        )
        jacobians_sparse.build(
            model=test.model,
            data=test.data,
            limits=test.limits.data,
            contacts=test.detector.contacts.data,
        )

        # Create sparse version of the dual problem
        problem_sparse = DualProblem(
            model=test.model,
            data=test.data,
            limits=test.limits,
            contacts=test.contacts,
            device=self.default_device,
            sparse=True,
        )
        problem_sparse.build(
            model=test.model,
            data=test.data,
            jacobians=jacobians_sparse,
            limits=test.limits,
            contacts=test.detector.contacts,
        )

        # Solve the test problem
        test.build()
        solver.reset()
        solver.coldstart()
        solver.solve(problem=test.problem)
        integrate_euler_semi_implicit(model=test.model, data=test.data)

        solver._initialize()
        solver._update_sparse_regularization(problem_sparse)
        problem_sparse.delassus.update()

        # Compute the metrics on the solution
        metrics_dense.evaluate(
            sigma=solver.data.state.sigma,
            lambdas=solver.data.solution.lambdas,
            v_plus=solver.data.solution.v_plus,
            model=test.model,
            data=test.data,
            state_p=test.state_p,
            problem=test.problem,
            jacobians=test.jacobians,
            limits=test.limits,
            contacts=test.contacts,
        )
        metrics_sparse.evaluate(
            sigma=solver.data.state.sigma,
            lambdas=solver.data.solution.lambdas,
            v_plus=solver.data.solution.v_plus,
            model=test.model,
            data=test.data,
            state_p=test.state_p,
            problem=problem_sparse,
            jacobians=jacobians_sparse,
            limits=test.limits,
            contacts=test.contacts,
        )

        def check_difference(
            val_A: wp.array | np.ndarray,
            val_B: wp.array | np.ndarray,
            rtol: float = 1e-5,
            atol: float = 1e-6,
        ):
            if isinstance(val_A, wp.array):
                val_A = val_A.numpy()
            if isinstance(val_B, wp.array):
                val_B = val_B.numpy()
            np.testing.assert_allclose(val_A, val_B, rtol=rtol, atol=atol)

        # Compare Jacobians
        J_cts_dense_np = extract_cts_jacobians(
            model=test.model,
            limits=test.limits,
            contacts=test.contacts,
            jacobians=test.jacobians,
            only_active_cts=True,
        )
        J_cts_sparse_np = jacobians_sparse._J_cts.bsm.numpy()
        for J_cts_dense_np_i, J_cts_sparse_np_i in zip(J_cts_dense_np, J_cts_sparse_np, strict=True):
            check_difference(J_cts_dense_np_i, J_cts_sparse_np_i)

        # Compare Delassus matrix
        D_dense_np = extract_delassus(delassus=test.problem.delassus, only_active_dims=True)
        D_sparse_np = extract_delassus(delassus=problem_sparse.delassus, only_active_dims=True)
        for D_dense_np_i, D_sparse_np_i in zip(D_dense_np, D_sparse_np, strict=True):
            check_difference(D_dense_np_i, D_sparse_np_i)

        # Somewhat hacky way to check `v_aug` computed in the metrics kernel, stored in `buffer_v`
        check_difference(metrics_dense._buffer_v, metrics_sparse._buffer_v)
        # Somewhat hacky way to check `s` computed in the metrics kernel, stored in `buffer_s`
        check_difference(metrics_dense._buffer_s, metrics_sparse._buffer_s)

        check_difference(metrics_dense.data.f_ncp, metrics_sparse.data.f_ncp)
        check_difference(metrics_dense.data.f_ccp, metrics_sparse.data.f_ccp)
        check_difference(metrics_dense.data.r_v_plus, metrics_sparse.data.r_v_plus)
        check_difference(metrics_dense.data.r_eom, metrics_sparse.data.r_eom)
        check_difference(metrics_dense.data.r_kinematics, metrics_sparse.data.r_kinematics)
        check_difference(metrics_dense.data.r_cts_joints, metrics_sparse.data.r_cts_joints)
        check_difference(metrics_dense.data.r_cts_limits, metrics_sparse.data.r_cts_limits)
        check_difference(metrics_dense.data.r_cts_contacts, metrics_sparse.data.r_cts_contacts)
        check_difference(metrics_dense.data.r_ncp_primal, metrics_sparse.data.r_ncp_primal)
        check_difference(metrics_dense.data.r_ncp_dual, metrics_sparse.data.r_ncp_dual)
        check_difference(metrics_dense.data.r_ncp_compl, metrics_sparse.data.r_ncp_compl)
        check_difference(metrics_dense.data.r_vi_natmap, metrics_sparse.data.r_vi_natmap)

        # Compute numpy solution to metrics
        metrics_np = compute_metrics_numpy(test.problem, solver.data)
        check_difference(metrics_np["r_v_plus"], metrics_sparse.data.r_v_plus)
        check_difference(metrics_np["f_ccp"], metrics_sparse.data.f_ccp)
        check_difference(metrics_np["f_ncp"], metrics_sparse.data.f_ncp)
        check_difference(
            metrics_np["r_ncp_p"],
            metrics_sparse.data.r_ncp_primal,
        )

        # Somewhat hacky way to check `v_aug` computed in the metrics kernel, stored in `buffer_v`,
        # and `s`, stored in `buffer_s`
        s = extract_problem_vector(test.problem.delassus, metrics_sparse._buffer_s.numpy(), True)
        v_aug = extract_problem_vector(test.problem.delassus, metrics_sparse._buffer_v.numpy(), True)
        for world_id in range(test.model.size.num_worlds):
            check_difference(metrics_np["s"][world_id], s[world_id])
            check_difference(metrics_np["v_aug"][world_id], v_aug[world_id])


###
# Test execution
###

if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
