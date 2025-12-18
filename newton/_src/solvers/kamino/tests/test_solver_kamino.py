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

import time
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino.core.control import Control
from newton._src.solvers.kamino.core.joints import JointCorrectionMode
from newton._src.solvers.kamino.core.model import Model, ModelData
from newton._src.solvers.kamino.core.state import State
from newton._src.solvers.kamino.core.types import float32
from newton._src.solvers.kamino.dynamics import DualProblem, DualProblemSettings
from newton._src.solvers.kamino.examples import print_progress_bar
from newton._src.solvers.kamino.geometry.contacts import Contacts
from newton._src.solvers.kamino.kinematics.jacobians import DenseSystemJacobians
from newton._src.solvers.kamino.kinematics.limits import Limits
from newton._src.solvers.kamino.linalg import ConjugateGradientSolver, LinearSolverType, LLTBlockedSolver
from newton._src.solvers.kamino.models.builders.basics import build_cartpole
from newton._src.solvers.kamino.models.builders.utils import make_homogeneous_builder
from newton._src.solvers.kamino.solver_kamino import SolverKamino, SolverKaminoSettings
from newton._src.solvers.kamino.solvers import PADMMSettings, PADMMSolver, PADMMWarmStartMode
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
    solver: SolverKamino, state_in: State, state_out: State, control: Control, contacts: Contacts
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
# Utils
###


def assert_solver_settings(testcase: unittest.TestCase, settings: SolverKaminoSettings):
    testcase.assertIsInstance(settings, SolverKaminoSettings)
    testcase.assertIsInstance(settings.problem, DualProblemSettings)
    testcase.assertIsInstance(settings.padmm, PADMMSettings)
    testcase.assertIsInstance(settings.warmstart, PADMMWarmStartMode)
    testcase.assertTrue(issubclass(settings.linear_solver_type, LinearSolverType))
    testcase.assertIsInstance(settings.rotation_correction, JointCorrectionMode)


def assert_solver_components(testcase: unittest.TestCase, solver: SolverKamino):
    testcase.assertIsInstance(solver, SolverKamino)
    testcase.assertIsInstance(solver.settings, SolverKaminoSettings)
    testcase.assertIsInstance(solver._model, Model)
    testcase.assertIsInstance(solver._data, ModelData)
    testcase.assertIsInstance(solver._state_pp_cache, State)
    testcase.assertIsInstance(solver._limits, Limits)
    testcase.assertIsInstance(solver._jacobians, DenseSystemJacobians)
    testcase.assertIsInstance(solver._problem_fd, DualProblem)
    testcase.assertIsInstance(solver._solver_fd, PADMMSolver)


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

    def test_00_make_default(self):
        settings = SolverKaminoSettings()
        assert_solver_settings(self, settings)
        self.assertEqual(settings.linear_solver_type, LLTBlockedSolver)

    def test_01_make_explicit(self):
        settings = SolverKaminoSettings(
            problem=DualProblemSettings(),
            padmm=PADMMSettings(),
            warmstart=PADMMWarmStartMode.CONTAINERS,
            linear_solver_type=ConjugateGradientSolver,
            rotation_correction=JointCorrectionMode.CONTINUOUS,
        )
        assert_solver_settings(self, settings)
        self.assertEqual(settings.linear_solver_type, ConjugateGradientSolver)


class TestSolverKamino(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        # self.verbose = test_context.verbose  # Set to True to enable verbose output
        self.verbose = True  # Set to True to enable verbose output
        self.progress = True  # Set to True for progress output
        self.seed = 42

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

    def test_00_make_default_invalid(self):
        """
        Test that creating a default Kamino solver without a model raises an error.
        """
        self.assertRaises(TypeError, lambda: SolverKamino())

    def test_01_make_default_valid_without_contacts(self):
        """
        Test creating a default Kamino solver without support for contacts.
        """
        builder = make_homogeneous_builder(num_worlds=1, build_fn=build_cartpole)
        model = builder.finalize(device=self.default_device)
        solver = SolverKamino(model=model)
        self.assertIsInstance(solver, SolverKamino)
        assert_solver_components(self, solver)

    def test_02_make_default_valid_with_contacts(self):
        """
        Test creating a default Kamino solver with support for contacts.
        """
        builder = make_homogeneous_builder(num_worlds=1, build_fn=build_cartpole)
        model = builder.finalize(device=self.default_device)
        _, world_max_contacts = builder.compute_required_contact_capacity(max_contacts_per_pair=16)
        contacts = Contacts(capacity=world_max_contacts, device=model.device)
        solver = SolverKamino(model=model, contacts=contacts)
        self.assertIsInstance(solver, SolverKamino)
        assert_solver_components(self, solver)

    # TODO: Test w/o limits, contacts
    # TODO: Test w/o limits
    # TODO: Test w/o contacts
    # TODO: Test w/ limits, contacts

    def test_03_step_multiple_cartpoles_all_from_initial_state(self):
        """
        Test stepping multiple cartpole solvers initialized
        uniformly from the default initial state multiple times.
        """

        # Create a single-instance system
        single_builder = build_cartpole(ground=False)
        for i, body in enumerate(single_builder.bodies):
            msg.info(f"[single]: [builder]: body {i}: q_i: {body.q_i_0}")
            msg.info(f"[single]: [builder]: body {i}: u_i: {body.u_i_0}")

        # Create a model and states from the builder
        single_model = single_builder.finalize(device=self.default_device)
        single_state_p = single_model.state()
        single_state_n = single_model.state()
        single_control = single_model.control()
        self.assertEqual(single_model.size.sum_of_num_bodies, 2)
        self.assertEqual(single_model.size.sum_of_num_joints, 2)
        for i, body in enumerate(single_builder.bodies):
            np.testing.assert_allclose(single_model.bodies.q_i_0.numpy()[i], body.q_i_0)
            np.testing.assert_allclose(single_model.bodies.u_i_0.numpy()[i], body.u_i_0)
            np.testing.assert_allclose(single_state_p.q_i.numpy()[i], body.q_i_0)
            np.testing.assert_allclose(single_state_p.u_i.numpy()[i], body.u_i_0)
            np.testing.assert_allclose(single_state_n.q_i.numpy()[i], body.q_i_0)
            np.testing.assert_allclose(single_state_n.u_i.numpy()[i], body.u_i_0)

        # Optional verbose output - enabled globally via self.verbose
        msg.info(f"[single]: [init]: model.size:\n{single_model.size}\n\n")
        msg.info(f"[single]: [init]: single_state_p.q_i:\n{single_state_p.q_i}\n\n")
        msg.info(f"[single]: [init]: single_state_p.u_i:\n{single_state_p.u_i}\n\n")
        msg.info(f"[single]: [init]: single_state_p.w_i:\n{single_state_p.w_i}\n\n")
        msg.info(f"[single]: [init]: single_state_p.q_j:\n{single_state_p.q_j}\n\n")
        msg.info(f"[single]: [init]: single_state_p.dq_j:\n{single_state_p.dq_j}\n\n")
        msg.info(f"[single]: [init]: single_state_p.lambda_j:\n{single_state_p.lambda_j}\n\n")

        # Create a contacts container for the single-instance system
        _, single_world_max_contacts = single_builder.compute_required_contact_capacity(max_contacts_per_pair=16)
        single_contacts = Contacts(capacity=single_world_max_contacts, device=single_model.device)

        # Create simulator and check if the initial state is consistent with the contents of the builder
        single_solver = SolverKamino(model=single_model, contacts=single_contacts)
        self.assertIsInstance(single_solver, SolverKamino)
        assert_solver_components(self, single_solver)
        self.assertIs(single_solver._model, single_model)

        # Define the total number of sample steps to collect, and the
        # total number of execution steps from which to collect them
        num_worlds = 42
        num_steps = 1000

        # Collect the initial states
        initial_q_i = single_state_p.q_i.numpy().copy()
        initial_u_i = single_state_p.u_i.numpy().copy()
        initial_q_j = single_state_p.q_j.numpy().copy()
        initial_dq_j = single_state_p.dq_j.numpy().copy()
        msg.info(f"[samples]: [single]: [init]: q_i (shape={initial_q_i.shape}):\n{initial_q_i}\n")
        msg.info(f"[samples]: [single]: [init]: u_i (shape={initial_u_i.shape}):\n{initial_u_i}\n")
        msg.info(f"[samples]: [single]: [init]: w_i (shape={initial_u_i.shape}):\n{initial_u_i}\n")
        msg.info(f"[samples]: [single]: [init]: q_j (shape={initial_q_j.shape}):\n{initial_q_j}\n")
        msg.info(f"[samples]: [single]: [init]: dq_j (shape={initial_dq_j.shape}):\n{initial_dq_j}\n")
        msg.info(f"[samples]: [single]: [init]: lambda_j (shape={initial_dq_j.shape}):\n{initial_dq_j}\n")

        # Run the simulation for the specified number of steps
        msg.info(f"[single]: Executing {num_steps} simulator steps")
        start_time = time.time()
        for step in range(num_steps):
            # Execute a single simulation step
            single_solver.step(single_state_p, single_state_n, single_control, contacts=single_contacts, dt=0.001)
            wp.synchronize()
            if self.verbose or self.progress:
                print_progress_bar(step + 1, num_steps, start_time, prefix="Progress", suffix="")

        # Collect the initial and final states
        final_q_i = single_state_n.q_i.numpy().copy()
        final_u_i = single_state_n.u_i.numpy().copy()
        final_w_i = single_state_n.w_i.numpy().copy()
        final_q_j = single_state_n.q_j.numpy().copy()
        final_dq_j = single_state_n.dq_j.numpy().copy()
        final_lambda_j = single_state_n.lambda_j.numpy().copy()
        msg.info(f"[samples]: [single]: [final]: q_i (shape={final_q_i.shape}):\n{final_q_i}\n")
        msg.info(f"[samples]: [single]: [final]: u_i (shape={final_u_i.shape}):\n{final_u_i}\n")
        msg.info(f"[samples]: [single]: [final]: w_i (shape={final_w_i.shape}):\n{final_w_i}\n")
        msg.info(f"[samples]: [single]: [final]: q_j (shape={final_q_j.shape}):\n{final_q_j}\n")
        msg.info(f"[samples]: [single]: [final]: dq_j (shape={final_dq_j.shape}):\n{final_dq_j}\n")
        msg.info(f"[samples]: [single]: [final]: lambda_j (shape={final_lambda_j.shape}):\n{final_lambda_j}\n")

        # Tile the collected states for comparison against the multi-instance simulator
        multi_init_q_i = np.tile(initial_q_i, (num_worlds, 1))
        multi_init_u_i = np.tile(initial_u_i, (num_worlds, 1))
        multi_init_q_j = np.tile(initial_q_j, (num_worlds, 1)).reshape(-1)
        multi_init_dq_j = np.tile(initial_dq_j, (num_worlds, 1)).reshape(-1)
        multi_final_q_i = np.tile(final_q_i, (num_worlds, 1))
        multi_final_u_i = np.tile(final_u_i, (num_worlds, 1))
        multi_final_q_j = np.tile(final_q_j, (num_worlds, 1)).reshape(-1)
        multi_final_dq_j = np.tile(final_dq_j, (num_worlds, 1)).reshape(-1)
        msg.info(f"[samples]: [multi] [init]: q_i (shape={multi_init_q_i.shape}):\n{multi_init_q_i}\n")
        msg.info(f"[samples]: [multi] [init]: u_i (shape={multi_init_u_i.shape}):\n{multi_init_u_i}\n")
        msg.info(f"[samples]: [multi] [init]: q_j (shape={multi_init_q_j.shape}):\n{multi_init_q_j}\n")
        msg.info(f"[samples]: [multi] [init]: dq_j (shape={multi_init_dq_j.shape}):\n{multi_init_dq_j}\n")
        msg.info(f"[samples]: [multi] [final]: q_i (shape={multi_final_q_i.shape}):\n{multi_final_q_i}\n")
        msg.info(f"[samples]: [multi] [final]: u_i (shape={multi_final_u_i.shape}):\n{multi_final_u_i}\n")
        msg.info(f"[samples]: [multi] [final]: q_j (shape={multi_final_q_j.shape}):\n{multi_final_q_j}\n")
        msg.info(f"[samples]: [multi] [final]: dq_j (shape={multi_final_dq_j.shape}):\n{multi_final_dq_j}\n")

        # Create a multi-instance system by replicating the single-instance builder
        multi_builder = make_homogeneous_builder(num_worlds=num_worlds, build_fn=build_cartpole, ground=False)
        for i, body in enumerate(multi_builder.bodies):
            msg.info(f"[multi]: [builder]: body {i}: bid: {body.bid}")
            msg.info(f"[multi]: [builder]: body {i}: q_i: {body.q_i_0}")
            msg.info(f"[multi]: [builder]: body {i}: u_i: {body.u_i_0}")

        # Create a model and states from the builder
        multi_model = multi_builder.finalize(device=self.default_device)
        multi_state_p = multi_model.state()
        multi_state_n = multi_model.state()
        multi_control = multi_model.control()

        # Create a contacts container for the multi-instance system
        _, multi_world_max_contacts = multi_builder.compute_required_contact_capacity(max_contacts_per_pair=16)
        multi_contacts = Contacts(capacity=multi_world_max_contacts, device=multi_model.device)

        # Create simulator and check if the initial state is consistent with the contents of the builder
        multi_solver = SolverKamino(model=multi_model, contacts=multi_contacts)
        self.assertEqual(multi_model.size.sum_of_num_bodies, single_model.size.sum_of_num_bodies * num_worlds)
        self.assertEqual(multi_model.size.sum_of_num_joints, single_model.size.sum_of_num_joints * num_worlds)
        for i, body in enumerate(multi_builder.bodies):
            np.testing.assert_allclose(multi_model.bodies.q_i_0.numpy()[i], body.q_i_0)
            np.testing.assert_allclose(multi_model.bodies.u_i_0.numpy()[i], body.u_i_0)
            np.testing.assert_allclose(multi_state_p.q_i.numpy()[i], body.q_i_0)
            np.testing.assert_allclose(multi_state_p.u_i.numpy()[i], body.u_i_0)
            np.testing.assert_allclose(multi_state_n.q_i.numpy()[i], body.q_i_0)
            np.testing.assert_allclose(multi_state_n.u_i.numpy()[i], body.u_i_0)

        # Optional verbose output - enabled globally via self.verbose
        msg.info(f"[multi]: [init]: sim.model.size:\n{multi_model.size}\n\n")
        msg.info(f"[multi]: [init]: sim.model.state_previous.q_i:\n{multi_state_p.q_i}\n\n")
        msg.info(f"[multi]: [init]: sim.model.state_previous.u_i:\n{multi_state_p.u_i}\n\n")
        msg.info(f"[multi]: [init]: sim.model.state_previous.q_j:\n{multi_state_p.q_j}\n\n")
        msg.info(f"[multi]: [init]: sim.model.state_previous.dq_j:\n{multi_state_p.dq_j}\n\n")
        msg.info(f"[multi]: [init]: sim.model.state.q_i:\n{multi_state_n.q_i}\n\n")
        msg.info(f"[multi]: [init]: sim.model.state.u_i:\n{multi_state_n.u_i}\n\n")
        msg.info(f"[multi]: [init]: sim.model.state.q_j:\n{multi_state_n.q_j}\n\n")
        msg.info(f"[multi]: [init]: sim.model.state.dq_j:\n{multi_state_n.dq_j}\n\n")
        msg.info(f"[multi]: [init]: sim.model.control.tau_j:\n{multi_control.tau_j}\n\n")

        # Check if the multi-instance simulator has initial states matching the tiled samples
        np.testing.assert_allclose(multi_state_p.q_i.numpy(), multi_init_q_i)
        np.testing.assert_allclose(multi_state_p.u_i.numpy(), multi_init_u_i)
        np.testing.assert_allclose(multi_state_n.q_i.numpy(), multi_init_q_i)
        np.testing.assert_allclose(multi_state_n.u_i.numpy(), multi_init_u_i)
        np.testing.assert_allclose(multi_state_p.q_j.numpy(), multi_init_q_j)
        np.testing.assert_allclose(multi_state_p.dq_j.numpy(), multi_init_dq_j)
        np.testing.assert_allclose(multi_state_n.q_j.numpy(), multi_init_q_j)
        np.testing.assert_allclose(multi_state_n.dq_j.numpy(), multi_init_dq_j)

        # Step the multi-instance simulator for the same number of steps
        msg.info(f"[multi]: Executing {num_steps} simulator steps")
        start_time = time.time()
        for step in range(num_steps):
            # Execute a single simulation step
            multi_solver.step(multi_state_p, multi_state_n, multi_control, contacts=multi_contacts, dt=0.001)
            wp.synchronize()
            if self.verbose or self.progress:
                print_progress_bar(step + 1, num_steps, start_time, prefix="Progress", suffix="")

        # Optional verbose output - enabled globally via self.verbose
        msg.info(f"[multi]: [final]: multi_state_n.q_i:\n{multi_state_n.q_i}\n\n")
        msg.info(f"[multi]: [final]: multi_state_n.u_i:\n{multi_state_n.u_i}\n\n")
        msg.info(f"[multi]: [final]: multi_state_n.q_j:\n{multi_state_n.q_j}\n\n")
        msg.info(f"[multi]: [final]: multi_state_n.dq_j:\n{multi_state_n.dq_j}\n\n")

        # Check that the next states match the collected samples
        np.testing.assert_allclose(multi_state_n.q_i.numpy(), multi_final_q_i)
        np.testing.assert_allclose(multi_state_n.u_i.numpy(), multi_final_u_i)
        np.testing.assert_allclose(multi_state_n.q_j.numpy(), multi_final_q_j)
        np.testing.assert_allclose(multi_state_n.dq_j.numpy(), multi_final_dq_j)


###
# Test execution
###

if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
