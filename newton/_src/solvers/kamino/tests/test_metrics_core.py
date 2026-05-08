# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :class:`SolutionMetricsNewton` class."""

import unittest

import numpy as np
import warp as wp

import newton
from newton import Contacts, Control, Model, ModelBuilder, State
from newton._src.solvers.kamino._src.core.data import DataKamino
from newton._src.solvers.kamino._src.metrics import SolutionMetricsNewton
from newton._src.solvers.kamino._src.utils import logger as msg
from newton._src.solvers.kamino.tests import setup_tests, test_context
from newton._src.solvers.kamino.tests.test_solvers_padmm import TestSetup
from newton.tests.utils import basics

###
# Helpers
###


class TestSetup:
    def __init__(
        self,
        builder_fn,
        builder_kwargs: dict | None = None,
        dt: float = 0.001,
        max_world_contacts: int = 32,
        gravity: float | None = None,
        device: wp.DeviceLike = None,
        request_state_attributes: tuple[str, ...] = (),
    ):
        # Cache the time-step size
        self.dt = dt

        # Cache the max contacts allocated for the test problem
        self.max_world_contacts = max_world_contacts

        # Construct the model description using model builders for different systems
        if builder_kwargs is None:
            builder_kwargs = {}
        self.builder: ModelBuilder = builder_fn(**builder_kwargs)
        self.builder.request_contact_attributes("force")

        # Optionally request extended state attributes
        if request_state_attributes:
            self.builder.request_state_attributes(*request_state_attributes)

        # Set the maximum number of rigid contacts per world
        self.builder.num_rigid_contacts_per_world = max_world_contacts

        # Set ad-hoc configurations
        if gravity and isinstance(gravity, float):
            self.builder.gravity = gravity

        # Create the model from the builder
        self.model: Model = self.builder.finalize(skip_validation_joints=True)

        # Create additional containers from the model
        self.state: State = self.model.state()
        self.state_p: State = self.model.state()
        self.control: Control = self.model.control()
        self.contacts: Contacts = self.model.contacts()

        # Create a Kamino solver from the model
        self.solver = newton.solvers.SolverKamino(model=self.model)


def assert_models_equal_but_not_same_malloc(testcase: unittest.TestCase, model_1: Model, model_2: Model):
    """
    Assert that two models are equal but do not share the same underlying memory allocation.
    """

    # Check that the models are not None
    testcase.assertIsNotNone(model_1)
    testcase.assertIsNotNone(model_2)

    # Check that the models are of the same type
    testcase.assertIsInstance(model_1, Model)
    testcase.assertIsInstance(model_2, Model)

    # Check that the models are of the same device
    testcase.assertEqual(model_1.device, model_2.device)

    # List of model attributes to check for equality
    model_attributes = [
        # Rigid bodies
        "body_q",
        "body_qd",
        "body_mass",
        "body_inertia",
        "body_inv_mass",
        "body_inv_inertia",
        "body_flags",
        "body_label",
        "body_world",
        "body_world_start",
        # TODO: Shapes
        # TODO: Joints
        # TODO: Metadata
    ]

    # Check that the models are equal and that the underlying memory allocation is not the same
    for attribute in model_attributes:
        if not hasattr(model_1, attribute) or not hasattr(model_2, attribute):
            testcase.fail(f"Model attribute '{attribute}' is not found in one of the models.")

        # Retrieve the attributes
        attr_1 = getattr(model_1, attribute)
        attr_2 = getattr(model_2, attribute)

        # Check the attributes for equality based on the attribute type
        if isinstance(attr_1, wp.array) and isinstance(attr_2, wp.array):
            np.testing.assert_equal(attr_1.numpy(), attr_2.numpy())
            testcase.assertNotEqual(attr_1.ptr, attr_2.ptr)
        elif isinstance(attr_1, np.ndarray) and isinstance(attr_2, np.ndarray):
            np.testing.assert_equal(attr_1, attr_2)
            testcase.assertNotEqual(attr_1.ptr, attr_2.ptr)
        elif isinstance(attr_1, list) and isinstance(attr_2, list):
            testcase.assertEqual(attr_1, attr_2)
        elif isinstance(attr_1, dict) and isinstance(attr_2, dict):
            testcase.assertEqual(attr_1, attr_2)
        elif isinstance(attr_1, tuple) and isinstance(attr_2, tuple):
            testcase.assertEqual(attr_1, attr_2)
        elif isinstance(attr_1, set) and isinstance(attr_2, set):
            testcase.assertEqual(attr_1, attr_2)
        else:
            testcase.fail(f"Model attribute '{attribute}' is not one of the supported model attribute types.")


def assert_kamino_data_allclose(testcase: unittest.TestCase, data_1: DataKamino, data_2: DataKamino):
    """
    Assert that two Kamino data containers are equal.
    """
    # Check that the data containers are not None
    testcase.assertIsNotNone(data_1)
    testcase.assertIsNotNone(data_2)

    # Check that the data containers are of the same type
    testcase.assertIsInstance(data_1, DataKamino)
    testcase.assertIsInstance(data_2, DataKamino)

    # Check that the data containers are of the same device
    testcase.assertEqual(data_1.device, data_2.device)

    # List of data attributes to check for equality
    data_attributes = [
        # Rigid bodies
        ["bodies", "q_i"],
        ["bodies", "u_i"],
        ["bodies", "I_i"],
        ["bodies", "inv_I_i"],
        # ["bodies", "w_i"],
        # ["bodies", "w_a_i"],
        # ["bodies", "w_j_i"],
        # ["bodies", "w_l_i"],
        # ["bodies", "w_c_i"],
        # ["bodies", "w_e_i"],
        # TODO: Shapes
        # TODO: Joints
        # TODO: Metadata
    ]

    # Check that the models are equal and that the underlying memory allocation is not the same
    for attribute in data_attributes:
        container_name, attribute_name = attribute

        # Check that the container is found in both data containers
        if not hasattr(data_1, container_name) or not hasattr(data_2, container_name):
            testcase.fail(f"DataKamino container '{container_name}' is not found in one of the data containers.")
        container_1 = getattr(data_1, container_name)
        container_2 = getattr(data_2, container_name)
        if not hasattr(container_1, attribute_name) or not hasattr(container_2, attribute_name):
            testcase.fail(f"DataKamino attribute '{attribute_name}' is not found in one of the data containers.")
        attr_1 = getattr(container_1, attribute_name)
        attr_2 = getattr(container_2, attribute_name)

        # Check that the attribute is equal and that the underlying memory allocation is not the same
        np.testing.assert_equal(
            actual=attr_1.numpy(),
            desired=attr_2.numpy(),
            err_msg=f"DataKamino attributes '{container_name}.{attribute_name}' are not equal.",
        )
        testcase.assertNotEqual(attr_1.ptr, attr_2.ptr)


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
        """Test creating a SolutionMetrics instance with default initialization."""
        metrics = SolutionMetricsNewton()
        self.assertIsNotNone(metrics)
        self.assertIsNone(metrics._model)
        self.assertIsNone(metrics._data)
        self.assertIsNone(metrics._limits)
        self.assertIsNone(metrics._contacts)
        self.assertIsNone(metrics._problem)
        self.assertIsNone(metrics._jacobians)
        self.assertIsNone(metrics._state)
        self.assertIsNone(metrics._state_p)
        self.assertIsNone(metrics._control)
        self.assertIsNone(metrics._v_plus)
        self.assertIsNone(metrics._lambdas)
        self.assertIsNone(metrics._sigma)
        self.assertIsNone(metrics._metrics)

    def test_01_finalize_default(self):
        """Test creating a SolutionMetrics instance with default initialization and then finalizing all memory allocations."""
        # Create the test setup containing a builder, model, data containers and a solver
        setup = TestSetup(builder_fn=basics.build_box_on_plane, max_world_contacts=8, device=self.default_device)

        # Create a SolutionMetricsNewton instance with the test model and time-step
        metrics = SolutionMetricsNewton(dt=0.001, model=setup.builder.finalize(skip_validation_joints=True))

        # Check that the model is identical to the test setup model
        # but does not share the same underlying memory allocation
        assert_models_equal_but_not_same_malloc(self, metrics._model._model, setup.model)

    def test_02_evaluate_on_box_on_plane(self):
        """TODO"""
        # Create the test setup containing a builder, model, data containers and a solver
        setup = TestSetup(
            builder_fn=basics.build_box_on_plane,
            builder_kwargs={"z_offset": -1e-5},
            max_world_contacts=8,
            device=self.default_device,
        )

        # Create a SolutionMetricsNewton instance with the test model and time-step
        metrics = SolutionMetricsNewton(dt=setup.dt, model=setup.builder.finalize(skip_validation_joints=True))

        # Execute a single time-step of the test problem
        setup.model.collide(setup.state_p, setup.contacts)
        setup.solver.step(
            state_in=setup.state_p,
            state_out=setup.state,
            control=setup.control,
            contacts=setup.contacts,
            dt=setup.dt,
        )

        # Ensure assumptions are true:
        # - that 4x contacts are active
        nc = int(setup.contacts.rigid_contact_count.numpy()[0])
        self.assertEqual(nc, 4)

        # Evaluate the metrics on the test problem
        metrics.evaluate(
            state=setup.state,
            state_p=setup.state_p,
            control=setup.control,
            contacts=setup.contacts,
        )

        # Check if the metrics._data contains the same data as the solver._solver_kamino._data
        assert_kamino_data_allclose(self, metrics._data, setup.solver._solver_kamino._data)

    def test_03_evaluate_on_boxes_hinged(self):
        """
        TODO
        """
        setup = TestSetup(
            builder_fn=basics.build_boxes_hinged,
            builder_kwargs={"z_offset": -1e-5},
            max_world_contacts=32,
            device=self.default_device,
        )

        # Create a SolutionMetricsNewton instance with the test model and time-step
        metrics = SolutionMetricsNewton(dt=setup.dt, model=setup.builder.finalize(skip_validation_joints=True))

        # Execute a single time-step of the test problem
        setup.model.collide(setup.state_p, setup.contacts)
        setup.solver.step(
            state_in=setup.state_p,
            state_out=setup.state,
            control=setup.control,
            contacts=setup.contacts,
            dt=setup.dt,
        )

        # Ensure assumptions are true:
        # - that 8x contacts are active
        nc = int(setup.contacts.rigid_contact_count.numpy()[0])
        self.assertEqual(nc, 8)

        # Evaluate the metrics on the test problem
        metrics.evaluate(
            state=setup.state,
            state_p=setup.state_p,
            control=setup.control,
            contacts=setup.contacts,
        )

        # Check if the metrics._data contains the same data as the solver._solver_kamino._data
        assert_kamino_data_allclose(self, metrics._data, setup.solver._solver_kamino._data)

    def test_04_evaluate_on_boxes_nunchaku_vertical(self):
        """
        TODO
        """
        self.skipTest("Not implemented")

    ###
    #
    ###

    def test_05_convert_body_parent_wrenches_to_joint_reactions(self):
        """
        Verifies that ``SolutionMetricsNewton._convert_body_parent_wrenches_to_joint_reactions``
        recovers the joint kinematic-constraint Lagrange multipliers produced by the Kamino
        solver, by inverting the joint-wrench accumulation that produced ``body_parent_f``.

        Uses the ``boxes_hinged`` system whose hinge contributes 5 kinematic constraints,
        and compares the recovered ``StateKamino.lambda_j`` to the solver's ground truth.
        """
        # Create the test setup and request the `body_parent_f` extended state attribute
        setup = TestSetup(
            builder_fn=basics.build_boxes_hinged,
            builder_kwargs={"z_offset": -1e-5},
            max_world_contacts=32,
            device=self.default_device,
            request_state_attributes=("body_parent_f",),
        )

        # Re-allocate state buffers because requesting attributes after `model()` was created
        setup.state = setup.model.state()
        setup.state_p = setup.model.state()

        # Create a SolutionMetricsNewton instance with the test model and time-step
        metrics = SolutionMetricsNewton(dt=setup.dt, model=setup.builder.finalize(skip_validation_joints=True))

        # Execute a single time-step of the test problem so that the solver populates
        # both the joint constraint reactions and the body parent wrenches.
        setup.model.collide(setup.state_p, setup.contacts)
        setup.solver.step(
            state_in=setup.state_p,
            state_out=setup.state,
            control=setup.control,
            contacts=setup.contacts,
            dt=setup.dt,
        )

        # Sanity checks on the simulation outputs needed by the metrics path
        self.assertIsNotNone(setup.state.body_parent_f)
        body_parent_f_np = setup.state.body_parent_f.numpy()
        self.assertGreater(np.linalg.norm(body_parent_f_np), 0.0)

        # Run the metrics evaluation, which internally invokes the method under test
        metrics.evaluate(
            state=setup.state,
            state_p=setup.state_p,
            control=setup.control,
            contacts=setup.contacts,
        )

        # Compare the recovered joint constraint Lagrange multipliers (kinematic constraints)
        # to the ground truth produced by the solver. Both are stored in force units.
        recovered_lambda_j = metrics._state.lambda_j.numpy()
        expected_lambda_j = setup.solver._solver_kamino._data.joints.lambda_j.numpy()
        msg.warning("Recovered lambda_j: %s", recovered_lambda_j)
        msg.warning("Expected lambda_j: %s", expected_lambda_j)
        np.testing.assert_allclose(
            recovered_lambda_j,
            expected_lambda_j,
            rtol=1e-5,
            atol=1e-5,
            err_msg="Recovered lambda_j does not match the solver's ground-truth lambda_j.",
        )

        # The hinge joint has 5 kinematic constraints; ensure the test was meaningful by
        # checking that at least one of them carries a non-trivial reaction.
        self.assertGreater(np.linalg.norm(expected_lambda_j), 0.0)

        # `boxes_hinged` defines no joint limits, so the limits container should be empty
        # and ``LimitsKamino.reaction`` should remain at its reset (zero) state.
        self.assertEqual(metrics._limits.model_max_limits_host, 0)

    def test_06_extract_constraint_reactions(self):
        """
        Verifies that ``SolutionMetricsNewton._extract_constraint_reactions`` packs the
        per-container constraint reactions (joint, limit, contact) back into the global
        ``lambdas`` array (in Lagrange impulse units), as the inverse of
        ``unpack_constraint_solutions``.

        Uses the ``boxes_hinged`` system, which produces both joint kinematic
        constraint reactions (5 per hinge) and contact reactions (8 contacts -> 24 cts).
        """
        # Create the test setup and request the `body_parent_f` extended state attribute
        # so that the upstream ``_convert_body_parent_wrenches_to_joint_reactions``
        # populates ``self._state.lambda_j`` before we extract.
        setup = TestSetup(
            builder_fn=basics.build_boxes_hinged,
            builder_kwargs={"z_offset": -1e-5},
            max_world_contacts=32,
            device=self.default_device,
            request_state_attributes=("body_parent_f",),
        )

        # Re-allocate state buffers because requesting attributes after `model()` was created
        setup.state = setup.model.state()
        setup.state_p = setup.model.state()

        # Create a SolutionMetricsNewton instance with the test model and time-step
        metrics = SolutionMetricsNewton(dt=setup.dt, model=setup.builder.finalize(skip_validation_joints=True))

        # Execute a single time-step of the test problem so that the solver populates
        # both joint constraint reactions and per-contact reactions.
        setup.model.collide(setup.state_p, setup.contacts)
        setup.solver.step(
            state_in=setup.state_p,
            state_out=setup.state,
            control=setup.control,
            contacts=setup.contacts,
            dt=setup.dt,
        )

        # Sanity check: confirm the expected number of contacts is active
        nc = int(setup.contacts.rigid_contact_count.numpy()[0])
        self.assertEqual(nc, 8)

        # The solver does not write per-contact forces back to ``setup.contacts.force``
        # automatically; explicitly request the conversion so that the downstream
        # ``convert_contacts_newton_to_kamino`` inside ``metrics.evaluate`` populates
        # ``self._contacts.reaction`` from non-zero values.
        setup.solver.update_contacts(setup.contacts, setup.state)

        # Run the metrics evaluation, which internally invokes the method under test
        metrics.evaluate(
            state=setup.state,
            state_p=setup.state_p,
            control=setup.control,
            contacts=setup.contacts,
        )

        # Build the active-index mask from the solver's data.info / model.info.
        # Both ``solution.lambdas`` (solver) and ``metrics._lambdas`` are sized
        # ``sum_of_max_total_cts``; the solver does not zero unused contact/limit slots
        # between iterations, so a full-array comparison would be brittle.
        solver_impl = setup.solver._solver_kamino
        total_cts_offset = solver_impl._model.info.total_cts_offset.numpy()
        limit_cts_group_offset = solver_impl._data.info.limit_cts_group_offset.numpy()
        contact_cts_group_offset = solver_impl._data.info.contact_cts_group_offset.numpy()
        num_limit_cts = solver_impl._data.info.num_limit_cts.numpy()
        num_contact_cts = solver_impl._data.info.num_contact_cts.numpy()

        active = []
        for w in range(solver_impl._model.size.num_worlds):
            block_start = int(total_cts_offset[w])
            njc = int(limit_cts_group_offset[w])  # joint cts occupy [0, njc) within the world block
            ccgo = int(contact_cts_group_offset[w])
            nlc = int(num_limit_cts[w])
            ncc = int(num_contact_cts[w])
            # Joint-constraint indices (always all of them, fixed in size per world)
            active.extend(range(block_start, block_start + njc))
            # Active limit-constraint indices (subset of the limit group)
            active.extend(range(block_start + njc, block_start + njc + nlc))
            # Active contact-constraint indices (subset of the contact group)
            active.extend(range(block_start + ccgo, block_start + ccgo + ncc))
        active_idx = np.array(active, dtype=int)

        # Compare recovered (packed) lambdas to the solver's ground-truth lambdas
        expected = solver_impl._solver_fd.data.solution.lambdas.numpy()
        recovered = metrics._lambdas.numpy()
        msg.warning("Recovered lambdas (active): %s", recovered[active_idx])
        msg.warning("Expected lambdas (active): %s", expected[active_idx])
        np.testing.assert_allclose(
            recovered[active_idx],
            expected[active_idx],
            rtol=1e-5,
            atol=1e-5,
            err_msg="Recovered lambdas do not match the solver's ground-truth lambdas at active indices.",
        )

        # Ensure the test was meaningful: at least one constraint carries a non-trivial reaction.
        self.assertGreater(np.linalg.norm(expected[active_idx]), 0.0)

        # `boxes_hinged` defines no joint limits, so the limits container should be empty.
        self.assertEqual(metrics._limits.model_max_limits_host, 0)


###
# Test execution
###

if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
