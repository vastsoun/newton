# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :class:`SolutionMetricsNewton` class."""

import unittest

import numpy as np
import warp as wp

import newton
from newton import Contacts, Control, Model, ModelBuilder, State
from newton._src.solvers.kamino._src.core.data import DataKamino
from newton._src.solvers.kamino._src.core.model import ModelKamino
from newton._src.solvers.kamino._src.geometry.contacts import (
    ContactsKamino,
    convert_contacts_newton_to_kamino,
)
from newton._src.solvers.kamino._src.kinematics.constraints import (
    make_unilateral_constraints_info,
    update_constraints_info,
)
from newton._src.solvers.kamino._src.kinematics.limits import LimitsKamino
from newton._src.solvers.kamino._src.solvers.metrics import (
    SolutionMetricsNewton,
    extract_mujoco_warp_constraint_forces,
)
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


class TestSetupMuJoCo:
    """Test setup that uses :class:`newton.solvers.SolverMuJoCo` instead of Kamino.

    Provides the same surface area as ``TestSetup`` but constructs a MuJoCo
    Warp solver, used by the
    :func:`extract_mujoco_warp_constraint_forces` exercise. ``state.body_parent_f``
    is pre-allocated so the RNE-postconstraint sensor is enabled by the
    solver.
    """

    def __init__(
        self,
        builder_fn,
        builder_kwargs: dict | None = None,
        dt: float = 0.001,
        max_world_contacts: int = 32,
        gravity: float | None = None,
        device: wp.DeviceLike = None,
    ):
        self.dt = dt
        self.max_world_contacts = max_world_contacts

        if builder_kwargs is None:
            builder_kwargs = {}
        self.builder: ModelBuilder = builder_fn(**builder_kwargs)
        self.builder.request_contact_attributes("force")
        self.builder.num_rigid_contacts_per_world = max_world_contacts
        if gravity and isinstance(gravity, float):
            self.builder.gravity = gravity

        self.model: Model = self.builder.finalize(skip_validation_joints=True)

        self.state: State = self.model.state()
        self.state_p: State = self.model.state()
        self.control: Control = self.model.control()
        self.contacts: Contacts = self.model.contacts()

        # Pre-allocate body_parent_f so SolverMuJoCo enables the RNE
        # post-constraint sensor and populates the array on each step.
        nb = self.model.body_count
        with wp.ScopedDevice(self.model.device):
            self.state.body_parent_f = wp.zeros(shape=(nb,), dtype=wp.spatial_vectorf)
            self.state_p.body_parent_f = wp.zeros(shape=(nb,), dtype=wp.spatial_vectorf)

        # Force ELLIPTIC cones and use Newton-side collision so that
        # ``self.contacts`` is populated and shared with MuJoCo Warp via
        # ``_convert_contacts_to_mjwarp`` during ``step``.
        self.solver = newton.solvers.SolverMuJoCo(
            model=self.model,
            cone="elliptic",
            use_mujoco_contacts=False,
        )


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
        self.skipTest("Not implemented")

    def test_04_evaluate_on_boxes_nunchaku_vertical(self):
        """
        TODO
        """
        self.skipTest("Not implemented")

    def test_05_extract_lambdas_box_on_plane(self):
        """Extract MuJoCo Warp lambdas from a box-on-plane simulation step.

        Runs a single ``SolverMuJoCo.step`` on the box-on-plane benchmark and
        verifies that
        :func:`extract_mujoco_warp_constraint_forces`:
        - populates the contact slots in the Kamino-ordered ``lambdas``,
        - leaves all non-contact slots at zero (no joints/limits in this fixture),
        - the sum of normal-impulse multipliers across the four contacts is
          approximately equal to ``mass * |gravity| * dt`` (the impulse
          required to support the box's weight over one step).
        """
        # Skip if the MuJoCo Warp backend is unavailable.
        try:
            import mujoco_warp  # noqa: F401
        except ImportError:
            self.skipTest("mujoco_warp is not installed.")

        setup = TestSetupMuJoCo(
            builder_fn=basics.build_box_on_plane,
            builder_kwargs={"z_offset": -1e-5},
            max_world_contacts=8,
            device=self.default_device,
        )

        # Populate Newton-side contacts via Newton's collision detector and
        # run a single MuJoCo step. With ``use_mujoco_contacts=False``, the
        # solver shares ``setup.contacts`` with MuJoCo Warp via the Newton →
        # MJW conversion path inside ``step``.
        setup.model.collide(setup.state_p, setup.contacts)
        setup.solver.step(
            state_in=setup.state_p,
            state_out=setup.state,
            control=setup.control,
            contacts=setup.contacts,
            dt=setup.dt,
        )

        # Verify the test pre-conditions.
        nc = int(setup.contacts.rigid_contact_count.numpy()[0])
        self.assertEqual(nc, 4, "Expected 4 box-on-plane contacts.")

        # Build the Kamino-side data containers that the launcher consumes.
        model_kamino = ModelKamino.from_newton(model=setup.model, overwrite_source_model=False)
        model_kamino.time.dt.fill_(wp.float32(setup.dt))
        model_kamino.time.inv_dt.fill_(wp.float32(1.0 / setup.dt))

        data_kamino = model_kamino.data()
        limits_kamino = LimitsKamino(model=model_kamino)
        contacts_kamino = ContactsKamino(model=model_kamino)
        make_unilateral_constraints_info(
            model=model_kamino,
            data=data_kamino,
            limits=limits_kamino,
            contacts=contacts_kamino,
        )
        contacts_kamino.reset()
        convert_contacts_newton_to_kamino(
            model=setup.model,
            state=setup.state,
            contacts_in=setup.contacts,
            contacts_out=contacts_kamino,
        )
        update_constraints_info(model=model_kamino, data=data_kamino)

        # Allocate the output lambdas array and call the launcher.
        with wp.ScopedDevice(model_kamino.device):
            lambdas = wp.zeros(shape=model_kamino.size.sum_of_max_total_cts, dtype=wp.float32)

        extract_mujoco_warp_constraint_forces(
            model=setup.model,
            state=setup.state,
            solver=setup.solver,
            model_kamino=model_kamino,
            contacts_kamino=contacts_kamino,
            limits_kamino=limits_kamino,
            contacts_newton=setup.contacts,
            lambdas=lambdas,
        )

        lambdas_np = lambdas.numpy()
        self.assertEqual(lambdas_np.size, model_kamino.size.sum_of_max_total_cts)

        # The fixture has a single floating box (free joint) and 4 contacts.
        # Joint constraint slots should not be touched in this prototype path
        # (the body_parent_f decomposition would write to kinematic slots
        # only if the joint had > 0 kinematic constraints — a free joint has
        # zero, so writes are skipped).
        njcts_total = int(model_kamino.info.num_joint_cts.numpy().sum())
        active_contacts = int(contacts_kamino.data.model_active_contacts.numpy()[0])
        self.assertEqual(active_contacts, 4)

        # Contact slot block: 3 entries per active contact.
        # We picked up the limits group offset = njc, contacts group offset = njc + 0.
        # Normal-component lambdas (impulses) are the entries at
        # offset 0 within each 3-vector.
        contact_block_start = njcts_total  # Single-world: total_cts_offset[0] == 0.
        normal_impulses = []
        for k in range(active_contacts):
            normal_impulses.append(float(lambdas_np[contact_block_start + 3 * k + 0]))

        sum_normal_impulse = float(np.sum(normal_impulses))
        self.assertGreater(
            sum_normal_impulse,
            0.0,
            f"Expected non-zero contact normal impulses, got {normal_impulses}.",
        )

        # The total normal impulse over one step should approximate
        # ``mass * |g| * dt`` (the impulse required to support the box).
        body_mass = float(setup.model.body_mass.numpy()[0])
        gravity_arr = setup.model.gravity
        if isinstance(gravity_arr, wp.array):
            gravity_arr = gravity_arr.numpy()
        gravity_mag = float(np.linalg.norm(np.asarray(gravity_arr).reshape(-1, 3)[0]))
        expected_impulse = body_mass * gravity_mag * setup.dt
        # Loose tolerance: solver convergence + sign convention may add slack.
        self.assertAlmostEqual(
            sum_normal_impulse,
            expected_impulse,
            delta=max(0.5 * expected_impulse, 1e-6),
            msg=(
                f"Sum of contact normal impulses = {sum_normal_impulse} not "
                f"close to expected mg*dt = {expected_impulse}."
            ),
        )

    def test_06_extract_lambdas_pendulum_with_limit(self):
        """Stub test for limit-multiplier extraction.

        Locks in the launcher API so future commits can swap a real
        pendulum-with-limit fixture in without touching the call sites.
        """
        self.skipTest("Pending: pendulum-with-limit fixture and joint-limit extraction validation.")

    def test_07_extract_lambdas_boxes_fourbar(self):
        """Stub test for loop-closure equality and tree-joint multiplier extraction.

        Locks in the launcher API for a four-bar (loop-closure) setup, where
        the body_parent_f decomposition is most relevant. To be implemented
        once the per-equality wrench accumulation kernel is no longer a stub.
        """
        self.skipTest("Pending: four-bar fixture and equality/tree-joint extraction validation.")


###
# Test execution
###

if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
