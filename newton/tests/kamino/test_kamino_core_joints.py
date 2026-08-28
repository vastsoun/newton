# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the `kamino.core.joints` module"""

import unittest

import numpy as np
import warp as wp

from newton._src.sim import JointType
from newton._src.solvers.kamino._src.core.joints import JointActuationType, JointDescriptor, JointDoFType
from newton._src.solvers.kamino._src.utils import logger as msg
from newton.tests.kamino import setup_tests, test_context

wp.set_module_options({"enable_backward": False})

###
# Kernels
###


###
# Tests
###


class TestCoreJoints(unittest.TestCase):
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

    def test_joint_dof_type_enum(self):
        """Verify metadata for an explicit revolute joint type."""
        doftype = JointDoFType.REVOLUTE

        # Optional verbose output
        msg.info(f"doftype: {doftype}")
        msg.info(f"doftype.value: {doftype.value}")
        msg.info(f"doftype.name: {doftype.name}")
        msg.info(f"doftype.num_cts: {doftype.num_cts}")
        msg.info(f"doftype.num_dofs: {doftype.num_dofs}")
        msg.info(f"doftype.cts_axes: {doftype.cts_axes}")

        # Check the enum values
        self.assertEqual(doftype.value, JointDoFType.REVOLUTE)
        self.assertEqual(doftype.name, "REVOLUTE")
        self.assertEqual(doftype.num_cts, 5)
        self.assertEqual(doftype.num_dofs, 1)
        self.assertEqual(doftype.cts_axes, (0, 1, 2, 4, 5))
    def test_newton_d6_classification(self):
        """Classify only canonical one-axis D6 layouts as explicit joint types."""
        for n_linear in range(4):
            for n_angular in range(4):
                with self.subTest(dof_dim=(n_linear, n_angular)):
                    dof_dim = (n_linear, n_angular)
                    count = n_linear + n_angular
                    limits = np.zeros(count, dtype=np.float32)
                    expected = JointDoFType.D6
                    if dof_dim == (0, 0):
                        expected = JointDoFType.FIXED
                    elif dof_dim == (1, 0):
                        expected = JointDoFType.PRISMATIC
                    elif dof_dim == (0, 1):
                        expected = JointDoFType.REVOLUTE
                    actual = JointDoFType.from_newton(
                        JointType.D6,
                        count,
                        count,
                        dof_dim,
                        limits,
                        limits,
                    )
                    self.assertEqual(actual, expected)

    def test_explicit_ball_and_free_classification(self):
        """Keep explicit Newton ball and free joints distinct from generic D6."""
        limits = np.zeros(6, dtype=np.float32)
        self.assertEqual(
            JointDoFType.from_newton(JointType.BALL, 4, 3, (0, 3), limits[:3], limits[:3]),
            JointDoFType.SPHERICAL,
        )
        self.assertEqual(
            JointDoFType.from_newton(JointType.FREE, 7, 6, (3, 3), limits, limits),
            JointDoFType.FREE,
        )

    def test_generic_d6_counts(self):
        """Compute generic D6 counts from per-joint dimensions."""
        for dof_dim in ((0, 2), (1, 1), (3, 0), (0, 3), (2, 3), (3, 3)):
            with self.subTest(dof_dim=dof_dim):
                num_dofs = sum(dof_dim)
                self.assertEqual(JointDoFType.num_coords_for(JointDoFType.D6, dof_dim), num_dofs)
                self.assertEqual(JointDoFType.num_dofs_for(JointDoFType.D6, dof_dim), num_dofs)
                self.assertEqual(JointDoFType.num_cts_for(JointDoFType.D6, dof_dim), 6 - num_dofs)

    def test_is_three_dof_rotation(self):
        """Identify spherical and D6 joints with three rotational DoFs."""
        self.assertTrue(JointDoFType.SPHERICAL.is_three_dof_rotation())
        for dof_dim in ((0, 3), (1, 3), (2, 3), (3, 3)):
            with self.subTest(dof_dim=dof_dim):
                self.assertTrue(JointDoFType.D6.is_three_dof_rotation(dof_dim))
        for dof_type in (JointDoFType.FREE, JointDoFType.REVOLUTE, JointDoFType.PRISMATIC, JointDoFType.FIXED):
            with self.subTest(dof_type=dof_type):
                self.assertFalse(dof_type.is_three_dof_rotation())
        for dof_dim in ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2)):
            with self.subTest(dof_dim=dof_dim):
                self.assertFalse(JointDoFType.D6.is_three_dof_rotation(dof_dim))
        self.assertFalse(JointDoFType.D6.is_three_dof_rotation())

    def test_generic_d6_descriptor_metadata(self):
        """Accept explicit generic D6 dimensions and separate orthonormal axis groups."""
        joint = JointDescriptor(
            name="d6",
            dof_type=JointDoFType.D6,
            dof_dim=(2, 2),
            dof_axes=[
                wp.vec3f(1.0, 0.0, 0.0),
                wp.vec3f(0.0, 1.0, 0.0),
                wp.vec3f(0.0, 0.0, 1.0),
                wp.vec3f(1.0, 0.0, 0.0),
            ],
            dof_act_types=[JointActuationType.PASSIVE] * 4,
        )
        self.assertEqual(joint.num_coords, 4)
        self.assertEqual(joint.num_dofs, 4)
        self.assertEqual(joint.num_kinematic_cts, 2)

    def test_generic_d6_descriptor_accepts_implicit_dynamics(self):
        """Accept runtime-sized armature, damping, and implicit PD on D6 joints."""
        joint = JointDescriptor(
            name="dynamic-d6",
            dof_type=JointDoFType.D6,
            dof_dim=(1, 1),
            dof_axes=[wp.vec3f(1.0, 0.0, 0.0), wp.vec3f(0.0, 0.0, 1.0)],
            dof_act_types=[JointActuationType.POSITION, JointActuationType.POSITION],
            a_j=np.array([0.2, 0.3]),
            b_j=np.array([0.4, 0.5]),
            k_p_j=np.array([10.0, 20.0]),
            k_d_j=np.array([1.0, 2.0]),
        )
        self.assertEqual(joint.num_dynamic_cts, 2)

    def test_generic_d6_descriptor_rejects_invalid_axes(self):
        """Reject missing, malformed, non-unit, and within-group nonorthogonal D6 axes."""
        valid = [wp.vec3f(1.0, 0.0, 0.0), wp.vec3f(0.0, 1.0, 0.0)]
        cases = (
            {"dof_dim": (1, 1), "dof_axes": None},
            {"dof_dim": (1, 1), "dof_axes": valid[:1]},
            {"dof_dim": (1, 1), "dof_axes": [wp.vec3f(2.0, 0.0, 0.0), valid[1]]},
            {"dof_dim": (2, 0), "dof_axes": [valid[0], valid[0]]},
            {"dof_dim": (4, 0), "dof_axes": [valid[0]] * 4},
        )
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                JointDescriptor(name="invalid_d6", dof_type=JointDoFType.D6, **kwargs)


###
# Test execution
###

if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
