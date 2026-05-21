# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for the `kamino.benchmark.metrics` module.
"""

import unittest
from collections.abc import Callable

import numpy as np
import warp as wp

from newton import Contacts, Model, ModelBuilder, State
from newton._src.solvers.kamino._src.utils import logger as msg
from newton._src.solvers.kamino.benchmark import metrics
from newton._src.solvers.kamino.tests import setup_tests, test_context
from newton.tests.utils import basics

###
# Scaffolding
###


class TestSetup:
    def __init__(
        self,
        builder_fn: Callable,
        builder_kwargs: dict | None = None,
        model_kwargs: dict | None = None,
        max_contacts: int = 32,
        margin: float = 0.0,
        gap: float = 0.0,
        device: wp.DeviceLike = None,
    ):
        if builder_kwargs is None:
            builder_kwargs = {}
        if model_kwargs is None:
            model_kwargs = {}
        self.builder: ModelBuilder = builder_fn(**builder_kwargs)
        self.builder.request_contact_attributes("force", "velocity")
        self.builder.default_shape_cfg.margin = margin
        self.builder.default_shape_cfg.gap = gap
        self.model: Model = self.builder.finalize(**model_kwargs)
        self.model.rigid_contact_max = max_contacts
        self.state: State = self.model.state()
        self.contacts: Contacts = self.model.contacts()


###
# Tests
###


class TestBenchmarkMetrics(unittest.TestCase):
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

    def test_01_contact_velocity_and_constraint_residuals_on_sphere_on_plane(self):
        """TODO"""
        # Define constants
        mass: float = 1.0
        radius: float = 0.1
        z_offset: float = -0.001
        mu: float = 0.5
        epsilon: float = 0.0
        margin: float = 0.0
        gap: float = 0.0
        max_contacts: int = 1

        # Create a test setup with a sphere on a plane model and data containers
        setup = TestSetup(
            builder_fn=basics.build_sphere_on_plane,
            builder_kwargs={
                "radius": radius,
                "mass": mass,
                "z_offset": z_offset,
                "friction": mu,
                "restitution": epsilon,
                "ground": True,
            },
            max_contacts=max_contacts,
            margin=margin,
            gap=gap,
            device=self.default_device,
        )
        msg.notif("setup.model.shape_gap: %s", setup.model.shape_gap)
        msg.notif("setup.model.shape_margin: %s", setup.model.shape_margin)
        msg.notif("setup.model.shape_material_mu: %s", setup.model.shape_material_mu)

        # Set the state to a non-trivial body twist
        body_q_np = np.array([[0.0, 0.0, radius + z_offset, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        body_qd_np = np.array([[1.0, 0.0, -1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        setup.state.body_q.assign(body_q_np)
        setup.state.body_qd.assign(body_qd_np)
        msg.warning("setup.state.body_q:\n%s", setup.state.body_q)
        msg.warning("setup.state.body_qd:\n%s\n", setup.state.body_qd)

        # Run collision detection
        setup.model.collide(setup.state, setup.contacts)
        nc = setup.contacts.rigid_contact_count.numpy()[0]
        msg.warning("setup.contacts.rigid_contact_count: %s", nc)
        msg.warning("setup.contacts.rigid_contact_shape0: %s", setup.contacts.rigid_contact_shape0)
        msg.warning("setup.contacts.rigid_contact_shape1: %s", setup.contacts.rigid_contact_shape1)
        msg.warning("setup.contacts.rigid_contact_margin0: %s", setup.contacts.rigid_contact_margin0)
        msg.warning("setup.contacts.rigid_contact_margin1: %s", setup.contacts.rigid_contact_margin1)
        msg.warning("setup.contacts.rigid_contact_offset0: %s", setup.contacts.rigid_contact_offset0)
        msg.warning("setup.contacts.rigid_contact_offset1: %s", setup.contacts.rigid_contact_offset1)
        msg.warning("setup.contacts.rigid_contact_point0:\n%s", setup.contacts.rigid_contact_point0)
        msg.warning("setup.contacts.rigid_contact_point1:\n%s", setup.contacts.rigid_contact_point1)
        msg.warning("setup.contacts.rigid_contact_normal:\n%s", setup.contacts.rigid_contact_normal)
        msg.warning("setup.contacts.force:\n%s", setup.contacts.force)
        msg.warning("setup.contacts.velocity:\n%s\n", setup.contacts.velocity)

        # Check that the number of contacts is as expected
        self.assertEqual(nc, 1)

        # Compute contact velocities
        metrics.compute_contact_velocities(setup.model, setup.state, setup.contacts)
        msg.error("setup.contacts.velocity:\n%s\n", setup.contacts.velocity)

        # Create metrics container
        m = metrics.PhysicsMetrics(model=setup.model)
        self.assertIsNotNone(m.contacts)
        msg.warning("metrics.contacts.r_cts_penetration: %s", m.contacts.r_cts_penetration)
        msg.warning("metrics.contacts.r_cts_velocity: %s", m.contacts.r_cts_velocity)
        msg.warning("metrics.contacts.r_ncp_primal: %s", m.contacts.r_ncp_primal)
        msg.warning("metrics.contacts.r_ncp_dual: %s", m.contacts.r_ncp_dual)
        msg.warning("metrics.contacts.r_ncp_compl: %s", m.contacts.r_ncp_compl)
        msg.warning("metrics.contacts.r_vi_natmap: %s\n", m.contacts.r_vi_natmap)

        # Compute contact constraint metrics
        metrics.compute_contact_constraint_metrics(setup.model, setup.state, setup.contacts, m)
        msg.error("metrics.contacts.r_cts_penetration: %s", m.contacts.r_cts_penetration)
        msg.error("metrics.contacts.r_cts_velocity: %s", m.contacts.r_cts_velocity)
        msg.error("metrics.contacts.r_ncp_primal: %s", m.contacts.r_ncp_primal)
        msg.error("metrics.contacts.r_ncp_dual: %s", m.contacts.r_ncp_dual)
        msg.error("metrics.contacts.r_ncp_compl: %s", m.contacts.r_ncp_compl)
        msg.error("metrics.contacts.r_vi_natmap: %s\n", m.contacts.r_vi_natmap)


###
# Test execution
###

if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
