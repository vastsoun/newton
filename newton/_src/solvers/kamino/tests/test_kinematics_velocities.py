# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :func:`compute_constraint_space_velocities`."""

import unittest

import numpy as np
import warp as wp

import newton
from newton import Contacts, Control, Model, ModelBuilder, State
from newton._src.solvers.kamino._src.core.bodies import update_body_inertias
from newton._src.solvers.kamino._src.core.data import DataKamino
from newton._src.solvers.kamino._src.core.model import ModelKamino
from newton._src.solvers.kamino._src.core.state import StateKamino
from newton._src.solvers.kamino._src.geometry.contacts import ContactsKamino, convert_contacts_newton_to_kamino
from newton._src.solvers.kamino._src.kinematics.constraints import (
    make_unilateral_constraints_info,
    update_constraints_info,
)
from newton._src.solvers.kamino._src.kinematics.jacobians import (
    DenseSystemJacobians,
    SparseSystemJacobians,
)
from newton._src.solvers.kamino._src.kinematics.joints import compute_joints_data
from newton._src.solvers.kamino._src.kinematics.limits import LimitsKamino
from newton._src.solvers.kamino._src.kinematics.velocities import compute_constraint_space_velocities
from newton._src.solvers.kamino._src.utils import logger as msg
from newton._src.solvers.kamino.tests import setup_tests, test_context
from newton._src.solvers.kamino.tests.utils.extract import extract_cts_jacobians
from newton.tests.utils import basics

###
# Helpers
###


class TestSetup:
    """Builds a Newton-side simulation harness alongside the equivalent Kamino containers.

    The Newton-side fields drive the reference :class:`SolverKamino` step that produces a
    realistic post-event state. The Kamino-side containers (``model_kamino``, ``data_kamino``,
    ``limits_kamino``, ``contacts_kamino``, ``jacobians_dense``, ``jacobians_sparse``) are used
    by tests to exercise lower-level Kamino primitives directly without going through
    :class:`SolutionMetricsNewton`.
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

        # Create additional Newton containers from the model
        self.state: State = self.model.state()
        self.state_p: State = self.model.state()
        self.control: Control = self.model.control()
        self.contacts: Contacts = self.model.contacts()

        # Create a Kamino solver from the model
        self.solver = newton.solvers.SolverKamino(model=self.model)

        # Build a parallel set of Kamino containers, mirroring the allocation order
        # used by SolutionMetricsNewton.finalize so we can exercise the same
        # constraint-space computations without going through the metrics wrapper.
        self.model_kamino: ModelKamino = ModelKamino.from_newton(model=self.model, overwrite_source_model=False)
        self.model_kamino.time.dt.fill_(wp.float32(dt))
        self.model_kamino.time.inv_dt.fill_(wp.float32(1.0 / dt))

        self.data_kamino: DataKamino = self.model_kamino.data()
        self.limits_kamino: LimitsKamino = LimitsKamino(model=self.model_kamino)
        self.contacts_kamino: ContactsKamino = ContactsKamino(model=self.model_kamino)
        self.limits_kamino.reset()
        self.contacts_kamino.reset()

        # Construct the unilateral constraints members in the model info
        make_unilateral_constraints_info(
            model=self.model_kamino,
            data=self.data_kamino,
            limits=self.limits_kamino,
            contacts=self.contacts_kamino,
        )

        # Allocate both Jacobian backends so tests can compare them against the same setup
        self.jacobians_dense: DenseSystemJacobians = DenseSystemJacobians(
            model=self.model_kamino, limits=self.limits_kamino, contacts=self.contacts_kamino
        )
        self.jacobians_sparse: SparseSystemJacobians = SparseSystemJacobians(
            model=self.model_kamino, limits=self.limits_kamino, contacts=self.contacts_kamino
        )

    def build_jacobians(self, state_p: State, contacts: Contacts) -> StateKamino:
        """Populate :attr:`data_kamino` from ``state_p`` and rebuild both Jacobian backends.

        Mirrors the data-population sequence from :meth:`SolutionMetricsNewton.evaluate`
        up to and including the Jacobian assembly step. Returns the :class:`StateKamino`
        view aliased over ``state_p`` for callers that need to read its arrays directly.
        """
        # Reset limits and contacts containers so prior contents do not bleed in
        self.limits_kamino.reset()
        self.contacts_kamino.reset()

        # Interface the Newton state and contacts to their Kamino equivalents
        state_p_kamino = StateKamino.from_newton(self.model_kamino.size, self.model, state_p)
        convert_contacts_newton_to_kamino(self.model, state_p, contacts, self.contacts_kamino)

        # Copy the relevant state arrays into the Kamino data container so that downstream
        # kinematics/jacobian builders see the post-step pre-event configuration.
        wp.copy(self.data_kamino.bodies.q_i, state_p_kamino.q_i)
        wp.copy(self.data_kamino.bodies.u_i, state_p_kamino.u_i)
        wp.copy(self.data_kamino.bodies.w_i, state_p_kamino.w_i)
        wp.copy(self.data_kamino.bodies.w_e_i, state_p_kamino.w_i_e)
        wp.copy(self.data_kamino.joints.q_j, state_p_kamino.q_j)
        wp.copy(self.data_kamino.joints.dq_j, state_p_kamino.dq_j)

        # Update the relevant data fields required for Jacobian assembly
        update_constraints_info(model=self.model_kamino, data=self.data_kamino)
        update_body_inertias(model=self.model_kamino.bodies, data=self.data_kamino.bodies)
        compute_joints_data(model=self.model_kamino, data=self.data_kamino, q_j_p=state_p_kamino.q_j)

        # Build both Jacobian backends in-place
        self.jacobians_dense.build(
            model=self.model_kamino,
            data=self.data_kamino,
            limits=self.limits_kamino,
            contacts=self.contacts_kamino,
            reset_to_zero=True,
        )
        self.jacobians_sparse.build(
            model=self.model_kamino,
            data=self.data_kamino,
            limits=self.limits_kamino,
            contacts=self.contacts_kamino,
            reset_to_zero=True,
        )

        return state_p_kamino


###
# Tests
###


class TestKinematicsVelocities(unittest.TestCase):
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

    def test_01_constraint_velocities(self):
        """Verify ``compute_constraint_space_velocities`` against a NumPy reference for both Jacobian flavors."""
        # Build a multi-body test setup with both joints and contacts so that
        # the constraint Jacobian exercises all constraint-block types
        setup = TestSetup(
            builder_fn=basics.build_boxes_hinged,
            builder_kwargs={"z_offset": -1e-5},
            max_world_contacts=32,
            device=self.default_device,
        )

        # Execute a single time-step so that the post-event state has a non-trivial body twist
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

        # Populate the Kamino data container from the pre-event state and assemble both
        # dense and sparse system Jacobians evaluated at q^- := state_p.body_q
        setup.build_jacobians(state_p=setup.state_p, contacts=setup.contacts)

        # Cross-check that the freshly-rebuilt Jacobians match the one the solver itself
        # assembled internally during ``step()``. Both are evaluated at the same q^- so
        # their per-world entries must agree up to floating-point tolerance.
        # NOTE: SolverKamino's default config uses :class:`DenseSystemJacobians`, but we
        # compare on a per-active-row basis so the comparison is valid against either
        # backend stored on the solver.
        solver_impl = setup.solver._solver_kamino
        solver_J_cts = extract_cts_jacobians(
            model=solver_impl._model,
            limits=solver_impl._limits,
            contacts=setup.solver._contacts_kamino,
            jacobians=solver_impl._jacobians,
            only_active_cts=True,
        )
        for backend, jacobians in (("dense", setup.jacobians_dense), ("sparse", setup.jacobians_sparse)):
            with self.subTest(jacobians_match_solver=backend):
                setup_J_cts = extract_cts_jacobians(
                    model=setup.model_kamino,
                    limits=setup.limits_kamino,
                    contacts=setup.contacts_kamino,
                    jacobians=jacobians,
                    only_active_cts=True,
                )
                for w in range(setup.model_kamino.size.num_worlds):
                    np.testing.assert_allclose(
                        actual=setup_J_cts[w],
                        desired=solver_J_cts[w],
                        atol=1e-5,
                        rtol=1e-5,
                        err_msg=f"J_cts mismatch for backend={backend} world={w}",
                    )

        # Build a StateKamino view over the post-event Newton state to read u^+ := body_qd
        state_kamino = StateKamino.from_newton(setup.model_kamino.size, setup.model, setup.state)

        # Compute the NumPy reference: v_plus_ref[w] = J_cts(q^-)[w] @ u^+[w], stored in
        # the same flat layout used by `compute_constraint_space_velocities`.
        # The dense extraction returns matrices with `max_total_cts` rows so the
        # inactive rows contribute zero, matching the kernel output layout exactly.
        J_cts_list = extract_cts_jacobians(
            model=setup.model_kamino,
            limits=setup.limits_kamino,
            contacts=setup.contacts_kamino,
            jacobians=setup.jacobians_dense,
            only_active_cts=False,
        )
        u_i_np = state_kamino.u_i.numpy().reshape(-1, 6)
        bodies_offset_np = setup.model_kamino.info.bodies_offset.numpy()
        max_total_cts_np = setup.model_kamino.info.max_total_cts.numpy()
        total_cts_offset_np = setup.model_kamino.info.total_cts_offset.numpy()

        v_plus_ref = np.zeros(setup.model_kamino.size.sum_of_max_total_cts, dtype=np.float32)
        for w in range(setup.model_kamino.size.num_worlds):
            bio = int(bodies_offset_np[w])
            nb = int(bodies_offset_np[w + 1]) - bio
            ncts = int(max_total_cts_np[w])
            if ncts == 0 or nb == 0:
                continue
            u_w = u_i_np[bio : bio + nb].reshape(6 * nb)
            vio = int(total_cts_offset_np[w])
            v_plus_ref[vio : vio + ncts] = J_cts_list[w] @ u_w

        # Verify that both dense and sparse Jacobian backends produce the expected v_plus
        for sparse, jacobians in ((False, setup.jacobians_dense), (True, setup.jacobians_sparse)):
            with self.subTest(sparse=sparse):
                v_plus = wp.zeros(
                    setup.model_kamino.size.sum_of_max_total_cts,
                    dtype=wp.float32,
                    device=setup.model_kamino.device,
                )
                compute_constraint_space_velocities(
                    model=setup.model_kamino,
                    jacobians=jacobians,
                    u=state_kamino.u_i,
                    v_start=setup.model_kamino.info.total_cts_offset,
                    v=v_plus,
                    reset_to_zero=True,
                )
                np.testing.assert_allclose(
                    actual=v_plus.numpy(),
                    desired=v_plus_ref,
                    atol=1e-5,
                    rtol=1e-5,
                    err_msg=f"v_plus mismatch for sparse={sparse}",
                )


###
# Test execution
###

if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
