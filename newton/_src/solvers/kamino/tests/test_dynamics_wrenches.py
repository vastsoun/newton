# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for `dynamics/wrenches.py`.
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton import Contacts, Control, Model, ModelBuilder, State
from newton._src.solvers.kamino._src.core.bodies import update_body_inertias
from newton._src.solvers.kamino._src.core.control import ControlKamino
from newton._src.solvers.kamino._src.core.data import DataKamino
from newton._src.solvers.kamino._src.core.model import ModelKamino
from newton._src.solvers.kamino._src.core.state import StateKamino
from newton._src.solvers.kamino._src.dynamics.wrenches import (
    compute_constraint_body_wrenches_dense,
    compute_constraint_body_wrenches_sparse,
    compute_joint_dof_body_wrenches_dense,
    compute_joint_dof_body_wrenches_sparse,
    convert_body_parent_wrenches_to_joint_reactions,
    convert_joint_parent_wrenches_to_joint_reactions,
)
from newton._src.solvers.kamino._src.geometry.contacts import ContactsKamino, convert_contacts_newton_to_kamino
from newton._src.solvers.kamino._src.kinematics.constraints import (
    make_unilateral_constraints_info,
    pack_constraint_solutions,
    unpack_constraint_solutions,
    update_constraints_info,
)
from newton._src.solvers.kamino._src.kinematics.jacobians import DenseSystemJacobians, SparseSystemJacobians
from newton._src.solvers.kamino._src.kinematics.joints import compute_joints_data
from newton._src.solvers.kamino._src.kinematics.limits import LimitsKamino
from newton._src.solvers.kamino._src.utils import logger as msg
from newton._src.solvers.kamino.tests import setup_tests, test_context
from newton._src.solvers.kamino.tests.utils.extract import (
    extract_active_constraint_vectors,
    extract_actuation_forces,
    extract_cts_jacobians,
    extract_dofs_jacobians,
)
from newton._src.solvers.kamino.tests.utils.make import (
    make_constraint_multiplier_arrays,
    make_test_problem_fourbar,
    make_test_problem_heterogeneous,
)
from newton.tests.utils import basics

###
# Constants
###

test_jacobian_rtol = 1e-7
test_jacobian_atol = 1e-7

# TODO: FIX THIS: sparse-dense differences are larger than expected,
# likely due to the sparse implementation not fully matching the dense
test_wrench_rtol = 1e-4  # TODO: Should be 1e-6
test_wrench_atol = 1e-4  # TODO: Should be 1e-6


###
# Helper functions for `TestDynamicsWrenches`
###


def compute_and_compare_dense_sparse_jacobian_wrenches(
    model: ModelKamino,
    data: DataKamino,
    limits: LimitsKamino,
    contacts: ContactsKamino,
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
    wp.synchronize()
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
    wp.synchronize()
    w_a_i_sparse_np = data.bodies.w_a_i.numpy().copy()
    w_j_i_sparse_np = data.bodies.w_j_i.numpy().copy()
    w_l_i_sparse_np = data.bodies.w_l_i.numpy().copy()
    w_c_i_sparse_np = data.bodies.w_c_i.numpy().copy()

    # TODO
    np.set_printoptions(precision=12, suppress=True, linewidth=20000, threshold=20000)

    # Extract the number of bodies and constraints for each world
    num_bodies_np = model.info.num_bodies.numpy().astype(int).tolist()
    num_joint_cts_np = model.info.num_joint_cts.numpy().astype(int).tolist()
    num_limit_cts_np = data.info.num_limit_cts.numpy().astype(int).tolist()
    num_contact_cts_np = data.info.num_contact_cts.numpy().astype(int).tolist()
    num_total_cts_np = data.info.num_total_cts.numpy().astype(int).tolist()
    msg.info("num_bodies_np: %s", num_bodies_np)
    msg.info("num_joint_cts_np: %s", num_joint_cts_np)
    msg.info("num_limit_cts_np: %s", num_limit_cts_np)
    msg.info("num_contact_cts_np: %s", num_contact_cts_np)
    msg.info("num_total_cts_np: %s\n", num_total_cts_np)

    # Extract the Jacobians and constraint multipliers as lists of numpy arrays (i.e. per world)
    J_cts_dense = extract_cts_jacobians(model, limits, contacts, jacobians_dense, only_active_cts=True)
    J_dofs_dense = extract_dofs_jacobians(model, jacobians_dense)
    J_cts_sparse = jacobians_sparse._J_cts.bsm.numpy()
    J_dofs_sparse = jacobians_sparse._J_dofs.bsm.numpy()
    lambdas_np = extract_active_constraint_vectors(model, data, lambdas)
    tau_j_np = extract_actuation_forces(model, data)
    for w in range(model.size.num_worlds):
        msg.info("[world='%d']: J_cts_dense:\n%s", w, J_cts_dense[w])
        msg.info("[world='%d']: J_cts_sparse:\n%s\n", w, J_cts_sparse[w])
        msg.info("[world='%d']: lambdas_np:\n%s\n\n", w, lambdas_np[w])
        msg.info("[world='%d']: J_dofs_dense:\n%s", w, J_dofs_dense[w])
        msg.info("[world='%d']: J_dofs_sparse:\n%s\n", w, J_dofs_sparse[w])
        msg.info("[world='%d']: tau_j_np:\n%s\n", w, tau_j_np[w])

    # Compute the wrenches manually using the extracted Jacobians and multipliers/forces for additional verification
    inv_dt_np = model.time.inv_dt.numpy().tolist()
    w_a_i_ref_np = [np.zeros((num_bodies_np[w], 6), dtype=np.float32) for w in range(model.size.num_worlds)]
    w_j_i_ref_np = [np.zeros((num_bodies_np[w], 6), dtype=np.float32) for w in range(model.size.num_worlds)]
    w_l_i_ref_np = [np.zeros((num_bodies_np[w], 6), dtype=np.float32) for w in range(model.size.num_worlds)]
    w_c_i_ref_np = [np.zeros((num_bodies_np[w], 6), dtype=np.float32) for w in range(model.size.num_worlds)]
    for w in range(model.size.num_worlds):
        joint_cts_start_w = 0
        joint_cts_end_w = num_joint_cts_np[w]
        limit_cts_start_w = joint_cts_end_w
        limit_cts_end_w = limit_cts_start_w + num_limit_cts_np[w]
        contact_cts_start_w = limit_cts_end_w
        contact_cts_end_w = contact_cts_start_w + num_contact_cts_np[w]
        J_cts_j = J_cts_dense[w][joint_cts_start_w:joint_cts_end_w, :]
        J_cts_l = J_cts_dense[w][limit_cts_start_w:limit_cts_end_w, :]
        J_cts_c = J_cts_dense[w][contact_cts_start_w:contact_cts_end_w, :]
        lambdas_j = lambdas_np[w][joint_cts_start_w:joint_cts_end_w]
        lambdas_l = lambdas_np[w][limit_cts_start_w:limit_cts_end_w]
        lambdas_c = lambdas_np[w][contact_cts_start_w:contact_cts_end_w]
        w_a_i_ref_np[w][:, :] = (J_dofs_dense[w].T @ tau_j_np[w]).reshape(num_bodies_np[w], 6)
        w_j_i_ref_np[w][:, :] = inv_dt_np[w] * (J_cts_j.T @ lambdas_j).reshape(num_bodies_np[w], 6)
        w_l_i_ref_np[w][:, :] = inv_dt_np[w] * (J_cts_l.T @ lambdas_l).reshape(num_bodies_np[w], 6)
        w_c_i_ref_np[w][:, :] = inv_dt_np[w] * (J_cts_c.T @ lambdas_c).reshape(num_bodies_np[w], 6)
    w_a_i_ref_np = wp.array(np.concatenate(w_a_i_ref_np, axis=0), device="cpu")
    w_j_i_ref_np = wp.array(np.concatenate(w_j_i_ref_np, axis=0), device="cpu")
    w_l_i_ref_np = wp.array(np.concatenate(w_l_i_ref_np, axis=0), device="cpu")
    w_c_i_ref_np = wp.array(np.concatenate(w_c_i_ref_np, axis=0), device="cpu")

    # Debug output
    msg.info("w_a_i_ref_np:\n%s", w_a_i_ref_np)
    msg.info("w_a_i_dense_np:\n%s", w_a_i_dense_np)
    msg.info("w_a_i_sparse_np:\n%s\n", w_a_i_sparse_np)
    msg.info("w_j_i_ref_np:\n%s", w_j_i_ref_np)
    msg.info("w_j_i_dense_np:\n%s", w_j_i_dense_np)
    msg.info("w_j_i_sparse_np:\n%s\n", w_j_i_sparse_np)
    msg.info("w_l_i_ref_np:\n%s", w_l_i_ref_np)
    msg.info("w_l_i_dense_np:\n%s", w_l_i_dense_np)
    msg.info("w_l_i_sparse_np:\n%s\n", w_l_i_sparse_np)
    msg.info("w_c_i_ref_np:\n%s", w_c_i_ref_np)
    msg.info("w_c_i_dense_np:\n%s", w_c_i_dense_np)
    msg.info("w_c_i_sparse_np:\n%s\n\n", w_c_i_sparse_np)

    # Check that the Jacobians computed using the dense and sparse implementations are close
    for w in range(model.size.num_worlds):
        np.testing.assert_allclose(J_cts_sparse[w], J_cts_dense[w], rtol=test_jacobian_rtol, atol=test_jacobian_atol)
        np.testing.assert_allclose(J_dofs_sparse[w], J_dofs_dense[w], rtol=test_jacobian_rtol, atol=test_jacobian_atol)

    # Check that the wrenches computed using the dense Jacobians match the reference wrenches
    np.testing.assert_allclose(w_a_i_dense_np, w_a_i_ref_np, rtol=test_wrench_rtol, atol=test_wrench_atol)
    np.testing.assert_allclose(w_j_i_dense_np, w_j_i_ref_np, rtol=test_wrench_rtol, atol=test_wrench_atol)
    np.testing.assert_allclose(w_l_i_dense_np, w_l_i_ref_np, rtol=test_wrench_rtol, atol=test_wrench_atol)
    np.testing.assert_allclose(w_c_i_dense_np, w_c_i_ref_np, rtol=test_wrench_rtol, atol=test_wrench_atol)

    # Check that the wrenches computed using the dense and sparse Jacobians are close
    np.testing.assert_allclose(w_a_i_sparse_np, w_a_i_dense_np, rtol=test_wrench_rtol, atol=test_wrench_atol)
    np.testing.assert_allclose(w_j_i_sparse_np, w_j_i_dense_np, rtol=test_wrench_rtol, atol=test_wrench_atol)
    np.testing.assert_allclose(w_l_i_sparse_np, w_l_i_dense_np, rtol=test_wrench_rtol, atol=test_wrench_atol)
    np.testing.assert_allclose(w_c_i_sparse_np, w_c_i_dense_np, rtol=test_wrench_rtol, atol=test_wrench_atol)


###
# Helpers for `TestDynamicsConvertWrenches`
###


class ConvertWrenchesTestSetup:
    """Builds a Newton-side simulation harness alongside the equivalent Kamino containers.

    The Newton-side fields drive the reference :class:`SolverKamino` step that produces the
    populated ``state.body_parent_f`` extended attribute and the solver-internal
    ``data.joints.lambda_j``. The Kamino-side containers (``model_kamino``, ``data_kamino``,
    ``limits_kamino``, ``contacts_kamino``, ``control_kamino``) are used by tests to exercise
    :func:`convert_body_parent_wrenches_to_joint_reactions` and
    :func:`convert_joint_parent_wrenches_to_joint_reactions` directly without going through
    :class:`SolutionMetricsNewton`.
    """

    def __init__(
        self,
        builder_fn,
        builder_kwargs: dict | None = None,
        dt: float = 0.001,
        max_world_contacts: int = 32,
        device: wp.DeviceLike = None,
    ):
        # Cache the time-step size and contact capacity
        self.dt = dt
        self.max_world_contacts = max_world_contacts

        # Construct the model description using the requested builder
        if builder_kwargs is None:
            builder_kwargs = {}
        self.builder: ModelBuilder = builder_fn(**builder_kwargs)
        self.builder.request_contact_attributes("force")
        self.builder.request_state_attributes("body_parent_f")

        # Set the maximum number of rigid contacts per world
        self.builder.num_rigid_contacts_per_world = max_world_contacts

        # Create the Newton-side model and runtime containers
        self.model: Model = self.builder.finalize(skip_validation_joints=True)
        self.state: State = self.model.state()
        self.state_p: State = self.model.state()
        self.control: Control = self.model.control()
        self.contacts: Contacts = self.model.contacts()

        # Create a Kamino solver from the model
        self.solver = newton.solvers.SolverKamino(model=self.model)

        # Build a parallel set of Kamino containers, mirroring the allocation order
        # used by ``SolutionMetricsNewton.finalize`` so we can exercise the convert
        # and pack/unpack primitives without going through the metrics wrapper.
        self.model_kamino: ModelKamino = ModelKamino.from_newton(model=self.model, overwrite_source_model=False)
        self.model_kamino.time.dt.fill_(wp.float32(dt))
        self.model_kamino.time.inv_dt.fill_(wp.float32(1.0 / dt))

        # ``joint_wrenches=True`` allocates ``data.joints.j_w_j`` which the convert
        # functions write to as a byproduct of recovering the joint-reaction lambdas.
        self.data_kamino: DataKamino = self.model_kamino.data(joint_wrenches=True)
        self.limits_kamino: LimitsKamino = LimitsKamino(model=self.model_kamino)
        self.contacts_kamino: ContactsKamino = ContactsKamino(model=self.model_kamino)
        self.limits_kamino.reset()
        self.contacts_kamino.reset()

        # Create and finalize the Kamino-side control container
        self.control_kamino: ControlKamino = ControlKamino()
        self.control_kamino.finalize(self.model_kamino)

        # Construct the unilateral constraints info on the Kamino model so that
        # ``model.joints.kinematic_cts_offset_joint_cts`` (consumed by the convert
        # kernels) and the per-world ``total_cts_offset`` (consumed by pack/unpack)
        # are populated.
        make_unilateral_constraints_info(
            model=self.model_kamino,
            data=self.data_kamino,
            limits=self.limits_kamino,
            contacts=self.contacts_kamino,
        )

    def step_and_populate(self):
        """Run a single solver step and populate Kamino data from the pre-event state.

        After this call:
        - ``self.state.body_parent_f`` is populated by the solver.
        - ``self.solver._solver_kamino._data.joints.lambda_j`` contains the reference
          joint-constraint Lagrange multipliers.
        - ``self.data_kamino`` mirrors the pre-event configuration (poses, velocities,
          joint frames, body inertias, active constraints) at which the body wrenches
          were evaluated.
        """
        # Step the solver. Newton's collide is needed before each step so that the
        # ``contacts`` container reflects the pre-event configuration.
        self.model.collide(self.state_p, self.contacts)
        self.solver.step(
            state_in=self.state_p,
            state_out=self.state,
            control=self.control,
            contacts=self.contacts,
            dt=self.dt,
        )

        # Reset the Kamino limits and contacts so prior contents do not bleed in
        self.limits_kamino.reset()
        self.contacts_kamino.reset()

        # Interface the Newton state and contacts to their Kamino equivalents
        state_p_kamino = StateKamino.from_newton(self.model_kamino.size, self.model, self.state_p)
        self.control_kamino.from_newton(self.control, self.model_kamino)
        convert_contacts_newton_to_kamino(self.model, self.state_p, self.contacts, self.contacts_kamino)

        # Copy the pre-event state arrays into the Kamino data container so that
        # downstream constraint and joint-data builders see the same configuration
        # at which the solver evaluated the body wrenches.
        wp.copy(self.data_kamino.bodies.q_i, state_p_kamino.q_i)
        wp.copy(self.data_kamino.bodies.u_i, state_p_kamino.u_i)
        wp.copy(self.data_kamino.bodies.w_i, state_p_kamino.w_i)
        wp.copy(self.data_kamino.bodies.w_e_i, state_p_kamino.w_i_e)
        wp.copy(self.data_kamino.joints.q_j, state_p_kamino.q_j)
        wp.copy(self.data_kamino.joints.dq_j, state_p_kamino.dq_j)
        self.data_kamino.joints.tau_j = self.control_kamino.tau_j

        # Refresh the active-constraint counts/group offsets and the per-joint
        # frame poses (``data.joints.p_j``) used by the convert kernels.
        update_constraints_info(model=self.model_kamino, data=self.data_kamino)
        update_body_inertias(model=self.model_kamino.bodies, data=self.data_kamino.bodies)
        compute_joints_data(model=self.model_kamino, data=self.data_kamino, q_j_p=state_p_kamino.q_j)


def make_joint_parent_f_from_body_parent_f(
    model_kamino: ModelKamino,
    body_parent_f: wp.array,
) -> wp.array:
    """Construct a per-joint ``joint_parent_f`` array from the per-body ``body_parent_f``.

    For systems where every non-FREE joint has a unique follower body (true for all builders
    used by :class:`TestDynamicsConvertWrenches`), the world-frame wrench applied on body
    ``bid_F`` by joint ``j`` equals ``body_parent_f[bid_F]``. Indexing this array per joint
    therefore yields the equivalent ``joint_parent_f`` representation.

    Args:
        model_kamino: The Kamino model providing the per-joint follower body indices.
        body_parent_f: The Newton-produced per-body parent wrench array.

    Returns:
        A ``wp.array`` of ``wp.spatial_vectorf`` with shape ``(num_joints,)``, allocated on
        the same device as ``model_kamino``.
    """
    bid_F_np = model_kamino.joints.bid_F.numpy()
    body_parent_f_np = body_parent_f.numpy()
    joint_parent_f_np = np.zeros((bid_F_np.shape[0], 6), dtype=np.float32)
    for jid, bid_F in enumerate(bid_F_np):
        if bid_F >= 0:
            joint_parent_f_np[jid] = body_parent_f_np[bid_F]
    return wp.array(joint_parent_f_np, dtype=wp.spatial_vectorf, device=model_kamino.device)


def get_leaf_joint_lambda_indices(model_kamino: ModelKamino) -> np.ndarray:
    """Return the ``lambda_j`` indices corresponding to leaf non-FREE joints.

    A non-FREE joint ``j`` is a *leaf joint* when its follower body ``bid_F`` is not the
    base body ``bid_B`` of any other non-FREE joint. The convert kernels recover these
    joints' kinematic-constraint Lagrange multipliers exactly, because the per-body
    accumulator in :func:`convert_joint_wrenches_to_body_parent_wrenches` only contains
    contributions from joint ``j`` for that body.

    For non-leaf joints the recovered lambdas mix contributions from descendant joints
    (their base-side reactions land on the same follower body), and an exact comparison
    against the solver's reference lambdas is not expected to hold.

    Args:
        model_kamino: The Kamino model whose joint topology is queried.

    Returns:
        A 1-D ``numpy.ndarray`` of ``int`` indices into ``data.joints.lambda_j``.
    """
    from newton._src.solvers.kamino._src.core.joints import JointDoFType  # noqa: PLC0415

    bid_F = model_kamino.joints.bid_F.numpy()
    bid_B = model_kamino.joints.bid_B.numpy()
    dof_type = model_kamino.joints.dof_type.numpy()
    kin_offset = model_kamino.joints.kinematic_cts_offset_joint_cts.numpy()
    num_kin_cts = model_kamino.joints.num_kinematic_cts.numpy()
    free_value = int(JointDoFType.FREE.value)

    # Compute the set of bodies that act as the base of at least one non-FREE joint;
    # any joint whose follower lies in this set is, by definition, not a leaf.
    bases_of_non_free = {int(b) for b, dt in zip(bid_B, dof_type, strict=True) if dt != free_value and b >= 0}

    leaf_indices = []
    for jid in range(len(dof_type)):
        if dof_type[jid] == free_value:
            continue
        if int(bid_F[jid]) in bases_of_non_free:
            continue
        # Add the kinematic-constraint indices for this leaf joint
        start = int(kin_offset[jid])
        leaf_indices.extend(range(start, start + int(num_kin_cts[jid])))
    return np.asarray(leaf_indices, dtype=np.int64)


###
# Tests
###


class TestDynamicsWrenches(unittest.TestCase):
    def setUp(self):
        # Configs
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        self.verbose = test_context.verbose  # Set to True for verbose output

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
        model, data, _state, limits, contacts = make_test_problem_fourbar(
            device=self.default_device,
            max_world_contacts=12,
            num_worlds=1,
            with_limits=True,
            with_contacts=True,
            with_implicit_joints=True,
            verbose=False,  # TODO
        )

        # Compute and compare the wrenches using the dense and sparse Jacobians
        compute_and_compare_dense_sparse_jacobian_wrenches(
            model=model,
            data=data,
            limits=limits,
            contacts=contacts,
        )

    def test_02_compute_wrenches_for_multiple_fourbars_with_limits_and_contacts(self):
        # Construct the test problem
        model, data, _state, limits, contacts = make_test_problem_fourbar(
            device=self.default_device,
            max_world_contacts=12,
            num_worlds=3,
            with_limits=True,
            with_contacts=True,
            with_implicit_joints=True,
            verbose=False,
        )

        # Compute and compare the wrenches using the dense and sparse Jacobians
        compute_and_compare_dense_sparse_jacobian_wrenches(
            model=model,
            data=data,
            limits=limits,
            contacts=contacts,
        )

    def test_03_compute_wrenches_heterogeneous_model_with_limits_and_contacts(self):
        # Construct the test problem
        model, data, _state, limits, contacts = make_test_problem_heterogeneous(
            device=self.default_device,
            max_world_contacts=12,
            with_limits=True,
            with_contacts=True,
            with_implicit_joints=True,
            verbose=False,
        )

        # Compute and compare the wrenches using the dense and sparse Jacobians
        compute_and_compare_dense_sparse_jacobian_wrenches(
            model=model,
            data=data,
            limits=limits,
            contacts=contacts,
        )


class TestDynamicsConvertWrenches(unittest.TestCase):
    """Tests :func:`convert_body_parent_wrenches_to_joint_reactions` and
    :func:`convert_joint_parent_wrenches_to_joint_reactions` against a reference
    :class:`SolverKamino` step (individual conversion) and a
    :func:`pack_constraint_solutions` / :func:`unpack_constraint_solutions` round-trip.

    All tests are passive (no actuated/dynamic joints, no implicit-PD), so the recovered
    multipliers populate only the kinematic-constraint slice of ``data.joints.lambda_j``.

    Leaf-joint comparison
    ---------------------
    The forward function :func:`convert_joint_wrenches_to_body_parent_wrenches` accumulates
    each body's *total* joint-constraint wrench (parent-side reaction + base-side reactions
    from any descendant joints sharing that body). Inverting only the parent-side reaction
    out of this sum is therefore exact only when the joint's follower body is a *leaf* of
    the non-FREE joint graph, i.e. it is not the base of any other non-FREE joint. The
    individual tests use :func:`get_leaf_joint_lambda_indices` to compare only those
    kinematic-constraint slices and merely sanity-check that the full recovered ``lambda_j``
    has non-trivial magnitude. The pack / unpack round-trip tests, in contrast, are
    independent of this issue and validate the full ``lambda_j``.
    """

    def setUp(self):
        # Mirrors ``TestDynamicsWrenches.setUp`` so the tests pick up the same
        # global device/verbosity configuration when launched via this module.
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        self.verbose = test_context.verbose

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

    @staticmethod
    def _builders_without_loops():
        """Builders whose multibody graph is a tree (no kinematic loops).

        Each entry is a ``(name, builder_fn, builder_kwargs)`` tuple. All three trees
        use ``z_offset=-1e-5`` so that the cart/base body is just barely penetrating the
        ground at ``t = 0``, which is enough for Kamino's narrowphase to register active
        contacts and for the constraint solver to produce non-zero kinematic-constraint
        Lagrange multipliers within a single time-step.
        """
        return [
            ("cartpole", basics.build_cartpole, {"z_offset": -1e-5}),
            ("boxes_hinged", basics.build_boxes_hinged, {"z_offset": -1e-5}),
            ("boxes_nunchaku_vertical", basics.build_boxes_nunchaku_vertical, {"z_offset": -1e-5}),
        ]

    @staticmethod
    def _builders_with_loops():
        """Tree builders plus the four-bar linkage (the only loop-joint case).

        ``boxes_fourbar`` requires ``floatingbase=True`` so Newton's articulation builder
        can reconcile the closed loop into a valid (rooted) articulation tree by attaching
        the base via a FREE joint.
        """
        return [
            ("cartpole", basics.build_cartpole, {"z_offset": -1e-5}),
            ("boxes_hinged", basics.build_boxes_hinged, {"z_offset": -1e-5}),
            ("boxes_nunchaku_vertical", basics.build_boxes_nunchaku_vertical, {"z_offset": -1e-5}),
            ("boxes_fourbar", basics.build_boxes_fourbar, {"z_offset": -1e-5, "floatingbase": True}),
        ]

    def _assert_leaf_kinematic_lambdas_close(
        self,
        recovered: np.ndarray,
        reference: np.ndarray,
        leaf_indices: np.ndarray,
        rtol: float = 1e-4,
        atol: float = 1e-5,
    ):
        """Compare two ``data.joints.lambda_j`` snapshots over the leaf-joint indices.

        - Asserts that the reference (solver-produced) lambdas have non-trivial norm so the
          test cannot pass vacuously.
        - Asserts ``np.testing.assert_allclose`` only on the leaf-joint kinematic-constraint
          indices, where the convert function exactly inverts the forward accumulation.
        - Sanity-checks that the recovered full ``lambda_j`` has non-trivial norm so the
          convert kernel is exercised end-to-end.
        """
        self.assertGreater(
            float(np.linalg.norm(reference)),
            0.0,
            msg="Reference lambda_j has zero norm; the test is vacuous.",
        )
        self.assertGreater(
            float(np.linalg.norm(recovered)),
            0.0,
            msg="Recovered lambda_j has zero norm; the convert kernel did not populate the output.",
        )
        if leaf_indices.size > 0:
            np.testing.assert_allclose(
                recovered[leaf_indices],
                reference[leaf_indices],
                rtol=rtol,
                atol=atol,
                err_msg="Recovered lambdas at leaf-joint indices do not match the solver's reference.",
            )

    def test_01_convert_body_parent_wrenches_individual(self):
        """
        Convert the solver-produced ``body_parent_f`` and compare
        against the solver's reference ``data.joints.lambda_j``.
        """
        for name, builder_fn, builder_kwargs in self._builders_without_loops():
            with self.subTest(builder=name):
                # Build the harness and run a step to populate `body_parent_f`
                # and the solver's reference lambdas.
                setup = ConvertWrenchesTestSetup(
                    builder_fn=builder_fn,
                    builder_kwargs=builder_kwargs,
                    device=self.default_device,
                )
                setup.step_and_populate()

                # Run the convert kernel on the populated `body_parent_f` to recover
                # the kinematic-constraint Lagrange multipliers from the wrenches.
                convert_body_parent_wrenches_to_joint_reactions(
                    body_parent_f=setup.state.body_parent_f,
                    model=setup.model_kamino,
                    data=setup.data_kamino,
                    control=setup.control_kamino,
                    limits=setup.limits_kamino,
                    reset_to_zero=True,
                )

                # Compare the recovered lambdas with the solver's internal reference,
                # restricting strict equality to the leaf-joint indices where the
                # body-wrench accumulation is exactly invertible.
                recovered = setup.data_kamino.joints.lambda_j.numpy()
                reference = setup.solver._solver_kamino._data.joints.lambda_j.numpy()
                leaf_indices = get_leaf_joint_lambda_indices(setup.model_kamino)
                msg.info(f"[{name}] Recovered lambda_j: {recovered}")
                msg.info(f"[{name}] Expected  lambda_j: {reference}")
                msg.info(f"[{name}] Leaf-joint lambda indices: {leaf_indices}")
                self._assert_leaf_kinematic_lambdas_close(recovered, reference, leaf_indices)

    def test_02_convert_body_parent_wrenches_round_trip(self):
        """
        Pack the recovered ``lambda_j`` into a global ``lambdas`` array and unpack
        it back into the joint-cts buffer, verifying ``pack`` / ``unpack`` symmetry.
        """
        for name, builder_fn, builder_kwargs in self._builders_without_loops():
            with self.subTest(builder=name):
                # Build the harness and run a step to populate `body_parent_f`
                # and the solver's reference lambdas.
                setup = ConvertWrenchesTestSetup(
                    builder_fn=builder_fn,
                    builder_kwargs=builder_kwargs,
                    device=self.default_device,
                )
                setup.step_and_populate()

                # Run the convert kernel to populate `data_kamino.joints.lambda_j`
                convert_body_parent_wrenches_to_joint_reactions(
                    body_parent_f=setup.state.body_parent_f,
                    model=setup.model_kamino,
                    data=setup.data_kamino,
                    control=setup.control_kamino,
                    limits=setup.limits_kamino,
                    reset_to_zero=True,
                )

                # Cache the lambdas before they are overwritten by `unpack`
                lambda_j_pre = setup.data_kamino.joints.lambda_j.numpy().copy()
                self.assertGreater(
                    float(np.linalg.norm(lambda_j_pre)),
                    0.0,
                    msg=f"[{name}] lambda_j has zero norm before round-trip; the test is vacuous.",
                )

                # Allocate the global multiplier and post-event constraint-velocity arrays.
                # ``pack_constraint_solutions`` writes only ``lambdas``; ``unpack`` reads both.
                lambdas = wp.zeros(setup.model_kamino.size.sum_of_max_total_cts, dtype=wp.float32)
                v_plus = wp.zeros_like(lambdas)

                # Pack the per-joint kinematic lambdas into the global lambdas array
                pack_constraint_solutions(
                    lambdas=lambdas,
                    model=setup.model_kamino,
                    data=setup.data_kamino,
                    limits=setup.limits_kamino,
                    contacts=setup.contacts_kamino,
                    reset_to_zero=True,
                )

                # Unpack back into the joint-cts buffer
                unpack_constraint_solutions(
                    lambdas=lambdas,
                    v_plus=v_plus,
                    model=setup.model_kamino,
                    data=setup.data_kamino,
                    limits=setup.limits_kamino,
                    contacts=setup.contacts_kamino,
                    reset_to_zero=True,
                )

                # The round-tripped lambdas should match the cached pre-pack lambdas
                lambda_j_post = setup.data_kamino.joints.lambda_j.numpy()
                np.testing.assert_allclose(lambda_j_post, lambda_j_pre, rtol=1e-5, atol=1e-6)

    def test_03_convert_joint_parent_wrenches_individual(self):
        """Convert a per-joint ``joint_parent_f`` constructed from the solver-produced
        ``body_parent_f`` and compare against the solver's reference ``data.joints.lambda_j``.

        ``boxes_fourbar`` is included as the only loop-joint case to ensure the convert kernel
        handles the closed-loop articulation correctly. For all builders considered here every
        non-FREE joint has a unique follower body, so the per-body and per-joint representations
        are equivalent.
        """
        for name, builder_fn, builder_kwargs in self._builders_with_loops():
            with self.subTest(builder=name):
                # Build the harness and run a step to populate `body_parent_f`
                # and the solver's reference lambdas.
                setup = ConvertWrenchesTestSetup(
                    builder_fn=builder_fn,
                    builder_kwargs=builder_kwargs,
                    device=self.default_device,
                )
                setup.step_and_populate()

                # Construct the per-joint parent wrench array from the per-body one
                joint_parent_f = make_joint_parent_f_from_body_parent_f(
                    setup.model_kamino,
                    setup.state.body_parent_f,
                )

                # Run the convert kernel on the synthesized `joint_parent_f`
                convert_joint_parent_wrenches_to_joint_reactions(
                    joint_parent_f=joint_parent_f,
                    model=setup.model_kamino,
                    data=setup.data_kamino,
                    control=setup.control_kamino,
                    limits=setup.limits_kamino,
                    reset_to_zero=True,
                )

                # Compare the recovered lambdas with the solver's internal reference,
                # restricting strict equality to the leaf-joint indices where the
                # body-wrench accumulation is exactly invertible.
                recovered = setup.data_kamino.joints.lambda_j.numpy()
                reference = setup.solver._solver_kamino._data.joints.lambda_j.numpy()
                leaf_indices = get_leaf_joint_lambda_indices(setup.model_kamino)
                msg.info(f"[{name}] Recovered lambda_j: {recovered}")
                msg.info(f"[{name}] Expected  lambda_j: {reference}")
                msg.info(f"[{name}] Leaf-joint lambda indices: {leaf_indices}")
                self._assert_leaf_kinematic_lambdas_close(recovered, reference, leaf_indices)

    def test_04_convert_joint_parent_wrenches_round_trip(self):
        """Same round-trip check as ``test_02`` but for the ``joint_parent_f`` convert path,
        including the ``boxes_fourbar`` loop-joint case.
        """
        for name, builder_fn, builder_kwargs in self._builders_with_loops():
            with self.subTest(builder=name):
                # Build the harness and run a step to populate `body_parent_f`
                # and the solver's reference lambdas.
                setup = ConvertWrenchesTestSetup(
                    builder_fn=builder_fn,
                    builder_kwargs=builder_kwargs,
                    device=self.default_device,
                )
                setup.step_and_populate()

                # Construct the per-joint parent wrench array from the per-body one
                joint_parent_f = make_joint_parent_f_from_body_parent_f(
                    setup.model_kamino,
                    setup.state.body_parent_f,
                )

                # Run the convert kernel to populate `data_kamino.joints.lambda_j`
                convert_joint_parent_wrenches_to_joint_reactions(
                    joint_parent_f=joint_parent_f,
                    model=setup.model_kamino,
                    data=setup.data_kamino,
                    control=setup.control_kamino,
                    limits=setup.limits_kamino,
                    reset_to_zero=True,
                )

                # Cache the lambdas before they are overwritten by `unpack`
                lambda_j_pre = setup.data_kamino.joints.lambda_j.numpy().copy()
                self.assertGreater(
                    float(np.linalg.norm(lambda_j_pre)),
                    0.0,
                    msg=f"[{name}] lambda_j has zero norm before round-trip; the test is vacuous.",
                )

                # Allocate the global multiplier and post-event constraint-velocity arrays.
                # ``pack_constraint_solutions`` writes only ``lambdas``; ``unpack`` reads both.
                lambdas = wp.zeros(setup.model_kamino.size.sum_of_max_total_cts, dtype=wp.float32)
                v_plus = wp.zeros_like(lambdas)

                # Pack the per-joint kinematic lambdas into the global lambdas array
                pack_constraint_solutions(
                    lambdas=lambdas,
                    model=setup.model_kamino,
                    data=setup.data_kamino,
                    limits=setup.limits_kamino,
                    contacts=setup.contacts_kamino,
                    reset_to_zero=True,
                )

                # Unpack back into the joint-cts buffer
                unpack_constraint_solutions(
                    lambdas=lambdas,
                    v_plus=v_plus,
                    model=setup.model_kamino,
                    data=setup.data_kamino,
                    limits=setup.limits_kamino,
                    contacts=setup.contacts_kamino,
                    reset_to_zero=True,
                )

                # The round-tripped lambdas should match the cached pre-pack lambdas
                lambda_j_post = setup.data_kamino.joints.lambda_j.numpy()
                np.testing.assert_allclose(lambda_j_post, lambda_j_pre, rtol=1e-5, atol=1e-6)


###
# Test execution
###

if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
