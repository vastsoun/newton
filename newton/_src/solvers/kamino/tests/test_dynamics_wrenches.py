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
    compute_joint_parent_wrenches_dense,
    compute_joint_parent_wrenches_sparse,
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

_TEST_JACOBIAN_RTOL = 1e-7
_TEST_JACOBIAN_ATOL = 1e-7

# TODO: FIX THIS: sparse-dense differences are larger than expected,
# likely due to the sparse implementation not fully matching the dense
_TEST_WRENCH_RTOL = 1e-4  # TODO: Should be 1e-6
_TEST_WRENCH_ATOL = 1e-4  # TODO: Should be 1e-6


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
        np.testing.assert_allclose(J_cts_sparse[w], J_cts_dense[w], rtol=_TEST_JACOBIAN_RTOL, atol=_TEST_JACOBIAN_ATOL)
        np.testing.assert_allclose(
            J_dofs_sparse[w], J_dofs_dense[w], rtol=_TEST_JACOBIAN_RTOL, atol=_TEST_JACOBIAN_ATOL
        )

    # Check that the wrenches computed using the dense Jacobians match the reference wrenches
    np.testing.assert_allclose(w_a_i_dense_np, w_a_i_ref_np, rtol=_TEST_WRENCH_RTOL, atol=_TEST_WRENCH_ATOL)
    np.testing.assert_allclose(w_j_i_dense_np, w_j_i_ref_np, rtol=_TEST_WRENCH_RTOL, atol=_TEST_WRENCH_ATOL)
    np.testing.assert_allclose(w_l_i_dense_np, w_l_i_ref_np, rtol=_TEST_WRENCH_RTOL, atol=_TEST_WRENCH_ATOL)
    np.testing.assert_allclose(w_c_i_dense_np, w_c_i_ref_np, rtol=_TEST_WRENCH_RTOL, atol=_TEST_WRENCH_ATOL)

    # Check that the wrenches computed using the dense and sparse Jacobians are close
    np.testing.assert_allclose(w_a_i_sparse_np, w_a_i_dense_np, rtol=_TEST_WRENCH_RTOL, atol=_TEST_WRENCH_ATOL)
    np.testing.assert_allclose(w_j_i_sparse_np, w_j_i_dense_np, rtol=_TEST_WRENCH_RTOL, atol=_TEST_WRENCH_ATOL)
    np.testing.assert_allclose(w_l_i_sparse_np, w_l_i_dense_np, rtol=_TEST_WRENCH_RTOL, atol=_TEST_WRENCH_ATOL)
    np.testing.assert_allclose(w_c_i_sparse_np, w_c_i_dense_np, rtol=_TEST_WRENCH_RTOL, atol=_TEST_WRENCH_ATOL)


def compute_and_compare_dense_sparse_joint_parent_wrenches(
    model: ModelKamino,
    data: DataKamino,
    limits: LimitsKamino,
    contacts: ContactsKamino,
):
    """Build dense+sparse Jacobians, populate ``tau_j`` and ``lambdas`` with ones, and verify
    that the dense and sparse :func:`compute_joint_parent_wrenches` kernels agree with a
    numpy reference computed from the extracted Jacobians.

    The reference per-joint wrench is

    .. code-block:: text

        joint_parent_f[jid] = J_dofs[joint_dof_rows, 6F:6F+6].T @ tau_j[joint_dofs]
                            + inv_dt * J_cts[joint_dyn_rows, 6F:6F+6].T @ lambdas[joint_dyn_rows]
                            + inv_dt * J_cts[joint_kin_rows, 6F:6F+6].T @ lambdas[joint_kin_rows]
                            + inv_dt * sum_{l in j} J_cts[limit_row_l, 6F:6F+6] * lambdas[limit_row_l]

    where ``F`` is the joint's follower body, all rows are world-local, and ``FREE`` joints are
    skipped (their ``joint_parent_f`` slot stays at zero).
    """
    # Lazy import to mirror the convention used elsewhere in this module
    from newton._src.solvers.kamino._src.core.joints import JointDoFType  # noqa: PLC0415

    # Create the Jacobians container
    jacobians_dense = DenseSystemJacobians(model=model, limits=limits, contacts=contacts)
    jacobians_sparse = SparseSystemJacobians(model=model, limits=limits, contacts=contacts)
    wp.synchronize()

    # Build the system Jacobians at the current configuration
    jacobians_dense.build(model=model, data=data, limits=limits.data, contacts=contacts.data)
    jacobians_sparse.build(model=model, data=data, limits=limits.data, contacts=contacts.data)
    wp.synchronize()

    # Create the global lambdas array and fill with ones
    lambdas_start, lambdas = make_constraint_multiplier_arrays(model)
    lambdas.fill_(1.0)

    # Initialize the generalized joint actuation forces to ones
    data.joints.tau_j.fill_(1.0)

    # Allocate two output arrays for dense and sparse to compare against each other
    num_joints = model.joints.num_joints
    joint_parent_f_dense = wp.zeros(shape=(num_joints,), dtype=wp.spatial_vectorf, device=model.device)
    joint_parent_f_sparse = wp.zeros(shape=(num_joints,), dtype=wp.spatial_vectorf, device=model.device)

    # Compute joint parent wrenches using the dense Jacobians
    compute_joint_parent_wrenches_dense(
        joint_parent_f=joint_parent_f_dense,
        model=model,
        data=data,
        jacobians=jacobians_dense,
        lambdas_offsets=lambdas_start,
        lambdas_data=lambdas,
        limits=limits.data,
        reset_to_zero=True,
    )
    wp.synchronize()
    joint_parent_f_dense_np = joint_parent_f_dense.numpy().copy()

    # Compute joint parent wrenches using the sparse Jacobians
    compute_joint_parent_wrenches_sparse(
        joint_parent_f=joint_parent_f_sparse,
        model=model,
        data=data,
        jacobians=jacobians_sparse,
        lambdas_offsets=lambdas_start,
        lambdas_data=lambdas,
        limits=limits.data,
        reset_to_zero=True,
    )
    wp.synchronize()
    joint_parent_f_sparse_np = joint_parent_f_sparse.numpy().copy()

    # Extract the per-world Jacobian and lambda data needed to build the numpy reference
    J_cts_dense = extract_cts_jacobians(model, limits.data, contacts.data, jacobians_dense, only_active_cts=True)
    J_dofs_dense = extract_dofs_jacobians(model, jacobians_dense)
    lambdas_np = extract_active_constraint_vectors(model, data, lambdas)

    # Per-world host views of the joint/dofs/cts metadata used by the reference computation
    inv_dt_np = model.time.inv_dt.numpy().tolist()
    bodies_offset_np = model.info.bodies_offset.numpy().tolist()
    joint_dofs_world_offset_np = model.info.joint_dofs_offset.numpy().tolist()
    joint_dyn_cts_world_offset_np = model.info.joint_dynamic_cts_offset.numpy().tolist()
    joint_kin_cts_world_offset_np = model.info.joint_kinematic_cts_offset.numpy().tolist()
    jdcgo_np = model.info.joint_dynamic_cts_group_offset.numpy().tolist()
    jkcgo_np = model.info.joint_kinematic_cts_group_offset.numpy().tolist()
    num_joint_cts_np = model.info.num_joint_cts.numpy().tolist()

    # Per-joint host views
    joint_dof_type_np = model.joints.dof_type.numpy().tolist()
    joint_wid_np = model.joints.wid.numpy().tolist()
    joint_bid_F_np = model.joints.bid_F.numpy().tolist()
    joint_dofs_offset_np = model.joints.dofs_offset.numpy().tolist()
    joint_num_dofs_np = model.joints.num_dofs.numpy().tolist()
    joint_dynamic_cts_offset_np = model.joints.dynamic_cts_offset.numpy().tolist()
    joint_kinematic_cts_offset_np = model.joints.kinematic_cts_offset.numpy().tolist()
    joint_num_dyn_cts_np = model.joints.num_dynamic_cts.numpy().tolist()
    joint_num_kin_cts_np = model.joints.num_kinematic_cts.numpy().tolist()
    tau_j_np = data.joints.tau_j.numpy()

    # Build the per-joint reference. Each non-FREE joint receives F-side actuation, dyn cts, and
    # kin cts contributions; FREE joints are left at zero (their slot is always zero by design).
    free_value = int(JointDoFType.FREE.value)
    joint_parent_f_ref_np = np.zeros((num_joints, 6), dtype=np.float32)
    for jid in range(num_joints):
        if joint_dof_type_np[jid] == free_value:
            continue

        wid = joint_wid_np[jid]
        bid_F_local = joint_bid_F_np[jid] - bodies_offset_np[wid]
        col_F_start = 6 * bid_F_local

        # Actuation: F-side dofs Jacobian columns times the joint's slice of tau_j
        d_io_world = joint_dofs_offset_np[jid] - joint_dofs_world_offset_np[wid]
        d_j = joint_num_dofs_np[jid]
        if d_j > 0:
            J_dofs_F_j = J_dofs_dense[wid][d_io_world : d_io_world + d_j, col_F_start : col_F_start + 6]
            tau_j_slice = tau_j_np[joint_dofs_offset_np[jid] : joint_dofs_offset_np[jid] + d_j]
            joint_parent_f_ref_np[jid] += J_dofs_F_j.T @ tau_j_slice

        # Dynamic constraints: F-side dyn cts Jacobian rows times the joint's lambda slice
        num_dyn_j = joint_num_dyn_cts_np[jid]
        if num_dyn_j > 0:
            local_dyn_offset = joint_dynamic_cts_offset_np[jid] - joint_dyn_cts_world_offset_np[wid]
            row_start = jdcgo_np[wid] + local_dyn_offset
            row_end = row_start + num_dyn_j
            J_cts_dyn_F_j = J_cts_dense[wid][row_start:row_end, col_F_start : col_F_start + 6]
            lambdas_dyn_j = lambdas_np[wid][row_start:row_end]
            joint_parent_f_ref_np[jid] += inv_dt_np[wid] * (J_cts_dyn_F_j.T @ lambdas_dyn_j)

        # Kinematic constraints: F-side kin cts Jacobian rows times the joint's lambda slice
        num_kin_j = joint_num_kin_cts_np[jid]
        if num_kin_j > 0:
            local_kin_offset = joint_kinematic_cts_offset_np[jid] - joint_kin_cts_world_offset_np[wid]
            row_start = jkcgo_np[wid] + local_kin_offset
            row_end = row_start + num_kin_j
            J_cts_kin_F_j = J_cts_dense[wid][row_start:row_end, col_F_start : col_F_start + 6]
            lambdas_kin_j = lambdas_np[wid][row_start:row_end]
            joint_parent_f_ref_np[jid] += inv_dt_np[wid] * (J_cts_kin_F_j.T @ lambdas_kin_j)

    # Add per-limit reactions to the limits' joint slots
    has_limits = limits is not None and limits.data is not None and limits.data.model_max_limits_host > 0
    if has_limits:
        active_limits_total = int(limits.data.model_active_limits.numpy()[0])
        if active_limits_total > 0:
            limits_wid_np = limits.data.wid.numpy().tolist()
            limits_jid_np = limits.data.jid.numpy().tolist()
            limits_lid_np = limits.data.lid.numpy().tolist()
            limits_bids_np = limits.data.bids.numpy()
            for tid in range(active_limits_total):
                wid_l = limits_wid_np[tid]
                jid_l = limits_jid_np[tid]
                lid_l = limits_lid_np[tid]
                bid_F_l_local = int(limits_bids_np[tid][1]) - bodies_offset_np[wid_l]
                col_F_l_start = 6 * bid_F_l_local

                # The limit cts block starts at row ``num_joint_cts[wid]`` within the world's
                # active-only J_cts matrix; within the limit block, rows are indexed by the
                # within-world limit id ``lid``.
                limit_row = num_joint_cts_np[wid_l] + lid_l
                lambda_l = lambdas_np[wid_l][limit_row]
                J_F_l = J_cts_dense[wid_l][limit_row, col_F_l_start : col_F_l_start + 6]
                joint_parent_f_ref_np[jid_l] += inv_dt_np[wid_l] * lambda_l * J_F_l

    # Debug output (gated on verbose mode upstream)
    msg.info("joint_parent_f_ref_np:\n%s", joint_parent_f_ref_np)
    msg.info("joint_parent_f_dense_np:\n%s", joint_parent_f_dense_np)
    msg.info("joint_parent_f_sparse_np:\n%s\n", joint_parent_f_sparse_np)

    # Check that the dense kernel matches the numpy reference
    np.testing.assert_allclose(
        joint_parent_f_dense_np,
        joint_parent_f_ref_np,
        rtol=_TEST_WRENCH_RTOL,
        atol=_TEST_WRENCH_ATOL,
    )

    # Check that the dense and sparse kernels agree
    np.testing.assert_allclose(
        joint_parent_f_sparse_np,
        joint_parent_f_dense_np,
        rtol=_TEST_WRENCH_RTOL,
        atol=_TEST_WRENCH_ATOL,
    )


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
        self.builder.request_state_attributes("body_parent_f", "joint_parent_f")

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

        # Create the data, limits and contacts containers.
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


def _non_contact_lambda_indices(
    model_kamino: ModelKamino,
    data_kamino: DataKamino,
) -> np.ndarray:
    """Return global ``lambdas`` indices that exclude the per-world ACTIVE contact-cts slice.

    The solver writes contact-cts reactions at::

        [total_cts_offset[w] + contact_cts_group_offset[w],
         total_cts_offset[w] + contact_cts_group_offset[w] + num_contact_cts[w])

    per world ``w``. Round-trips through :attr:`State.body_parent_f` or
    :attr:`State.joint_parent_f` cannot recover those slots because contact forces do
    not flow through joints, so this helper builds the index mask of slots that ARE
    recoverable (joint dyn-cts, joint kin-cts, limit cts, and the unused inactive
    tail). Inactive limit and contact slots remain zero in both the solver-produced
    and recovered ``lambdas`` arrays after a single warm-started step, so including
    them in the comparison is safe.

    Args:
        model_kamino: The Kamino model providing per-world ``total_cts_offset``.
        data_kamino: The Kamino data container; must have been populated via
            :func:`update_constraints_info` against the same active-contact set as the
            ``lambdas`` being compared.

    Returns:
        A 1-D ``numpy.ndarray`` of ``int`` indices into the global ``lambdas`` array.
    """
    total_offsets = model_kamino.info.total_cts_offset.numpy()
    ccgo = data_kamino.info.contact_cts_group_offset.numpy()
    nccts = data_kamino.info.num_contact_cts.numpy()

    total_size = int(model_kamino.size.sum_of_max_total_cts)
    mask = np.ones(total_size, dtype=bool)
    for w in range(int(ccgo.shape[0])):
        start = int(total_offsets[w]) + int(ccgo[w])
        end = start + int(nccts[w])
        if end > start:
            mask[start:end] = False
    return np.flatnonzero(mask)


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


class TestComputeJointParentWrenches(unittest.TestCase):
    """Tests :func:`compute_joint_parent_wrenches` (dense and sparse) against a numpy
    reference computed from the extracted Jacobians.

    Each test exercises a non-trivial topology (four-bar with kinematic loop, multi-world
    four-bar, and a heterogeneous model that mixes FREE/REVOLUTE/PRISMATIC joints), with
    joint limits and contacts both active and ``with_implicit_joints=True`` so that the
    actuation, dynamic-constraint, kinematic-constraint, and limit contributions are all
    non-zero.
    """

    def setUp(self):
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

    def test_01_compute_joint_parent_wrenches_for_single_fourbar_with_limits_and_contacts(self):
        # Construct the test problem
        model, data, _state, limits, contacts = make_test_problem_fourbar(
            device=self.default_device,
            max_world_contacts=12,
            num_worlds=1,
            with_limits=True,
            with_contacts=True,
            with_implicit_joints=True,
            verbose=False,
        )

        # Compute and compare the joint parent wrenches using the dense and sparse Jacobians
        compute_and_compare_dense_sparse_joint_parent_wrenches(
            model=model,
            data=data,
            limits=limits,
            contacts=contacts,
        )

    def test_02_compute_joint_parent_wrenches_for_multiple_fourbars_with_limits_and_contacts(self):
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

        # Compute and compare the joint parent wrenches using the dense and sparse Jacobians
        compute_and_compare_dense_sparse_joint_parent_wrenches(
            model=model,
            data=data,
            limits=limits,
            contacts=contacts,
        )

    def test_03_compute_joint_parent_wrenches_heterogeneous_model_with_limits_and_contacts(self):
        # Construct the test problem
        model, data, _state, limits, contacts = make_test_problem_heterogeneous(
            device=self.default_device,
            max_world_contacts=12,
            with_limits=True,
            with_contacts=True,
            with_implicit_joints=True,
            verbose=False,
        )

        # Compute and compare the joint parent wrenches using the dense and sparse Jacobians
        compute_and_compare_dense_sparse_joint_parent_wrenches(
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


class TestDynamicsRoundTripWrenches(unittest.TestCase):
    """Round-trip the solver's global ``lambdas`` through ``body_parent_f`` and
    ``joint_parent_f`` and verify exact recovery (modulo numerical precision).

    Pipeline (per test):
        1. Build a model via :class:`ConvertWrenchesTestSetup` and apply explicit
           actuation via ``setup.control.joint_f`` (see :meth:`_apply_actuation`).
        2. Run one :meth:`SolverKamino.step` call. The solver writes
           ``setup.state.body_parent_f``, ``setup.state.joint_parent_f``, and the
           reference global ``lambdas`` at
           ``setup.solver._solver_kamino._solver_fd.data.solution.lambdas``.
        3. Convert the chosen wrench back into ``data.joints.lambda_j`` and
           ``limits.reaction`` via :func:`convert_body_parent_wrenches_to_joint_reactions`
           or :func:`convert_joint_parent_wrenches_to_joint_reactions`. The solver's
           internal limits container is passed so the recovered limit reactions land
           in the same array, on the same active-limit set used by the step.
        4. Pack the recovered per-joint and per-limit reactions back into a global
           ``recovered_lambdas`` array via :func:`pack_constraint_solutions`. Contacts
           are intentionally NOT passed because contact forces don't flow through
           joints and therefore aren't recoverable from these wrench representations.
        5. Compare ``recovered_lambdas`` against ``original_lambdas``:

           - When ``z_offset > 0`` (no ground contacts), the entire array round-trips
             exactly: the joint kinematic-cts slots match, dynamic-cts and limit-cts
             slots are zero, and the contact-cts slots are zero in both.
           - When ``z_offset < 0`` (ground contacts active in the solver), the per-world
             active contact-cts slice is excluded from the comparison since the
             recovered array has zeros there but the solver's array does not.

    Builder coverage follows the user constraints:

    - ``body_parent_f`` round-trip is restricted to tree topologies because the
      per-body accumulator only inverts cleanly when every non-FREE joint has a
      unique follower body (true for tree builders, not for the four-bar loop).
    - ``joint_parent_f`` round-trip exercises every builder, including the
      ``boxes_fourbar`` linkage with a closed kinematic loop.

    Why every test applies explicit actuation
        Most builders here root their base body to the world via a FREE joint. Under
        pure gravity those systems sit in trivial equilibrium — every body free-falls
        at the same rate, all joint constraints are satisfied without any reaction,
        and ``lambdas`` collapses to zero, leaving the round-trip vacuous. Applying a
        small effort on a single DoF (typically the FREE joint's first translation
        axis or, for ``cartpole``, the cart's prismatic axis) creates relative motion
        that propagates through the joint chain, forcing the kinematic-cts reactions
        to be non-zero. This satisfies the project guidance that "an example with
        explicit actuation via ``control.joint_f`` is fine" and ensures every test
        is non-vacuous.

        All builders run with their default ``dynamic_joints=False`` and
        ``implicit_pd=False`` configurations, so the joint dynamic-cts slice is empty
        (``num_dynamic_cts == 0`` per joint).
    """

    Z_OFFSET_NO_CONTACTS = 0.5
    Z_OFFSET_WITH_CONTACTS = -1e-5
    JOINT_F_VALUE = 5.0

    # Tolerances for the lambdas comparison. The recovered values pass through one
    # impulse-to-force scale (`inv_dt`), one Jacobian-T multiply (in the convert
    # kernel), and one force-to-impulse scale (`dt`) on the way back, so the bound
    # is set well above the floor of `float32` round-off.
    _RTOL = 1e-4
    _ATOL = 1e-5

    def setUp(self):
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
    def _tree_builders():
        """Builders whose multibody graph is a tree (no kinematic loops)."""
        return [
            ("cartpole", basics.build_cartpole, {}),
            ("boxes_hinged", basics.build_boxes_hinged, {}),
            ("boxes_nunchaku", basics.build_boxes_nunchaku, {}),
            ("boxes_nunchaku_vertical", basics.build_boxes_nunchaku_vertical, {}),
        ]

    @staticmethod
    def _all_builders():
        """Tree builders plus the four-bar linkage (loop topology)."""
        return [
            *TestDynamicsRoundTripWrenches._tree_builders(),
            ("boxes_fourbar", basics.build_boxes_fourbar, {"floatingbase": True}),
        ]

    def _apply_actuation(self, setup):
        """Apply ``setup.control.joint_f[0] = JOINT_F_VALUE`` to break trivial equilibrium.

        For ``cartpole`` slot 0 is the cart's prismatic axis; for the other builders
        it is the FREE joint's first DoF (X-translation on the base body). Either
        way the resulting motion propagates through the downstream joints and forces
        non-zero reactions in their kinematic-cts slots.

        ``control.joint_f`` is sized ``(joint_dof_count,)`` and aliased verbatim onto
        ``control_kamino.tau_j`` by :meth:`ControlKamino.from_newton`, so this single
        write drives both the solver step and the convert kernel (which subtracts
        ``tau_j`` when recovering limit reactions).
        """
        joint_f_np = setup.control.joint_f.numpy()
        joint_f_np[0] = self.JOINT_F_VALUE
        setup.control.joint_f.assign(joint_f_np)

    def _setup(self, builder_fn, builder_kwargs, z_offset):
        kwargs = dict(builder_kwargs)
        kwargs["z_offset"] = z_offset
        setup = ConvertWrenchesTestSetup(
            builder_fn=builder_fn,
            builder_kwargs=kwargs,
            device=self.default_device,
        )
        self._apply_actuation(setup)
        return setup

    def _run_round_trip(self, setup, *, convert_via, with_contacts, label):
        """Step the solver, convert + pack, and assert agreement with the reference."""
        # Run a single step to populate `body_parent_f`, `joint_parent_f` and the
        # solver's reference global `lambdas`.
        setup.step_and_populate()

        solver_impl = setup.solver._solver_kamino
        original_lambdas = solver_impl._solver_fd.data.solution.lambdas.numpy().copy()
        msg.info(f"[{label}] Original lambdas: {original_lambdas}")
        self.assertGreater(
            float(np.linalg.norm(original_lambdas)),
            0.0,
            msg=f"[{label}] Reference lambdas have zero norm; the round-trip test would be vacuous.",
        )

        # Recover per-joint lambdas (and per-limit reactions) from the chosen wrench.
        # The SOLVER's internal limits container is reused so the limit-recovery
        # kernel writes into the same array that `pack_constraint_solutions` reads
        # below, against the same active-limit set used by the step.
        if convert_via == "body":
            convert_body_parent_wrenches_to_joint_reactions(
                body_parent_f=setup.state.body_parent_f,
                model=setup.model_kamino,
                data=setup.data_kamino,
                control=setup.control_kamino,
                limits=solver_impl._limits,
                reset_to_zero=True,
            )
        elif convert_via == "joint":
            convert_joint_parent_wrenches_to_joint_reactions(
                joint_parent_f=setup.state.joint_parent_f,
                model=setup.model_kamino,
                data=setup.data_kamino,
                control=setup.control_kamino,
                limits=solver_impl._limits,
                reset_to_zero=True,
            )
        else:
            raise ValueError(f"Unknown convert_via: {convert_via!r}")

        # Pack the recovered per-joint + per-limit reactions back into a global
        # lambdas vector. Contacts are NOT passed because contact-cts cannot be
        # recovered from the joint-mediated wrench representations.
        recovered_lambdas = wp.zeros(
            setup.model_kamino.size.sum_of_max_total_cts,
            dtype=wp.float32,
            device=self.default_device,
        )
        pack_constraint_solutions(
            lambdas=recovered_lambdas,
            model=setup.model_kamino,
            data=setup.data_kamino,
            limits=solver_impl._limits,
            contacts=None,
            reset_to_zero=True,
        )
        recovered_np = recovered_lambdas.numpy()
        msg.info(f"[{label}] Recovered lambdas: {recovered_np}")
        self.assertGreater(
            float(np.linalg.norm(recovered_np)),
            0.0,
            msg=f"[{label}] Recovered lambdas have zero norm; the convert/pack pipeline did not populate the output.",
        )

        if with_contacts:
            # With ground contact active, exclude the per-world ACTIVE contact-cts
            # slice from the comparison — the solver populated those slots but the
            # recovered array zeroed them out by design.
            indices = _non_contact_lambda_indices(setup.model_kamino, setup.data_kamino)
            np.testing.assert_allclose(
                recovered_np[indices],
                original_lambdas[indices],
                rtol=self._RTOL,
                atol=self._ATOL,
                err_msg=f"[{label}] Recovered lambdas (joint+limit slice) do not match the solver reference.",
            )
        else:
            np.testing.assert_allclose(
                recovered_np,
                original_lambdas,
                rtol=self._RTOL,
                atol=self._ATOL,
                err_msg=f"[{label}] Recovered lambdas do not match the solver reference.",
            )

    def test_01_round_trip_via_body_parent_f_no_contacts(self):
        """Round-trip via ``body_parent_f`` for tree topologies with no ground contact.

        The full global ``lambdas`` array is compared.
        """
        for name, builder_fn, builder_kwargs in self._tree_builders():
            with self.subTest(builder=name):
                setup = self._setup(builder_fn, builder_kwargs, self.Z_OFFSET_NO_CONTACTS)
                self._run_round_trip(setup, convert_via="body", with_contacts=False, label=name)

    def test_02_round_trip_via_body_parent_f_with_contacts(self):
        """Round-trip via ``body_parent_f`` for tree topologies with active ground contact.

        The active contact-cts slice is excluded from the comparison.
        """
        for name, builder_fn, builder_kwargs in self._tree_builders():
            with self.subTest(builder=name):
                setup = self._setup(builder_fn, builder_kwargs, self.Z_OFFSET_WITH_CONTACTS)
                self._run_round_trip(setup, convert_via="body", with_contacts=True, label=name)

    def test_03_round_trip_via_joint_parent_f_no_contacts(self):
        """Round-trip via ``joint_parent_f`` for all builders with no ground contact.

        Includes the ``boxes_fourbar`` closed-loop case.
        """
        for name, builder_fn, builder_kwargs in self._all_builders():
            with self.subTest(builder=name):
                setup = self._setup(builder_fn, builder_kwargs, self.Z_OFFSET_NO_CONTACTS)
                self._run_round_trip(setup, convert_via="joint", with_contacts=False, label=name)

    def test_04_round_trip_via_joint_parent_f_with_contacts(self):
        """Round-trip via ``joint_parent_f`` for all builders with active ground contact."""
        for name, builder_fn, builder_kwargs in self._all_builders():
            with self.subTest(builder=name):
                setup = self._setup(builder_fn, builder_kwargs, self.Z_OFFSET_WITH_CONTACTS)
                self._run_round_trip(setup, convert_via="joint", with_contacts=True, label=name)


###
# Test execution
###

if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
