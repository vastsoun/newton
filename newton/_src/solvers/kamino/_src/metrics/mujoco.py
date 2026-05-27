# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Provides mechanisms to extract constraint reactions from SolverMuJoCo (MuJoCo Warp backend).
"""

from functools import cache

import warp as wp

from newton.solvers import SolverMuJoCo

from .....sim import Model, State
from .....solvers.solver import SolverBase
from ..core.model import ModelKamino
from ..kinematics.limits import LimitsKamino

###
# Module interface
###

__all__ = [
    "extract_constraint_reactions_mujoco_warp",
    "extract_joint_wrenches_solvermujoco",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Local types
###

# Layout-compatible with ``mujoco_warp._src.types.vec11`` so kernels can read
# ``efc_data`` rows without forcing a top-level ``mujoco_warp`` import.
_vec11 = wp.types.vector(length=11, dtype=wp.float32)


###
# Functions
###

# TODO: IMPLEMENT THIS: FUNCTIONS FOR EACH CONSTRAINT TYPE


###
# Kernels
###


def _mjw_eq_type_connect() -> int:
    """Lazy import of ``mujoco_warp._src.types.EqType.CONNECT`` as an int."""
    from mujoco_warp._src.types import EqType

    return int(EqType.CONNECT)


def _mjw_constraint_type_equality() -> int:
    """Lazy import of ``mujoco_warp._src.types.ConstraintType.EQUALITY`` as an int."""
    from mujoco_warp._src.types import ConstraintType

    return int(ConstraintType.EQUALITY)


@wp.kernel
def _extract_mjw_connect_constraint_force(
    # Inputs:
    mjw_eq_type: wp.array[wp.int32],
    mjw_efc_type: wp.array2d[wp.int32],
    mjw_efc_id: wp.array2d[wp.int32],
    mjw_efc_force: wp.array2d[wp.float32],
    mjw_ne: wp.array[wp.int32],
    # Outputs:
    connect_constraint_force: wp.array2d[wp.spatial_vectorf],
):
    """
    Reads the cartesian constraint force of each MuJoCo CONNECT equality from
    ``efc.force`` and stores it (as a 3-D force in ``spatial_top``, zeros in
    ``spatial_bottom``) in a per-equality output buffer.

    The kernel scans the ``efc`` rows up to ``ne`` and matches by ``(efc.type ==
    EQUALITY, efc.id == eq)`` because mujoco_warp allocates ``efc`` rows via
    ``atomic_add`` -- the row order is non-deterministic.
    """
    wid, eq = wp.tid()
    if mjw_eq_type[eq] != wp.static(_mjw_eq_type_connect()):
        return

    ne = mjw_ne[wid]
    efcid = int(-1)
    for i in range(ne):
        if mjw_efc_type[wid, i] == wp.static(_mjw_constraint_type_equality()) and mjw_efc_id[wid, i] == eq:
            efcid = i
            break
    if efcid < 0:
        return

    fx = mjw_efc_force[wid, efcid + 0]
    fy = mjw_efc_force[wid, efcid + 1]
    fz = mjw_efc_force[wid, efcid + 2]
    connect_constraint_force[wid, eq] = wp.spatial_vector(wp.vec3(fx, fy, fz), wp.vec3(0.0))


@wp.kernel
def _accumulate_mjw_connect_force_to_joint_parent_f(
    # Inputs:
    mjc_eq_to_newton_jnt: wp.array2d[wp.int32],
    mjw_eq_data: wp.array2d[_vec11],
    connect_constraint_force: wp.array2d[wp.spatial_vectorf],
    joint_parent: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    # Outputs:
    joint_parent_f: wp.array[wp.spatial_vectorf],
):
    """
    Accumulates the per-equality CONNECT constraint force into ``state.joint_parent_f``
    (Newton convention: world frame, moment referenced to the child body's COM).

    For each CONNECT equality mapped to a Newton loop-closure joint via
    ``mjc_eq_to_newton_jnt``, the force on the child body (= negation of mujoco_warp's
    ``efc.force``, since the CONNECT Jacobian is ``pos1 - pos2``) is applied at
    ``anchor1`` (in body1=parent's local frame, mapped to world). The moment is
    computed about the child body's COM in world frame.

    Multiple equalities mapping to the same joint accumulate via ``wp.atomic_add``.
    The output buffer must be zeroed by the caller before invocation.
    """
    wid, eq = wp.tid()

    jnt = mjc_eq_to_newton_jnt[wid, eq]
    if jnt == -1:
        return

    # Force on child body. mujoco_warp's CONNECT Jacobian is ``pos1 - pos2``, so
    # ``efc.force`` is the force on body1 (parent); negate to get the force on
    # body2 (child).
    f = -wp.spatial_top(connect_constraint_force[wid, eq])

    # ``eq_data[0:3]`` is anchor1 in body1=parent's local frame. Map to world.
    data = mjw_eq_data[wid % mjw_eq_data.shape[0], eq]
    anchor1_local = wp.vec3(data[0], data[1], data[2])

    parent = joint_parent[jnt]
    if parent >= 0:
        anchor1 = wp.transform_point(body_q[parent], anchor1_local)
    else:
        anchor1 = anchor1_local

    # Child body COM in world frame.
    child = joint_child[jnt]
    child_com_world = wp.transform_point(body_q[child], body_com[child])

    r = anchor1 - child_com_world
    moment = wp.cross(r, f)

    wp.atomic_add(joint_parent_f, jnt, wp.spatial_vector(f, moment))


@cache
def make_unpack_mjw_equalities_to_joint_reactions_kamino():
    """
    Makes a kernel to unpack mujoco_warp equalities to joint constraint reactions.
    """
    from mujoco_warp._src.types import ConstraintType as MjwConstraintType

    @wp.kernel
    def _unpack_mjw_equalities_to_joint_reactions_kamino(
        # Inputs:
        mjc_eq_to_newton_jnt: wp.array2d[wp.int32],
        mjw_nefc: wp.array[wp.int32],
        mjw_efc_type: wp.array2d[wp.int32],
        mjw_efc_id: wp.array2d[wp.int32],
        mjw_efc_pos: wp.array2d[wp.float32],
        mjw_efc_force: wp.array2d[wp.float32],
        model_time_dt: wp.array[wp.float32],
        model_joint_world: wp.array[wp.int32],
        model_joint_type: wp.array[wp.int32],
        model_joint_parent: wp.array[wp.int32],
        model_joint_child: wp.array[wp.int32],
        model_joint_X_p: wp.array[wp.transform],
        model_joint_X_c: wp.array[wp.transform],
        model_joint_axis: wp.array[wp.vec3],
        model_joint_dof_dim: wp.array2d[wp.int32],
        # TODO: other ModelKamino arrays
        # Outputs:
        state_lambda_j: wp.array[wp.float32],
    ):
        # Retrieve the world and constraint
        # row index from the thread grid
        wid, rid = wp.tid()

        # Skip if the constraint row index exceeds the
        # maximum number of constraints for this world
        if rid >= mjw_nefc[wid]:
            return

        # Skip if the constraint row is not a limit joint
        if mjw_efc_type[wid, rid] != MjwConstraintType.EQUALITY:
            return

        # Retrieve the constraint id
        mj_efc_eq_id = mjw_efc_id[wid, rid]
        if mj_efc_eq_id < 0:
            return

        # TODO: IMPLEMENT THIS

    return _unpack_mjw_equalities_to_joint_reactions_kamino


@cache
def make_unpack_mjw_joint_limited_to_limits_kamino():
    """
    Makes a kernel to unpack mujoco_warp joint limited to limits.
    """
    from mujoco_warp._src.types import ConstraintType as MjwConstraintType

    @wp.kernel
    def _unpack_mjw_joint_limited_to_limits_kamino(
        # Inputs:
        mjc_jnt_to_newton_jnt: wp.array2d[wp.int32],
        mjc_jnt_dofadr: wp.array[wp.int32],
        mjw_nefc: wp.array[wp.int32],
        mjw_efc_type: wp.array2d[wp.int32],
        mjw_efc_id: wp.array2d[wp.int32],
        mjw_efc_pos: wp.array2d[wp.float32],
        mjw_efc_force: wp.array2d[wp.float32],
        model_time_dt: wp.array[wp.float32],
        # Outputs:
        limit_model_max: wp.array[wp.int32],
        limit_world_max: wp.array[wp.int32],
        limit_model_active: wp.array[wp.int32],
        limit_world_active: wp.array[wp.int32],
        limit_wid: wp.array[wp.int32],
        limit_lid: wp.array[wp.int32],
        limit_jid: wp.array[wp.int32],
        limit_dof: wp.array[wp.int32],
        limit_r_q: wp.array[wp.float32],
        limit_reaction: wp.array[wp.float32],
    ):
        ###
        # TODO: DOES THIS ALSO WORK FOR BALL JOINT LIMITS?
        ###

        # Retrieve the world and constraint
        # row index from the thread grid
        wid, rid = wp.tid()

        # Skip if the constraint row index exceeds the
        # maximum number of constraints for this world
        if rid >= mjw_nefc[wid]:
            return

        # Skip if the constraint row is not a limit joint
        if mjw_efc_type[wid, rid] != MjwConstraintType.LIMIT_JOINT:
            return

        # Retrieve the constraint id
        # NOTE: This is also the joint id for the limit joint
        mj_efc_jnt_id = mjw_efc_id[wid, rid]
        if mj_efc_jnt_id < 0:
            return

        # Retrieve the Newton joint id
        newton_jnt_id = mjc_jnt_to_newton_jnt[wid, mj_efc_jnt_id]
        if newton_jnt_id < 0:
            return

        # TODO: CHECK THE NEWTON JOINT TYPE AND SKIP IF IT IS NOT SUPPORTED BY KAMINO

        # Retrieve the DoF index
        dofid = mjc_jnt_dofadr[mj_efc_jnt_id]
        if dofid < 0:
            return

        # Retrieve the constraint position, i.e. the constraint violation
        pos = mjw_efc_pos[wid, rid]

        # Retrieve the max limits of the model and world
        model_max_limits = limit_model_max[0]
        world_max_limits = limit_world_max[wid]
        mlid = wp.atomic_add(limit_model_active, 0, 1)
        wlid = wp.atomic_add(limit_world_active, wid, 1)
        if mlid < model_max_limits and wlid < world_max_limits:
            # Store the limit data
            limit_wid[mlid] = wid
            limit_lid[mlid] = wlid
            limit_jid[mlid] = newton_jnt_id
            limit_dof[mlid] = dofid
            limit_r_q[mlid] = pos
            limit_reaction[mlid] = model_time_dt[wid] * mjw_efc_force[wid, rid]

    return _unpack_mjw_joint_limited_to_limits_kamino


@cache
def make_unpack_mjw_efc_to_lambdas_kamino():
    """
    Makes a kernel to unpack mujoco_warp constraint forces into Kamino constraint lambdas.
    """
    from mujoco_warp._src.types import ConstraintType as MjwConstraintType

    @wp.kernel
    def _unpack_mjw_efc_to_lambdas_kamino(
        # Inputs:
        newton_dof_to_body: wp.array[wp.int32],
        mjc_jnt_to_newton_jnt: wp.array2d[wp.int32],
        mjc_jnt_to_newton_dof: wp.array2d[wp.int32],
        mjc_dof_to_newton_dof: wp.array2d[wp.int32],
        mjc_eq_to_newton_eq: wp.array2d[wp.int32],
        mjc_eq_to_newton_jnt: wp.array2d[wp.int32],
        mjc_jnt_type: wp.array[wp.int32],
        mjc_jnt_qposadr: wp.array[wp.int32],
        mjc_jnt_dofadr: wp.array[wp.int32],
        mjc_jnt_bodyid: wp.array[wp.int32],
        mjc_eq_type: wp.array[wp.int32],
        # TODO: other mjc or SolverMuJoCo model arrays
        mjw_nefc: wp.array[wp.int32],
        mjw_efc_type: wp.array2d[wp.int32],
        mjw_efc_id: wp.array2d[wp.int32],
        mjw_efc_pos: wp.array2d[wp.float32],
        mjw_efc_margin: wp.array2d[wp.float32],
        mjw_efc_force: wp.array2d[wp.float32],
        model_time_dt: wp.array[wp.float32],
        # TODO: other ModelKamino arrays
        # Outputs:
        lambdas: wp.array[wp.float32],
    ):
        """
        All-in-one kernel to unpack mujoco_warp constraint forces into Kamino lambdas.
        """
        # Retrieve the world and constraint
        # row index from the thread grid
        wid, rid = wp.tid()

        # Skip if the constraint row index exceeds the
        # maximum number of constraints for this world
        if rid >= mjw_nefc[wid]:
            return

        # Retrieve the constraint id and skip if it is invalid
        mj_efc_id = mjw_efc_id[wid, rid]
        if mj_efc_id < 0:
            return

        # Unpack the constraint force into a Kamino lambda
        if mjw_efc_type[wid, rid] == MjwConstraintType.EQUALITY:
            pass  # TODO: IMPLEMENT THIS WITH HELPER FUNCTION

        elif mjw_efc_type[wid, rid] == MjwConstraintType.LIMIT_JOINT:
            pass  # TODO: IMPLEMENT THIS WITH HELPER FUNCTION

        elif mjw_efc_type[wid, rid] == MjwConstraintType.FRICTION_DOF:
            pass  # TODO: IMPLEMENT THIS WITH HELPER FUNCTION

        elif mjw_efc_type[wid, rid] == MjwConstraintType.CONTACT_ELLIPTIC:
            pass  # TODO: IMPLEMENT THIS WITH HELPER FUNCTION


###
# Launchers
###


def unpack_mjw_joint_limited_to_limits_kamino(
    solver: SolverBase,
    model_kamino: ModelKamino,
    limits_kamino: LimitsKamino,
):
    """
    TODO
    """
    from newton._src.solvers.mujoco.solver_mujoco import SolverMuJoCo  # noqa: PLC0415

    if not isinstance(solver, SolverMuJoCo):
        raise TypeError(f"`solver` must be an instance of `newton.solvers.SolverMuJoCo`; got {type(solver).__name__}.")
    if solver.use_mujoco_cpu:
        raise NotImplementedError(
            "unpack_mjw_joint_limited_to_limits_kamino only supports the "
            "mujoco_warp GPU backend (SolverMuJoCo(use_mujoco_cpu=False))."
        )

    kernel = make_unpack_mjw_joint_limited_to_limits_kamino()
    wp.launch(
        kernel=kernel,
        dim=(model_kamino.info.num_worlds, solver.mjw_model.njmax),
        inputs=[
            solver.mjc_jnt_to_newton_jnt,
            solver.mjw_model.jnt_dofadr,
            solver.mjw_data.nefc,
            solver.mjw_data.efc.type,
            solver.mjw_data.efc.id,
            solver.mjw_data.efc.pos,
            solver.mjw_data.efc.force,
            model_kamino.time.dt,
            limits_kamino.data.model_max_limits,
            limits_kamino.data.world_max_limits,
        ],
        outputs=[
            limits_kamino.data.model_active_limits,
            limits_kamino.data.world_active_limits,
            limits_kamino.data.wid,
            limits_kamino.data.lid,
            limits_kamino.data.jid,
            limits_kamino.data.dof,
            limits_kamino.data.r_q,
            limits_kamino.data.reaction,
        ],
        device=model_kamino.device,
    )


@wp.kernel
def populate_joint_parent_f_from_body_parent_f_kernel(
    model_joint_parent: wp.array[int],
    model_joint_child: wp.array[int],
    model_joint_articulation: wp.array[wp.int32],
    state_body_parent_f: wp.array[wp.spatial_vector],
    state_joint_parent_f: wp.array[wp.spatial_vector],
):
    jnt = wp.tid()
    # no articulation - no body_parent_f
    if model_joint_articulation[jnt] == -1:
        return
    body = model_joint_child[jnt]
    # translate
    state_joint_parent_f[jnt] = state_body_parent_f[body]


def populate_joint_parent_f_from_body_parent_f(
    solver: SolverBase,
    model: Model,
    state: State,
):
    wp.launch(
        populate_joint_parent_f_from_body_parent_f_kernel,
        model.joint_count,
        inputs=[model.joint_parent, model.joint_child, model.joint_articulation, state.body_parent_f],
        outputs=[state.joint_parent_f],
    )


def populate_joint_parent_f_from_mjw_connect_equalities(
    solver: SolverBase,
    model: Model,
    state: State,
    *,
    connect_constraint_force: wp.array | None = None,
):
    """
    Populates ``state.joint_parent_f`` with the per-joint constraint wrench induced
    by MuJoCo CONNECT equality constraints (loop-closure ball joints).

    The output convention matches :attr:`newton.State.joint_parent_f`: world frame,
    moment referenced to the child body's center of mass. Joints not associated with
    a CONNECT equality are left untouched -- the caller is expected to zero
    ``state.joint_parent_f`` (and the optional ``connect_constraint_force`` buffer)
    before invocation.

    Args:
        solver: The :class:`SolverMuJoCo` instance whose ``mjw_data.efc.force`` is read.
        model: The Newton :class:`Model` providing joint topology and body COMs.
        state: The Newton :class:`State` whose ``joint_parent_f`` is accumulated into.
            Must have been allocated via ``Model.request_state_attributes("joint_parent_f")``.
        connect_constraint_force: Optional pre-allocated per-equality scratch buffer
            of shape ``(nworld, neq)`` and dtype ``wp.spatial_vectorf``. If ``None``,
            a fresh buffer is allocated on the model's device.
    """
    from newton._src.solvers.mujoco.solver_mujoco import SolverMuJoCo  # noqa: PLC0415

    if not isinstance(solver, SolverMuJoCo):
        raise TypeError(f"`solver` must be an instance of `newton.solvers.SolverMuJoCo`; got {type(solver).__name__}.")
    if solver.use_mujoco_cpu:
        raise NotImplementedError("populate_joint_parent_f_from_mjw_connect_equalities only supports mjwarp")

    if state.joint_parent_f is None or model.joint_count == 0:
        return

    nworld = solver.mjw_data.nworld
    neq = solver.mjw_model.neq
    if neq == 0:
        return

    if connect_constraint_force is None:
        connect_constraint_force = wp.zeros(shape=(nworld, neq), dtype=wp.spatial_vectorf, device=model.device)
    else:
        connect_constraint_force.zero_()

    wp.launch(
        kernel=_extract_mjw_connect_constraint_force,
        dim=(nworld, neq),
        inputs=[
            solver.mjw_model.eq_type,
            solver.mjw_data.efc.type,
            solver.mjw_data.efc.id,
            solver.mjw_data.efc.force,
            solver.mjw_data.ne,
        ],
        outputs=[connect_constraint_force],
        device=model.device,
    )

    wp.launch(
        kernel=_accumulate_mjw_connect_force_to_joint_parent_f,
        dim=(nworld, neq),
        inputs=[
            solver.mjc_eq_to_newton_jnt,
            solver.mjw_model.eq_data,
            connect_constraint_force,
            model.joint_parent,
            model.joint_child,
            state.body_q,
            model.body_com,
        ],
        outputs=[state.joint_parent_f],
        device=model.device,
    )


###
# Extractors
###


def extract_joint_wrenches_solvermujoco(
    solver: SolverMuJoCo,
    model: Model,
    state: State,
):
    if getattr(state, "joint_parent_f", None) is None:
        state.joint_parent_f = wp.zeros_like(model.joint_count, dtype=wp.spatial_vector)

    # Populate proper joints from body_parent_f
    populate_joint_parent_f_from_body_parent_f(solver, model, state)
    # Populate inequality constraints from mjwarp dta
    populate_joint_parent_f_from_mjw_connect_equalities(solver, model, state)


def extract_constraint_reactions_mujoco_warp(
    # Inputs:
    solver: SolverBase,
    model_kamino: ModelKamino,
    # Outputs:
    lambdas_kamino: wp.array,
    limits_kamino: LimitsKamino | None = None,
) -> None:
    """
    TODO
    """
    # Import SolverMuJoCo from upstream newton to check model assumptions
    from ......_src.solvers.mujoco.solver_mujoco import SolverMuJoCo  # noqa: PLC0415

    # Check that the solver is a SolverMuJoCo instance and that all assumptions are met, namely:
    # - The solver is a SolverMuJoCo instance
    # - The solver is using the mujoco_warp GPU backend
    # - The solver is using ELLIPTIC friction cones
    if not isinstance(solver, SolverMuJoCo):
        raise TypeError(f"`solver` must be an instance of `newton.solvers.SolverMuJoCo`; got {type(solver).__name__}.")
    if solver.use_mujoco_cpu:
        raise NotImplementedError(
            "extract_mujoco_warp_constraint_forces only supports the "
            "mujoco_warp GPU backend (SolverMuJoCo(use_mujoco_cpu=False))."
        )
    if int(solver.mjw_model.opt.cone) != int(mujoco_warp.ConeType.ELLIPTIC):
        raise NotImplementedError(
            "extract_mujoco_warp_constraint_forces currently supports only "
            f"ELLIPTIC friction cones;got cone type {solver.mjw_model.opt.cone}."
        )

    # Update limits container from MuJoCo limits if provided
    if limits_kamino is not None:
        unpack_mjw_joint_limited_to_limits_kamino(solver, model_kamino, limits_kamino)

    # Update lambdas array from MuJoCo Warp equality constraints
    # TODO
