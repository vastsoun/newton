# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Provides mechanisms to extract constraint reactions from SolverMuJoCo (MuJoCo Warp backend).
"""

from functools import cache

import warp as wp

from .....solvers.solver import SolverBase
from ..core.model import ModelKamino
from ..kinematics.limits import LimitsKamino

###
# Module interface
###

__all__ = [
    "extract_constraint_reactions_mujoco_warp",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Functions
###

# TODO: IMPLEMENT THIS: FUNCTIONS FOR EACH CONSTRAINT TYPE


###
# Kernels
###


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


###
# Extractors
###


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
    # # Use lazy imports so that this module does not require ``mujoco_warp``
    import mujoco_warp

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
