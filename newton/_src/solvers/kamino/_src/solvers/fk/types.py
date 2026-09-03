# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Defines data types used by the Forward Kinematics solver."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp

if TYPE_CHECKING:
    from newton._src.solvers.kamino._src.linalg.conjugate import BatchedLinearOperator  # noqa: F401
    from newton._src.solvers.kamino._src.linalg.sparse_matrix import BlockSparseMatrices  # noqa: F401

###
# Module interface
###

__all__ = [
    "FKData",
    "FKDimensions",
    "FKGaussNewtonData",
    "FKJointDoFType",
    "FKJointsModel",
    "FKLineSearchData",
    "FKLinearSystemData",
    "FKProblemData",
    "FKVelocitySolveData",
    "ForwardKinematicsPreconditionerType",
    "ForwardKinematicsStatus",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Types
###


class FKJointDoFType(IntEnum):
    """
    Joint dof types for the FK solver, which currently differs from Kamino's main joint types
    by the addition of the axis joint, used to regularize tie rods between two spherical joints
    (taking out the rotation dof about their own axis).

    Importantly, the integer value is the same for all joints not specific to FK, allowing seamless
    conversions.
    """

    FREE = 0
    REVOLUTE = 1
    PRISMATIC = 2
    CYLINDRICAL = 3
    UNIVERSAL = 4
    SPHERICAL = 5
    CARTESIAN = 6
    FIXED = 7
    GIMBAL = 8
    GIMBAL_LEFT_HANDED = 9
    AXIS = 10


class ForwardKinematicsPreconditionerType(IntEnum):
    """Conjugate gradient preconditioning options of the FK solver, if sparsity is enabled."""

    NONE = 0
    """No preconditioning"""

    JACOBI_DIAGONAL = 1
    """Diagonal Jacobi preconditioner"""

    JACOBI_BLOCK_DIAGONAL = 2
    """Blockwise-diagonal Jacobi preconditioner, alternating blocks of size 3 and 4 along the diagonal,
    corresponding to the position and orientation (quaternion) of individual rigid bodies."""

    @classmethod
    def from_string(cls, s: str) -> ForwardKinematicsPreconditionerType:
        """Converts a string to a ForwardKinematicsPreconditionerType enum value."""
        try:
            return cls[s.upper()]
        except KeyError as e:
            raise ValueError(
                f"Invalid ForwardKinematicsPreconditionerType: {s}. Valid options are: {[e.name for e in cls]}"
            ) from e


@dataclass
class ForwardKinematicsStatus:
    """
    Container holding detailed information on the success/failure status of a forward kinematics solve.
    """

    success: np.ndarray(dtype=np.int32)
    """
    Solver success flag per world, as an integer array (0 = failure, 1 = success).
    Shape of `(num_worlds,)`.

    Note that in some cases the solver may fail to converge within the maximum number
    of iterations, but still produce a solution with a reasonable residual.
    In such cases, the success flag will be set to 0, but the `max_residual` field
    can be inspected to check the actual residuals and determine if the solution is acceptable
    for the intended application.
    """

    iterations: np.ndarray(dtype=np.int32)
    """
    Number of Gauss-Newton iterations executed per world.
    Shape of `(num_worlds,)`.
    """

    max_residual: np.ndarray(dtype=np.float32)
    """
    Maximal absolute residual at the final solution, per world. In the general case, the residual vector
    is the kinematic constraints vector; if regularization is enabled, it is the penalty gradient.

    Shape of `(num_worlds,)`.
    """


###
# Internal solver data (FKData and sub-containers)
###


@dataclass
class FKDimensions:
    """Sizes, index maps and world/tile bookkeeping used across the FK solver."""

    # Scalar counts derived at finalize time
    num_bodies_max: int = 0  # Max number of bodies across worlds
    num_states_tot: int = 0  # State dims (7 * num bodies) for the whole model
    num_states_max: int = 0  # Max state dim across worlds
    num_joints_tot: int = 0  # Total joints (FK model, including axis joints)
    num_joints_max: int = 0  # Max joints across worlds
    num_axis_joints: int = 0  # Number of FK-specific axis joints across worlds
    num_actuated_coords: int = 0  # Total actuated coords across worlds
    num_actuated_coords_max: int = 0  # Max actuated coords across worlds
    num_actuated_dofs: int = 0  # Total actuated dofs across worlds
    num_constraints_max: int = 0  # Max kinematic constraints across worlds

    # Tile sizes for tile-based kernels
    tile_size_cts_1d: int = 0
    num_tiles_cts_1d: int = 0
    tile_size_vrs_1d: int = 0
    num_tiles_vrs_1d: int = 0
    tile_size_cts_2d: int = 0
    num_tiles_cts_2d: int = 0
    tile_size_vrs_2d: int = 0
    num_tiles_vrs_2d: int = 0
    tile_size_coords: int = 0
    num_tiles_coords: int = 0

    # Warp int32 arrays with per-world counts and offsets
    num_joints: wp.array | None = None  # Number of joints per world; shape (num_worlds,)
    joint_offset: wp.array | None = None  # Joint offset per world; shape (num_worlds + 1,)
    num_states: wp.array | None = None  # Number of states per world; shape (num_worlds,)
    num_constraints: wp.array | None = None  # Constraints per world; shape (num_worlds,)
    actuated_coord_offset: wp.array | None = None  # Per-joint actuated coord offset; shape (num_joints_tot + 1,)
    world_actuated_coord_offset: wp.array | None = None  # Per-world actuated coord offset; shape (num_worlds + 1,)
    actuated_coords_map: wp.array | None = None  # Actuated coord id -> global joint coord id
    actuated_dof_offset: wp.array | None = None  # Per-joint actuated dof offset; shape (num_joints_tot + 1,)
    actuated_dofs_map: wp.array | None = None  # Actuated dof id -> global joint dof id
    constraint_full_to_red_map: wp.array | None = None  # Full to reduced constraint index map

    # Sparse Jacobian tile pattern
    tile_sparsity_pattern: wp.array | None = None  # Nonzero tile indicators per world; shape (num_worlds, T_c, T_v)
    rb_nzb_id: wp.array | None = None  # Rigid body row nzb id per body; shape (num_bodies_tot,)
    ct_nzb_id_base: wp.array | None = None  # Constraint nzb id (base side); shape (6 * num_joints_tot,)
    ct_nzb_id_follower: wp.array | None = None  # Constraint nzb id (follower side); shape (6 * num_joints_tot,)

    # Convenience mask covering all worlds
    all_worlds_mask: wp.array | None = None  # True for all worlds; shape (num_worlds,)


@dataclass
class FKJointsModel:
    """FK-local joints model (may differ from the main model's joints; adds axis joints and base joint)."""

    dof_type: wp.array | None = None  # Joint dof type per joint; shape (num_joints_tot,)
    act_type: wp.array | None = None  # Joint actuation type per joint; shape (num_joints_tot,)
    bid_B: wp.array | None = None  # Base body id per joint; shape (num_joints_tot,)
    bid_F: wp.array | None = None  # Follower body id per joint; shape (num_joints_tot,)
    B_r_Bj: wp.array | None = None  # Joint local origin on base body; shape (num_joints_tot,)
    F_r_Fj: wp.array | None = None  # Joint local origin on follower body; shape (num_joints_tot,)
    X_Bj: wp.array | None = None  # Joint local frame on base body; shape (num_joints_tot,)
    X_Fj: wp.array | None = None  # Joint local frame on follower body; shape (num_joints_tot,)
    source_id: wp.array | None = None  # Corresponding joint id in the main model (-1 for FK-only joints)
    world_id: wp.array | None = None  # World id per joint; shape (num_joints_tot,)
    axis_joint: wp.array | None = None  # FK axis joint ids across worlds; shape (num_axis_joints,)
    axis_body: wp.array | None = None  # Body id targeted by each axis joint; shape (num_axis_joints,)
    axis_source_joint_0: wp.array | None = None  # First source joint id per axis joint
    axis_source_joint_1: wp.array | None = None  # Second source joint id per axis joint
    base_joint_id: wp.array | None = None  # Base joint id per world (-1 = none); shape (num_worlds,)
    has_universal_actuators: bool = False  # True iff the model has at least one actuated universal joint
    _built_actuated: wp.array | None = None  # Built-time actuated flags per joint; shape (num_joints_tot,)
    _actuation_violations: wp.array | None = None  # Violations counter for model-change validation


@dataclass
class FKProblemData:
    """Problem inputs, current estimates, and constraint/Jacobian storage for the FK Gauss-Newton solve."""

    actuator_q_next: wp.array | None = None  # Target actuated coords for the FK model; shape (num_actuated_coords,)
    actuator_q_prev: wp.array | None = None  # Previous actuated coords (incremental solve)
    actuator_q_curr: wp.array | None = None  # Current incremental target actuated coords
    target_rel_transforms: wp.array | None = None  # Target relative transforms per joint; shape (num_joints_tot,)
    body_q_ref: wp.array | None = None  # Reference body poses for the regularizer; shape (num_bodies_tot,)
    base_q_default: wp.array | None = None  # Default base pose per world; shape (num_worlds,)
    base_u_default: wp.array | None = None  # Default base twist per world; shape (num_worlds,)
    constraints: wp.array | None = None  # Kinematic constraints vector; shape (num_worlds, num_constraints_max)
    jacobian: wp.array | None = None  # Dense constraints Jacobian; shape (num_worlds, C_max, S_max)
    sparse_jacobian: Any = None  # Sparse constraints Jacobian (BlockSparseMatrices)
    sparse_jacobian_op: Any = None  # Sparse Jacobian linear operator


@dataclass
class FKGaussNewtonData:
    """State and companion arrays driving the outer Gauss-Newton loop."""

    iteration: wp.array | None = None  # Iteration count per world; shape (num_worlds,)
    success: wp.array | None = None  # Convergence flag per world; shape (num_worlds,)
    mask: wp.array | None = None  # Continue-iterating flag per world; shape (num_worlds,)
    loop_condition: wp.array | None = None  # Global loop condition; shape (1,)
    min_iterations: wp.array | None = None  # Min iterations per world; shape (num_worlds,)
    max_iterations: wp.array | None = None  # Max iterations (shared); shape (1,)
    tolerance: wp.array | None = None  # Tolerance on max residual (shared); shape (1,)
    jacobian_early_update_mask: wp.array | None = None  # Optional: worlds needing early Jacobian update
    jacobian_late_update_mask: wp.array | None = None  # Optional: worlds needing late Jacobian update
    delta_q_max: wp.array | None = None  # Max step in actuated coords (incremental solve)
    grad: wp.array | None = None  # Merit function gradient w.r.t. state; shape (num_worlds, S_max)
    step: wp.array | None = None  # State step per world; shape (num_worlds, S_max)
    max_residual: wp.array | None = None  # Max constraint / gradient residual per world; shape (num_worlds,)


@dataclass
class FKLineSearchData:
    """State and companion arrays driving the inner backtracking line search."""

    iteration: wp.array | None = None  # Iteration count per world; shape (num_worlds,)
    success: wp.array | None = None  # Line-search success flag per world; shape (num_worlds,)
    mask: wp.array | None = None  # Continue-iterating flag per world; shape (num_worlds,)
    loop_condition: wp.array | None = None  # Global loop condition; shape (1,)
    max_iterations: wp.array | None = None  # Max iterations (shared); shape (1,)
    alpha: wp.array | None = None  # Current step size per world; shape (num_worlds,)
    body_q_alpha: wp.array | None = None  # Trial body poses at alpha; shape (num_bodies_tot,)
    val_0: wp.array | None = None  # Merit function value at 0; shape (num_worlds,)
    val_alpha: wp.array | None = None  # Merit function value at alpha; shape (num_worlds,)
    grad_0: wp.array | None = None  # Merit function directional gradient at 0; shape (num_worlds,)


@dataclass
class FKLinearSystemData:
    """Left/right-hand sides, temporaries and preconditioner for the linear system in each Newton step."""

    lhs: wp.array | None = None  # Dense Gauss-Newton LHS per world; shape (num_worlds, S_max, S_max)
    rhs: wp.array | None = None  # Gauss-Newton RHS per world (= -grad); shape (num_worlds, S_max)
    jacobian_times_vector: wp.array | None = None  # J * x intermediary; shape (num_worlds, C_max)
    lhs_times_vector: wp.array | None = None  # (J^T J + regI) * x intermediary; shape (num_worlds, S_max)
    jacobian_diag_inv: wp.array | None = None  # Diagonal Jacobi preconditioner (CG); shape (num_worlds, S_max)
    inv_blocks_3: wp.array | None = None  # Block-diagonal preconditioner 3x3 blocks
    inv_blocks_4: wp.array | None = None  # Block-diagonal preconditioner 4x4 blocks
    cg_atol: wp.array | None = None  # Absolute tolerance per world (CG); shape (num_worlds,)
    cg_rtol: wp.array | None = None  # Relative tolerance per world (CG); shape (num_worlds,)
    cg_max_iter: wp.array | None = None  # Max iterations per world (CG); shape (num_worlds,)
    preconditioner_type: ForwardKinematicsPreconditionerType = ForwardKinematicsPreconditionerType.NONE


@dataclass
class FKVelocitySolveData:
    """Preallocated buffers for one batch size of the post-FK velocity solve."""

    fk_actuator_u: wp.array | None = None  # Actuated dof velocities (FK model); shape (batch, num_actuated_dofs)
    target_cts_u: wp.array | None = None  # Target constraint velocities; shape (num_worlds, C_max, batch)
    rhs: wp.array | None = None  # Body-space RHS; shape (num_worlds, S_max, batch)
    body_q_dot: wp.array | None = None  # Body pose time derivatives; shape (num_worlds, S_max, batch)


@dataclass
class FKData:
    """Aggregated internal data of the FK solver, split into coherent sub-containers."""

    dimensions: FKDimensions = field(default_factory=FKDimensions)
    joints: FKJointsModel = field(default_factory=FKJointsModel)
    problem: FKProblemData = field(default_factory=FKProblemData)
    gauss_newton: FKGaussNewtonData = field(default_factory=FKGaussNewtonData)
    line_search: FKLineSearchData = field(default_factory=FKLineSearchData)
    linear_system: FKLinearSystemData = field(default_factory=FKLinearSystemData)
    velocity_solve: dict[int, FKVelocitySolveData] = field(default_factory=dict)
