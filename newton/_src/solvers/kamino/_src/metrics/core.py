# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
TODO
"""

import warp as wp

from .....sim import Contacts, Control, Model, State
from ..core.bodies import update_body_inertias, update_body_wrenches
from ..core.control import ControlKamino
from ..core.data import DataKamino
from ..core.joints import JointDoFType
from ..core.math import (
    concat6d,
    expand6d,
    screw_transform_matrix_from_points,
)
from ..core.model import ModelKamino
from ..core.state import StateKamino
from ..core.types import float32, int32, mat33f, transformf, vec6f
from ..dynamics.dual import DualProblem
from ..dynamics.wrenches import (
    compute_constraint_body_wrenches,
    compute_joint_dof_body_wrenches,
)
from ..geometry.contacts import ContactsKamino, convert_contacts_newton_to_kamino
from ..kinematics.constraints import (
    make_unilateral_constraints_info,
    update_constraints_info,
)
from ..kinematics.jacobians import (
    DenseSystemJacobians,
    SparseSystemJacobians,
    SystemJacobiansType,
    compute_intermediate_body_frame_universal_joint,
    compute_joint_relative_quaternion,
)
from ..kinematics.joints import compute_joints_data
from ..kinematics.limits import LimitsKamino
from ..kinematics.velocities import compute_constraint_space_velocities
from ..solvers.metrics import SolutionMetrics

###
# Module interface
###

__all__ = ["SolutionMetricsNewton"]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Functions
###


@wp.func
def joint_dof_axis_from_index(dof_type: int32, dof_within_joint: int32) -> int32:
    """
    Maps a joint's local DoF index (i.e. ``dof_within_joint``) to the corresponding
    6D axis index in the joint frame, based on the joint's DoF type.
    """
    if dof_type == JointDoFType.REVOLUTE:
        return 3
    elif dof_type == JointDoFType.PRISMATIC:
        return 0
    elif dof_type == JointDoFType.CYLINDRICAL:
        # CYLINDRICAL DoFs are: T_x (axis 0), R_x (axis 3)
        if dof_within_joint == 0:
            return 0
        return 3
    elif dof_type == JointDoFType.UNIVERSAL:
        return 3 + dof_within_joint
    elif dof_type == JointDoFType.SPHERICAL:
        return 3 + dof_within_joint
    elif dof_type == JointDoFType.GIMBAL:
        return 3 + dof_within_joint
    elif dof_type == JointDoFType.CARTESIAN:
        return dof_within_joint
    elif dof_type == JointDoFType.FREE:
        return dof_within_joint
    return -1


def make_typed_write_joint_kinematic_lambdas(dof_type: JointDoFType):
    """
    Generates a per-joint-type Warp function that writes the kinematic-constraint
    Lagrange multipliers of a single joint into the global ``state_lambda_j`` array.
    """
    cts_axes = dof_type.cts_axes
    num_cts = dof_type.num_cts

    @wp.func
    def _typed_write_joint_kinematic_lambdas(
        cts_offset_j: int32,
        j_w_j: vec6f,
        state_lambda_j: wp.array[float32],
    ):
        for k in range(num_cts):
            state_lambda_j[cts_offset_j + k] = j_w_j[cts_axes[k]]

    return _typed_write_joint_kinematic_lambdas


def make_write_joint_kinematic_lambdas():
    """
    Generates a Warp function that dispatches the per-joint-type writer of the
    joint's kinematic-constraint Lagrange multipliers, based on the joint's DoF type.
    """

    @wp.func
    def _write_joint_kinematic_lambdas(
        dof_type: int32,
        cts_offset_j: int32,
        j_w_j: vec6f,
        state_lambda_j: wp.array[float32],
    ):
        if dof_type == JointDoFType.REVOLUTE:
            wp.static(make_typed_write_joint_kinematic_lambdas(JointDoFType.REVOLUTE))(
                cts_offset_j, j_w_j, state_lambda_j
            )
        elif dof_type == JointDoFType.PRISMATIC:
            wp.static(make_typed_write_joint_kinematic_lambdas(JointDoFType.PRISMATIC))(
                cts_offset_j, j_w_j, state_lambda_j
            )
        elif dof_type == JointDoFType.CYLINDRICAL:
            wp.static(make_typed_write_joint_kinematic_lambdas(JointDoFType.CYLINDRICAL))(
                cts_offset_j, j_w_j, state_lambda_j
            )
        elif dof_type == JointDoFType.UNIVERSAL:
            wp.static(make_typed_write_joint_kinematic_lambdas(JointDoFType.UNIVERSAL))(
                cts_offset_j, j_w_j, state_lambda_j
            )
        elif dof_type == JointDoFType.SPHERICAL:
            wp.static(make_typed_write_joint_kinematic_lambdas(JointDoFType.SPHERICAL))(
                cts_offset_j, j_w_j, state_lambda_j
            )
        elif dof_type == JointDoFType.GIMBAL:
            wp.static(make_typed_write_joint_kinematic_lambdas(JointDoFType.GIMBAL))(
                cts_offset_j, j_w_j, state_lambda_j
            )
        elif dof_type == JointDoFType.CARTESIAN:
            wp.static(make_typed_write_joint_kinematic_lambdas(JointDoFType.CARTESIAN))(
                cts_offset_j, j_w_j, state_lambda_j
            )
        elif dof_type == JointDoFType.FIXED:
            wp.static(make_typed_write_joint_kinematic_lambdas(JointDoFType.FIXED))(cts_offset_j, j_w_j, state_lambda_j)
        # FREE: no kinematic constraints; nothing to write

    return _write_joint_kinematic_lambdas


###
# Kernels
###


@wp.kernel
def _compute_joint_wrenches_from_body_parent_wrenches(
    # Inputs:
    model_joints_dof_type: wp.array[int32],
    model_joints_kinematic_cts_offset_joint_cts: wp.array[int32],
    model_joints_bid_F: wp.array[int32],
    model_joints_bid_B: wp.array[int32],
    model_joints_X_j: wp.array[mat33f],
    data_joints_p_j: wp.array[transformf],
    data_bodies_q_i: wp.array[transformf],
    state_body_parent_f: wp.array[wp.spatial_vectorf],
    # Outputs:
    data_joints_j_w_j: wp.array[vec6f],
    state_lambda_j: wp.array[float32],
):
    # Retrieve the thread index as the joint index
    jid = wp.tid()

    # Retrieve the joint model data
    dof_type = model_joints_dof_type[jid]

    # Skip FREE joints: they have no kinematic constraints and `body_parent_f`
    # is not accumulated for FREE joints (see `_convert_joint_wrenches_to_body_parent_wrenches`).
    if dof_type == JointDoFType.FREE:
        return

    # Retrieve the body indices of the joint
    bid_F = model_joints_bid_F[jid]
    bid_B = model_joints_bid_B[jid]

    # Retrieve the joint frame pose (in world coords)
    T_j = data_joints_p_j[jid]
    r_j = wp.transform_get_translation(T_j)
    R_X_j = wp.quat_to_matrix(wp.transform_get_rotation(T_j))

    # Retrieve the follower body's pose (CoM in world coords)
    T_F_j = data_bodies_q_i[bid_F]
    r_F_j = wp.transform_get_translation(T_F_j)

    # Compute the inverse wrench-transform from the follower CoM to the joint frame.
    # Since `W_j_F = screw_transform_matrix_from_points(r_j, r_F_j)` transforms a wrench
    # from the joint frame to the body's CoM, its inverse swaps the role of the two points.
    inv_W_j_F = screw_transform_matrix_from_points(r_F_j, r_j)

    # General case: 6D extension of the constant joint-frame rotation matrix
    if dof_type != JointDoFType.UNIVERSAL:
        R_X_bar_j = expand6d(R_X_j)
    # Universal joint: replace R_X_j with the frame of the intermediate body for rotation constraints
    else:
        # The base body's pose is needed to compute the relative quaternion;
        # for unary joints (bid_B == -1), use the world identity transform.
        T_B_j = wp.transform_identity()
        if bid_B > -1:
            T_B_j = data_bodies_q_i[bid_B]
        j_q_j = compute_joint_relative_quaternion(T_B_j, T_F_j, model_joints_X_j[jid])
        R_intermediate = compute_intermediate_body_frame_universal_joint(j_q_j)
        R_X_bar_j = concat6d(R_X_j, R_X_j @ R_intermediate)

    # Read the world-frame wrench applied on body F by joint j (at body F's CoM).
    w_ij_sv = state_body_parent_f[bid_F]
    w_ij = vec6f(w_ij_sv[0], w_ij_sv[1], w_ij_sv[2], w_ij_sv[3], w_ij_sv[4], w_ij_sv[5])

    # Transform the wrench from body-F CoM to the joint frame (world-aligned),
    # then express it in the joint-local frame.
    w_j = inv_W_j_F @ w_ij
    j_w_j = wp.transpose(R_X_bar_j) @ w_j

    # Store the joint-local wrench
    data_joints_j_w_j[jid] = j_w_j

    # Write the kinematic-constraint Lagrange multipliers for this joint
    cts_offset_j = model_joints_kinematic_cts_offset_joint_cts[jid]
    wp.static(make_write_joint_kinematic_lambdas())(dof_type, cts_offset_j, j_w_j, state_lambda_j)


@wp.kernel
def _compute_limit_reactions_from_joint_wrenches(
    # Inputs:
    model_joints_dof_type: wp.array[int32],
    model_joints_dofs_offset: wp.array[int32],
    limits_model_num: wp.array[int32],
    limits_model_max: int32,
    limits_jid: wp.array[int32],
    limits_dof: wp.array[int32],
    limits_side: wp.array[float32],
    data_joints_j_w_j: wp.array[vec6f],
    control_tau_j: wp.array[float32],
    # Outputs:
    limits_reaction: wp.array[float32],
):
    # Retrieve the limit index from the thread grid
    lid = wp.tid()

    # Skip if lid is greater than the number of active limits in the model
    if lid >= wp.min(limits_model_num[0], limits_model_max):
        return

    # Retrieve the joint and DoF indices for this active limit
    jid = limits_jid[lid]
    dof_l = limits_dof[lid]
    side_l = limits_side[lid]

    # Map the global DoF index to the joint-local DoF index, then to the 6D joint-frame axis
    dof_within_joint = dof_l - model_joints_dofs_offset[jid]
    axis = joint_dof_axis_from_index(model_joints_dof_type[jid], dof_within_joint)

    # Recover the limit reaction: the joint-frame total wrench at the DoF axis is
    # `tau_total = tau_actuation + side * lambda_l`, so `lambda_l = side * (tau_total - tau_actuation)`.
    j_w_j = data_joints_j_w_j[jid]
    tau_total = j_w_j[axis]
    tau_act = control_tau_j[dof_l]
    limits_reaction[lid] = side_l * (tau_total - tau_act)


###
# Launchers
###


def convert_body_parent_wrenches_to_joint_reactions(
    model: ModelKamino,
    data: DataKamino,
    limits: LimitsKamino | None,
    control: ControlKamino,
    body_parent_f: wp.array,
    state_lambda_j: wp.array,
):
    """
    Inverts the joint-wrench accumulation to recover the joint constraint
    Lagrange multipliers (``state.lambda_j``) and joint-limit reactions
    (``limits.reaction``) from the per-body parent wrenches ``body_parent_f``.

    As a byproduct, the per-joint wrench in joint-local coordinates is stored
    in ``data.joints.j_w_j``.

    Args:
        model: The model containing the time-invariant data of the simulation.
        data: The solver data container holding the time-varying data of the simulation.
        limits: The active joint-limits container. Optional; if ``None`` (or empty),
            ``limits.reaction`` is not updated.
        control: The control inputs container, used to read joint actuation forces ``tau_j``.
        body_parent_f: The input array of per-body parent wrenches (world frame, at body CoM).
        state_lambda_j: The output array of joint constraint Lagrange multipliers.
    """
    if model.size.sum_of_num_joints == 0:
        return

    wp.launch(
        kernel=_compute_joint_wrenches_from_body_parent_wrenches,
        dim=model.size.sum_of_num_joints,
        inputs=[
            model.joints.dof_type,
            model.joints.kinematic_cts_offset_joint_cts,
            model.joints.bid_F,
            model.joints.bid_B,
            model.joints.X_j,
            data.joints.p_j,
            data.bodies.q_i,
            body_parent_f,
        ],
        outputs=[data.joints.j_w_j, state_lambda_j],
        device=model.device,
    )

    if limits is not None and limits.model_max_limits_host > 0:
        wp.launch(
            kernel=_compute_limit_reactions_from_joint_wrenches,
            dim=limits.model_max_limits_host,
            inputs=[
                model.joints.dof_type,
                model.joints.dofs_offset,
                limits.model_active_limits,
                limits.model_max_limits_host,
                limits.jid,
                limits.dof,
                limits.side,
                data.joints.j_w_j,
                control.tau_j,
            ],
            outputs=[limits.reaction],
            device=model.device,
        )


###
# Interfaces
###


class SolutionMetricsNewton:
    """
    SolutionMetrics wrapper to interface with Newton's front-end API.
    """

    def __init__(
        self,
        dt: float | None = None,
        model: Model | None = None,
        model_kamino: ModelKamino | None = None,
        sparse: bool = True,
    ):
        """
        Initializes the SolutionMetricsNewton wrapper.

        Args:
            dt: The time-step size of the simulation.
            model: The model container holding the time-invariant data of the simulation.
            model_kamino: The Kamino model container holding the time-invariant data of the simulation.
        """
        # Declare internal Kamino data containers
        self._model: ModelKamino | None = None
        self._data: DataKamino | None = None
        self._limits: LimitsKamino | None = None
        self._contacts: ContactsKamino | None = None
        self._problem: DualProblem | None = None
        self._jacobians: SystemJacobiansType | None = None
        self._state: StateKamino | None = None
        self._state_p: StateKamino | None = None
        self._control: ControlKamino | None = None

        # Declare additional buffers for metrics computations
        self._v_plus: wp.array | None = None
        self._lambdas: wp.array | None = None
        self._sigma: wp.array | None = None

        # Declare the metrics data container
        self._metrics: SolutionMetrics | None = None

        # If a model is provided, finalize the metrics data allocations
        if dt is not None and model is not None:
            self.finalize(dt=dt, model=model, model_kamino=model_kamino, sparse=sparse)

    ###
    # Properties
    ###

    @property
    def device(self) -> wp.DeviceLike:
        """
        Returns the device where the metrics data is allocated.
        """
        if self._model is None:
            raise RuntimeError("SolutionMetricsNewton data is not initialized. Call finalize() first.")
        return self._model.device

    ###
    # Operations
    ###

    def finalize(self, dt: float, model: Model, model_kamino: ModelKamino | None = None, sparse: bool = True):
        """
        Finalizes the SolutionMetricsNewton wrapper.

        Args:
            dt: The time-step size of the simulation.
            model: The model container holding the time-invariant data of the simulation.
            model_kamino: The Kamino model container holding the time-invariant data of the simulation.
        """
        # Ensure the model is valid
        if dt is None or not isinstance(dt, float):
            raise ValueError("Expected 'dt' argument to be a non-None float.")
        if model is None or not isinstance(model, Model):
            raise TypeError("Expected 'model' argument to be of type `newton.Model`")

        # If a model_kamino is provided, use it; otherwise, convert the model to Kamino format
        if model_kamino is not None:
            if not isinstance(model_kamino, ModelKamino):
                raise TypeError(
                    f"Expected 'model_kamino' argument to be of type `ModelKamino`, got {type(model_kamino)}"
                )
            self._model = model_kamino
        else:
            # TODO: We need a mechanism to force all joints being only kinematic, i.e. no dynamic constraints
            self._model = ModelKamino.from_newton(model=model, overwrite_source_model=False)

        # Configure model time-steps
        self._model.time.dt.fill_(wp.float32(dt))
        self._model.time.inv_dt.fill_(wp.float32(1.0 / dt))

        # Create the data, limits and contacts containers. ``joint_wrenches=True`` allocates
        # the per-joint local-frame wrench buffers (e.g. ``data.joints.j_w_j``) used by the
        # joint-reaction recovery in :meth:`_convert_body_parent_wrenches_to_joint_reactions`.
        self._data = self._model.data(joint_wrenches=True)
        self._limits = LimitsKamino(model=self._model)
        self._contacts = ContactsKamino(model=self._model)

        # Reset limits and contacts containers
        self._limits.reset()
        self._contacts.reset()

        # Create and finalize the control container
        self._control = ControlKamino()
        self._control.finalize(self._model)

        # Construct the unilateral constraints members in the model info
        make_unilateral_constraints_info(
            model=self._model, data=self._data, limits=self._limits, contacts=self._contacts
        )

        # Create the Jacobians container
        if sparse:
            self._jacobians = SparseSystemJacobians(model=self._model, limits=self._limits, contacts=self._contacts)
        else:
            self._jacobians = DenseSystemJacobians(model=self._model, limits=self._limits, contacts=self._contacts)

        # Create the DualProblem container
        problem_cfg = DualProblem.Config()
        problem_cfg.constraints.alpha = 0.0
        problem_cfg.constraints.beta = 0.0
        problem_cfg.constraints.gamma = 0.0
        problem_cfg.constraints.delta = 0.0
        problem_cfg.dynamics.preconditioning = False
        self._problem = DualProblem(
            model=self._model,
            data=self._data,
            limits=self._limits,
            contacts=self._contacts,
            jacobians=self._jacobians,
            config=problem_cfg,
            sparse=False,
        )

        # Finalize the internal SolutionMetrics instance
        self._metrics = SolutionMetrics(model=self._model)

        # Allocate metrics data on the target device
        with wp.ScopedDevice(self.device):
            self._v_plus = wp.zeros(self._model.size.sum_of_max_total_cts, dtype=wp.float32)
            self._lambdas = wp.zeros(self._model.size.sum_of_max_total_cts, dtype=wp.float32)
            self._sigma = wp.zeros(self._model.size.sum_of_max_total_cts, dtype=wp.float32)

    def evaluate(
        self,
        state: State,
        state_p: State,
        control: Control | None = None,
        contacts: Contacts | None = None,
    ):
        """
        Evaluates the solution metrics on the provided Newton state and control data.

        Args:
            state: The Newton state data containing the current state of the simulation.
            state_p: The previous state of the simulation.
            control: The Newton control data containing the current control inputs of the simulation.
            contacts: The Newton contacts data containing the current contacts of the simulation.
        """
        # Reset limits and contacts containers
        self._limits.reset()
        self._contacts.reset()

        # Interface the input state containers to Kamino's equivalents
        self._state = StateKamino.from_newton(self._model.size, self._model._model, state)
        self._state_p = StateKamino.from_newton(self._model.size, self._model._model, state_p)
        self._control.from_newton(control, self._model)
        convert_contacts_newton_to_kamino(self._model._model, state_p, contacts, self._contacts)

        # TODO: ENABLE THIS WHEN WE EXTEND TO SUPPORT JOINT LIMITS
        # # Run limit detection to generate active limits
        # self._limits.detect(q_j=self._state_p.q_j)

        # Update the relevant data fields of `DataKamino` and system Jacobians required
        # for the metrics computations, using the provided `StateKamino` instances.
        self._read_step_inputs(self._state_p, self._control)
        update_constraints_info(model=self._model, data=self._data)
        update_body_inertias(model=self._model.bodies, data=self._data.bodies)
        compute_joints_data(model=self._model, data=self._data, q_j_p=self._state_p.q_j)
        self._update_jacobians()
        self._update_dynamics()

        # Compute the post-event constraint-space velocities given
        # the pre- and post-event state and constraint Jacobians
        self._compute_postevent_constraint_velocities(
            model=self._model,
            state=self._state,
            state_p=self._state_p,
            jacobians=self._jacobians,
            v_plus=self._v_plus,
        )

        # Perform the necessary conversions and extractions to obtain the
        # solver data in the expected format for the metrics computations
        self._convert_body_parent_wrenches_to_joint_reactions(
            model=self._model,
            state_in=state,
            control_in=self._control,
            limits_out=self._limits,
            state_out=self._state,
            data_out=self._data,
        )
        self._extract_constraint_reactions(
            model=self._model,
            state=self._state,
            limits=self._limits,
            contacts=self._contacts,
            lambdas=self._lambdas,
        )

        # Update all dynamics quantities based
        # on the extracted constraint reactions
        self._read_step_inputs(self._state, self._control)
        self._update_body_wrenches()

        # # Evaluate the metrics using the extracted solver data
        # self._metrics.evaluate(
        #     sigma=self._sigma,
        #     lambdas=self._lambdas,
        #     v_plus=self._v_plus,
        #     data=self._data,
        #     state_p=self._state_p,
        #     problem=self._problem,
        #     jacobians=self._jacobians,
        #     limits=self._limits,
        #     contacts=self._contacts,
        # )

    ###
    # Internals
    ###

    def _read_step_inputs(self, state_in: StateKamino, control_in: ControlKamino):
        wp.copy(self._data.bodies.q_i, state_in.q_i)
        wp.copy(self._data.bodies.u_i, state_in.u_i)
        wp.copy(self._data.bodies.w_i, state_in.w_i)
        wp.copy(self._data.bodies.w_e_i, state_in.w_i_e)
        wp.copy(self._data.joints.q_j, state_in.q_j)
        wp.copy(self._data.joints.dq_j, state_in.dq_j)
        self._data.joints.tau_j = control_in.tau_j

    def _update_jacobians(self):
        self._jacobians.build(
            model=self._model,
            data=self._data,
            limits=self._limits,
            contacts=self._contacts,
            reset_to_zero=True,
        )

    def _update_dynamics(self):
        self._problem.build(
            model=self._model,
            data=self._data,
            limits=self._limits,
            contacts=self._contacts,
            jacobians=self._jacobians,
            reset_to_zero=True,
        )

    def _update_body_wrenches(self):
        # Compute the per-body actuation wrenches: `DataKamino.bodies.w_a_i`
        # in world coordinates from the current joint torques
        compute_joint_dof_body_wrenches(self._model, self._data, self._jacobians)

        # Compute the per-body constraint wrenches: `w_j_i`, `w_l_i`,
        # and `w_c_i` of `DataKamino.bodies` in world coordinates
        compute_constraint_body_wrenches(
            model=self._model,
            data=self._data,
            limits=self._limits,
            contacts=self._contacts,
            jacobians=self._jacobians,
            lambdas_offsets=self._problem.data.vio,
            lambdas_data=self._lambdas,
        )

        # Compute the total applied wrenches per body by summing up all individual contributions,
        # from joint actuation, joint limits, contacts, and purely external effects.
        update_body_wrenches(self._model.bodies, self._data.bodies)

    ###
    # TODO: TO BE IMPLEMENTED
    ###

    def _compute_postevent_constraint_velocities(
        self,
        model: ModelKamino,
        state: StateKamino,
        state_p: StateKamino,
        jacobians: SystemJacobiansType,
        v_plus: wp.array,
    ):
        """
        Computes the constraint-space velocities `v = J @ u`, where `J := J_cts` and `u := State.bodies.u_i`

        This realizes the `v^{+/-} = J(q^{+/-}) @ u^{+/-}` operation.

        Depending on whether we use the pre-event (-) or post-event (+) state,
        we will compute the respective constraint-space velocities as:
        - `v^{+} = J(q) @ u^{+}`, i.e. `v^{+} := v_plus`
        - `v^{-} = J(q) @ u^{-}`, i.e. `v^{-} := v_minus`
        - All computations also depend on whether the pre- or post-event coordinates are use
          to evaluate the constraint Jacobian `J(q)`. However, for the purposes of evaluating
          physical correctness, we will use the system coordinates `q` that are coincident
          with the given state.

        Args:
            model:
                The model containing the time-invariant data of the simulation.
            state:
                The input post-event state data containing the current state of the simulation.
            state_p:
                The input pre-event state data containing the initial state of the simulation.
            jacobians:
                The system Jacobians.
            v_plus:
                The output array to store the post-event constraint-space velocities.
        """
        compute_constraint_space_velocities(
            model=model,
            jacobians=jacobians,
            u=state.u_i,
            v_start=self._problem.data.vio,
            v=v_plus,
            reset_to_zero=True,
        )

    def _convert_body_parent_wrenches_to_joint_reactions(
        self,
        model: ModelKamino,
        state_in: State,
        control_in: ControlKamino,
        data_out: DataKamino,
        limits_out: LimitsKamino,
        state_out: StateKamino,
    ):
        """
        Converts Newton body-parent wrenches `newton.State.body_parent_f` data
        to Kamino `StateKamino.lambda_j` and `DataKamino.joints.lambda_l_j`.

        This operation also updates per-joint wrenches arrays `DataKamino.joints.j_w_j` as a byproduct.

        Definitions:
        - `body_parent_f` contains the wrench applied on each body by its parent body, referenced w.r.
           the child body's center of mass (COM) and expressed in the world frame (i.e. world coordinates).
           Each entry is equal to `w_ij`, the world wrench applied by parent body `i` joint `j`.
        - `w_j` is the wrench applied by joint `j` on its follower/child
           body, referenced w.r.t. the joint frame in world coordinates.
        - `j_w_j` is the wrench applied by joint `j` on its follower/child
          body, expressed in the local coordinates of the joint frame.
        - `lambda_j` contains the constraint reaction impulses
          applied by each joint, expressed in the joint frame.
        - `lambda_l_j` contains the joint-limit constraint reactions.
        - `tau_c_j` is the joint-space actuation generalized forces.
        - `tau_j` is the joint-space generaralized forces. However, as any acting joint-limit constraint
          reactions also lie in the same space (i.e. DoF-space), we will consider this to be equal to
          the total joint-space generalized forces `tau_j := tau_c_j + lambda_l_j`
        - `dt` is the simulation time step.

        The conversion is performed parallel over joints as follows:
        - We use the relation `w_j = inv(W_ij) @ w_ij` to compute `w_j`, i.e. the joint wrench
          referenced w.r.t. the joint frame in world coordinates, where `W_ij` is the `6x6` wrench
          transform matrix transforming `w_j` from the joint frame to the COM frame of body `i`.
          When body `i` is the  follower/child we use the absolute pose of the body and joint
          frames to compute `W_ij`.
        - Having `w_j`, we compute `j_w_j` as `j_w_j = X_bar_j.T @ R_bar_j.T @ w_j`, where `X_bar_j`
          is the `6x6` constant joint frame transform matrix extended to 6D (via 3x3 on both diagonals)
          and similarly `R_bar_j` is the `6x6` extended joint frame rotation matrix extended to 6D
          computed from the absolute pose of the joint frame `p_j`.
        - Having `j_w_j`, we compute `lambda_j` as `[lambda_j; tau_j] = dt * inv(S_j) @ j_w_j`, where `S_j`
          is the `6x6` joint constraint/dof selection matrix. `tau_j` is the sum of the joint-space actuation
          generalized forces plus the joint-limit constraint reactions. Thus to recover `lambda_l_j`, and
          assuming we know `tau_c_j`, we can simply compute `lambda_l_j := tau_j - tau_c_j`.

        Correspondences between data containers and conversion inputs/outputs:
        - state_in.body_parent_f --> w_ij
        - control_in.tau_j --> tau_c_j
        - data_out.joints.j_w_j --> j_w_j
        - limits_out.reaction --> lambda_l_j
        - state_out.lambda_j --> lambda_j

        Args:
            model:
                The model containing the time-invariant data of the simulation.
            state_in:
                The input state data containing the current state of the simulation.
                Used to read the body-parent wrenches `body_parent_f`.
            control_in:
                The input control data containing the current control inputs of
                the simulation. Used to compute the joint actuation forces `tau_j`.
            state_out:
                The output state data containing the current state of the
                simulation. Used to store the joint reactions `lambda_j`.
            data_out:
                The output data containing the current data of the simulation.
                Used to store the joint wrenches `j_w_j and joint-limit reactions `lambda_l_j`.
        """
        # Skip the conversion if the required `body_parent_f` extended state attribute
        # was not allocated. Callers that intend to evaluate metrics on joint reactions
        # must request it via `ModelBuilder.request_state_attributes("body_parent_f")`.
        if state_in.body_parent_f is None:
            return
        convert_body_parent_wrenches_to_joint_reactions(
            model=model,
            data=data_out,
            limits=limits_out,
            control=control_in,
            body_parent_f=state_in.body_parent_f,
            state_lambda_j=state_out.lambda_j,
        )

    def _extract_constraint_reactions(
        self,
        model: ModelKamino,
        state: StateKamino,
        limits: LimitsKamino,
        contacts: ContactsKamino,
        lambdas: wp.array,
    ):
        """
        Fills in `lambdas` from:
        - `State.joints.lambda_j` containing the joint constraint reactions.
        - `LimitsKamino.reaction` containing the joint-limit constraint reactions.
        - `Contacts.reaction` containing the contact constraint reactions.

        Args:
            model:
                The model containing the time-invariant data of the simulation.
            state:
                The input state data containing the current state of the simulation.
            limits:
                The input limits data containing the joint-limit data.
            contacts:
                The input contacts data containing the contact data.
            lambdas:
                The output array to store the constraint reactions.
        """
        pass  # TODO: TO BE IMPLEMENTED
