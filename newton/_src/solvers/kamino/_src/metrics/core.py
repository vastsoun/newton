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
from ..core.model import ModelKamino
from ..core.state import StateKamino
from ..core.types import float32
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
)
from ..kinematics.joints import compute_joints_data
from ..kinematics.limits import LimitsKamino
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
# Types
###


# TODO


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

        # Create the data, limits and contacts containers
        self._data = self._model.data()
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
        # TODO: Create a solver config and set constraint stabilization to zero
        self._problem = DualProblem(
            model=self._model,
            data=self._data,
            limits=self._limits,
            contacts=self._contacts,
            jacobians=self._jacobians,
            sparse=False,
        )

        # Finalize the internal SolutionMetrics instance
        self._metrics = SolutionMetrics(model=self._model)

        # Allocate metrics data on the target device
        with wp.ScopedDevice(self.device):
            self._v_plus = wp.zeros(self._model.size.sum_of_max_total_cts, dtype=float32)
            self._lambdas = wp.zeros(self._model.size.sum_of_max_total_cts, dtype=float32)
            self._sigma = wp.zeros(self._model.size.sum_of_max_total_cts, dtype=float32)

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
        pass  # TODO: TO BE IMPLEMENTED

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
        - Having `j_w_j`, we compute `lambda_j` as `[lambda_j; tau_j] = (1/dt) inv(S_j) @ j_w_j`, where `S_j`
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
        pass  # TODO: TO BE IMPLEMENTED

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
