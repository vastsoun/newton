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
from ..dynamics.dual import DualProblem
from ..dynamics.wrenches import (
    compute_constraint_body_wrenches,
    compute_joint_dof_body_wrenches,
    convert_body_parent_wrenches_to_joint_reactions,
    convert_joint_parent_wrenches_to_joint_reactions,
)
from ..geometry.contacts import ContactsKamino, convert_contacts_newton_to_kamino
from ..kinematics.constraints import (
    make_unilateral_constraints_info,
    pack_constraint_solutions,
    update_constraints_info,
)
from ..kinematics.jacobians import (
    DenseSystemJacobians,
    SparseSystemJacobians,
    SystemJacobiansType,
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

        ###
        # Computations using self._data synchronized with state_p
        ###

        # Update the relevant data fields of `DataKamino` and system Jacobians required
        # for the metrics computations, using the previous-state `StateKamino` instance.
        wp.copy(self._data.bodies.q_i, self._state_p.q_i)
        wp.copy(self._data.bodies.u_i, self._state_p.u_i)
        wp.copy(self._data.bodies.w_e_i, self._state_p.w_i_e)
        wp.copy(self._data.joints.tau_j, self._control.tau_j)

        # Update the relevant data intermediate quantities such as:
        # - active constraint info
        # - body inertias
        # - joint frame poses, DoF velocities, coordinates, and constraint residuals
        update_constraints_info(model=self._model, data=self._data)
        update_body_inertias(model=self._model.bodies, data=self._data.bodies)
        compute_joints_data(model=self._model, data=self._data, q_j_p=self._state_p.q_j)

        # Update the forward kinematics and dynamics quantities
        self._update_jacobians()
        self._update_dynamics()

        ###
        # Computations using self._data synchronized with state_p
        ###

        # Compute the post-event constraint-space velocities given
        # the pre- and post-event state and constraint Jacobians
        self._compute_constraint_velocities(u=self._state.u_i)

        # Perform the necessary conversions and extractions to obtain the
        # solver data in the expected format for the metrics computations
        if state.joint_parent_f is not None:
            self._convert_joint_parent_wrenches_to_joint_reactions(state.joint_parent_f)
            self._extract_constraint_reactions()
        elif state.body_parent_f is not None:
            self._convert_body_parent_wrenches_to_joint_reactions(state.body_parent_f)
            self._extract_constraint_reactions()
        else:
            # TODO: Add an additional solver-specific path to extract the constraint reactions from the internal solver data.
            raise ValueError("Expected either 'state.body_parent_f' or 'state.joint_parent_f', but both are None.")

        ###
        # Computations using self._data synchronized with state
        ###

        # Update the relevant data fields of `DataKamino` to synchronize it with the current-state
        # `StateKamino` instance, emulating state integration after the forward dynamics solve.
        wp.copy(self._data.bodies.q_i, self._state.q_i)
        wp.copy(self._data.bodies.u_i, self._state.u_i)
        wp.copy(self._data.joints.q_j, self._state.q_j)
        wp.copy(self._data.joints.dq_j, self._state.dq_j)

        # Update all dynamics quantities based
        # on the extracted constraint reactions
        self._update_body_wrenches()

        ###
        # Run metrics evaluation back-end
        ###

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

    def _update_jacobians(self):
        """
        Builds the system Jacobians based on the current state of the system,
        the set of active constraints, and the previous-state `StateKamino` instance.
        """
        self._jacobians.build(
            model=self._model,
            data=self._data,
            limits=self._limits,
            contacts=self._contacts,
            reset_to_zero=True,
        )

    def _update_dynamics(self):
        """
        Constructs the forward dynamics problem quantities based on the current state of
        the system, the set of active constraints, and the updated system Jacobians.
        """
        self._problem.build(
            model=self._model,
            data=self._data,
            limits=self._limits,
            contacts=self._contacts,
            jacobians=self._jacobians,
            reset_to_zero=True,
        )

    def _update_body_wrenches(self):
        """
        Computes the per-body actuation and constraint wrenches based on the
        extracted constraint reactions and the current system Jacobians.
        """
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

    def _compute_constraint_velocities(self, u: wp.array[wp.spatial_vectorf]):
        """
        Computes the constraint-space velocities`v = J_cts @ u_i`, where `J_cts` is the constraint Jacobian.

        Args:
            u: The input per-body twists `u_i`.
        """
        compute_constraint_space_velocities(
            model=self._model,
            jacobians=self._jacobians,
            u=u,
            v_start=self._problem.data.vio,
            v=self._v_plus,
            reset_to_zero=True,
        )

    def _convert_body_parent_wrenches_to_joint_reactions(self, body_parent_f: wp.array[wp.spatial_vectorf]):
        """
        Converts Newton body-parent wrenches `newton.State.body_parent_f` data
        to Kamino `StateKamino.lambda_j` and `DataKamino.joints.lambda_l_j`.

        This operation also updates per-joint wrenches arrays `DataKamino.joints.j_w_j` as a byproduct.

        Args:
            body_parent_f: The input array of per-body parent wrenches (world frame, at body CoM).
        """
        convert_body_parent_wrenches_to_joint_reactions(
            body_parent_f=body_parent_f,
            model=self._model,
            data=self._data,
            control=self._control,
            limits=self._limits,
            reset_to_zero=True,
        )

    def _convert_joint_parent_wrenches_to_joint_reactions(self, joint_parent_f: wp.array[wp.spatial_vectorf]):
        """
        Converts Newton joint-parent wrenches `newton.State.joint_parent_f` data
        to Kamino `StateKamino.lambda_j` and `DataKamino.joints.lambda_l_j`.
        """
        convert_joint_parent_wrenches_to_joint_reactions(
            joint_parent_f=joint_parent_f,
            model=self._model,
            data=self._data,
            control=self._control,
            limits=self._limits,
            reset_to_zero=True,
        )

    def _extract_constraint_reactions(self):
        """
        Fills in `lambdas` from:
        - `DataKamino.joints.lambda_j` containing the joint constraint reactions.
        - `LimitsKamino.reaction` containing the joint-limit constraint reactions.
        - `ContactsKamino.reaction` containing the contact constraint reactions.
        """
        pack_constraint_solutions(
            lambdas=self._lambdas,
            model=self._model,
            data=self._data,
            limits=self._limits,
            contacts=self._contacts,
            reset_to_zero=True,
        )
