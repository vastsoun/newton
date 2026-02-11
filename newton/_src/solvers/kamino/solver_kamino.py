# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Defines the :class:`SolverKamino` class, providing a physics backend for
simulating constrained multi-body systems for arbitrary mechanical assemblies.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import warp as wp
from warp.context import Devicelike

# Newton imports
from ...core.types import override
from ..solver import SolverBase

# Kamino imports
from .core.bodies import update_body_inertias, update_body_wrenches
from .core.control import Control
from .core.joints import JointCorrectionMode
from .core.model import Model, ModelData
from .core.state import State
from .core.time import advance_time
from .core.types import float32, int32, transformf, vec6f
from .dynamics.dual import DualProblem, DualProblemSettings
from .dynamics.wrenches import (
    compute_constraint_body_wrenches,
    compute_joint_dof_body_wrenches,
)
from .geometry.contacts import Contacts
from .integrators.euler import integrate_euler_semi_implicit
from .kinematics.constraints import (
    make_unilateral_constraints_info,
    unpack_constraint_solutions,
    update_constraints_info,
)
from .kinematics.jacobians import DenseSystemJacobians, SparseSystemJacobians
from .kinematics.joints import (
    compute_joints_data,
    extract_actuators_state_from_joints,
    extract_joints_state_from_actuators,
)
from .kinematics.limits import Limits
from .kinematics.resets import (
    reset_body_net_wrenches,
    reset_joint_constraint_reactions,
    reset_state_from_base_state,
    reset_state_to_model_default,
    reset_time,
)
from .linalg import ConjugateGradientSolver, IterativeSolver, LinearSolverType, LLTBlockedSolver
from .solvers.fk import ForwardKinematicsSolver, ForwardKinematicsSolverSettings
from .solvers.metrics import SolutionMetrics
from .solvers.padmm import PADMMSettings, PADMMSolver, PADMMWarmStartMode
from .solvers.warmstart import WarmstarterContacts, WarmstarterLimits
from .utils import logger as msg

###
# Types
###


@dataclass
class SolverKaminoSettings:
    """
    A container to hold configurations for :class:`SolverKamino`.
    """

    problem: DualProblemSettings = field(default_factory=DualProblemSettings)
    """
    Settings for the dynamics problem.\n
    See :class:`DualProblemSettings` for more details.
    """

    padmm: PADMMSettings = field(default_factory=PADMMSettings)
    """
    Settings for the dynamics solver.\n
    See :class:`PADMMSettings` for more details.
    """

    fk: ForwardKinematicsSolverSettings = field(default_factory=ForwardKinematicsSolverSettings)
    """
    Settings for the forward kinematics solver.\n
    See :class:`ForwardKinematicsSolverSettings` for more details.
    """

    warmstart_mode: PADMMWarmStartMode = PADMMWarmStartMode.CONTAINERS
    """
    Warmstart mode to be used for the dynamics solver.\n
    See :class:`PADMMWarmStartMode` for the available options.\n
    Defaults to `PADMMWarmStartMode.CONTAINERS to warmstart from the solver data containers.
    """

    contact_warmstart_method: WarmstarterContacts.Method = WarmstarterContacts.Method.KEY_AND_POSITION
    """
    Method to be used for warm-starting contacts.\n
    See :class:`WarmstarterContacts.Method` for available options.\n
    Defaults to `WarmstarterContacts.Method.KEY_AND_POSITION`.
    """

    use_solver_acceleration: bool = True
    """
    Enables Nesterov-type acceleration, i.e. use APADMM instead of standard PADMM.\n
    Defaults to `True`.
    """

    collect_solver_info: bool = False
    """
    Enables collection of dynamics solver convergence and performance info at each simulation step.\n
    Defaults to `False`.
    """

    compute_metrics: bool = False
    """
    Enables computation of solution metrics at each simulation step.\n
    Defaults to `False`.
    """

    avoid_graph_conditionals: bool = False
    """
    Avoids CUDA graph conditional nodes in iterative solvers.\n
    When enabled, replaces `wp.capture_while` with unrolled for-loops over max iterations.\n
    Defaults to `False`.
    """

    linear_solver_type: type[LinearSolverType] = LLTBlockedSolver
    """
    The type of linear solver to use for the dynamics problem.\n
    See :class:`LinearSolverType` for available options.\n
    Defaults to :class:`LLTBlockedSolver`.
    """

    linear_solver_kwargs: dict[str, Any] = field(default_factory=dict)
    """
    Additional keyword arguments to pass to the linear solver.\n
    Defaults to an empty dictionary.
    """

    rotation_correction: JointCorrectionMode = JointCorrectionMode.TWOPI
    """
    The rotation correction mode to use for rotational DoFs.\n
    See :class:`JointCorrectionMode` for available options.\n
    Defaults to `JointCorrectionMode.TWOPI`.
    """

    sparse: bool = True
    """
    Flag to indicate whether the solver should use sparse data representations.
    """

    def check(self) -> None:
        """Validates relevant solver settings."""
        if not issubclass(self.linear_solver_type, LinearSolverType):
            raise TypeError(
                "Invalid linear solver type: Expected a subclass of `LinearSolverType`, "
                f"but got {type(self.linear_solver_type)}."
            )
        if not isinstance(self.warmstart_mode, PADMMWarmStartMode):
            raise TypeError(
                "Invalid warmstart mode: Expected a `PADMMWarmStartMode` enum value, "
                f"but got {type(self.warmstart_mode)}."
            )
        if not isinstance(self.contact_warmstart_method, WarmstarterContacts.Method):
            raise TypeError(
                "Invalid contact warmstart method: Expected a `WarmstarterContacts.Method` enum value, "
                f"but got {type(self.contact_warmstart_method)}."
            )
        if not isinstance(self.rotation_correction, JointCorrectionMode):
            raise TypeError(
                "Invalid rotation correction mode: Expected a `JointCorrectionMode` enum value, "
                f"but got {type(self.rotation_correction)}."
            )
        self.problem.check()
        self.padmm.check()
        self.fk.check()

    def __post_init__(self):
        """Post-initialization to validate settings."""
        self.check()


###
# Interfaces
###


class SolverKamino(SolverBase):
    """
    A physics solver for simulating constrained multi-body systems for arbitrary mechanical assemblies.

    This solver uses the Proximal-ADMM algorithm to solve the forward dynamics formulated
    as a Nonlinear Complementarity Problem (NCP) over the set of bilateral kinematic joint
    constraints and unilateral constraints that include joint-limits and contacts.

    References:
        - Tsounis, Vassilios, Ruben Grandia, and Moritz Bächer.
          On Solving the Dynamics of Constrained Rigid Multi-Body Systems with Kinematic Loops.
          arXiv preprint arXiv:2504.19771 (2025).
          https://doi.org/10.48550/arXiv.2504.19771
        - Carpentier, Justin, Quentin Le Lidec, and Louis Montaut.
          From Compliant to Rigid Contact Simulation: a Unified and Efficient Approach.
          20th edition of the “Robotics: Science and Systems”(RSS) Conference. 2024.
          https://roboticsproceedings.org/rss20/p108.pdf
        - Tasora, A., Mangoni, D., Benatti, S., & Garziera, R. (2021).
          Solving variational inequalities and cone complementarity problems in
          nonsmooth dynamics using the alternating direction method of multipliers.
          International Journal for Numerical Methods in Engineering, 122(16), 4093-4113.
          https://onlinelibrary.wiley.com/doi/full/10.1002/nme.6693

    After constructing :class:`Model`, :class:`State`, :class:`Control` and :class:`Contacts`
    objects, this physics solver may be used to advance the simulation state forward in time.

    Example
    -------

    .. code-block:: python

        contacts = ...
        settings = newton.solvers.kamino.SolverKaminoSettings()
        solver = newton.solvers.SolverKamino(model, contacts, settings)

        # simulation loop
        for i in range(100):
            solver.step(state_in, state_out, control, contacts, dt)
            state_in, state_out = state_out, state_in

    """

    ResetCallbackType = Callable[["SolverKamino", State], None]
    """Defines the type signature for reset callback functions."""

    StepCallbackType = Callable[["SolverKamino", State, State, Control, Contacts], None]
    """Defines the type signature for step callback functions."""

    def __init__(
        self,
        model: Model,
        contacts: Contacts | None = None,
        settings: SolverKaminoSettings | None = None,
    ):
        """
        Initializes the Kamino physics solver for the given set of multi-body systems
        defined in `model`, and the total contact allocations defined in `contacts`.

        Explicit solver settings may be provided through the `settings` argument. If no
        settings are provided, default settings will be used.

        Args:
            model (Model): The multi-body systems model to simulate.
            contacts (Contacts): The contact data container for the simulation.
            settings (SolverKaminoSettings | None): Optional solver settings.
        """
        # Ensure the input containers are valid
        if not isinstance(model, Model):
            raise TypeError(f"Invalid model container: Expected a `Model` instance, but got {type(model)}.")
        if contacts is not None and not isinstance(contacts, Contacts):
            raise TypeError(f"Invalid contacts container: Expected a `Contacts` instance, but got {type(contacts)}.")
        if settings is not None and not isinstance(settings, SolverKaminoSettings):
            raise TypeError(
                f"Invalid solver settings: Expected a `SolverKaminoSettings` instance, but got {type(settings)}."
            )

        # First initialize the base solver
        # NOTE: Although we pass the model here, we will re-assign it below
        # since currently Kamino defines its own :class`Model` class.
        super().__init__(model=model)
        self._model = model

        # Cache solver settings: If no settings are provided, use defaults
        if settings is None:
            settings = SolverKaminoSettings()
        settings.check()
        self._settings: SolverKaminoSettings = settings

        if self._settings.sparse and not issubclass(self._settings.linear_solver_type, IterativeSolver):
            msg.warning(
                f"Sparse problem requires iterative solver, but got '{self._settings.linear_solver_type.__name__}'. Switching to 'ConjugateGradientSolver'."
            )
            self._settings.linear_solver_type = ConjugateGradientSolver

        # Allocate internal time-varying solver data
        self._data = self._model.data()

        # Allocate a joint-limits interface
        self._limits = Limits(model=self._model, device=self._model.device)

        # Construct the unilateral constraints members in the model info
        make_unilateral_constraints_info(model=self._model, data=self._data, limits=self._limits, contacts=contacts)

        # Allocate Jacobians data on the device
        if self._settings.sparse:
            self._jacobians = SparseSystemJacobians(
                model=self._model,
                limits=self._limits,
                contacts=contacts,
                device=self._model.device,
            )
        else:
            self._jacobians = DenseSystemJacobians(
                model=self._model,
                limits=self._limits,
                contacts=contacts,
                device=self._model.device,
            )

        # Allocate the dual problem data on the device
        linear_solver_kwargs = dict(self._settings.linear_solver_kwargs)
        if self._settings.avoid_graph_conditionals and issubclass(self._settings.linear_solver_type, IterativeSolver):
            linear_solver_kwargs.setdefault("avoid_graph_conditionals", True)
        self._problem_fd = DualProblem(
            model=self._model,
            data=self._data,
            limits=self._limits,
            contacts=contacts,
            solver=self._settings.linear_solver_type,
            solver_kwargs=linear_solver_kwargs,
            settings=self._settings.problem,
            device=self._model.device,
            sparse=self._settings.sparse,
        )

        # Allocate the forward dynamics solver on the device
        self._solver_fd = PADMMSolver(
            model=self._model,
            settings=self._settings.padmm,
            warmstart=self._settings.warmstart_mode,
            use_acceleration=self._settings.use_solver_acceleration,
            collect_info=self._settings.collect_solver_info,
            avoid_graph_conditionals=self._settings.avoid_graph_conditionals,
            device=self._model.device,
        )

        # Allocate the forward kinematics solver on the device
        self._solver_fk = ForwardKinematicsSolver(model=self._model, settings=self._settings.fk)

        # Allocate additional internal data for reset operations
        with wp.ScopedDevice(self._model.device):
            self._all_worlds_mask = wp.ones(shape=(self._model.size.num_worlds,), dtype=int32)
            self._base_q = wp.zeros(shape=(self._model.size.num_worlds,), dtype=transformf)
            self._base_u = wp.zeros(shape=(self._model.size.num_worlds,), dtype=vec6f)
            self._actuators_q = wp.zeros(shape=(self._model.size.sum_of_num_actuated_joint_coords,), dtype=float32)
            self._actuators_u = wp.zeros(shape=(self._model.size.sum_of_num_actuated_joint_dofs,), dtype=float32)

        # Allocate the contacts warmstarter if enabled
        self._ws_limits: WarmstarterLimits | None = None
        self._ws_contacts: WarmstarterContacts | None = None
        if self._settings.warmstart_mode == PADMMWarmStartMode.CONTAINERS:
            self._ws_limits = WarmstarterLimits(limits=self._limits)
            self._ws_contacts = WarmstarterContacts(
                contacts=contacts,
                method=self._settings.contact_warmstart_method,
            )

        # Allocate the solution metrics evaluator if enabled
        self._metrics: SolutionMetrics | None = None
        if self._settings.compute_metrics:
            self._metrics = SolutionMetrics(model=self._model)

        # Initialize callbacks
        self._pre_reset_cb: SolverKamino.ResetCallbackType | None = None
        self._post_reset_cb: SolverKamino.ResetCallbackType | None = None
        self._pre_step_cb: SolverKamino.StepCallbackType | None = None
        self._mid_step_cb: SolverKamino.StepCallbackType | None = None
        self._post_step_cb: SolverKamino.StepCallbackType | None = None

        # Initialize all internal solver data
        with wp.ScopedDevice(self._model.device):
            self._reset()

    ###
    # Properties
    ###

    @property
    def settings(self) -> SolverKaminoSettings:
        """
        Returns the host-side cache of high-level solver settings.
        """
        return self._settings

    @property
    def device(self) -> Devicelike:
        """
        Returns the device where the solver data is allocated.
        """
        return self._model.device

    @property
    def data(self) -> ModelData:
        """
        Returns the internal solver data container.
        """
        return self._data

    @property
    def problem_fd(self) -> DualProblem:
        """
        Returns the dual forward dynamics problem.
        """
        return self._problem_fd

    @property
    def solver_fd(self) -> PADMMSolver:
        """
        Returns the forward dynamics solver.
        """
        return self._solver_fd

    @property
    def solver_fk(self) -> ForwardKinematicsSolver:
        """
        Returns the forward kinematics solver backend.
        """
        return self._solver_fk

    @property
    def metrics(self) -> SolutionMetrics | None:
        """
        Returns the solution metrics evaluator, if enabled.
        """
        return self._metrics

    ###
    # Configurations
    ###

    def set_pre_reset_callback(self, callback: ResetCallbackType):
        """
        Set a reset callback to be called at the beginning of each call to `reset_*()` methods.
        """
        self._pre_reset_cb = callback

    def set_post_reset_callback(self, callback: ResetCallbackType):
        """
        Set a reset callback to be called at the end of each call to to `reset_*()` methods.
        """
        self._post_reset_cb = callback

    def set_pre_step_callback(self, callback: StepCallbackType):
        """
        Sets a callback to be called before forward dynamics solve.
        """
        self._pre_step_cb = callback

    def set_mid_step_callback(self, callback: StepCallbackType):
        """
        Sets a callback to be called between forward dynamics solver and state integration.
        """
        self._mid_step_cb = callback

    def set_post_step_callback(self, callback: StepCallbackType):
        """
        Sets a callback to be called after state integration.
        """
        self._post_step_cb = callback

    ###
    # Solver API
    ###

    def reset(
        self,
        state_out: State,
        world_mask: wp.array | None = None,
        actuator_q: wp.array | None = None,
        actuator_u: wp.array | None = None,
        joint_q: wp.array | None = None,
        joint_u: wp.array | None = None,
        base_q: wp.array | None = None,
        base_u: wp.array | None = None,
    ):
        """
        Resets the simulation state given a combination of desired base body
        and joint states, as well as an optional per-world mask array indicating
        which worlds should be reset. The reset state is written to `state_out`.

        For resets given absolute quantities like base body poses, the
        `state_out` must initially contain the current state of the simulation.

        Args:
            state_out (State):
                The output state container to which the reset state data is written.
            world_mask (wp.array, optional):
                Optional array of per-world masks indicating which worlds should be reset.\n
                Shape of `(num_worlds,)` and type :class:`wp.int8 | wp.bool`
            actuator_q (wp.array, optional):
                Optional array of target actuated joint coordinates.\n
                Shape of `(num_actuated_joint_coords,)` and type :class:`wp.float32`
            actuator_u (wp.array, optional):
                Optional array of target actuated joint DoF velocities.\n
                Shape of `(num_actuated_joint_dofs,)` and type :class:`wp.float32`
            joint_q (wp.array, optional):
                Optional array of target joint coordinates.\n
                Shape of `(num_joint_coords,)` and type :class:`wp.float32`
            joint_qd (wp.array, optional):
                Optional array of target joint DoF velocities.\n
                Shape of `(num_joint_dofs,)` and type :class:`wp.float32`
            base_q (wp.array, optional):
                Optional array of target base body poses.\n
                Shape of `(num_worlds,)` and type :class:`wp.transformf`
            base_qd (wp.array, optional):
                Optional array of target base body twists.\n
                Shape of `(num_worlds,)` and type :class:`wp.spatial_vectorf`
        """
        # Ensure the input reset targets are valid
        if joint_q is not None and joint_q.shape[0] != self._model.size.sum_of_num_joint_coords:
            raise ValueError(
                f"Invalid joint_q shape: Expected ({self._model.size.sum_of_num_joint_coords},),"
                f" but got {joint_q.shape}."
            )
        if joint_u is not None and joint_u.shape[0] != self._model.size.sum_of_num_joint_dofs:
            raise ValueError(
                f"Invalid joint_u shape: Expected ({self._model.size.sum_of_num_joint_dofs},), but got {joint_u.shape}."
            )
        if actuator_q is not None and actuator_q.shape[0] != self._model.size.sum_of_num_actuated_joint_coords:
            raise ValueError(
                f"Invalid actuator_q shape: Expected ({self._model.size.sum_of_num_actuated_joint_coords},),"
                f" but got {actuator_q.shape}."
            )
        if actuator_u is not None and actuator_u.shape[0] != self._model.size.sum_of_num_actuated_joint_dofs:
            raise ValueError(
                f"Invalid actuator_u shape: Expected ({self._model.size.sum_of_num_actuated_joint_dofs},),"
                f" but got {actuator_u.shape}."
            )
        if base_q is not None and base_q.shape[0] != self._model.size.num_worlds:
            raise ValueError(
                f"Invalid base_q shape: Expected ({self._model.size.num_worlds},), but got {base_q.shape}."
            )
        if base_u is not None and base_u.shape[0] != self._model.size.num_worlds:
            raise ValueError(
                f"Invalid base_u shape: Expected ({self._model.size.num_worlds},), but got {base_u.shape}."
            )
        if world_mask is not None and world_mask.shape[0] != self._model.size.num_worlds:
            raise ValueError(
                f"Invalid world_mask shape: Expected ({self._model.size.num_worlds},), but got {world_mask.shape}."
            )

        # Ensure that only joint or actuator targets are provided
        if (joint_q is not None or joint_u is not None) and (actuator_q is not None or actuator_u is not None):
            raise ValueError("Combined joint and actuator targets are not supported. Only one type may be provided.")

        # Run the pre-reset callback if it has been set
        self._run_pre_reset_callback(state_out=state_out)

        # Determine the effective world mask to use for the reset operation
        _world_mask = world_mask if world_mask is not None else self._all_worlds_mask

        # If no reset targets are provided, reset all bodies to the model default state
        if (
            (base_q is None and base_u is None)
            and (joint_q is None and joint_u is None)
            and (actuator_q is None and actuator_u is None)
        ):
            self._reset_to_default_state(
                state_out=state_out,
                world_mask=_world_mask,
            )

        # If only base targets are provided, uniformly reset all bodies to the given base states
        elif (
            (base_q is not None or base_u is not None)
            and (joint_q is None and joint_u is None)
            and (actuator_q is None and actuator_u is None)
        ):
            self._reset_to_base_state(
                state_out=state_out,
                world_mask=_world_mask,
                base_q=base_q,
                base_u=base_u,
            )

        # If a joint target is provided, use the FK solver to reset the bodies accordingly
        elif joint_q is not None or actuator_q is not None:
            self._reset_with_fk_solve(
                state_out=state_out,
                world_mask=_world_mask,
                actuator_q=actuator_q,
                actuator_u=actuator_u,
                joint_q=joint_q,
                joint_u=joint_u,
                base_q=base_q,
                base_u=base_u,
            )

        # If no valid combination of reset targets is provided, raise an error
        else:
            raise ValueError(
                "Unsupported reset combination with: "
                f" actuator_q: {actuator_q is not None}, actuator_u: {actuator_u is not None},"
                f" joint_q: {joint_q is not None}, joint_u: {joint_u is not None},"
                f" base_q: {base_q is not None}, base_u: {base_u is not None}."
            )

        # Post-process the reset operation
        self._reset_post_process(world_mask=_world_mask)

        # Run the post-reset callback if it has been set
        self._run_post_reset_callback(state_out=state_out)

    @override
    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control,
        contacts: Contacts | None = None,
        dt: float | None = None,
    ):
        """
        Progresses the simulation by a single time-step `dt` given the current
        state `state_in`, control inputs `control`, and set of active contacts
        `contacts`. The updated state is written to `state_out`.

        Args:
            state_in (State):
                The input current state of the simulation.
            state_out (State):
                The output next state after time integration.
            control (Control):
                The input controls applied to the system.
            contacts (Contacts, optional):
                The set of active contacts.
            dt (float, optional):
                A uniform time-step to apply uniformly to all worlds of the simulation.
        """
        # If specified, configure the internal per-world solver time-step uniformly from the input argument
        if dt is not None:
            self._model.time.set_uniform_timestep(dt)

        # Copy the new input state and control to the internal solver data
        self._read_step_inputs(state_in=state_in, control_in=control)

        # Update intermediate quantities of the bodies and joints
        self._update_intermediates(state_in=state_in)

        # Run limit detection to generate active joint limits
        self._update_limits()

        # Update the constraint state info
        self._update_constraint_info()

        # Update the differential forward kinematics to compute system Jacobians
        self._update_jacobians(contacts=contacts)

        # Compute the body actuation wrenches based on the current control inputs
        self._update_actuation_wrenches()

        # Run the pre-step callback if it has been set
        self._run_prestep_callback(state_in, state_out, control, contacts)

        # Solve the forward dynamics sub-problem to compute constraint reactions and body wrenches
        self._forward(contacts=contacts)

        # Run the mid-step callback if it has been set
        self._run_midstep_callback(state_in, state_out, control, contacts)

        # Solve the time integration sub-problem to compute the next state of the system
        self._integrate()

        # Compute solver solution metrics if enabled
        self._compute_metrics(state_in=state_in, contacts=contacts)

        # Update time-keeping (i.e. physical time and discrete steps)
        self._advance_time()

        # Run the post-step callback if it has been set
        self._run_poststep_callback(state_in, state_out, control, contacts)

        # Copy the updated internal solver state to the output state
        self._write_step_output(state_out=state_out)

    ###
    # Internals - Callback Operations
    ###

    def _run_pre_reset_callback(self, state_out: State):
        """
        Runs the pre-reset callback if it has been set.
        """
        if self._pre_reset_cb is not None:
            self._pre_reset_cb(self, state_out)

    def _run_post_reset_callback(self, state_out: State):
        """
        Runs the post-reset callback if it has been set.
        """
        if self._post_reset_cb is not None:
            self._post_reset_cb(self, state_out)

    def _run_prestep_callback(self, state_in: State, state_out: State, control: Control, contacts: Contacts):
        """
        Runs the pre-step callback if it has been set.
        """
        if self._pre_step_cb is not None:
            self._pre_step_cb(self, state_in, state_out, control, contacts)

    def _run_midstep_callback(self, state_in: State, state_out: State, control: Control, contacts: Contacts):
        """
        Runs the mid-step callback if it has been set.
        """
        if self._mid_step_cb is not None:
            self._mid_step_cb(self, state_in, state_out, control, contacts)

    def _run_poststep_callback(self, state_in: State, state_out: State, control: Control, contacts: Contacts):
        """
        Executes the post-step callback if it has been set.
        """
        if self._post_step_cb is not None:
            self._post_step_cb(self, state_in, state_out, control, contacts)

    ###
    # Internals - Input/Output Operations
    ###

    def _read_step_inputs(self, state_in: State, control_in: Control):
        """
        Updates the internal solver data from the input state and control.
        """
        # TODO: Remove corresponding data copies
        # by directly using the input containers
        wp.copy(self._data.bodies.q_i, state_in.q_i)
        wp.copy(self._data.bodies.u_i, state_in.u_i)
        wp.copy(self._data.bodies.w_i, state_in.w_i)
        wp.copy(self._data.joints.q_j, state_in.q_j)
        wp.copy(self._data.joints.q_j_p, state_in.q_j_p)
        wp.copy(self._data.joints.dq_j, state_in.dq_j)
        wp.copy(self._data.joints.lambda_j, state_in.lambda_j)
        wp.copy(self._data.joints.tau_j, control_in.tau_j)

    def _write_step_output(self, state_out: State):
        """
        Updates the output state from the internal solver data.
        """
        # TODO: Remove corresponding data copies
        # by directly using the input containers
        wp.copy(state_out.q_i, self._data.bodies.q_i)
        wp.copy(state_out.u_i, self._data.bodies.u_i)
        wp.copy(state_out.w_i, self._data.bodies.w_i)
        wp.copy(state_out.q_j, self._data.joints.q_j)
        wp.copy(state_out.q_j_p, self._data.joints.q_j_p)
        wp.copy(state_out.dq_j, self._data.joints.dq_j)
        wp.copy(state_out.lambda_j, self._data.joints.lambda_j)

    ###
    # Internals - Reset Operations
    ###

    def _reset(self):
        """
        Performs a hard-reset of all solver internal data.
        """
        # Reset internal time-keeping data
        self._data.time.reset()

        # Reset all bodies to their model default states
        self._data.bodies.clear_all_wrenches()
        wp.copy(self._data.bodies.q_i, self._model.bodies.q_i_0)
        wp.copy(self._data.bodies.u_i, self._model.bodies.u_i_0)
        update_body_inertias(model=self._model.bodies, data=self._data.bodies)

        # Reset all joints to their model default states
        self._data.joints.reset_state(q_j_ref=self._model.joints.q_j_ref)
        self._data.joints.clear_all()

        # Reset the joint-limits interface
        self._limits.reset()

        # Initialize the constraint state info
        self._data.info.num_limits.zero_()
        self._data.info.num_contacts.zero_()
        update_constraints_info(model=self._model, data=self._data)

        # Initialize the system Jacobians so that they may be available after reset
        # NOTE: This is not strictly necessary, but serves advanced users who may
        # want to query Jacobians in controllers immediately after a reset operation.
        self._jacobians.build(
            model=self._model,
            data=self._data,
            limits=None,
            contacts=None,
            reset_to_zero=True,
        )

        # Reset the forward dynamics solver
        self._solver_fd.reset()

    def _reset_to_default_state(self, state_out: State, world_mask: wp.array):
        """
        Resets the simulation to the default state defined in the model.
        """
        reset_state_to_model_default(
            model=self._model,
            state_out=state_out,
            world_mask=world_mask,
        )

    def _reset_to_base_state(
        self,
        state_out: State,
        world_mask: wp.array,
        base_q: wp.array | None = None,
        base_u: wp.array | None = None,
    ):
        """
        Resets the simulation to the given base body states by
        uniformly applying the necessary transform across all bodies.
        """
        # First determine the effective base states to use
        _base_q = base_q if base_q is not None else self._base_q
        _base_u = base_u if base_u is not None else self._base_u

        # Uniformly reset all bodies according to the transform between the given
        # base state and the existing body states contained in `state_out`
        reset_state_from_base_state(
            model=self._model,
            state_out=state_out,
            world_mask=world_mask,
            base_q=_base_q,
            base_u=_base_u,
            q_i_cache=self._data.bodies.q_i,
        )

    def _reset_with_fk_solve(
        self,
        state_out: State,
        world_mask: wp.array,
        joint_q: wp.array | None = None,
        joint_u: wp.array | None = None,
        actuator_q: wp.array | None = None,
        actuator_u: wp.array | None = None,
        base_q: wp.array | None = None,
        base_u: wp.array | None = None,
    ):
        """
        Resets the simulation to the given joint states by solving
        the forward kinematics to compute the corresponding body states.
        """
        # Detect if joint or actuator targets are provided
        with_joint_targets = joint_q is not None and (actuator_q is None and actuator_u is None)

        # Unpack the actuated joint states from the input joint states
        if with_joint_targets:
            extract_actuators_state_from_joints(
                model=self._model,
                world_mask=world_mask,
                joint_q=joint_q,
                joint_u=joint_u if joint_u is not None else state_out.dq_j,
                actuator_q=self._actuators_q,
                actuator_u=self._actuators_u,
            )

        # Determine the actuator state arrays to use for the FK solve
        _actuator_q = actuator_q if actuator_q is not None else self._actuators_q
        _actuator_u = actuator_u if actuator_u is not None else self._actuators_u

        # TODO: We need a graph-capturable mechanism to detect solver errors
        # Solve the forward kinematics to compute the body states
        self._solver_fk.run_fk_solve(
            world_mask=world_mask,
            bodies_q=state_out.q_i,
            bodies_u=state_out.u_i if joint_u is not None or actuator_u is not None else None,
            actuators_q=_actuator_q,
            actuators_u=_actuator_u,
            base_q=base_q,
            base_u=base_u,
        )

        # Reset net body wrenches and joint constraint reactions to zero
        # NOTE: This is necessary to ensure proper solver behavior after resets
        reset_body_net_wrenches(model=self._model, body_w=state_out.w_i, world_mask=world_mask)
        reset_joint_constraint_reactions(model=self._model, lambda_j=state_out.lambda_j, world_mask=world_mask)

        # If joint targets were provided, copy them to the output state
        if with_joint_targets:
            # Copy the joint states to the output state
            wp.copy(state_out.q_j_p, joint_q)
            wp.copy(state_out.q_j, joint_q)
            if joint_u is not None:
                wp.copy(state_out.dq_j, joint_u)
        # Otherwise, extract the joint states from the actuators
        else:
            extract_joints_state_from_actuators(
                model=self._model,
                world_mask=world_mask,
                actuator_q=_actuator_q,
                actuator_u=_actuator_u,
                joint_q=state_out.q_j,
                joint_u=state_out.dq_j,
            )
            wp.copy(state_out.q_j_p, state_out.q_j)

    def _reset_post_process(self, world_mask: wp.array | None = None):
        """
        Resets solver internal data and calls reset callbacks.

        This is a common operation that must be called after resetting bodies and joints,
        that ensures that all state and control data are synchronized with the internal
        solver state, and that intermediate quantities are updated accordingly.
        """
        # Reset the solver-internal time-keeping data
        reset_time(
            model=self._model,
            world_mask=world_mask,
            time=self._data.time.time,
            steps=self._data.time.steps,
        )

        # Reset the forward dynamics solver to clear internal state
        # NOTE: This will cause the solver to perform a cold-start
        # on the first call to `step()`
        self._solver_fd.reset(problem=self._problem_fd, world_mask=world_mask)

    ###
    # Internals - Step Operations
    ###

    def _update_joints_data(self, q_j_p: wp.array):
        """
        Updates the joint states based on the current body states.
        """
        # Update the joint states based on the updated body states
        # NOTE: We use the previous state `state_p` for post-processing
        # purposes, e.g. account for roll-over of revolute joints etc
        compute_joints_data(
            model=self._model,
            q_j_ref=q_j_p,
            data=self._data,
            correction=self._settings.rotation_correction,
        )

    def _update_intermediates(self, state_in: State):
        """
        Updates intermediate quantities required for the forward dynamics solve.
        """
        self._update_joints_data(q_j_p=state_in.q_j_p)
        update_body_inertias(model=self._model.bodies, data=self._data.bodies)

    def _update_limits(self):
        """
        Runs limit detection to generate active joint limits.
        """
        self._limits.detect(self._model, self._data)

    def _update_constraint_info(self):
        """
        Updates the state info with the set of active constraints resulting from limit and collision detection.
        """
        update_constraints_info(model=self._model, data=self._data)

    def _update_jacobians(self, contacts: Contacts | None = None):
        """
        Updates the forward kinematics by building the system Jacobians (of actuation and
        constraints) based on the current state of the system and set of active constraints.
        """
        self._jacobians.build(
            model=self._model,
            data=self._data,
            limits=self._limits,
            contacts=contacts,
            reset_to_zero=True,
        )

    def _update_actuation_wrenches(self):
        """
        Updates the actuation wrenches based on the current control inputs.
        """
        compute_joint_dof_body_wrenches(self._model, self._data, self._jacobians)

    def _update_dynamics(self, contacts: Contacts | None = None):
        """
        Constructs the forward dynamics problem quantities based on the current state of
        the system, the set of active constraints, and the updated system Jacobians.
        """
        self._problem_fd.build(
            model=self._model,
            data=self._data,
            limits=self._limits,
            contacts=contacts,
            jacobians=self._jacobians,
            reset_to_zero=True,
        )

    def _update_constraints(self, contacts: Contacts | None = None):
        """
        Solves the forward dynamics sub-problem to compute constraint
        reactions and body wrenches effected through constraints.
        """
        # If warm-starting is enabled, initialize unilateral
        # constraints containers from the current solver data
        if self._settings.warmstart_mode > PADMMWarmStartMode.NONE:
            if self._settings.warmstart_mode == PADMMWarmStartMode.CONTAINERS:
                self._ws_limits.warmstart(self._limits)
                self._ws_contacts.warmstart(self._model, self._data, contacts)
            self._solver_fd.warmstart(
                problem=self._problem_fd,
                model=self._model,
                data=self._data,
                limits=self._limits,
                contacts=contacts,
            )
        # Otherwise, perform a cold-start of the dynamics solver
        else:
            self._solver_fd.coldstart()

        # Solve the dual problem to compute the constraint reactions
        self._solver_fd.solve(problem=self._problem_fd)

        # Compute the effective body wrenches applied by the set of
        # active constraints from the respective reaction multipliers
        compute_constraint_body_wrenches(
            model=self._model,
            data=self._data,
            limits=self._limits,
            contacts=contacts,
            jacobians=self._jacobians,
            lambdas_offsets=self._problem_fd.data.vio,
            lambdas_data=self._solver_fd.data.solution.lambdas,
        )

        # Unpack the computed constraint multipliers to the respective joint-limit
        # and contact data for post-processing and optional solver warm-starting
        unpack_constraint_solutions(
            lambdas=self._solver_fd.data.solution.lambdas,
            v_plus=self._solver_fd.data.solution.v_plus,
            model=self._model,
            data=self._data,
            limits=self._limits,
            contacts=contacts,
        )

        # If warmstarting is enabled, update the limits and contacts caches
        # with the constraint reactions generated by the dynamics solver
        # NOTE: This needs to happen after unpacking the multipliers
        if self._settings.warmstart_mode == PADMMWarmStartMode.CONTAINERS:
            self._ws_limits.update(self._limits)
            self._ws_contacts.update(contacts)

    def _update_wrenches(self):
        """
        Computes the total (i.e. net) body wrenches by summing up all individual contributions,
        from joint actuation, joint limits, contacts, and purely external effects.
        """
        update_body_wrenches(self._model.bodies, self._data.bodies)

    def _forward(self, contacts: Contacts | None = None):
        """
        Solves the forward dynamics sub-problem to compute constraint reactions
        and total effective body wrenches applied to each body of the system.
        """
        # Update the dynamics
        self._update_dynamics(contacts=contacts)

        # Compute constraint reactions
        self._update_constraints(contacts=contacts)

        # Post-processing
        self._update_wrenches()

    def _integrate(self):
        """
        Solves the time integration sub-problem to compute the next state of the system.
        """
        # Integrate the state of the system (i.e. of the bodies) to compute the next state
        integrate_euler_semi_implicit(model=self._model, data=self._data)

        # Update the internal joint states based on the current and next body states
        wp.copy(self._data.joints.q_j_p, self._data.joints.q_j)
        self._update_joints_data(q_j_p=self._data.joints.q_j_p)

    def _compute_metrics(self, state_in: State, contacts: Contacts | None = None):
        """
        Computes performance metrics measuring the physical fidelity of the dynamics solver solution.
        """
        if self._settings.compute_metrics:
            self.metrics.reset()
            self._metrics.evaluate(
                sigma=self._solver_fd.data.state.sigma,
                lambdas=self._solver_fd.data.solution.lambdas,
                v_plus=self._solver_fd.data.solution.v_plus,
                model=self._model,
                data=self._data,
                state_p=state_in,
                problem=self._problem_fd,
                jacobians=self._jacobians,
                limits=self._limits,
                contacts=contacts,
            )

    def _advance_time(self):
        """
        Updates simulation time-keeping (i.e. physical time and discrete steps).
        """
        advance_time(self._model.time, self._data.time)
