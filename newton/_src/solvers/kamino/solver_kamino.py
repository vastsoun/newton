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

import warp as wp

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
from .kinematics.jacobians import DenseSystemJacobians
from .kinematics.joints import compute_joints_data
from .kinematics.limits import Limits
from .linalg import LinearSolverType, LLTBlockedSolver
from .solvers.metrics import SolutionMetrics
from .solvers.padmm import PADMMSettings, PADMMSolver, PADMMWarmStartMode
from .solvers.warmstart import WarmstarterContacts, WarmstarterLimits

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
    The settings for the dynamics problem.\n
    See :class:`DualProblemSettings` for more details.
    """

    padmm: PADMMSettings = field(default_factory=PADMMSettings)
    """
    The settings for the dynamics solver.\n
    See :class:`PADMMSettings` for more details.
    """

    warmstart: PADMMWarmStartMode = PADMMWarmStartMode.CONTAINERS
    """
    The warmstart mode to be used for the dynamics solver.\n
    See :class:`PADMMWarmStartMode` for the available options.\n
    Defaults to `PADMMWarmStartMode.CONTAINERS to warmstart from the solver data containers.
    """

    contact_warmstart_method: WarmstarterContacts.Method = WarmstarterContacts.Method.KEY_AND_POSITION
    """
    The method to be used for warm-starting contacts.\n
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

    linear_solver_type: type[LinearSolverType] = LLTBlockedSolver
    """
    The type of linear solver to use for the dynamics problem.\n
    See :class:`LinearSolverType` for available options.\n
    Defaults to :class:`LLTBlockedSolver`.
    """

    rotation_correction: JointCorrectionMode = JointCorrectionMode.TWOPI
    """
    The rotation correction mode to use for rotational DoFs.\n
    See :class:`JointCorrectionMode` for available options.\n
    Defaults to `JointCorrectionMode.TWOPI`.
    """

    def check(self) -> None:
        """Validates relevant solver settings."""
        if not issubclass(self.linear_solver_type, LinearSolverType):
            raise ValueError(
                f"Invalid linear solver type: Expected a subclass of `LinearSolverType`, "
                f"but got {self.linear_solver_type}."
            )
        if not isinstance(self.warmstart, PADMMWarmStartMode):
            raise ValueError(
                f"Invalid warmstart mode: Expected a `PADMMWarmStartMode` enum value, but got {self.warmstart}."
            )
        if not isinstance(self.contact_warmstart_method, WarmstarterContacts.Method):
            raise ValueError(
                f"Invalid contact warmstart method: Expected a `WarmstarterContacts.Method` enum value, "
                f"but got {self.contact_warmstart_method}."
            )
        if not isinstance(self.rotation_correction, JointCorrectionMode):
            raise ValueError(
                f"Invalid rotation correction mode: Expected a `JointCorrectionMode` enum value, "
                f"but got {self.rotation_correction}."
            )
        self.problem.check()
        self.padmm.check()

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

    def __init__(
        self,
        model: Model,
        contacts: Contacts,
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
        # First initialize the base solver
        # NOTE: Although we pass the model here, we will re-assign it below
        # since currently the Kamino defines its own :class`Model` class.
        super().__init__(model=model)
        self._model = model

        # Cache solver settings: If no settings are provided, use defaults
        if settings is None:
            settings = SolverKaminoSettings()
        settings.check()
        self._settings: SolverKaminoSettings = settings

        # Allocate internal time-varying solver data
        self._data = self._model.data()

        # Allocate an internal cache for previous states
        # NOTE: this is used as state[k-1] when computing state[k+1] at step k
        self._state_pp_cache = self._model.state()

        # Allocate a joint-limits interface
        self._limits = Limits(model=self._model, device=self._model.device)

        # Construct the unilateral constraints members in the model info
        make_unilateral_constraints_info(
            model=self._model, data=self._data, limits=self._limits, contacts=contacts, device=self._model.device
        )

        # Allocate Jacobians data on the device
        self._jacobians = DenseSystemJacobians(
            model=self._model,
            limits=self._limits,
            contacts=contacts,
            device=self._model.device,
        )

        # Allocate the dual problem data on the device
        self._problem_fd = DualProblem(
            model=self._model,
            data=self._data,
            limits=self._limits,
            contacts=contacts,
            solver=settings.linear_solver_type,
            settings=settings.problem,
            device=self._model.device,
        )

        # Allocate the forward dynamics solver on the device
        self._solver_fd = PADMMSolver(
            model=self._model,
            settings=settings.padmm,
            warmstart=settings.warmstart,
            use_acceleration=settings.use_solver_acceleration,
            collect_info=settings.collect_solver_info,
            device=self._model.device,
        )

        # Allocate the contacts warmstarter if enabled
        self._ws_limits: WarmstarterLimits | None = None
        self._ws_contacts: WarmstarterContacts | None = None
        if self._settings.warmstart == PADMMWarmStartMode.CONTAINERS:
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
        self._pre_step_cb: Callable[[SolverKamino], None] = None
        self._mid_step_cb: Callable[[SolverKamino], None] = None
        self._post_step_cb: Callable[[SolverKamino], None] = None

        # Initialize the solver internal data
        with wp.ScopedDevice(self._model.device):
            self.reset()

    ###
    # Properties
    ###

    @property
    def data(self) -> ModelData:
        """
        Returns the internal solver data container.
        """
        return self._data

    @property
    def jacobians(self) -> DenseSystemJacobians:
        """
        Returns the system Jacobians container.
        """
        return self._jacobians

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
    def metrics(self) -> SolutionMetrics | None:
        """
        Returns the solution metrics evaluator, if enabled.
        """
        return self._metrics

    ###
    # Configurations
    ###

    def set_pre_step_callback(self, callback: Callable[[SolverKamino], None]):
        """
        Sets a callback to be called before forward dynamics solve.
        """
        self._pre_step_cb = callback

    def set_mid_step_callback(self, callback: Callable[[SolverKamino], None]):
        """
        Sets a callback to be called between forward dynamics solver and state integration.
        """
        self._mid_step_cb = callback

    def set_post_step_callback(self, callback: Callable[[SolverKamino], None]):
        """
        Sets a callback to be called after state integration.
        """
        self._post_step_cb = callback

    ###
    # Solver API
    ###

    def reset(self):
        """
        Resets the internal solver data and cache from the initial state defined in the model.

        **WARNING**: This method is NOT intended for actual selective resetting
        of the simulation state, as would be necessary for Reinforcement Learning
        (RL) applications. It is provided solely for the purpose of performing
        hard-resets of the engine.
        """
        # First copy the initial state to the internal data and cache
        wp.copy(self._data.bodies.q_i, self._model.bodies.q_i_0)
        wp.copy(self._data.bodies.u_i, self._model.bodies.u_i_0)
        wp.copy(self._state_pp_cache.q_i, self._model.bodies.q_i_0)
        wp.copy(self._state_pp_cache.u_i, self._model.bodies.u_i_0)

        # Reset all remaining internal body and joints data
        self._data.bodies.clear_all_wrenches()
        self._data.joints.clear_all()
        self._state_pp_cache.w_i.zero_()
        self._state_pp_cache.lambda_j.zero_()

        # Initialize joints data from the initial state
        self._data.joints.tau_j.zero_()
        self._update_joints_data(state_p=self._state_pp_cache)
        wp.copy(self._state_pp_cache.q_j, self._data.joints.q_j)
        wp.copy(self._state_pp_cache.dq_j, self._data.joints.dq_j)

    @override
    def step(self, state_in: State, state_out: State, control: Control, contacts: Contacts, dt: float):
        """
        Progresses the simulation by a single time-step `dt` given the current
        state `state_in`, control inputs `control`, and set of active contacts
        `contacts`. The updated state is written to `state_out`.

        Args:
            state_in (State): The input current state of the simulation.
            state_out (State): The output next state after time integration.
            control (Control): The input controls applied to the system.
            contacts (Contacts): The set of active contacts.
            dt (float): A uniform time-step to apply uniformly to all worlds of the simulation.
        """
        # Configure the internal per-world solver time-step uniformly from the input argument
        self._model.time.set_uniform_timestep(dt)

        # Copy the new input state and control to the internal solver data
        self._read_current_data(state_in=state_in, control_in=control)

        # Update intermediate quantities of the bodies and joints
        self._update_intermediates()

        # Run limit detection to generate active joint limits
        self._update_limits()

        # Update the constraint state info
        self._update_constraint_info()

        # Update the differential forward kinematics to compute system Jacobians
        self._update_jacobians(contacts=contacts)

        # Compute the body actuation wrenches based on the current control inputs
        self._update_actuation_wrenches()

        # Run the pre-step callback if it has been set
        self._run_prestep_callback()

        # Solve the forward dynamics sub-problem to compute constraint reactions and body wrenches
        self._forward(contacts=contacts)

        # Run the mid-step callback if it has been set
        self._run_midstep_callback()

        # Solve the time integration sub-problem to compute the next state of the system
        self._integrate(state_p=state_in)

        # Compute solver solution metrics if enabled
        self._compute_metrics(state_p=state_in, contacts=contacts)

        # Update time-keeping (i.e. physical time and discrete steps)
        self._advance_time()

        # Run the post-step callback if it has been set
        self._run_poststep_callback()

        # Copy the updated internal solver state to the output state
        self._write_next_state(state_out=state_out)

        # Cache the current state for the next step
        self._state_pp_cache.copy_from(state_in)

    ###
    # Internals - Callback Operations
    ###

    def _run_prestep_callback(self):
        """
        Runs the pre-step callback if it has been set.
        """
        if self._pre_step_cb is not None:
            self._pre_step_cb(self)

    def _run_midstep_callback(self):
        """
        Runs the mid-step callback if it has been set.
        """
        if self._mid_step_cb is not None:
            self._mid_step_cb(self)

    def _run_poststep_callback(self):
        """
        Executes the post-step callback if it has been set.
        """
        if self._post_step_cb is not None:
            self._post_step_cb(self)

    ###
    # Internals - Input/Output Operations
    ###

    def _read_current_data(self, state_in: State, control_in: Control):
        """
        Updates the internal solver data from the input state and control.
        """
        wp.copy(self._data.bodies.q_i, state_in.q_i)
        wp.copy(self._data.bodies.u_i, state_in.u_i)
        wp.copy(self._data.bodies.w_i, state_in.w_i)
        wp.copy(self._data.joints.q_j, state_in.q_j)
        wp.copy(self._data.joints.dq_j, state_in.dq_j)
        wp.copy(self._data.joints.lambda_j, state_in.lambda_j)
        wp.copy(self._data.joints.tau_j, control_in.tau_j)

    def _write_next_state(self, state_out: State):
        """
        Updates the output state from the internal solver data.
        """
        wp.copy(state_out.q_i, self._data.bodies.q_i)
        wp.copy(state_out.u_i, self._data.bodies.u_i)
        wp.copy(state_out.w_i, self._data.bodies.w_i)
        wp.copy(state_out.q_j, self._data.joints.q_j)
        wp.copy(state_out.dq_j, self._data.joints.dq_j)
        wp.copy(state_out.lambda_j, self._data.joints.lambda_j)

    ###
    # Internals - Update Operations
    ###

    def _update_joints_data(self, state_p: State):
        """
        Updates the joint states based on the current body states.
        """
        # Update the joint states based on the updated body states
        # NOTE: We use the previous state `state_p` for post-processing
        # purposes, e.g. account for roll-over of revolute joints etc
        compute_joints_data(
            model=self._model,
            q_j_ref=state_p.q_j,
            data=self._data,
            correction=self._settings.rotation_correction,
        )

    def _update_intermediates(self):
        """
        Updates intermediate quantities required for the forward dynamics solve.
        """
        self._update_joints_data(state_p=self._state_pp_cache)
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

    def _update_jacobians(self, contacts: Contacts):
        """
        Updates the forward kinematics by building the system Jacobians (of actuation and
        constraints) based on the current state of the system and set of active constraints.
        """
        self._jacobians.build(
            model=self._model,
            data=self._data,
            limits=self._limits.data,
            contacts=contacts.data,
            reset_to_zero=True,
        )

    def _update_actuation_wrenches(self):
        """
        Updates the actuation wrenches based on the current control inputs.
        """
        compute_joint_dof_body_wrenches(self._model, self._data, self._jacobians.data)

    def _update_dynamics(self, contacts: Contacts):
        """
        Constructs the forward dynamics problem quantities based on the current state of
        the system, the set of active constraints, and the updated system Jacobians.
        """
        self._problem_fd.build(
            model=self._model,
            data=self._data,
            limits=self._limits.data,
            contacts=contacts.data,
            jacobians=self.jacobians.data,
            reset_to_zero=True,
        )

    def _update_constraints(self, contacts: Contacts):
        """
        Solves the forward dynamics sub-problem to compute constraint
        reactions and body wrenches effected through constraints.
        """
        # If warm-starting is enabled, initialize unilateral
        # constraints containers from the current solver data
        if self._settings.warmstart > PADMMWarmStartMode.NONE:
            if self._settings.warmstart == PADMMWarmStartMode.CONTAINERS:
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
            limits=self._limits.data,
            contacts=contacts.data,
            jacobians=self._jacobians.data,
            lambdas_offsets=self._problem_fd.data.vio,
            lambdas_data=self._solver_fd.data.solution.lambdas,
        )

        # TODO: Could this operation be combined with computing body wrenches to optimize kernel launches?
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
        if self._settings.warmstart == PADMMWarmStartMode.CONTAINERS:
            self._ws_limits.update(self._limits)
            self._ws_contacts.update(contacts)

    def _update_wrenches(self):
        """
        Computes the total (i.e. net) body wrenches by summing up all individual contributions,
        from joint actuation, joint limits, contacts, and purely external effects.
        """
        update_body_wrenches(self._model.bodies, self._data.bodies)

    def _forward(self, contacts: Contacts):
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

    def _integrate(self, state_p: State):
        """
        Solves the time integration sub-problem to compute the next state of the system.
        """
        # Integrate the state of the system (i.e. of the bodies) to compute the next state
        integrate_euler_semi_implicit(model=self._model, data=self._data)

        # Update the internal joint states based on the current and next body states
        self._update_joints_data(state_p=state_p)

    def _compute_metrics(self, state_p: State, contacts: Contacts):
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
                state_p=state_p,
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
