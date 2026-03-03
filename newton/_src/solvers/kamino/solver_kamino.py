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
from typing import Any, Literal

import warp as wp

# Newton imports
from ...core.types import override
from ...sim import (
    Contacts,
    Control,
    Model,
    ModelBuilder,
    State,
)
from ...sim.joints import JointType
from ..flags import SolverNotifyFlags
from ..solver import SolverBase

# Kamino imports
from .core.bodies import update_body_inertias, update_body_wrenches
from .core.control import ControlKamino
from .core.data import DataKamino
from .core.joints import JointCorrectionMode
from .core.model import ModelKamino
from .core.state import StateKamino, compute_body_com_state, compute_body_frame_state
from .core.time import advance_time
from .core.types import float32, int32, transformf, vec3f, vec4f, vec6f
from .dynamics.dual import DualProblem, DualProblemConfig
from .dynamics.wrenches import (
    compute_constraint_body_wrenches,
    compute_joint_dof_body_wrenches,
)
from .geometry.contacts import ContactsKamino, convert_contacts_kamino_to_newton, convert_contacts_newton_to_kamino
from .geometry.detector import CollisionDetector, CollisionDetectorConfig
from .integrators import IntegratorEuler, IntegratorMoreauJean
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
from .kinematics.limits import LimitsKamino
from .kinematics.resets import (
    reset_body_net_wrenches,
    reset_joint_constraint_reactions,
    reset_state_from_base_state,
    reset_state_from_bodies_state,
    reset_state_to_model_default,
    reset_time,
)
from .linalg import ConjugateResidualSolver, IterativeSolver, LinearSolverType, LLTBlockedSolver
from .solvers.fk import ForwardKinematicsSolver, ForwardKinematicsSolverConfig
from .solvers.metrics import SolutionMetrics
from .solvers.padmm import PADMMConfig, PADMMSolver, PADMMWarmStartMode
from .solvers.warmstart import WarmstarterContacts, WarmstarterLimits
from .utils import logger as msg

###
# Module interface
###

__all__ = [
    "SolverKamino",
    "SolverKaminoConfig",
    "SolverKaminoImpl",
]


###
# Types
###


@dataclass
class SolverKaminoConfig:
    """
    A container to hold configurations for :class:`SolverKamino`.
    """

    integrator: Literal["euler", "moreau"] | None = "euler"
    """
    The time-integrator to use for state integration.\n
    See available options in the `integrators` module.\n
    Defaults to `"euler"`.
    """

    problem: DualProblemConfig = field(default_factory=DualProblemConfig)
    """
    Config for the dynamics problem.\n
    See :class:`DualProblemConfig` for more details.
    """

    padmm: PADMMConfig = field(default_factory=PADMMConfig)
    """
    Config for the dynamics solver.\n
    See :class:`PADMMConfig` for more details.
    """

    fk: ForwardKinematicsSolverConfig = field(default_factory=ForwardKinematicsSolverConfig)
    """
    Config for the forward kinematics solver.\n
    See :class:`ForwardKinematicsSolverConfig` for more details.
    """

    warmstart_mode: Literal["none", "internal", "containers"] = "containers"
    """
    Warmstart mode to be used for the dynamics solver.\n
    See :class:`PADMMWarmStartMode` for the available options.\n
    Defaults to `containers` to warmstart from the solver data containers.
    """

    contact_warmstart_method: Literal[
        "key_and_position",
        "geom_pair_net_force",
        "geom_pair_net_wrench",
        "key_and_position_with_net_force_backup",
        "key_and_position_with_net_wrench_backup",
    ] = "key_and_position"
    """
    Method to be used for warm-starting contacts.\n
    See :class:`WarmstarterContacts.Method` for available options.\n
    Defaults to `key_and_position`.
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

    rotation_correction: Literal["twopi", "continuous", "none"] = "twopi"
    """
    The rotation correction mode to use for rotational DoFs.\n
    See :class:`JointCorrectionMode` for available options.\n
    Defaults to `twopi`.
    """

    angular_velocity_damping: float = 0.0
    """
    A damping factor applied to the angular velocity of bodies during state integration.\n
    This can help stabilize simulations with large time steps or high angular velocities.\n
    Defaults to `0.0` (i.e. no damping).
    """

    sparse: bool = False
    """
    Flag to indicate whether the solver should use sparse data representations.
    """

    sparse_jacobian: bool = False
    """
    Flag to indicate whether the solver should use sparse data representations for the Jacobian.
    """

    def check(self) -> None:
        """Validates relevant solver config."""
        if not issubclass(self.linear_solver_type, LinearSolverType):
            raise TypeError(
                "Invalid linear solver type: Expected a subclass of `LinearSolverType`, "
                f"but got {type(self.linear_solver_type)}."
            )
        # Conversion to PADMMWarmStartMode will raise an error if the input string is invalid.
        PADMMWarmStartMode.from_string(self.warmstart_mode)
        # Conversion to WarmstarterContacts.Method will raise an error if the input string is invalid.
        WarmstarterContacts.Method.from_string(self.contact_warmstart_method)
        # Conversion to JointCorrectionMode will raise an error if the input string is invalid.
        JointCorrectionMode.from_string(self.rotation_correction)
        if self.sparse and not self.sparse_jacobian:
            raise ValueError(
                "Sparsity setting mismatch: `sparse` solver option requires that `sparse_jacobian` is set to `True`."
            )
        self.problem.check()
        self.padmm.check()
        self.fk.check()

    def __post_init__(self):
        """Post-initialization to validate config."""
        self.check()


###
# Interfaces
###


class SolverKaminoImpl(SolverBase):
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

    After constructing :class:`ModelKamino`, :class:`StateKamino`, :class:`ControlKamino` and :class:`ContactsKamino`
    objects, this physics solver may be used to advance the simulation state forward in time.

    Example
    -------

    .. code-block:: python

        contacts = ...
        config = newton.solvers.kamino.SolverKaminoConfig()
        solver = newton.solvers.SolverKamino(model, contacts, config)

        # simulation loop
        for i in range(100):
            solver.step(state_in, state_out, control, contacts, dt)
            state_in, state_out = state_out, state_in
    """

    ResetCallbackType = Callable[["SolverKamino", StateKamino], None]
    """Defines the type signature for reset callback functions."""

    StepCallbackType = Callable[["SolverKamino", StateKamino, StateKamino, ControlKamino, ContactsKamino], None]
    """Defines the type signature for step callback functions."""

    def __init__(
        self,
        model: ModelKamino,
        contacts: ContactsKamino | None = None,
        config: SolverKaminoConfig | None = None,
    ):
        """
        Initializes the Kamino physics solver for the given set of multi-body systems
        defined in `model`, and the total contact allocations defined in `contacts`.

        Explicit solver config may be provided through the `config` argument. If no
        config is provided, a default config will be used.

        Args:
            model (ModelKamino): The multi-body systems model to simulate.
            contacts (ContactsKamino): The contact data container for the simulation.
            config (SolverKaminoConfig | None): Optional solver config.
        """
        # Ensure the input containers are valid
        if not isinstance(model, ModelKamino):
            raise TypeError(f"Invalid model container: Expected a `ModelKamino` instance, but got {type(model)}.")
        if contacts is not None and not isinstance(contacts, ContactsKamino):
            raise TypeError(
                f"Invalid contacts container: Expected a `ContactsKamino` instance, but got {type(contacts)}."
            )
        if config is not None and not isinstance(config, SolverKaminoConfig):
            raise TypeError(f"Invalid solver config: Expected a `SolverKaminoConfig` instance, but got {type(config)}.")

        # First initialize the base solver
        # NOTE: Although we pass the model here, we will re-assign it below
        # since currently Kamino defines its own :class`ModelKamino` class.
        super().__init__(model=model)
        self._model = model

        # Cache solver config: If no config is provided, use default
        if config is None:
            config = SolverKaminoConfig()
        config.check()
        self._config: SolverKaminoConfig = config
        self._warmstart_mode: PADMMWarmStartMode = PADMMWarmStartMode.from_string(config.warmstart_mode)
        self._rotation_correction: JointCorrectionMode = JointCorrectionMode.from_string(config.rotation_correction)

        # TODO: We need to rework these checks and potentially handle this check with the dynamics problem
        # TODO: Also consider raising an error here instead of a warning
        # Override the linear solver type to an iterative solver if
        # sparsity is enabled but the provided solver is not iterative
        if self._config.sparse and not issubclass(self._config.linear_solver_type, IterativeSolver):
            msg.warning(
                f"Sparse dynamics requires an iterative solver, but got '{self._config.linear_solver_type.__name__}'."
                " Defaulting to 'ConjugateResidualSolver' as the PADMM linear solver."
            )
            self._config.linear_solver_type = ConjugateResidualSolver

        # Allocate internal time-varying solver data
        self._data = self._model.data()

        # Allocate a joint-limits interface
        self._limits = LimitsKamino(model=self._model, device=self._model.device)

        # Construct the unilateral constraints members in the model info
        make_unilateral_constraints_info(model=self._model, data=self._data, limits=self._limits, contacts=contacts)

        msg.error("model_kamino.info.num_joint_cts: %s", self._model.info.num_joint_cts)
        msg.error("model_kamino.info.num_joint_dynamic_cts: %s", self._model.info.num_joint_dynamic_cts)
        msg.error("model_kamino.info.num_joint_kinematic_cts: %s", self._model.info.num_joint_kinematic_cts)
        msg.error("model_kamino.info.total_cts_offset: %s", self._model.info.total_cts_offset)
        msg.error("model_kamino.info.joint_dynamic_cts_group_offset: %s", self._model.info.joint_dynamic_cts_group_offset)
        msg.error("model_kamino.info.joint_kinematic_cts_group_offset: %s", self._model.info.joint_kinematic_cts_group_offset)
        msg.error("model_kamino.joints.act_type: %s", self._model.joints.act_type)
        msg.error("model_kamino.joints.a_j: %s", self._model.joints.a_j)
        msg.error("model_kamino.joints.k_p_j: %s", self._model.joints.k_p_j)
        msg.error("model_kamino.joints.k_d_j: %s", self._model.joints.k_d_j)
        msg.error("model_kamino.joints.num_cts: %s", self._model.joints.num_cts)
        msg.error("model_kamino.joints.num_dynamic_cts: %s", self._model.joints.num_dynamic_cts)
        msg.error("model_kamino.joints.num_kinematic_cts: %s", self._model.joints.num_kinematic_cts)
        msg.error("model_kamino.joints.cts_offset: %s", self._model.joints.cts_offset)
        msg.error("model_kamino.joints.dynamic_cts_offset: %s", self._model.joints.dynamic_cts_offset)
        msg.error("model_kamino.joints.kinematic_cts_offset: %s", self._model.joints.kinematic_cts_offset)

        # Allocate Jacobians data on the device
        if self._config.sparse_jacobian:
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
        linear_solver_kwargs = dict(self._config.linear_solver_kwargs)
        if self._config.avoid_graph_conditionals and issubclass(self._config.linear_solver_type, IterativeSolver):
            linear_solver_kwargs.setdefault("avoid_graph_conditionals", True)
        self._problem_fd = DualProblem(
            model=self._model,
            data=self._data,
            limits=self._limits,
            contacts=contacts,
            solver=self._config.linear_solver_type,
            solver_kwargs=linear_solver_kwargs,
            config=self._config.problem,
            device=self._model.device,
            sparse=self._config.sparse,
        )

        # Allocate the forward dynamics solver on the device
        self._solver_fd = PADMMSolver(
            model=self._model,
            config=self._config.padmm,
            warmstart=self._warmstart_mode,
            use_acceleration=self._config.use_solver_acceleration,
            collect_info=self._config.collect_solver_info,
            avoid_graph_conditionals=self._config.avoid_graph_conditionals,
            device=self._model.device,
        )

        # Allocate the forward kinematics solver on the device
        self._solver_fk = ForwardKinematicsSolver(model=self._model, config=self._config.fk)

        # Create the time-integrator instance based on the config
        if self._config.integrator == "euler":
            self._integrator = IntegratorEuler(model=self._model)
        elif self._config.integrator == "moreau":
            self._integrator = IntegratorMoreauJean(model=self._model)
        else:
            raise ValueError(
                f"Unsupported integrator type: Expected 'euler' or 'moreau', but got {self._config.integrator}."
            )

        # Allocate additional internal data for reset operations
        with wp.ScopedDevice(self._model.device):
            self._all_worlds_mask = wp.ones(shape=(self._model.size.num_worlds,), dtype=int32)
            self._base_q = wp.zeros(shape=(self._model.size.num_worlds,), dtype=transformf)
            self._base_u = wp.zeros(shape=(self._model.size.num_worlds,), dtype=vec6f)
            self._bodies_u_zeros = wp.zeros(shape=(self._model.size.sum_of_num_bodies,), dtype=vec6f)
            self._actuators_q = wp.zeros(shape=(self._model.size.sum_of_num_actuated_joint_coords,), dtype=float32)
            self._actuators_u = wp.zeros(shape=(self._model.size.sum_of_num_actuated_joint_dofs,), dtype=float32)

        # Allocate the contacts warmstarter if enabled
        self._ws_limits: WarmstarterLimits | None = None
        self._ws_contacts: WarmstarterContacts | None = None
        if self._warmstart_mode == PADMMWarmStartMode.CONTAINERS:
            self._ws_limits = WarmstarterLimits(limits=self._limits)
            self._ws_contacts = WarmstarterContacts(
                contacts=contacts,
                method=WarmstarterContacts.Method.from_string(self._config.contact_warmstart_method),
            )

        # Allocate the solution metrics evaluator if enabled
        self._metrics: SolutionMetrics | None = None
        if self._config.compute_metrics:
            self._metrics = SolutionMetrics(model=self._model)

        # Initialize callbacks
        self._pre_reset_cb: SolverKaminoImpl.ResetCallbackType | None = None
        self._post_reset_cb: SolverKaminoImpl.ResetCallbackType | None = None
        self._pre_step_cb: SolverKaminoImpl.StepCallbackType | None = None
        self._mid_step_cb: SolverKaminoImpl.StepCallbackType | None = None
        self._post_step_cb: SolverKaminoImpl.StepCallbackType | None = None

        # Initialize all internal solver data
        with wp.ScopedDevice(self._model.device):
            self._reset()

    ###
    # Properties
    ###

    @property
    def config(self) -> SolverKaminoConfig:
        """
        Returns the host-side cache of high-level solver config.
        """
        return self._config

    @property
    def device(self) -> wp.DeviceLike:
        """
        Returns the device where the solver data is allocated.
        """
        return self._model.device

    @property
    def data(self) -> DataKamino:
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
        state_out: StateKamino,
        world_mask: wp.array | None = None,
        actuator_q: wp.array | None = None,
        actuator_u: wp.array | None = None,
        joint_q: wp.array | None = None,
        joint_u: wp.array | None = None,
        base_q: wp.array | None = None,
        base_u: wp.array | None = None,
        bodies_q: wp.array | None = None,
        bodies_u: wp.array | None = None,
    ):
        """
        Resets the simulation state given a combination of desired base body
        and joint states, as well as an optional per-world mask array indicating
        which worlds should be reset. The reset state is written to `state_out`.

        For resets given absolute quantities like base body poses, the
        `state_out` must initially contain the current state of the simulation.

        Args:
            state_out (StateKamino):
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
            bodies_q (wp.array, optional):
                Optional array of target body poses.\n
                Shape of `(num_bodies,)` and type :class:`wp.transformf`
            bodies_u (wp.array, optional):
                Optional array of target body twists.\n
                Shape of `(num_bodies,)` and type :class:`wp.spatial_vectorf`
        """

        # Ensure the input reset targets are valid
        def _check_length(data: wp.array, name: str, expected: int):
            if data is not None and data.shape[0] != expected:
                raise ValueError(f"Invalid {name} shape: Expected ({expected},), but got {data.shape}.")

        _check_length(joint_q, "joint_q", self._model.size.sum_of_num_joint_coords)
        _check_length(joint_u, "joint_u", self._model.size.sum_of_num_joint_dofs)
        _check_length(actuator_q, "actuator_q", self._model.size.sum_of_num_actuated_joint_coords)
        _check_length(actuator_u, "actuator_u", self._model.size.sum_of_num_actuated_joint_dofs)
        _check_length(base_q, "base_q", self._model.size.num_worlds)
        _check_length(base_u, "base_u", self._model.size.num_worlds)
        _check_length(bodies_q, "bodies_q", self._model.size.sum_of_num_bodies)
        _check_length(bodies_u, "bodies_u", self._model.size.sum_of_num_bodies)
        _check_length(world_mask, "world_mask", self._model.size.num_worlds)

        # Ensure that only joint or actuator targets are provided
        if (joint_q is not None or joint_u is not None) and (actuator_q is not None or actuator_u is not None):
            raise ValueError("Combined joint and actuator targets are not supported. Only one type may be provided.")

        # Ensure that joint/actuator velocity-only resets are prevented
        if (joint_q is None and joint_u is not None) or (actuator_q is None and actuator_u is not None):
            raise ValueError("Velocity-only joint or actuator resets are not supported.")

        # Run the pre-reset callback if it has been set
        self._run_pre_reset_callback(state_out=state_out)

        # Determine the effective world mask to use for the reset operation
        _world_mask = world_mask if world_mask is not None else self._all_worlds_mask

        # Detect mode
        base_reset = base_q is not None or base_u is not None
        joint_reset = joint_q is not None or actuator_q is not None
        bodies_reset = bodies_q is not None or bodies_u is not None

        # If no reset targets are provided, reset all bodies to the model default state
        if not base_reset and not joint_reset and not bodies_reset:
            self._reset_to_default_state(
                state_out=state_out,
                world_mask=_world_mask,
            )

        # If only base targets are provided, uniformly reset all bodies to the given base states
        elif base_reset and not joint_reset and not bodies_reset:
            self._reset_to_base_state(
                state_out=state_out,
                world_mask=_world_mask,
                base_q=base_q,
                base_u=base_u,
            )

        # If a joint target is provided, use the FK solver to reset the bodies accordingly
        elif joint_reset and not bodies_reset:
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

        # If body targets are provided, reset bodies directly
        elif not base_reset and not joint_reset and bodies_reset:
            self._reset_to_bodies_state(
                state_out=state_out,
                world_mask=_world_mask,
                bodies_q=bodies_q,
                bodies_u=bodies_u,
            )

        # If no valid combination of reset targets is provided, raise an error
        else:
            raise ValueError(
                "Unsupported reset combination with: "
                f" actuator_q: {actuator_q is not None}, actuator_u: {actuator_u is not None},"
                f" joint_q: {joint_q is not None}, joint_u: {joint_u is not None},"
                f" base_q: {base_q is not None}, base_u: {base_u is not None}."
                f" bodies_q: {bodies_q is not None}, bodies_u: {bodies_u is not None}."
            )

        # Post-process the reset operation
        self._reset_post_process(world_mask=_world_mask)

        # Run the post-reset callback if it has been set
        self._run_post_reset_callback(state_out=state_out)

    @override
    def step(
        self,
        state_in: StateKamino,
        state_out: StateKamino,
        control: ControlKamino,
        contacts: ContactsKamino | None = None,
        detector: CollisionDetector | None = None,
        dt: float | None = None,
    ):
        """
        Progresses the simulation by a single time-step `dt` given the current
        state `state_in`, control inputs `control`, and set of active contacts
        `contacts`. The updated state is written to `state_out`.

        Args:
            state_in (StateKamino):
                The input current state of the simulation.
            state_out (StateKamino):
                The output next state after time integration.
            control (ControlKamino):
                The input controls applied to the system.
            contacts (ContactsKamino, optional):
                The set of active contacts.
            detector (CollisionDetector, optional):
                An optional collision detector to use for generating contacts at the current state.\n
                If `None`, the `contacts` data will be used as the current set of active contacts.
            dt (float, optional):
                A uniform time-step to apply uniformly to all worlds of the simulation.
        """
        # If specified, configure the internal per-world solver time-step uniformly from the input argument
        if dt is not None:
            self._model.time.set_uniform_timestep(dt)

        # Copy the new input state and control to the internal solver data
        self._read_step_inputs(state_in=state_in, control_in=control)

        # Execute state integration:
        #  - Optionally calls limit and contact detection to generate unilateral constraints
        #  - Solves the forward dynamics sub-problem to compute constraint reactions
        #  - Integrates the state forward in time
        self._integrator.integrate(
            forward=self._solve_forward_dynamics,
            model=self._model,
            data=self._data,
            state_in=state_in,
            state_out=state_out,
            control=control,
            limits=self._limits,
            contacts=contacts,
            detector=detector,
        )

        # Update the internal joint states from the
        # updated body states after time-integration
        self._update_joints_data()

        # Compute solver solution metrics if enabled
        self._compute_metrics(state_in=state_in, contacts=contacts)

        # Update time-keeping (i.e. physical time and discrete steps)
        self._advance_time()

        # Run the post-step callback if it has been set
        self._run_poststep_callback(state_in, state_out, control, contacts)

        # Copy the updated internal solver state to the output state
        self._write_step_output(state_out=state_out)

    @override
    def notify_model_changed(self, flags: int):
        pass  # TODO

    @override
    def update_contacts(self, contacts: Contacts) -> None:
        pass  # TODO

    @override
    @classmethod
    def register_custom_attributes(cls, flags: int):
        pass  # TODO

    ###
    # Internals - Callback Operations
    ###

    def _run_pre_reset_callback(self, state_out: StateKamino):
        """
        Runs the pre-reset callback if it has been set.
        """
        if self._pre_reset_cb is not None:
            self._pre_reset_cb(self, state_out)

    def _run_post_reset_callback(self, state_out: StateKamino):
        """
        Runs the post-reset callback if it has been set.
        """
        if self._post_reset_cb is not None:
            self._post_reset_cb(self, state_out)

    def _run_prestep_callback(
        self, state_in: StateKamino, state_out: StateKamino, control: ControlKamino, contacts: ContactsKamino
    ):
        """
        Runs the pre-step callback if it has been set.
        """
        if self._pre_step_cb is not None:
            self._pre_step_cb(self, state_in, state_out, control, contacts)

    def _run_midstep_callback(
        self, state_in: StateKamino, state_out: StateKamino, control: ControlKamino, contacts: ContactsKamino
    ):
        """
        Runs the mid-step callback if it has been set.
        """
        if self._mid_step_cb is not None:
            self._mid_step_cb(self, state_in, state_out, control, contacts)

    def _run_poststep_callback(
        self, state_in: StateKamino, state_out: StateKamino, control: ControlKamino, contacts: ContactsKamino
    ):
        """
        Executes the post-step callback if it has been set.
        """
        if self._post_step_cb is not None:
            self._post_step_cb(self, state_in, state_out, control, contacts)

    ###
    # Internals - Input/Output Operations
    ###

    def _read_step_inputs(self, state_in: StateKamino, control_in: ControlKamino):
        """
        Updates the internal solver data from the input state and control.
        """
        # TODO: Remove corresponding data copies
        # by directly using the input containers
        wp.copy(self._data.bodies.q_i, state_in.q_i)
        wp.copy(self._data.bodies.u_i, state_in.u_i)
        wp.copy(self._data.bodies.w_i, state_in.w_i)
        wp.copy(self._data.bodies.w_e_i, state_in.w_i_e)
        wp.copy(self._data.joints.q_j, state_in.q_j)
        wp.copy(self._data.joints.q_j_p, state_in.q_j_p)
        wp.copy(self._data.joints.dq_j, state_in.dq_j)
        wp.copy(self._data.joints.lambda_j, state_in.lambda_j)
        wp.copy(self._data.joints.tau_j, control_in.tau_j)
        wp.copy(self._data.joints.q_j_ref, control_in.q_j_ref)
        wp.copy(self._data.joints.dq_j_ref, control_in.dq_j_ref)
        wp.copy(self._data.joints.tau_j_ref, control_in.tau_j_ref)

    def _write_step_output(self, state_out: StateKamino):
        """
        Updates the output state from the internal solver data.
        """
        # TODO: Remove corresponding data copies
        # by directly using the input containers
        wp.copy(state_out.q_i, self._data.bodies.q_i)
        wp.copy(state_out.u_i, self._data.bodies.u_i)
        wp.copy(state_out.w_i, self._data.bodies.w_i)
        wp.copy(state_out.w_i_e, self._data.bodies.w_e_i)
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
        self._data.joints.reset_state(q_j_0=self._model.joints.q_j_0)
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

    def _reset_to_default_state(self, state_out: StateKamino, world_mask: wp.array):
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
        state_out: StateKamino,
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
        )

    def _reset_to_bodies_state(
        self,
        state_out: StateKamino,
        world_mask: wp.array,
        bodies_q: wp.array | None = None,
        bodies_u: wp.array | None = None,
    ):
        """
        Resets the simulation to the given rigid body states.
        There is no check that the provided states satisfy any kinematic constraints.
        """

        # use initial model poses if not provided
        _bodies_q = bodies_q if bodies_q is not None else self._model.bodies.q_i_0
        # use zero body velocities if not provided
        _bodies_u = bodies_u if bodies_u is not None else self._bodies_u_zeros

        reset_state_from_bodies_state(
            model=self._model,
            state_out=state_out,
            world_mask=world_mask,
            bodies_q=_bodies_q,
            bodies_u=_bodies_u,
        )

    def _reset_with_fk_solve(
        self,
        state_out: StateKamino,
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

    def _update_joints_data(self, q_j_p: wp.array | None = None):
        """
        Updates the joint states based on the current body states.
        """
        # Use the provided previous joint states if given,
        # otherwise use the internal cached joint states
        if q_j_p is not None:
            _q_j_p = q_j_p
        else:
            wp.copy(self._data.joints.q_j_p, self._data.joints.q_j)
            _q_j_p = self._data.joints.q_j_p

        # Update the joint states based on the updated body states
        # NOTE: We use the previous state `state_p` for post-processing
        # purposes, e.g. account for roll-over of revolute joints etc
        compute_joints_data(
            model=self._model,
            data=self._data,
            q_j_p=_q_j_p,
            correction=self._rotation_correction,
        )

    def _update_intermediates(self, state_in: StateKamino):
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

    def _update_jacobians(self, contacts: ContactsKamino | None = None):
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

    def _update_dynamics(self, contacts: ContactsKamino | None = None):
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

    def _update_constraints(self, contacts: ContactsKamino | None = None):
        """
        Solves the forward dynamics sub-problem to compute constraint
        reactions and body wrenches effected through constraints.
        """
        # If warm-starting is enabled, initialize unilateral
        # constraints containers from the current solver data
        if self._warmstart_mode > PADMMWarmStartMode.NONE:
            if self._warmstart_mode == PADMMWarmStartMode.CONTAINERS:
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
        if self._warmstart_mode == PADMMWarmStartMode.CONTAINERS:
            self._ws_limits.update(self._limits)
            self._ws_contacts.update(contacts)

    def _update_wrenches(self):
        """
        Computes the total (i.e. net) body wrenches by summing up all individual contributions,
        from joint actuation, joint limits, contacts, and purely external effects.
        """
        update_body_wrenches(self._model.bodies, self._data.bodies)

    def _forward(self, contacts: ContactsKamino | None = None):
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

    def _solve_forward_dynamics(
        self,
        state_in: StateKamino,
        state_out: StateKamino,
        control: ControlKamino,
        limits: LimitsKamino | None = None,  # TODO: Fix this interface
        contacts: ContactsKamino | None = None,
        detector: CollisionDetector | None = None,
    ):
        """
        TODO
        """
        # Update intermediate quantities of the bodies and joints
        # NOTE: We update the intermediate joint and body data here
        # to ensure that they consistent with the current state.
        # This is to handle cases when the forward dynamics may be
        # evaluated at intermediate points of the discrete time-step
        # (and potentially multiple times). The intermediate data is
        # then used to perform limit and contact detection, as well
        # as to evaluate kinematics and dynamics quantities such as
        # the system Jacobians and generalized mass matrix.
        self._update_intermediates(state_in=state_in)

        # If a collision detector is provided, use it to generate
        # update the set of active contacts at the current state
        if detector is not None:
            detector.collide(data=self._data, state=state_in, contacts=contacts)

        # If a limits container/detector is provided, run joint-limit
        # detection to generate active joint limits at the current state
        if limits is not None:
            limits.detect(self._model, self._data)

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

    def _compute_metrics(self, state_in: StateKamino, contacts: ContactsKamino | None = None):
        """
        Computes performance metrics measuring the physical fidelity of the dynamics solver solution.
        """
        if self._config.compute_metrics:
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


class SolverKamino(SolverBase):
    """
    TODO
    """

    def __init__(
        self,
        model: Model,
        solver_config: SolverKaminoConfig | None = None,
        collision_detector_config: CollisionDetectorConfig | None = None,
    ):
        """
        TODO
        """
        # Initialize the base solver
        super().__init__(model=model)

        # Validate that the model does not contain unsupported components
        self._validate_model_compatibility(model)

        # Create a Kamino model from the Newton model
        self._model_kamino = ModelKamino.from_newton(model)

        # Create a collision detector
        self._collision_detector_kamino = CollisionDetector(
            model=self._model_kamino,
            config=collision_detector_config,
        )

        # Capture a reference to the contacts container
        self._contacts_kamino: ContactsKamino = self._collision_detector_kamino.contacts

        # Initialize the internal Kamino solver
        self._solver_kamino = SolverKaminoImpl(
            model=self._model_kamino,
            contacts=self._contacts_kamino,
            config=solver_config,
        )

        # Reference to the latest state from the latest step output, used by update_contacts()
        self._state_p: State | None = None

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
        self._solver_kamino.reset(
            state_out=StateKamino.from_newton(self._model_kamino.size, self.model, state_out),
            world_mask=world_mask,
            actuator_q=actuator_q,
            actuator_u=actuator_u,
            joint_q=joint_q,
            joint_u=joint_u,
            base_q=base_q,
            base_u=base_u,
        )

    @override
    def step(self, state_in: State, state_out: State, control: Control, contacts: Contacts, dt: float):
        """
        Simulate the model for a given time step using the given control input.

        When ``contacts`` is not ``None`` (i.e. produced by :meth:`Model.collide`),
        those contacts are converted to Kamino's internal format and used directly,
        bypassing Kamino's own collision detector.  When ``contacts`` is ``None``,
        Kamino's internal collision pipeline runs as a fallback.

        Args:
            state_in (State): The input state.
            state_out (State): The output state.
            control (Control): The control input.
                Defaults to `None` which means the control values from the
                :class:`Model` are used.
            contacts (Contacts): The contact information from Newton's collision
                pipeline, or ``None`` to use Kamino's internal collision detector.
            dt (float): The time step (typically in seconds).
        """
        # Interface the input state and control
        # containers to Kamino's equivalents
        # NOTE: These should produce zero-copy views/references
        # to the arrays of the source Newton containers.
        state_in_kamino = StateKamino.from_newton(self._model_kamino.size, self.model, state_in)
        state_out_kamino = StateKamino.from_newton(self._model_kamino.size, self.model, state_out)
        control_kamino = ControlKamino.from_newton(control)

        # If contacts are provided, use them directly, bypassing Kamino's collision detector
        if contacts is not None:
            convert_contacts_newton_to_kamino(self.model, state_in, contacts, self._contacts_kamino)
            _detector = None
        # Otherwise, use Kamino's internal collision detector to generate contacts
        else:
            _detector = self._collision_detector_kamino

        # Convert Newton body-frame poses to Kamino CoM-frame poses using
        # Kamino's corrected body-com offsets (can differ from Newton model data).
        # TODO: state_in_kamino.convert_to_body_com_state(model=self.model)
        compute_body_com_state(
            body_com=self._model_kamino.bodies.i_r_com_i,
            body_q=state_in_kamino.q_i,
            body_q_com=state_in_kamino.q_i,
        )

        # Step the physics solver
        self._solver_kamino.step(
            state_in=state_in_kamino,
            state_out=state_out_kamino,
            control=control_kamino,
            contacts=self._contacts_kamino,
            detector=_detector,
            dt=dt,
        )

        # Convert back from Kamino CoM-frame to Newton body-frame poses using
        # the same corrected body-com offsets as the forward conversion.
        # state_in_kamino.convert_to_body_frame_state(model=self.model)
        # state_out_kamino.convert_to_body_frame_state(model=self.model)
        compute_body_frame_state(
            body_com=self._model_kamino.bodies.i_r_com_i,
            body_q_com=state_in_kamino.q_i,
            body_q=state_in_kamino.q_i,
        )
        compute_body_frame_state(
            body_com=self._model_kamino.bodies.i_r_com_i,
            body_q_com=state_out_kamino.q_i,
            body_q=state_out_kamino.q_i,
        )

        # Keep a reference for update_contacts() which needs body_q to
        # transform world-space contact positions to body-local frame.
        self._state_p = state_out

    @override
    def notify_model_changed(self, flags: int):
        """Propagate Newton model property changes to Kamino's internal ModelKamino.

        Args:
            flags: Bitmask of :class:`SolverNotifyFlags` indicating which properties changed.
        """
        if flags & SolverNotifyFlags.MODEL_PROPERTIES:
            self._update_gravity()

        if flags & SolverNotifyFlags.BODY_PROPERTIES:
            pass  # TODO: convert to CoM-frame if body_q_i_0 is changed at runtime?

        if flags & SolverNotifyFlags.BODY_INERTIAL_PROPERTIES:
            # Kamino's RigidBodiesModel references Newton's arrays directly
            # (m_i, inv_m_i, i_I_i, inv_i_I_i, i_r_com_i), so no copy needed.
            pass

        if flags & SolverNotifyFlags.SHAPE_PROPERTIES:
            pass  # TODO: ???

        if flags & SolverNotifyFlags.JOINT_PROPERTIES:
            self._update_joint_transforms()

        if flags & SolverNotifyFlags.JOINT_DOF_PROPERTIES:
            # Joint limits (q_j_min, q_j_max, dq_j_max, tau_j_max) are direct
            # references to Newton's arrays, so no copy needed.
            pass

        if flags & SolverNotifyFlags.ACTUATOR_PROPERTIES:
            pass  # TODO: ???

        if flags & SolverNotifyFlags.CONSTRAINT_PROPERTIES:
            pass  # TODO: ???

        unsupported = flags & ~(
            SolverNotifyFlags.MODEL_PROPERTIES
            | SolverNotifyFlags.BODY_INERTIAL_PROPERTIES
            | SolverNotifyFlags.JOINT_PROPERTIES
            | SolverNotifyFlags.JOINT_DOF_PROPERTIES
        )
        if unsupported:
            msg.warning(
                "SolverKamino.notify_model_changed: flags 0x%x not yet supported",
                unsupported,
            )

    def _update_gravity(self):
        """Re-derive Kamino's GravityModel from Newton's model.gravity."""
        import numpy as np  # noqa: PLC0415

        gravity_np = self.model.gravity.numpy()
        num_worlds = self.model.num_worlds
        g_dir_acc_np = np.zeros((num_worlds, 4), dtype=np.float32)
        vector_np = np.zeros((num_worlds, 4), dtype=np.float32)

        for w in range(num_worlds):
            g_vec = gravity_np[w, :]
            accel = float(np.linalg.norm(g_vec))
            if accel > 0.0:
                direction = g_vec / accel
            else:
                direction = np.array([0.0, 0.0, -1.0])
            g_dir_acc_np[w, :3] = direction
            g_dir_acc_np[w, 3] = accel
            vector_np[w, :3] = g_vec
            vector_np[w, 3] = 1.0

        device = self.model.device
        wp.copy(self._model_kamino.gravity.g_dir_acc, wp.array(g_dir_acc_np, dtype=vec4f, device=device))
        wp.copy(self._model_kamino.gravity.vector, wp.array(vector_np, dtype=vec4f, device=device))

    def _update_joint_transforms(self):
        """
        Re-derive Kamino joint anchors and axes from Newton's joint_X_p / joint_X_c.

        Called when :data:`SolverNotifyFlags.JOINT_PROPERTIES` is raised,
        indicating that ``model.joint_X_p`` or ``model.joint_X_c`` may have
        changed at runtime (e.g. animated root transforms).
        """
        import numpy as np  # noqa: PLC0415

        from .core.joints import JointDoFType  # noqa: PLC0415

        model = self.model
        joints_km = self._model_kamino.joints

        joint_X_p_np = model.joint_X_p.numpy()
        joint_X_c_np = model.joint_X_c.numpy()
        body_com_np = model.body_com.numpy()
        joint_parent_np = model.joint_parent.numpy()
        joint_child_np = model.joint_child.numpy()
        joint_axis_np = model.joint_axis.numpy()
        joint_dof_dim_np = model.joint_dof_dim.numpy()
        joint_qd_start_np = model.joint_qd_start.numpy()
        joint_limit_lower_np = model.joint_limit_lower.numpy()
        joint_limit_upper_np = model.joint_limit_upper.numpy()
        dof_type_np = joints_km.dof_type.numpy()

        n_joints = model.joint_count
        B_r_Bj_np = np.zeros((n_joints, 3), dtype=np.float32)
        F_r_Fj_np = np.zeros((n_joints, 3), dtype=np.float32)
        X_j_np = np.zeros((n_joints, 9), dtype=np.float32)

        for j in range(n_joints):
            dof_type_j = JointDoFType(int(dof_type_np[j]))
            dof_dim_j = (int(joint_dof_dim_np[j][0]), int(joint_dof_dim_np[j][1]))
            dofs_start_j = int(joint_qd_start_np[j])
            ndofs_j = dof_type_j.num_dofs
            joint_axes_j = joint_axis_np[dofs_start_j : dofs_start_j + ndofs_j]
            joint_q_min_j = joint_limit_lower_np[dofs_start_j : dofs_start_j + ndofs_j]
            joint_q_max_j = joint_limit_upper_np[dofs_start_j : dofs_start_j + ndofs_j]
            R_axis_j = JointDoFType.from_newton(dof_type_j, dof_dim_j, joint_axes_j, joint_q_min_j, joint_q_max_j)

            parent_bid = int(joint_parent_np[j])
            p_r_p_com = wp.vec3f(body_com_np[parent_bid]) if parent_bid >= 0 else wp.vec3f(0.0, 0.0, 0.0)
            c_r_c_com = wp.vec3f(body_com_np[int(joint_child_np[j])])

            X_p_j = wp.transformf(*joint_X_p_np[j, :])
            X_c_j = wp.transformf(*joint_X_c_np[j, :])
            q_p_j = wp.transform_get_rotation(X_p_j)
            p_r_p_j = wp.transform_get_translation(X_p_j)
            c_r_c_j = wp.transform_get_translation(X_c_j)

            B_r_Bj_np[j, :] = p_r_p_j - p_r_p_com
            F_r_Fj_np[j, :] = c_r_c_j - c_r_c_com
            X_j_np[j, :] = wp.quat_to_matrix(q_p_j) @ R_axis_j

        device = model.device
        joints_km.B_r_Bj.assign(wp.array(B_r_Bj_np, dtype=vec3f, device=device))
        joints_km.F_r_Fj.assign(wp.array(F_r_Fj_np, dtype=vec3f, device=device))
        joints_km.X_j.assign(wp.array(X_j_np.reshape((n_joints, 3, 3)), dtype=wp.mat33f, device=device))

    @override
    def update_contacts(self, contacts: Contacts, state: State | None = None) -> None:
        """
        Converts Kamino contacts to Newton's Contacts format.

        Args:
            contacts: The Newton Contacts object to populate.
            state: Optional simulation state providing ``body_q`` for converting
                world-space contact positions to body-local frame. Falls back to
                the last ``state_out`` from :meth:`step` if not provided.
        """
        # Determine the source state to use for contact conversion
        if state is not None:
            _state = state
        elif self._state_p is not None:
            _state = self._state_p
        # Skip contact conversion if no state is provided and no previous state is available
        else:
            msg.warning(
                "SolverKamino.update_contacts: no state provided and "
                "no previous state available, cannot convert contacts"
            )
            return

        # Convert Kamino's internal contact representation to Newton's format
        convert_contacts_kamino_to_newton(self.model, _state, self._contacts_kamino, contacts)

    @override
    @classmethod
    def register_custom_attributes(cls, builder: ModelBuilder) -> None:
        """
        Register custom attributes for SolverKamino.

        Args:
            builder (ModelBuilder): The model builder to register the custom attributes to.
        """
        # State attributes
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="body_f_total",
                assignment=Model.AttributeAssignment.STATE,
                frequency=Model.AttributeFrequency.BODY,
                dtype=vec6f,
                default=vec6f(0.0),
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="joint_q_prev",
                assignment=Model.AttributeAssignment.STATE,
                frequency=Model.AttributeFrequency.JOINT_COORD,
                dtype=wp.float32,
                default=0.0,
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="joint_lambdas",
                assignment=Model.AttributeAssignment.STATE,
                frequency=Model.AttributeFrequency.JOINT_CONSTRAINT,
                dtype=wp.float32,
                default=0.0,
            )
        )

    @staticmethod
    def _validate_model_compatibility(model: Model):
        """
        Validates that the model does not contain components unsupported by SolverKamino:
        - particles
        - springs
        - triangles, edges, tetrahedra
        - muscles
        - equality constraints
        - distance, cable, or gimbal joints

        Args:
            model (Model): The Newton model to validate.

        Raises:
            ValueError: If the model contains unsupported components.
        """

        unsupported_features = []
        if model.particle_count > 0:
            unsupported_features.append(f"particles (found {model.particle_count})")
        if model.spring_count > 0:
            unsupported_features.append(f"springs (found {model.spring_count})")
        if model.tri_count > 0:
            unsupported_features.append(f"triangle elements (found {model.tri_count})")
        if model.edge_count > 0:
            unsupported_features.append(f"edge elements (found {model.edge_count})")
        if model.tet_count > 0:
            unsupported_features.append(f"tetrahedral elements (found {model.tet_count})")
        if model.muscle_count > 0:
            unsupported_features.append(f"muscles (found {model.muscle_count})")
        if model.equality_constraint_count > 0:
            unsupported_features.append(f"equality constraints (found {model.equality_constraint_count})")

        # Check for unsupported joint types
        if model.joint_count > 0:
            joint_type_np = model.joint_type.numpy()
            joint_dof_dim_np = model.joint_dof_dim.numpy()
            joint_q_start_np = model.joint_q_start.numpy()
            joint_qd_start_np = model.joint_qd_start.numpy()

            unsupported_joint_types = {}

            for j in range(model.joint_count):
                joint_type = int(joint_type_np[j])
                dof_dim = (int(joint_dof_dim_np[j][0]), int(joint_dof_dim_np[j][1]))
                q_count = int(joint_q_start_np[j + 1] - joint_q_start_np[j])
                qd_count = int(joint_qd_start_np[j + 1] - joint_qd_start_np[j])

                # Check for explicitly unsupported joint types
                if joint_type == JointType.DISTANCE:
                    unsupported_joint_types["DISTANCE"] = unsupported_joint_types.get("DISTANCE", 0) + 1
                elif joint_type == JointType.CABLE:
                    unsupported_joint_types["CABLE"] = unsupported_joint_types.get("CABLE", 0) + 1
                # Check for GIMBAL configuration (3 coords, 3 DoFs, 0 linear/3 angular)
                elif joint_type == JointType.D6 and q_count == 3 and qd_count == 3 and dof_dim == (0, 3):
                    unsupported_joint_types["D6 (GIMBAL)"] = unsupported_joint_types.get("D6 (GIMBAL)", 0) + 1

            if len(unsupported_joint_types) > 0:
                joint_desc = [f"{name} ({count} instances)" for name, count in unsupported_joint_types.items()]
                unsupported_features.append("joint types: " + ", ".join(joint_desc))

        # If any unsupported features were found, raise an error
        if len(unsupported_features) > 0:
            error_msg = "SolverKamino cannot simulate this model due to unsupported features:"
            for feature in unsupported_features:
                error_msg += "\n  - " + feature
            raise ValueError(error_msg)
