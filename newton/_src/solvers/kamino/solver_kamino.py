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
from ...sim import (
    Contacts,
    Control,
    Model,
    ModelBuilder,
    State,
)
from ...sim.joints import ActuatorMode
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
from .core.types import float32, int32, quatf, transformf, uint32, vec2f, vec2i, vec3f, vec4f, vec6f
from .dynamics.dual import DualProblem, DualProblemSettings
from .dynamics.wrenches import (
    compute_constraint_body_wrenches,
    compute_joint_dof_body_wrenches,
)
from .geometry import CollisionDetector, CollisionDetectorSettings
from .geometry.contacts import ContactsKamino, make_contact_frame_znorm
from .geometry.keying import build_pair_key2
from .integrators.euler import integrate_euler_semi_implicit
from .kinematics.constraints import (
    make_unilateral_constraints_info,
    unpack_constraint_solutions,
    update_constraints_info,
)
from .kinematics.jacobians import DenseSystemJacobians
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
    reset_state_to_model_default,
    reset_time,
)
from .linalg import LinearSolverType, LLTBlockedSolver
from .solvers.fk import ForwardKinematicsSolver, ForwardKinematicsSolverSettings
from .solvers.metrics import SolutionMetrics
from .solvers.padmm import PADMMSettings, PADMMSolver, PADMMWarmStartMode
from .solvers.warmstart import WarmstarterContacts, WarmstarterLimits
from .utils import logger as msg

###
# Kernels
###


@wp.kernel
def _apply_pd_torques(
    joint_q: wp.array(dtype=float32),
    joint_qd: wp.array(dtype=float32),
    joint_target_pos: wp.array(dtype=float32),
    joint_target_ke: wp.array(dtype=float32),
    joint_target_kd: wp.array(dtype=float32),
    dof_has_pd: wp.array(dtype=int32),
    dof_to_coord: wp.array(dtype=int32),
    joint_f: wp.array(dtype=float32),
):
    """Computes PD torques for DOFs that have non-zero position/velocity gains.

    Overwrites (not accumulates) ``joint_f`` for PD-controlled DOFs so that
    stale torques from a previous step are not carried over.  Any user-supplied
    external joint forces should be applied *after* this kernel via
    ``control.joint_f``.

    Note:
        This is a temporary workaround.  Kamino's solver core does not natively
        consume Newton's ``joint_target_ke / joint_target_kd`` gains, unlike the
        Featherstone, XPBD, and MuJoCo solvers which apply PD drives internally.
        Until Kamino gains native PD support, this kernel bridges the gap so that
        Newton examples (e.g. ANYmal) work without modification.
    """
    tid = wp.tid()
    if dof_has_pd[tid] == 0:
        return
    coord_idx = dof_to_coord[tid]
    if coord_idx < 0:
        return
    ke = joint_target_ke[tid]
    kd = joint_target_kd[tid]
    pos_err = joint_target_pos[tid] - joint_q[coord_idx]
    vel_err = -joint_qd[tid]
    joint_f[tid] = ke * pos_err + kd * vel_err


@wp.kernel
def _convert_kamino_contacts_to_newton(
    n_active: wp.array(dtype=int32),
    kamino_wid: wp.array(dtype=int32),
    kamino_gid_AB: wp.array(dtype=vec2i),
    kamino_position_A: wp.array(dtype=vec3f),
    kamino_position_B: wp.array(dtype=vec3f),
    kamino_gapfunc: wp.array(dtype=vec4f),
    world_geom_offset: wp.array(dtype=int32),
    shape_body: wp.array(dtype=int32),
    body_q: wp.array(dtype=wp.transformf),
    max_output: int32,
    # outputs
    rigid_contact_count: wp.array(dtype=int32),
    rigid_contact_shape0: wp.array(dtype=int32),
    rigid_contact_shape1: wp.array(dtype=int32),
    rigid_contact_point0: wp.array(dtype=wp.vec3),
    rigid_contact_point1: wp.array(dtype=wp.vec3),
    rigid_contact_normal: wp.array(dtype=wp.vec3),
):
    """Converts Kamino's internal contact representation to Newton's Contacts format."""
    tid = wp.tid()
    n = wp.min(n_active[0], max_output)

    if tid == 0:
        rigid_contact_count[0] = n

    if tid >= n:
        return

    wid = kamino_wid[tid]
    offset = world_geom_offset[wid]
    gids = kamino_gid_AB[tid]
    shape0 = offset + gids[0]
    shape1 = offset + gids[1]

    rigid_contact_shape0[tid] = shape0
    rigid_contact_shape1[tid] = shape1

    normal = wp.vec3(
        float(kamino_gapfunc[tid][0]),
        float(kamino_gapfunc[tid][1]),
        float(kamino_gapfunc[tid][2]),
    )
    rigid_contact_normal[tid] = normal

    pos_a = wp.vec3(
        float(kamino_position_A[tid][0]),
        float(kamino_position_A[tid][1]),
        float(kamino_position_A[tid][2]),
    )
    pos_b = wp.vec3(
        float(kamino_position_B[tid][0]),
        float(kamino_position_B[tid][1]),
        float(kamino_position_B[tid][2]),
    )

    body_a = shape_body[shape0]
    body_b = shape_body[shape1]

    X_inv_a = wp.transform_identity()
    if body_a >= 0:
        X_inv_a = wp.transform_inverse(body_q[body_a])
    X_inv_b = wp.transform_identity()
    if body_b >= 0:
        X_inv_b = wp.transform_inverse(body_q[body_b])

    rigid_contact_point0[tid] = wp.transform_point(X_inv_a, pos_a)
    rigid_contact_point1[tid] = wp.transform_point(X_inv_b, pos_b)


@wp.kernel
def _convert_newton_contacts_to_kamino(
    # Newton contact inputs
    newton_contact_count: wp.array(dtype=int32),
    newton_shape0: wp.array(dtype=int32),
    newton_shape1: wp.array(dtype=int32),
    newton_point0: wp.array(dtype=wp.vec3),
    newton_point1: wp.array(dtype=wp.vec3),
    newton_normal: wp.array(dtype=wp.vec3),
    newton_thickness0: wp.array(dtype=float32),
    newton_thickness1: wp.array(dtype=float32),
    # Model lookups
    shape_body: wp.array(dtype=int32),
    shape_world: wp.array(dtype=int32),
    shape_mu: wp.array(dtype=float32),
    shape_restitution: wp.array(dtype=float32),
    body_q: wp.array(dtype=wp.transformf),
    kamino_max_contacts: int32,
    kamino_num_worlds: int32,
    kamino_world_max_contacts: wp.array(dtype=int32),
    # Kamino contact outputs
    kamino_model_active: wp.array(dtype=int32),
    kamino_world_active: wp.array(dtype=int32),
    kamino_wid: wp.array(dtype=int32),
    kamino_cid: wp.array(dtype=int32),
    kamino_gid_AB: wp.array(dtype=vec2i),
    kamino_bid_AB: wp.array(dtype=vec2i),
    kamino_position_A: wp.array(dtype=vec3f),
    kamino_position_B: wp.array(dtype=vec3f),
    kamino_gapfunc: wp.array(dtype=vec4f),
    kamino_frame: wp.array(dtype=quatf),
    kamino_material: wp.array(dtype=vec2f),
    kamino_key: wp.array(dtype=wp.uint64),
):
    """Convert Newton Contacts to Kamino's ContactsKamino format.

    Reads body-local contact points from Newton, transforms them to world-space,
    and populates the Kamino contact arrays with the A/B convention that Kamino's
    solver core expects (bid_B >= 0, normal points A -> B).

    Newton's ``rigid_contact_normal`` points from shape1 toward shape0 (the
    direction that pushes shape0 away from shape1).
    """
    tid = wp.tid()
    nc = newton_contact_count[0]
    if tid >= nc or tid >= kamino_max_contacts:
        return

    s0 = newton_shape0[tid]
    s1 = newton_shape1[tid]
    b0 = shape_body[s0]
    b1 = shape_body[s1]

    # Determine the world index.  Global shapes (shape_world == -1) can
    # collide with shapes from any world, so fall back to the other shape.
    w0 = shape_world[s0]
    w1 = shape_world[s1]
    wid = w0
    if w0 < 0:
        wid = w1
    if wid < 0 or wid >= kamino_num_worlds:
        return

    # Body-local → world-space
    X0 = wp.transform_identity()
    if b0 >= 0:
        X0 = body_q[b0]
    X1 = wp.transform_identity()
    if b1 >= 0:
        X1 = body_q[b1]

    p0_world = wp.transform_point(X0, newton_point0[tid])
    p1_world = wp.transform_point(X1, newton_point1[tid])

    # Newton normal points from shape1 → shape0.
    # Kamino convention: normal points A → B, with bid_B >= 0.
    n_newton = newton_normal[tid]

    # Reconstruct Newton signed contact distance d from exported fields:
    # d = dot((p1 - p0), n_a_to_b) - (offset0 + offset1),
    # with n_newton = -n_a_to_b and offset* stored in rigid_contact_thickness*.
    d_newton = -wp.dot(p1_world - p0_world, n_newton) - (newton_thickness0[tid] + newton_thickness1[tid])

    if b1 < 0:
        # shape1 is world-static → make it A, shape0 becomes B.
        # Newton normal already points from shape1 (A) to shape0 (B).
        gid_A = s1
        gid_B = s0
        bid_A = b1
        bid_B = b0
        pos_A = p1_world
        pos_B = p0_world
        normal = vec3f(n_newton[0], n_newton[1], n_newton[2])
    else:
        # Both dynamic or shape0 is static → keep A=shape0, B=shape1.
        # Newton normal goes shape1→shape0 = B→A, need A→B so negate.
        gid_A = s0
        gid_B = s1
        bid_A = b0
        bid_B = b1
        pos_A = p0_world
        pos_B = p1_world
        normal = vec3f(-n_newton[0], -n_newton[1], -n_newton[2])

    distance = d_newton
    if distance > 0.0:
        return
    gapfunc = vec4f(normal[0], normal[1], normal[2], float32(distance))
    q_frame = wp.quat_from_matrix(make_contact_frame_znorm(normal))

    mu = float32(0.5) * (shape_mu[s0] + shape_mu[s1])
    rest = float32(0.5) * (shape_restitution[s0] + shape_restitution[s1])

    mcid = wp.atomic_add(kamino_model_active, 0, 1)
    wcid = wp.atomic_add(kamino_world_active, wid, 1)

    world_max = kamino_world_max_contacts[wid]
    if mcid < kamino_max_contacts and wcid < world_max:
        kamino_wid[mcid] = wid
        kamino_cid[mcid] = wcid
        kamino_gid_AB[mcid] = vec2i(gid_A, gid_B)
        kamino_bid_AB[mcid] = vec2i(bid_A, bid_B)
        kamino_position_A[mcid] = pos_A
        kamino_position_B[mcid] = pos_B
        kamino_gapfunc[mcid] = gapfunc
        kamino_frame[mcid] = q_frame
        kamino_material[mcid] = vec2f(mu, rest)
        kamino_key[mcid] = build_pair_key2(uint32(gid_A), uint32(gid_B))
    else:
        wp.atomic_sub(kamino_model_active, 0, 1)
        wp.atomic_sub(kamino_world_active, wid, 1)


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
        settings = newton.solvers.kamino.SolverKaminoSettings()
        solver = newton.solvers.SolverKamino(model, contacts, settings)

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
        settings: SolverKaminoSettings | None = None,
    ):
        """
        Initializes the Kamino physics solver for the given set of multi-body systems
        defined in `model`, and the total contact allocations defined in `contacts`.

        Explicit solver settings may be provided through the `settings` argument. If no
        settings are provided, default settings will be used.

        Args:
            model (ModelKamino): The multi-body systems model to simulate.
            contacts (ContactsKamino): The contact data container for the simulation.
            settings (SolverKaminoSettings | None): Optional solver settings.
        """
        # Ensure the input containers are valid
        if not isinstance(model, ModelKamino):
            raise TypeError(f"Invalid model container: Expected a `ModelKamino` instance, but got {type(model)}.")
        if contacts is not None and not isinstance(contacts, ContactsKamino):
            raise TypeError(
                f"Invalid contacts container: Expected a `ContactsKamino` instance, but got {type(contacts)}."
            )
        if settings is not None and not isinstance(settings, SolverKaminoSettings):
            raise TypeError(
                f"Invalid solver settings: Expected a `SolverKaminoSettings` instance, but got {type(settings)}."
            )

        # First initialize the base solver
        # NOTE: Although we pass the model here, we will re-assign it below
        # since currently Kamino defines its own :class`ModelKamino` class.
        super().__init__(model=model)
        self._model = model

        # Cache solver settings: If no settings are provided, use defaults
        if settings is None:
            settings = SolverKaminoSettings()
        settings.check()
        self._settings: SolverKaminoSettings = settings

        # Allocate internal time-varying solver data
        self._data = self._model.data()

        # Allocate a joint-limits interface
        self._limits = LimitsKamino(model=self._model, device=self._model.device)

        # Construct the unilateral constraints members in the model info
        make_unilateral_constraints_info(model=self._model, data=self._data, limits=self._limits, contacts=contacts)

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
            solver=self._settings.linear_solver_type,
            solver_kwargs=self._settings.linear_solver_kwargs,
            settings=self._settings.problem,
            device=self._model.device,
        )

        # Allocate the forward dynamics solver on the device
        self._solver_fd = PADMMSolver(
            model=self._model,
            settings=self._settings.padmm,
            warmstart=self._settings.warmstart_mode,
            use_acceleration=self._settings.use_solver_acceleration,
            collect_info=self._settings.collect_solver_info,
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
        state_in: StateKamino,
        state_out: StateKamino,
        control: ControlKamino,
        contacts: ContactsKamino | None = None,
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
        wp.copy(self._data.bodies.w_e_i, state_in.w_i)
        wp.copy(self._data.joints.q_j, state_in.q_j)
        wp.copy(self._data.joints.q_j_p, state_in.q_j_p)
        wp.copy(self._data.joints.dq_j, state_in.dq_j)
        wp.copy(self._data.joints.lambda_j, state_in.lambda_j)
        wp.copy(self._data.joints.tau_j, control_in.tau_j)

    def _write_step_output(self, state_out: StateKamino):
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
        self._data.joints.reset_state(q_j_ref=self._model.joints.q_j_0)
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
            q_i_cache=self._data.bodies.q_i,
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

    def _integrate(self):
        """
        Solves the time integration sub-problem to compute the next state of the system.
        """
        # Integrate the state of the system (i.e. of the bodies) to compute the next state
        integrate_euler_semi_implicit(model=self._model, data=self._data)

        # Update the internal joint states based on the current and next body states
        wp.copy(self._data.joints.q_j_p, self._data.joints.q_j)
        self._update_joints_data(q_j_p=self._data.joints.q_j_p)

    def _compute_metrics(self, state_in: StateKamino, contacts: ContactsKamino | None = None):
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


class SolverKamino(SolverBase):
    """
    TODO
    """

    def __init__(
        self,
        model: Model,
        solver_settings: SolverKaminoSettings | None = None,
        collision_detector_settings: CollisionDetectorSettings | None = None,
    ):
        """
        TODO
        """
        # Initialize the base solver
        super().__init__(model=model)

        # Create a Kamino model from the Newton model
        self._model_kamino = ModelKamino.from_newton(model)

        # Create a collision detector
        self._collision_detector_kamino = CollisionDetector(
            model=self._model_kamino,
            settings=collision_detector_settings,
        )

        # Capture a reference to the contacts container
        self._contacts_kamino: ContactsKamino = self._collision_detector_kamino.contacts

        # Initialize the internal Kamino solver
        self._solver_kamino = SolverKaminoImpl(
            model=self._model_kamino,
            contacts=self._contacts_kamino,
            settings=solver_settings,
        )

        # Build per-world geom offset array for contact conversion
        import numpy as np  # noqa: PLC0415

        geom_offsets = np.array(
            [w.geoms_idx_offset for w in self._model_kamino.worlds],
            dtype=np.int32,
        )
        self._world_geom_offset = wp.array(geom_offsets, dtype=int32, device=model.device)

        # Reference to body_q from the latest step output, used by update_contacts()
        self._last_state_body_q: wp.array | None = None

        # Pre-compute PD control arrays for DOFs that have non-zero gains.
        # This allows automatic PD torque computation in step() without
        # requiring the user to implement a custom PD kernel.
        self._setup_pd_control(model)

    _DEFAULT_EFFORT_KE: float = 150.0
    _DEFAULT_EFFORT_KD: float = 20.0

    def _setup_pd_control(self, model: Model):
        """Pre-compute arrays for automatic PD torque computation.

        When Newton joints have non-zero ``joint_target_ke`` or ``joint_target_kd``
        gains, this method builds the lookup tables needed to apply PD torques
        automatically in :meth:`step`, so that user code does not need to
        implement a custom PD kernel.

        For joints in ``EFFORT`` mode (drive present but zero gains), default PD
        gains are applied automatically so that the robot holds its pose without
        requiring the user to set gains manually.  This matches the behaviour of
        other Newton solvers which natively consume ``joint_target_ke / kd``.

        Note:
            This is a temporary workaround until Kamino's solver core natively
            supports Newton's PD drive model.  It exists so that Newton examples
            (e.g. ANYmal) work out of the box with ``SolverKamino`` without any
            example-side modifications.
        """
        import numpy as np  # noqa: PLC0415

        ke_np = model.joint_target_ke.numpy().copy()
        kd_np = model.joint_target_kd.numpy().copy()
        act_mode_np = model.joint_act_mode.numpy()

        effort_mask = (act_mode_np == int(ActuatorMode.EFFORT)) & (ke_np == 0.0) & (kd_np == 0.0)
        if np.any(effort_mask):
            ke_np[effort_mask] = self._DEFAULT_EFFORT_KE
            kd_np[effort_mask] = self._DEFAULT_EFFORT_KD
            msg.info(
                "Auto-applied default PD gains (ke=%.1f, kd=%.1f) for %d EFFORT-mode DOFs",
                self._DEFAULT_EFFORT_KE,
                self._DEFAULT_EFFORT_KD,
                int(np.sum(effort_mask)),
            )

        has_pd = ((ke_np != 0.0) | (kd_np != 0.0)) & (act_mode_np != 0)
        self._has_pd_dofs = bool(np.any(has_pd))

        if not self._has_pd_dofs:
            return

        device = model.body_q.device
        self._pd_dof_has_pd = wp.array(has_pd.astype(np.int32), dtype=int32, device=device)
        self._pd_target_ke = wp.array(ke_np.astype(np.float32), dtype=float32, device=device)
        self._pd_target_kd = wp.array(kd_np.astype(np.float32), dtype=float32, device=device)

        q_start = model.joint_q_start.numpy()
        qd_start = model.joint_qd_start.numpy()
        dof_to_coord_np = np.full(model.joint_dof_count, -1, dtype=np.int32)
        for j in range(model.joint_count):
            ndofs = int(qd_start[j + 1] - qd_start[j])
            ncoords = int(q_start[j + 1] - q_start[j])
            if ndofs == ncoords:
                for d in range(ndofs):
                    dof_to_coord_np[int(qd_start[j]) + d] = int(q_start[j]) + d
        self._pd_dof_to_coord = wp.array(dof_to_coord_np, dtype=int32, device=device)
        self._pd_num_dofs = model.joint_dof_count

    def _apply_pd_control(self, state: State, control: Control):
        """Apply PD torques to ``control.joint_f`` based on current state."""
        if not self._has_pd_dofs:
            return
        wp.launch(
            _apply_pd_torques,
            dim=self._pd_num_dofs,
            inputs=[
                state.joint_q,
                state.joint_qd,
                control.joint_target_pos,
                self._pd_target_ke,
                self._pd_target_kd,
                self._pd_dof_has_pd,
                self._pd_dof_to_coord,
                control.joint_f,
            ],
        )

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
            state_out=StateKamino.from_newton(self.model, state_out),
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
        # Apply PD torques from joint_target_ke / joint_target_kd gains
        self._apply_pd_control(state_in, control)

        # Interface the input state and control
        # containers to Kamino's equivalents
        # NOTE: These should produce zero-copy views/references
        # to the arrays of the source Newton containers.
        state_in_kamino = StateKamino.from_newton(self.model, state_in)
        state_out_kamino = StateKamino.from_newton(self.model, state_out)
        control_kamino = ControlKamino.from_newton(control)

        if contacts is not None:
            self._ingest_newton_contacts(contacts, state_in)
        else:
            self._collision_detector_kamino.collide(
                self._model_kamino, self._solver_kamino.data, state_in_kamino
            )

        # Convert Newton body-frame poses to Kamino CoM-frame poses using
        # Kamino's corrected body-com offsets (can differ from Newton model data).
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
            dt=dt,
        )

        # Convert back from Kamino CoM-frame to Newton body-frame poses using
        # the same corrected body-com offsets as the forward conversion.
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
        self._last_state_body_q = state_out.body_q

    @override
    def notify_model_changed(self, flags: int):
        """Propagate Newton model property changes to Kamino's internal ModelKamino.

        Args:
            flags: Bitmask of :class:`SolverNotifyFlags` indicating which properties changed.
        """
        if flags & SolverNotifyFlags.MODEL_PROPERTIES:
            self._update_gravity()

        if flags & SolverNotifyFlags.BODY_INERTIAL_PROPERTIES:
            # Kamino's RigidBodiesModel references Newton's arrays directly
            # (m_i, inv_m_i, i_I_i, inv_i_I_i, i_r_com_i), so no copy needed.
            pass

        if flags & SolverNotifyFlags.JOINT_PROPERTIES:
            self._update_joint_transforms()

        if flags & SolverNotifyFlags.JOINT_DOF_PROPERTIES:
            # Joint limits (q_j_min, q_j_max, dq_j_max, tau_j_max) are direct
            # references to Newton's arrays, so no copy needed.
            # Re-run PD setup in case target gains changed.
            self._setup_pd_control(self.model)

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
        """Re-derive Kamino joint anchors and axes from Newton's joint_X_p / joint_X_c.

        Called when :data:`SolverNotifyFlags.JOINT_PROPERTIES` is raised,
        indicating that ``model.joint_X_p`` or ``model.joint_X_c`` may have
        changed at runtime (e.g. animated root transforms).
        """
        import numpy as np  # noqa: PLC0415

        from .core.joints import (  # noqa: PLC0415
            JointDoFType,
            axes_matrix_from_joint_type,
        )

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
            R_axis_j = axes_matrix_from_joint_type(dof_type_j, dof_dim_j, joint_axes_j)

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

    def _ingest_newton_contacts(self, contacts: Contacts, state: State):
        """Convert Newton's Contacts to Kamino's ContactsKamino for the solver core.

        Transforms body-local contact points to world-space, applies the A/B
        convention expected by Kamino (bid_B >= 0, normal A -> B), and populates
        all required ContactsKamino fields.
        """
        kc = self._contacts_kamino
        kc.clear()

        max_kamino = kc.data.model_max_contacts_host
        if max_kamino == 0:
            return

        dim = min(contacts.rigid_contact_max, max_kamino)
        if dim == 0:
            return

        num_worlds = self._model_kamino.size.num_worlds
        wp.launch(
            _convert_newton_contacts_to_kamino,
            dim=dim,
            inputs=[
                contacts.rigid_contact_count,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.rigid_contact_normal,
                contacts.rigid_contact_thickness0,
                contacts.rigid_contact_thickness1,
                self.model.shape_body,
                self.model.shape_world,
                self.model.shape_material_mu,
                self.model.shape_material_restitution,
                state.body_q,
                int32(max_kamino),
                int32(num_worlds),
                kc.data.world_max_contacts,
            ],
            outputs=[
                kc.data.model_active_contacts,
                kc.data.world_active_contacts,
                kc.data.wid,
                kc.data.cid,
                kc.data.gid_AB,
                kc.data.bid_AB,
                kc.data.position_A,
                kc.data.position_B,
                kc.data.gapfunc,
                kc.data.frame,
                kc.data.material,
                kc.data.key,
            ],
            device=self.model.device,
        )

    @override
    def update_contacts(self, contacts: Contacts, state: State | None = None) -> None:
        """Convert Kamino contacts to Newton's Contacts format for viewer visualization.

        Args:
            contacts: The Newton Contacts object to populate.
            state: Optional simulation state providing ``body_q`` for converting
                world-space contact positions to body-local frame. Falls back to
                the last ``state_out`` from :meth:`step` if not provided.
        """
        body_q = state.body_q if state is not None else self._last_state_body_q
        if body_q is None:
            return

        kc = self._contacts_kamino
        max_contacts = kc.data.model_max_contacts_host

        if max_contacts == 0:
            return

        if max_contacts > contacts.rigid_contact_max:
            msg.warning(
                "Kamino max contacts (%d) exceeds Newton rigid_contact_max (%d); "
                "contacts will be truncated.",
                max_contacts,
                contacts.rigid_contact_max,
            )

        dim = min(max_contacts, contacts.rigid_contact_max)

        wp.launch(
            _convert_kamino_contacts_to_newton,
            dim=dim,
            inputs=[
                kc.data.model_active_contacts,
                kc.data.wid,
                kc.data.gid_AB,
                kc.data.position_A,
                kc.data.position_B,
                kc.data.gapfunc,
                self._world_geom_offset,
                self.model.shape_body,
                body_q,
                int32(contacts.rigid_contact_max),
            ],
            outputs=[
                contacts.rigid_contact_count,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.rigid_contact_normal,
            ],
            device=self.model.device,
        )

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
