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

###########################################################################
# RigidBodySim: Generic Kamino rigid body simulator for RL
#
# Consolidates common simulation infrastructure (model building, solver
# setup, state/control tensor wiring, contact aggregation, selective
# resets, CUDA graphs) into a single reusable class.  Any robot RL
# example can use this instead of duplicating ~300 lines of boilerplate.
###########################################################################

from __future__ import annotations

import torch
import warp as wp
from warp.context import Devicelike

from newton._src.solvers.kamino.core.builder import ModelBuilder
from newton._src.solvers.kamino.core.types import transformf, vec6f
from newton._src.solvers.kamino.geometry.aggregation import ContactAggregation
from newton._src.solvers.kamino.models.builders.utils import (
    build_usd,
    make_homogeneous_builder,
    set_uniform_body_pose_offset,
)
from newton._src.solvers.kamino.utils import logger as msg
from newton._src.solvers.kamino.utils.sim import Simulator, SimulatorSettings, ViewerKamino


class RigidBodySim:
    """Generic Kamino rigid body simulator for RL.


    Features:
        * USD model loading via ``make_homogeneous_builder``
        * Configurable solver settings with sensible RL defaults
        * Zero-copy PyTorch views of state, control and contact arrays
        * Automatic extraction of actuated joint metadata
        * Selective per-world reset infrastructure (world mask + deferred buffers)
        * Optional CUDA graph capture for step and reset
        * Optional ViewerKamino

    Args:
        usd_model_path: Full filesystem path to the USD model file.
        num_worlds: Number of parallel simulation worlds.
        sim_dt: Physics timestep in seconds.
        device: Warp device (e.g. ``"cuda:0"``).  ``None`` → preferred device.
        headless: If ``True``, skip viewer creation.
        load_drive_dynamics: Load PD gains from USD (for implicit PD control).
        body_pose_offset: Optional ``(x, y, z, qx, qy, qz, qw)`` tuple to
            offset every body's initial pose (e.g. to place the robot above
            the ground plane).
        add_ground: Add a ground-plane box to each world.
        enable_gravity: Enable gravity in every world.
        settings: Solver settings.  ``None`` uses ``default_settings(sim_dt)``.
        use_cuda_graph: Capture CUDA graphs for step and reset (requires
            CUDA device with memory pool enabled).
    """

    def __init__(
        self,
        usd_model_path: str,
        num_worlds: int = 1,
        sim_dt: float = 0.01,
        device: Devicelike = None,
        headless: bool = False,
        load_drive_dynamics: bool = True,
        body_pose_offset: tuple | None = None,
        add_ground: bool = True,
        enable_gravity: bool = True,
        settings: SimulatorSettings | None = None,
        use_cuda_graph: bool = False,
    ):
        # ----- Device setup -----
        self._device = wp.get_device(device)
        self._torch_device: str = "cuda" if self._device.is_cuda else "cpu"
        self._use_cuda_graph = use_cuda_graph
        self._sim_dt = sim_dt

        # ----- Model builder from USD -----
        msg.notif("Constructing builder from imported USD ...")
        self.builder: ModelBuilder = make_homogeneous_builder(
            num_worlds=num_worlds,
            build_fn=build_usd,
            source=usd_model_path,
            load_static_geometry=True,
            load_drive_dynamics=load_drive_dynamics,
            ground=add_ground,
        )

        # Apply body pose offset if provided
        if body_pose_offset is not None:
            offset = wp.transformf(*body_pose_offset)
            set_uniform_body_pose_offset(builder=self.builder, offset=offset)

        # Enable gravity per world
        if enable_gravity:
            for w in range(self.builder.num_worlds):
                self.builder.gravity[w].enabled = True

        # ----- Solver settings -----
        if settings is None:
            settings = self.default_settings(sim_dt)
        else:
            settings.dt = sim_dt

        # ----- Create simulator -----
        msg.notif("Building Kamino simulator ...")
        self.sim = Simulator(builder=self.builder, settings=settings, device=self._device)

        # Empty control callback (torques applied directly via control arrays)
        self.sim.set_control_callback(lambda _: None)

        # ----- Wire RL interface (zero-copy tensors) -----
        self._make_rl_interface()

        # ----- Extract metadata -----
        self._extract_metadata()

        # ----- Viewer -----
        self.viewer: ViewerKamino | None = None
        if not headless:
            msg.notif("Creating the 3D viewer ...")
            self.viewer = ViewerKamino(builder=self.builder, simulator=self.sim)

        # ----- CUDA graphs -----
        self._reset_graph = None
        self._step_graph = None
        self._capture_graphs()

        # ----- Warm-up (compiles Warp kernels) -----
        msg.notif("Warming up simulator ...")
        self.step()
        self.reset()

    # ------------------------------------------------------------------
    # RL interface wiring
    # ------------------------------------------------------------------

    def _make_rl_interface(self):
        """Create zero-copy PyTorch views of simulator state, control and contact arrays."""
        nw = self.sim.model.size.num_worlds
        njc = self.sim.model.size.max_of_num_joint_coords
        njd = self.sim.model.size.max_of_num_joint_dofs
        nb = self.sim.model.size.max_of_num_bodies

        # State tensors (read-only views into simulator)
        self._q_j = wp.to_torch(self.sim.state.q_j).reshape(nw, njc)
        self._dq_j = wp.to_torch(self.sim.state.dq_j).reshape(nw, njd)
        self._q_i = wp.to_torch(self.sim.state.q_i).reshape(nw, nb, 7)
        self._u_i = wp.to_torch(self.sim.state.u_i).reshape(nw, nb, 6)

        # Control tensors (writable views)
        self._q_j_ref = wp.to_torch(self.sim.control.q_j_ref).reshape(nw, njc)
        self._dq_j_ref = wp.to_torch(self.sim.control.dq_j_ref).reshape(nw, njd)
        self._tau_j_ref = wp.to_torch(self.sim.control.tau_j_ref).reshape(nw, njd)

        # World mask for selective resets
        self._world_mask_wp = wp.zeros((nw,), dtype=wp.int32, device=self._device)
        self._world_mask = wp.to_torch(self._world_mask_wp)

        # Reset buffers
        self._reset_base_q_wp = wp.zeros(nw, dtype=transformf, device=self._device)
        self._reset_base_u_wp = wp.zeros(nw, dtype=vec6f, device=self._device)
        self._reset_q_j = torch.zeros((nw, njd), device=self._torch_device)
        self._reset_dq_j = torch.zeros((nw, njd), device=self._torch_device)
        self._reset_base_q = wp.to_torch(self._reset_base_q_wp).reshape(nw, 7)
        self._reset_base_u = wp.to_torch(self._reset_base_u_wp).reshape(nw, 6)

        # Reset flags
        self._update_q_j = False
        self._update_dq_j = False
        self._update_base_q = False
        self._update_base_u = False

        # Contact aggregation
        ground_geom_ids = [self.sim.model.cgeoms.num_geoms - 1]
        self._contact_aggregation = ContactAggregation(
            model=self.sim.model,
            contacts=self.sim.contacts,
            static_geom_ids=ground_geom_ids,
            device=self._device,
        )
        self._contact_flags = wp.to_torch(self._contact_aggregation.body_contact_flag).reshape(nw, nb)
        self._ground_contact_flags = wp.to_torch(self._contact_aggregation.body_static_contact_flag).reshape(nw, nb)
        self._net_contact_forces = wp.to_torch(self._contact_aggregation.body_net_force).reshape(nw, nb, 3)

        # Default joint positions (cloned from initial state)
        self._default_q_j = self._q_j.clone()

    # ------------------------------------------------------------------
    # Metadata extraction
    # ------------------------------------------------------------------

    def _extract_metadata(self):
        """Extract joint/body names, actuated DOF indices, and joint limits from the builder."""
        max_joints = self.sim.model.size.max_of_num_joints
        max_bodies = self.sim.model.size.max_of_num_bodies

        # Joint names and actuated indices
        self._joint_names: list[str] = []
        self._actuated_joint_names: list[str] = []
        self._actuated_dof_indices: list[int] = []
        dof_offset = 0
        for j in range(max_joints):
            joint = self.builder.joints[j]
            self._joint_names.append(joint.name)
            if joint.is_actuated:
                self._actuated_joint_names.append(joint.name)
                for dof_idx in range(joint.num_dofs):
                    self._actuated_dof_indices.append(dof_offset + dof_idx)
            dof_offset += joint.num_dofs

        self._actuated_dof_indices_tensor = torch.tensor(
            self._actuated_dof_indices, device=self._torch_device, dtype=torch.long
        )

        msg.info(f"Actuated joints ({self.num_actuated}): {self._actuated_joint_names}")

        # Body names
        self._body_names: list[str] = []
        for b in range(max_bodies):
            self._body_names.append(self.builder.bodies[b].name)

        # Joint limits
        self._joint_limits: list[list[float]] = []
        for j in range(max_joints):
            joint = self.builder.joints[j]
            lower = float(joint.q_j_min[0])
            upper = float(joint.q_j_max[0])
            self._joint_limits.append([lower, upper])

    # ------------------------------------------------------------------
    # CUDA graph capture
    # ------------------------------------------------------------------

    def _capture_graphs(self):
        """Capture CUDA graphs for step and reset if requested and available."""
        if not self._use_cuda_graph:
            return
        if not (self._device.is_cuda and wp.is_mempool_enabled(self._device)):
            msg.warning("CUDA graphs requested but not available (need CUDA device with mempool). Using kernels.")
            return

        msg.notif("Capturing CUDA graphs ...")
        with wp.ScopedCapture(device=self._device) as reset_capture:
            self._reset_worlds()
        self._reset_graph = reset_capture.graph

        with wp.ScopedCapture(device=self._device) as step_capture:
            self.sim.step()
            self._contact_aggregation.compute()
        self._step_graph = step_capture.graph

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def step(self):
        """Execute a single physics step (simulator + contact aggregation).

        Uses CUDA graph replay if available.
        """
        if self._step_graph:
            wp.capture_launch(self._step_graph)
        else:
            self.sim.step()
            self._contact_aggregation.compute()

    def reset(self):
        """Full reset of all worlds to initial state."""
        self._world_mask.fill_(1)
        self._reset_worlds()
        self._world_mask.zero_()

    def apply_resets(self):
        """Apply pending selective resets staged via :meth:`set_dof` / :meth:`set_root`.

        Applies resets for worlds marked in :attr:`world_mask`, then clears
        the mask and all update flags.  Call this before :meth:`step` when
        using deferred resets.
        """
        if self._reset_graph:
            wp.capture_launch(self._reset_graph)
        else:
            self._reset_worlds()
        self._world_mask.zero_()
        self._update_q_j = False
        self._update_dq_j = False
        self._update_base_q = False
        self._update_base_u = False

    def _reset_worlds(self):
        """Reset selected worlds based on world_mask."""
        self.sim.reset(
            world_mask=self._world_mask_wp,
            joint_q=wp.from_torch(self._reset_q_j.view(-1)) if self._update_q_j else None,
            joint_u=wp.from_torch(self._reset_dq_j.view(-1)) if self._update_dq_j else None,
            base_q=wp.from_torch(self._reset_base_q.view(-1, 7)) if self._update_base_q else None,
            base_u=wp.from_torch(self._reset_base_u.view(-1, 6)) if self._update_base_u else None,
        )

    def render(self):
        """Render the current frame if viewer exists."""
        if self.viewer is not None:
            self.viewer.render_frame()

    def is_running(self) -> bool:
        """Check if the viewer is still running (always ``True`` in headless mode)."""
        if self.viewer is None:
            return True
        return self.viewer.is_running()

    # ------------------------------------------------------------------
    # Deferred reset staging
    # ------------------------------------------------------------------

    def set_dof(
        self,
        dof_positions: torch.Tensor | None = None,
        dof_velocities: torch.Tensor | None = None,
        env_ids: torch.Tensor | list[int] | None = None,
    ):
        """Stage joint state for deferred reset.

        The actual reset happens on the next call to :meth:`apply_resets`.

        Args:
            dof_positions: Joint positions ``(len(env_ids), num_joint_dofs)``.
            dof_velocities: Joint velocities ``(len(env_ids), num_joint_dofs)``.
            env_ids: Which worlds to reset.  ``None`` resets all.
        """
        if env_ids is None:
            self._world_mask.fill_(1)
            ids = slice(None)
        else:
            self._world_mask[env_ids] = 1
            ids = env_ids

        if dof_positions is not None:
            self._update_q_j = True
            self._reset_q_j[ids] = dof_positions
        if dof_velocities is not None:
            self._update_dq_j = True
            self._reset_dq_j[ids] = dof_velocities

    def set_root(
        self,
        root_positions: torch.Tensor | None = None,
        root_orientations: torch.Tensor | None = None,
        root_linear_velocities: torch.Tensor | None = None,
        root_angular_velocities: torch.Tensor | None = None,
        env_ids: torch.Tensor | list[int] | None = None,
    ):
        """Stage root body state for deferred reset.

        The actual reset happens on the next call to :meth:`apply_resets`.

        Args:
            root_positions: Root positions ``(len(env_ids), 3)``.
            root_orientations: Root orientations ``(len(env_ids), 4)`` (quaternion).
            root_linear_velocities: Root linear velocities ``(len(env_ids), 3)``.
            root_angular_velocities: Root angular velocities ``(len(env_ids), 3)``.
            env_ids: Which worlds to reset.  ``None`` resets all.
        """
        if env_ids is None:
            self._world_mask.fill_(1)
            ids = slice(None)
        else:
            self._world_mask[env_ids] = 1
            ids = env_ids

        if root_positions is not None or root_orientations is not None:
            self._update_base_q = True
            # Copy current state as baseline
            self._reset_base_q[ids] = self._q_i[ids, 0, :7]
            if root_positions is not None:
                self._reset_base_q[ids, :3] = root_positions
            if root_orientations is not None:
                self._reset_base_q[ids, 3:] = root_orientations

        if root_linear_velocities is not None or root_angular_velocities is not None:
            self._update_base_u = True
            self._reset_base_u[ids] = self._u_i[ids, 0, :6]
            if root_linear_velocities is not None:
                self._reset_base_u[ids, :3] = root_linear_velocities
            if root_angular_velocities is not None:
                self._reset_base_u[ids, 3:] = root_angular_velocities

    # ------------------------------------------------------------------
    # State properties (zero-copy torch views)
    # ------------------------------------------------------------------

    @property
    def q_j(self) -> torch.Tensor:
        """Joint positions ``(num_worlds, num_joint_coords)``."""
        return self._q_j

    @property
    def dq_j(self) -> torch.Tensor:
        """Joint velocities ``(num_worlds, num_joint_dofs)``."""
        return self._dq_j

    @property
    def q_i(self) -> torch.Tensor:
        """Body poses ``(num_worlds, num_bodies, 7)`` — position + quaternion."""
        return self._q_i

    @property
    def u_i(self) -> torch.Tensor:
        """Body twists ``(num_worlds, num_bodies, 6)`` — linear + angular velocity."""
        return self._u_i

    # ------------------------------------------------------------------
    # Control properties (zero-copy torch views)
    # ------------------------------------------------------------------

    @property
    def q_j_ref(self) -> torch.Tensor:
        """Joint position reference ``(num_worlds, num_joint_coords)`` for implicit PD."""
        return self._q_j_ref

    @property
    def dq_j_ref(self) -> torch.Tensor:
        """Joint velocity reference ``(num_worlds, num_joint_dofs)`` for implicit PD."""
        return self._dq_j_ref

    @property
    def tau_j_ref(self) -> torch.Tensor:
        """Joint torque reference ``(num_worlds, num_joint_dofs)`` for feed-forward control."""
        return self._tau_j_ref

    # ------------------------------------------------------------------
    # Contact properties
    # ------------------------------------------------------------------

    @property
    def contact_flags(self) -> torch.Tensor:
        """Per-body contact flags ``(num_worlds, num_bodies)``."""
        return self._contact_flags

    @property
    def ground_contact_flags(self) -> torch.Tensor:
        """Per-body ground contact flags ``(num_worlds, num_bodies)``."""
        return self._ground_contact_flags

    @property
    def net_contact_forces(self) -> torch.Tensor:
        """Net contact forces ``(num_worlds, num_bodies, 3)``."""
        return self._net_contact_forces

    # ------------------------------------------------------------------
    # Metadata properties
    # ------------------------------------------------------------------

    @property
    def num_worlds(self) -> int:
        return self.sim.model.size.num_worlds

    @property
    def num_joint_coords(self) -> int:
        return self.sim.model.size.max_of_num_joint_coords

    @property
    def num_joint_dofs(self) -> int:
        return self.sim.model.size.max_of_num_joint_dofs

    @property
    def num_bodies(self) -> int:
        return self.sim.model.size.max_of_num_bodies

    @property
    def joint_names(self) -> list[str]:
        return self._joint_names

    @property
    def body_names(self) -> list[str]:
        return self._body_names

    @property
    def actuated_joint_names(self) -> list[str]:
        return self._actuated_joint_names

    @property
    def actuated_dof_indices(self) -> list[int]:
        return self._actuated_dof_indices

    @property
    def actuated_dof_indices_tensor(self) -> torch.Tensor:
        """Actuated DOF indices as a ``torch.long`` tensor on the simulation device."""
        return self._actuated_dof_indices_tensor

    @property
    def num_actuated(self) -> int:
        return len(self._actuated_dof_indices)

    @property
    def default_q_j(self) -> torch.Tensor:
        """Default joint positions ``(num_worlds, num_joint_coords)`` cloned at init."""
        return self._default_q_j

    @property
    def joint_limits(self) -> list[list[float]]:
        """Per-joint ``[lower, upper]`` limits."""
        return self._joint_limits

    @property
    def torch_device(self) -> str:
        """Torch device string (``"cuda"`` or ``"cpu"``)."""
        return self._torch_device

    @property
    def device(self):
        """Warp device."""
        return self._device

    @property
    def sim_dt(self) -> float:
        return self._sim_dt

    @property
    def world_mask(self) -> torch.Tensor:
        """World mask ``(num_worlds,)`` int32 for selective resets."""
        return self._world_mask

    # ------------------------------------------------------------------
    # Default solver settings
    # ------------------------------------------------------------------

    @staticmethod
    def default_settings(sim_dt: float = 0.01) -> SimulatorSettings:
        """Return sensible default solver settings for RL."""
        settings = SimulatorSettings()
        settings.dt = sim_dt
        settings.solver.integrator = "moreau"
        settings.solver.problem.alpha = 0.1
        settings.solver.padmm.primal_tolerance = 1e-4
        settings.solver.padmm.dual_tolerance = 1e-4
        settings.solver.padmm.compl_tolerance = 1e-4
        settings.solver.padmm.max_iterations = 200
        settings.solver.padmm.eta = 1e-5
        settings.solver.padmm.rho_0 = 0.05
        settings.solver.use_solver_acceleration = True
        settings.solver.collect_solver_info = False
        settings.solver.compute_metrics = False
        return settings
