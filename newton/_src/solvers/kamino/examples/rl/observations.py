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

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
import warp as wp

from newton._src.solvers.kamino.utils.sim import Simulator
from rcp_python_utils.rotations import (
    euler_ZYX_to_quat,
    quat_inv_apply,
    quat_inv_mul,
    quat_to_euler_ZYX,
    quat_to_projected_yaw,
    quat_to_rotation9D,
    yaw_inv_apply_2d,
    yaw_to_quat,
)
from rcp_python_utils.stacked_indices import StackedIndices
from newton._src.solvers.kamino.examples.rl.simulation import RigidBodySim


class ObservationBuilder(ABC):
    """Base class for building observation tensors from a Kamino Simulator.

    Subclasses define which signals to extract and concatenate.  The builder
    maintains internal buffers (e.g. action history) and provides a uniform
    ``compute()`` interface suitable for inference loops.

    Args:
        sim: A Kamino ``Simulator`` instance.
        num_worlds: Number of parallel simulation worlds.
        device: Torch device string (e.g. ``"cuda:0"``).
        command_dim: Dimensionality of the external command vector
            (0 means no command input; todo reserved for future joystick / keyboard velocity commands).
    """

    def __init__(
        self,
        sim: Simulator,
        num_worlds: int,
        device: str,
        command_dim: int = 0,
    ) -> None:
        self._sim = sim
        self._num_worlds = num_worlds
        self._device = device
        self._command_dim = command_dim

        # Pre-allocated command tensor (empty if command_dim == 0).
        self._command: torch.Tensor = torch.zeros(
            (num_worlds, max(command_dim, 0)),
            device=device,
            dtype=torch.float32,
        )

    @property
    @abstractmethod
    def num_observations(self) -> int:
        """Total observation dimensionality (per environment)."""
        ...

    @property
    def command_dim(self) -> int:
        """Dimensionality of the external command vector."""
        return self._command_dim

    @property
    def command(self) -> torch.Tensor:
        """Current command tensor, shape ``(num_worlds, command_dim)``.

        External code (keyboard handler, joystick) writes into this tensor
        so that ``compute()`` can include it in the observation.
        """
        return self._command

    @command.setter
    def command(self, value: torch.Tensor) -> None:
        self._command = value

    @abstractmethod
    def compute(self, actions: torch.Tensor | None = None) -> torch.Tensor:
        """Build the observation tensor from the current simulator state.

        Args:
            actions: The most recent actions applied, shape
                ``(num_worlds, num_actions)``.  Pass ``None`` on the very
                first step (before any action has been applied).

        Returns:
            Observation tensor of shape ``(num_worlds, num_observations)``.
        """
        ...

    @abstractmethod
    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset internal buffers (e.g. action history) for given envs.

        Args:
            env_ids: Which environments to reset.  ``None`` resets all.
        """
        ...

    def _get_joint_positions(self) -> torch.Tensor:
        """Joint positions as a PyTorch tensor ``(num_worlds, num_joint_coords)``.

        Zero-copy view of the underlying Warp array.
        """
        num_joint_coords = self._sim.model.size.max_of_num_joint_coords
        return wp.to_torch(self._sim.state.q_j).reshape(self._num_worlds, num_joint_coords)

    def _get_joint_velocities(self) -> torch.Tensor:
        """Joint velocities as a PyTorch tensor ``(num_worlds, num_joint_dofs)``.

        Zero-copy view of the underlying Warp array.
        """
        num_joint_dofs = self._sim.model.size.max_of_num_joint_dofs
        return wp.to_torch(self._sim.state.dq_j).reshape(self._num_worlds, num_joint_dofs)


class DrlegsStanceObservation(ObservationBuilder):
    """Observation builder drlegs_stance

    Observation vector (48 dims):
        * 36 DOF positions  (all joints, including passive linkages)
        * 12 action history (actuated joints only, most recent step)

    Args:
        sim: Kamino Simulator.
        num_worlds: Number of parallel worlds.
        device: Torch device string.
        actuated_joint_indices: Per-world DOF indices of the 12 actuated
            joints within the joint-coord vector of length 36.
        num_actions: Number of policy action outputs (12 for DR Legs).
        action_scale: Scale applied to raw actions before storing in history.
    """

    def __init__(
        self,
        sim: Simulator,
        num_worlds: int,
        device: str,
        actuated_joint_indices: list[int],
        num_actions: int = 12,
        action_scale: float = 0.25,
    ) -> None:
        super().__init__(sim=sim, num_worlds=num_worlds, device=device, command_dim=0)
        self._actuated_joint_indices = torch.tensor(
            actuated_joint_indices,
            device=device,
            dtype=torch.long,
        )
        self._num_actions = num_actions
        self._num_dofs = sim.model.size.max_of_num_joint_coords
        self._action_scale = action_scale

        # Action history buffer (most recent scaled actions for actuated joints).
        self._action_history: torch.Tensor = torch.zeros(
            (num_worlds, num_actions),
            device=device,
            dtype=torch.float32,
        )

    @property
    def num_observations(self) -> int:
        return self._num_dofs + self._num_actions  # 36 + 12 = 48

    def compute(self, actions: torch.Tensor | None = None) -> torch.Tensor:
        q_j = self._get_joint_positions()  # (num_worlds, 36)

        if actions is not None:
            # Scale to match training observation (action_transform output)
            self._action_history[:] = self._action_scale * actions

        obs = torch.cat([q_j, self._action_history], dim=-1)  # (num_worlds, 48)
        return obs

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._action_history.zero_()
        else:
            self._action_history[env_ids] = 0.0


# ---------------------------------------------------------------------------
# BDX helpers
# ---------------------------------------------------------------------------


def _periodic_encoding(k: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute phase-encoding frequencies and offsets.

    Returns ``(freq_2pi, offset)`` arrays of length ``2*k``.  Each pair
    encodes ``[cos(n*2π*φ), sin(n*2π*φ)]`` for ``n = 1 … k``.
    """
    dim = k * 2
    freq_2pi = np.zeros((dim,))
    offset = np.zeros((dim,))
    for i in range(k):
        freq = 2.0 * np.pi * (1 + i)
        freq_2pi[2 * i] = freq
        freq_2pi[2 * i + 1] = freq
        offset[2 * i] = 0.5 * np.pi
    return freq_2pi, offset


def _phase_encoding(
    phase: torch.Tensor,
    freq_2pi: torch.Tensor,
    offset: torch.Tensor,
) -> torch.Tensor:
    """Encode a scalar phase into a sin/cos feature vector."""
    return torch.sin(torch.outer(phase, freq_2pi) + offset)

# ---------------------------------------------------------------------------
# BdxObservation — standalone BDX observation builder
# ---------------------------------------------------------------------------


class BdxObservation(ObservationBuilder, torch.nn.Module):
    """Standalone BDX observation builder for inference.

    Reads commands from :pyattr:`command` (shape ``(num_worlds, 11)``),
    simulator state from a :class:`RigidBodySim`, and maintains action
    history internally.

    Command tensor layout (11 dims)::

         [0]      path_heading         (1)
         [1:3]    path_position_2d     (2)
         [3:5]    cmd_vel_xy           (2)   
         [5]      cmd_yaw_rate         (1)   
         [6]      phase                (1)
         [7:11]   neck_cmd             (4)
    """

    # -- Command tensor indices --
    CMD_DIM = 11
    CMD_PATH_HEADING = 0
    CMD_PATH_POSITION = slice(1, 3)
    CMD_VEL = slice(3, 5)
    CMD_YAW_RATE = 5
    CMD_PHASE = 6
    CMD_NECK = slice(7, 11)

    def __init__(
        self,
        body_sim: RigidBodySim,
        joint_position_default: list[float],
        joint_position_range: list[float],
        joint_velocity_scale: float,
        path_deviation_scale: float,
        phase_embedding_dim: int,
        num_joints: int = 14,
        root_orientation_sigma: float = 0.0,
        root_linear_velocity_sigma: float = 0.0,
        root_angular_velocity_sigma: float = 0.0,
    ) -> None:
        torch.nn.Module.__init__(self)
        ObservationBuilder.__init__(
            self,
            sim=body_sim.sim,
            num_worlds=body_sim.num_worlds,
            device=body_sim.torch_device,
            command_dim=self.CMD_DIM,
        )
        self._body_sim = body_sim

        self.joint_velocity_scale = joint_velocity_scale
        self.path_deviation_scale = path_deviation_scale
        self.root_orientation_sigma = root_orientation_sigma
        self.root_linear_velocity_sigma = root_linear_velocity_sigma
        self.root_angular_velocity_sigma = root_angular_velocity_sigma

        # Joint normalization
        self.register_buffer(
            "_joint_position_default",
            torch.tensor(joint_position_default, dtype=torch.float),
        )
        self.register_buffer(
            "_joint_position_range",
            torch.tensor(joint_position_range, dtype=torch.float),
        )

        # Phase encoding
        freq_2pi, offset = _periodic_encoding(k=phase_embedding_dim // 2)
        self.register_buffer("_freq_2pi", torch.from_numpy(freq_2pi).to(dtype=torch.float))
        self.register_buffer("_offset", torch.from_numpy(offset).to(dtype=torch.float))

        phase_encoding_size = len(freq_2pi)

        # History size
        self.history_size = num_joints * 2

        # Observation structure (matches the GTC training layout)
        self.obs_idx = StackedIndices(
            [
                ("orientation_root_to_path", 9),
                ("path_deviation", 2),
                ("path_deviation_in_heading", 2),
                ("path_cmd", 3),
                ("path_lin_vel_in_root", 3),
                ("path_ang_vel_in_root", 3),
                ("phase_encoding", phase_encoding_size),
                ("neck_cmd", 4),
                ("root_lin_vel_in_root", 3),
                ("root_ang_vel_in_root", 3),
                ("normalized_joint_positions", num_joints),
                ("normalized_joint_velocities", num_joints),
                ("history", self.history_size),
            ]
        )
        self.num_obs = len(self.obs_idx)

        # Action history (normalised joint-position setpoints)
        self._action_hist_0 = torch.zeros(
            self._num_worlds, num_joints, device=self._device, dtype=torch.float
        )
        self._action_hist_1 = torch.zeros(
            self._num_worlds, num_joints, device=self._device, dtype=torch.float
        )

        # Cached intermediates (populated by compute, used by subclasses
        # for privileged observations)
        self._root_orientation_in_path: torch.Tensor | None = None
        self._root_lin_vel_in_root: torch.Tensor | None = None
        self._root_ang_vel_in_root: torch.Tensor | None = None
        self._skip_obs = torch.empty(
            (self._num_worlds, 0), dtype=torch.float, device=self._device
        )

        # Move registered buffers to device
        self.to(self._device)

    def get_feature_module(self) -> BdxObservation:
        return self

    @property
    def num_observations(self) -> int:
        return self.num_obs

    def compute(self, setpoints: torch.Tensor | None = None) -> torch.Tensor:
        """Build the observation tensor from current simulator state.

        Args:
            setpoints: Latest joint-position setpoints (raw, un-normalised),
                shape ``(num_worlds, num_joints)``.  ``None`` on the very
                first step before any action has been applied.
        """
        sim = self._body_sim
        nw = self._num_worlds
        device = self._device

        # -- Read commands (pre-clipped by caller) --
        path_heading = self.command[:, self.CMD_PATH_HEADING]
        path_position_2d = self.command[:, self.CMD_PATH_POSITION]
        path_cmd_vel = self.command[:, self.CMD_VEL]
        path_cmd_yaw_rate = self.command[:, self.CMD_YAW_RATE]
        phase = self.command[:, self.CMD_PHASE]
        neck_cmd = self.command[:, self.CMD_NECK]

        # -- Root orientation --
        root_orientations = sim.q_i[:, 0, 3:]
        path_orientation = yaw_to_quat(path_heading)
        root_orientation_in_path = quat_inv_mul(path_orientation, root_orientations)

        # Add noise to roll and pitch
        root_orientation_in_path_euler = quat_to_euler_ZYX(root_orientation_in_path)
        if self.root_orientation_sigma > 0.0:
            s = self.root_orientation_sigma
            root_orientation_in_path_euler[:, 1:] += (
                (2.0 * s) * torch.rand((nw, 2), device=device) - s
            )
        noisy_root_orientation_in_path = euler_ZYX_to_quat(root_orientation_in_path_euler)

        # Orientation as 9D rotation matrix
        root_orientation_in_path_9d = quat_to_rotation9D(noisy_root_orientation_in_path)

        # Heading (from noisy orientation)
        root_heading_in_path = quat_to_projected_yaw(noisy_root_orientation_in_path)

        # -- Position --
        root_translation_in_path = yaw_inv_apply_2d(
            path_heading, sim.q_i[:, 0, :2] - path_position_2d
        )
        path_translation_in_heading = yaw_inv_apply_2d(
            root_heading_in_path, -root_translation_in_path
        )

        # -- Velocities --
        root_lin_vel_in_root = quat_inv_apply(root_orientations, sim.u_i[:, 0, :3])
        root_ang_vel_in_root = quat_inv_apply(root_orientations, sim.u_i[:, 0, 3:])

        noisy_root_lin_vel_in_root = root_lin_vel_in_root
        noisy_root_ang_vel_in_root = root_ang_vel_in_root
        if self.root_linear_velocity_sigma > 0.0:
            s = self.root_linear_velocity_sigma
            noisy_root_lin_vel_in_root = root_lin_vel_in_root + (
                (2.0 * s) * torch.rand((nw, 3), device=device) - s
            )
        if self.root_angular_velocity_sigma > 0.0:
            s = self.root_angular_velocity_sigma
            noisy_root_ang_vel_in_root = root_ang_vel_in_root + (
                (2.0 * s) * torch.rand((nw, 3), device=device) - s
            )

        # Path velocities rotated to root frame
        path_lin_vel_in_path = torch.cat(
            (
                path_cmd_vel,
                torch.zeros((nw, 1), dtype=torch.float, device=device),
            ),
            dim=-1,
        )
        path_ang_vel_in_path = torch.cat(
            (
                torch.zeros((nw, 2), dtype=torch.float, device=device),
                path_cmd_yaw_rate.view(-1, 1),
            ),
            dim=-1,
        )
        path_lin_vel_in_root = quat_inv_apply(noisy_root_orientation_in_path, path_lin_vel_in_path)
        path_ang_vel_in_root = quat_inv_apply(noisy_root_orientation_in_path, path_ang_vel_in_path)

        # -- Phase encoding --
        phase_enc_vector = _phase_encoding(phase, self._freq_2pi, self._offset)

        # -- Joint state --
        dof_positions = sim.q_j
        dof_velocity_estimate = sim.dq_j
        relative_dof_positions = (dof_positions - self._joint_position_default) / self._joint_position_range

        # -- Action history --
        if setpoints is not None:
            self._action_hist_1.copy_(self._action_hist_0)
            self._action_hist_0[:] = (
                setpoints - self._joint_position_default
            ) / self._joint_position_range

        # -- Build observation --
        obs = torch.cat(
            (
                root_orientation_in_path_9d,
                root_translation_in_path / self.path_deviation_scale,
                path_translation_in_heading / self.path_deviation_scale,
                path_cmd_vel,
                path_cmd_yaw_rate.view(-1, 1),
                path_lin_vel_in_root,
                path_ang_vel_in_root,
                phase_enc_vector,
                neck_cmd,
                noisy_root_lin_vel_in_root,
                noisy_root_ang_vel_in_root,
                relative_dof_positions,
                dof_velocity_estimate / self.joint_velocity_scale,
                self._action_hist_0,
                self._action_hist_1,
            ),
            dim=-1,
        )

        # Cache intermediates for privileged observations (subclasses)
        self._root_orientation_in_path = root_orientation_in_path
        self._root_lin_vel_in_root = root_lin_vel_in_root
        self._root_ang_vel_in_root = root_ang_vel_in_root

        return obs

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset action history for the given environments."""
        if env_ids is None:
            normalized = (
                self._body_sim.q_j - self._joint_position_default
            ) / self._joint_position_range
            self._action_hist_0[:] = normalized
            self._action_hist_1[:] = normalized
        else:
            normalized = (
                self._body_sim.q_j[env_ids] - self._joint_position_default
            ) / self._joint_position_range
            self._action_hist_0[env_ids] = normalized
            self._action_hist_1[env_ids] = normalized
