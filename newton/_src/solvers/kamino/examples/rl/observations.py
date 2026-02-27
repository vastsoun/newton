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

import torch
import warp as wp

from newton._src.solvers.kamino.utils.sim import Simulator


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
    """

    def __init__(
        self,
        sim: Simulator,
        num_worlds: int,
        device: str,
        actuated_joint_indices: list[int],
        num_actions: int = 12,
    ) -> None:
        super().__init__(sim=sim, num_worlds=num_worlds, device=device, command_dim=0)
        self._actuated_joint_indices = torch.tensor(
            actuated_joint_indices,
            device=device,
            dtype=torch.long,
        )
        self._num_actions = num_actions
        self._num_dofs = sim.model.size.max_of_num_joint_coords

        # Action history buffer (most recent actions for actuated joints).
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
            self._action_history[:] = actions

        obs = torch.cat([q_j, self._action_history], dim=-1)  # (num_worlds, 48)
        return obs

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._action_history.zero_()
        else:
            self._action_history[env_ids] = 0.0
