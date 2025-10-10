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

"""PID Controller Interfaces"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp
from warp.context import Devicelike

from ..core.control import Control
from ..core.joints import JointActuationType
from ..core.model import Model
from ..core.state import State
from ..core.time import TimeData
from ..core.types import float32, int32

###
# Module interface
###


__all__ = [
    "JointSpacePIDController",
    "PIDControllerData",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Types
###


ArrayLike = np.ndarray | list[float]
"""An Array-like structure for aliasing various data types compatible with numpy."""


@dataclass
class PIDControllerData:
    """A data container for joint-space PID controller parameters and state."""

    q_j_ref: wp.array | None = None
    """The reference actuator joint positions."""
    dq_j_ref: wp.array | None = None
    """The reference actuator joint velocities."""
    tau_j_ref: wp.array | None = None
    """The feedforward actuator joint torques."""
    K_p: wp.array | None = None
    """The proportional gains."""
    K_i: wp.array | None = None
    """The integral gains."""
    K_d: wp.array | None = None
    """The derivative gains."""
    integrator: wp.array | None = None
    """Integrator of joint-space position tracking error."""
    decimation: wp.array | None = None
    """The control decimation for each world expressed as a multiple of simulation steps."""


###
# Kernels
###


@wp.kernel
def _reset_jointspace_pid_references(
    # Inputs
    model_info_joint_dofs_offset: wp.array(dtype=int32),
    model_info_joint_actuated_dofs_offset: wp.array(dtype=int32),
    model_joints_wid: wp.array(dtype=int32),
    model_joints_act_type: wp.array(dtype=int32),
    model_joints_num_dofs: wp.array(dtype=int32),
    model_joints_dofs_offset: wp.array(dtype=int32),
    model_joints_actuated_dofs_offset: wp.array(dtype=int32),
    state_joints_q_j: wp.array(dtype=float32),
    state_joints_dq_j: wp.array(dtype=float32),
    # Outputs
    controller_q_j_ref: wp.array(dtype=float32),
    controller_dq_j_ref: wp.array(dtype=float32),
):
    """
    A kernel to compute joint-space PID control torques for force-actuated joints.
    """
    """
    A kernel to compute joint-space PID control torques for force-actuated joints.
    """
    # Retrieve the the joint index from the thread indices
    jid = wp.tid()

    # Retrieve the joint actuation type
    act_type = model_joints_act_type[jid]

    # Retrieve the world index from the thread indices
    wid = model_joints_wid[jid]

    # Only procceed for force actuated joints and at
    # simulation steps matching the control decimation
    if act_type != JointActuationType.FORCE:
        return

    # Retrieve the offset of the world's joints in the global DoF vector
    world_dof_offset = model_info_joint_dofs_offset[wid]
    world_actuated_dof_offset = model_info_joint_actuated_dofs_offset[wid]

    # Retrieve the number of DoFs and offset of the joint
    num_dofs = model_joints_num_dofs[jid]
    dofs_offset = model_joints_dofs_offset[jid]
    actuated_dofs_offset = model_joints_actuated_dofs_offset[jid]

    # Compute the global DoF offset of the joint
    dofs_offset += world_dof_offset
    actuated_dofs_offset += world_actuated_dof_offset

    # Iterate over the DoFs of the joint
    for dof in range(num_dofs):
        # Compute the DoF index in the global DoF vector
        dof_index = dofs_offset + dof

        # Compute the actuator index in the controller vectors
        actuator_dof_index = actuated_dofs_offset + dof

        # Retrieve the current joint state
        q_j = state_joints_q_j[dof_index]
        dq_j = state_joints_dq_j[dof_index]

        # Retrieve the current controller references
        controller_q_j_ref[actuator_dof_index] = q_j
        controller_dq_j_ref[actuator_dof_index] = dq_j


@wp.kernel
def _compute_jointspace_pid_control(
    # Inputs
    model_info_joint_dofs_offset: wp.array(dtype=int32),
    model_info_joint_actuated_dofs_offset: wp.array(dtype=int32),
    model_joints_wid: wp.array(dtype=int32),
    model_joints_act_type: wp.array(dtype=int32),
    model_joints_num_dofs: wp.array(dtype=int32),
    model_joints_dofs_offset: wp.array(dtype=int32),
    model_joints_actuated_dofs_offset: wp.array(dtype=int32),
    model_joints_tau_j_max: wp.array(dtype=float32),
    model_time_dt: wp.array(dtype=float32),
    state_time_steps: wp.array(dtype=int32),
    state_joints_q_j: wp.array(dtype=float32),
    state_joints_dq_j: wp.array(dtype=float32),
    controller_q_j_ref: wp.array(dtype=float32),
    controller_dq_j_ref: wp.array(dtype=float32),
    controller_tau_j_ref: wp.array(dtype=float32),
    controller_K_p: wp.array(dtype=float32),
    controller_K_i: wp.array(dtype=float32),
    controller_K_d: wp.array(dtype=float32),
    controller_integrator: wp.array(dtype=float32),
    controller_decimation: wp.array(dtype=int32),
    # Outputs
    control_tau_j: wp.array(dtype=float32),
):
    """
    A kernel to compute joint-space PID control torques for force-actuated joints.
    """
    # Retrieve the the joint index from the thread indices
    jid = wp.tid()

    # Retrieve the joint actuation type
    act_type = model_joints_act_type[jid]

    # Retrieve the world index from the thread indices
    wid = model_joints_wid[jid]

    # Retrieve the current simulation step
    step = state_time_steps[wid]

    # Retrieve the control decimation for the world
    decimation = controller_decimation[wid]

    # Only procceed for force actuated joints and at
    # simulation steps matching the control decimation
    if act_type != JointActuationType.FORCE or step % decimation != 0:
        return

    # Retrieve the time step and current time
    dt = model_time_dt[wid]

    # TODO: Enable this
    # Decimate the simulation time-step by the control
    # decimation to get the effective control time-step
    dt *= float32(decimation)

    # Retrieve the offset of the world's joints in the global DoF vector
    world_dof_offset = model_info_joint_dofs_offset[wid]
    world_actuated_dof_offset = model_info_joint_actuated_dofs_offset[wid]

    # Retrieve the number of DoFs and offset of the joint
    num_dofs = model_joints_num_dofs[jid]
    dofs_offset = model_joints_dofs_offset[jid]
    actuated_dofs_offset = model_joints_actuated_dofs_offset[jid]

    # Compute the global DoF offset of the joint
    dofs_offset += world_dof_offset
    actuated_dofs_offset += world_actuated_dof_offset

    # Iterate over the DoFs of the joint
    for dof in range(num_dofs):
        # Compute the DoF index in the global DoF vector
        dof_index = dofs_offset + dof

        # Compute the actuator index in the controller vectors
        actuator_dof_index = actuated_dofs_offset + dof
        # wp.printf("[step=%d][jid=%d]: dof_index: %d, actuator_dof_index: %d\n", step, jid, dof_index, actuator_dof_index)

        # Retrieve the maximum limit of the generalized actuator forces
        tau_j_max = model_joints_tau_j_max[dof_index]
        # wp.printf("[step=%d][jid=%d]: dof_index: %d, tau_j_max: %f\n", step, jid, dof_index, tau_j_max)

        # Retrieve the current joint state
        q_j = state_joints_q_j[dof_index]
        dq_j = state_joints_dq_j[dof_index]

        # Retrieve the current controller references
        q_j_ref = controller_q_j_ref[actuator_dof_index]
        dq_j_ref = controller_dq_j_ref[actuator_dof_index]
        tau_j_ref = controller_tau_j_ref[actuator_dof_index]

        # Retrieve the controller gains and integrator state
        K_p = controller_K_p[actuator_dof_index]
        K_i = controller_K_i[actuator_dof_index]
        K_d = controller_K_d[actuator_dof_index]
        integrator = controller_integrator[actuator_dof_index]
        wp.printf(
            "[step=%d][aid=%d]: K_p: %f, K_i: %f, K_d: %f, q_j_err: %f, dq_j_err: %f\n",
            step,
            actuator_dof_index,
            K_p,
            K_i,
            K_d,
            q_j_ref - q_j,
            dq_j_ref - dq_j,
        )

        # Update the integrator state with anti-windup clamping
        integrator += (q_j_ref - q_j) * dt
        integrator = wp.clamp(integrator, -tau_j_max, tau_j_max)

        # Compute the Feed-Forward + PID control generalized forces
        tau_j_c = tau_j_ref + K_p * (q_j_ref - q_j) + K_d * (dq_j_ref - dq_j) + K_i * integrator

        # Clamp the generalized control forces to the joint limits
        tau_j_c = wp.clamp(tau_j_c, -tau_j_max, tau_j_max)

        # Store the updated integrator state and actuator control forces
        controller_integrator[actuator_dof_index] = integrator
        control_tau_j[dof_index] = tau_j_c


###
# Launchers
###


def reset_jointspace_pid_references(
    controller: PIDControllerData,
    model: Model,
    state: State,
) -> None:
    """
    A kernel launcher to compute joint-space PID control torques for force-actuated joints.
    """
    wp.launch(
        _reset_jointspace_pid_references,
        dim=model.size.sum_of_num_joints,
        inputs=[
            # Inputs
            model.info.joint_dofs_offset,
            model.info.joint_actuated_dofs_offset,
            model.joints.wid,
            model.joints.act_type,
            model.joints.num_dofs,
            model.joints.dofs_offset,
            model.joints.actuated_dofs_offset,
            state.q_j,
            state.dq_j,
            # Outputs
            controller.q_j_ref,
            controller.dq_j_ref,
        ],
    )


def compute_jointspace_pid_control(
    # Inputs:
    controller: PIDControllerData,
    model: Model,
    state: State,
    time: TimeData,
    # Outputs:
    control: Control,
) -> None:
    """
    A kernel launcher to compute joint-space PID control torques for force-actuated joints.
    """
    wp.launch(
        _compute_jointspace_pid_control,
        dim=model.size.sum_of_num_joints,
        inputs=[
            # Inputs
            model.info.joint_dofs_offset,
            model.info.joint_actuated_dofs_offset,
            model.joints.wid,
            model.joints.act_type,
            model.joints.num_dofs,
            model.joints.dofs_offset,
            model.joints.actuated_dofs_offset,
            model.joints.tau_j_max,
            model.time.dt,
            time.steps,
            state.q_j,
            state.dq_j,
            controller.q_j_ref,
            controller.dq_j_ref,
            controller.tau_j_ref,
            controller.K_p,
            controller.K_i,
            controller.K_d,
            controller.integrator,
            controller.decimation,
            # Outputs
            control.tau_j,
        ],
    )


###
# Interfaces
###


class JointSpacePIDController:
    """
    A simple PID controller in joint space.
    """

    def __init__(
        self,
        model: Model | None = None,
        K_p: ArrayLike | None = None,
        K_i: ArrayLike | None = None,
        K_d: ArrayLike | None = None,
        decimation: ArrayLike | None = None,
        device: Devicelike = None,
    ):
        """
        A simple PID controller in joint space.

        Args:
            model (Model | None): Model used to size and allocate controller buffers.
                If None, call ``allocate()`` later.
            K_p (ArrayLike | None): Proportional gains per actuated joint DoF.
            K_i (ArrayLike | None): Integral gains per actuated joint DoF.
            K_d (ArrayLike | None): Derivative gains per actuated joint DoF.
            decimation (ArrayLike | None): Control decimation for each world
                expressed as a multiple of simulation steps.
            device (Devicelike | None): Device to use for allocations and execution.
        """

        # Cache the device
        self._device: Devicelike = device

        # Declare the internal controller data
        self._data: PIDControllerData | None = None

        # If a model is provided, allocate the controller data
        if model is not None:
            self.allocate(model, K_p, K_i, K_d, decimation, device)

    ###
    # Properties
    ###

    @property
    def data(self) -> PIDControllerData:
        """The internal controller data."""
        if self._data is None:
            raise RuntimeError("Controller data is not allocated. Call allocate() first.")
        return self._data

    @property
    def device(self) -> Devicelike:
        """The device used for allocations and execution."""
        return self._device

    ###
    # Operations
    ###

    def allocate(
        self,
        model: Model,
        K_p: ArrayLike | None = None,
        K_i: ArrayLike | None = None,
        K_d: ArrayLike | None = None,
        decimation: ArrayLike | None = None,
        device: Devicelike = None,
    ) -> None:
        """
        Allocate the controller data.
        """

        # Get the number of actuated coordinates and DoFs
        num_actuated_coords = model.size.sum_of_num_actuated_joint_coords
        num_actuated_dofs = model.size.sum_of_num_actuated_joint_dofs

        # Check if there are any actuated DoFs
        if num_actuated_dofs == 0:
            raise ValueError("Model has no actuated DoFs.")

        # Ensure the model has only 1-DoF actuated joints
        if num_actuated_coords != num_actuated_dofs:
            raise ValueError(
                f"Model has {num_actuated_coords} actuated coordinates but {num_actuated_dofs} actuated DoFs. "
                "Joint-space PID control is currently incompatible with multi-DoF actuated joints."
            )

        # Check length of gain arrays
        if K_p is not None and len(K_p) != num_actuated_dofs:
            raise ValueError(f"K_p must have length {num_actuated_dofs}, but has length {len(K_p)}")
        if K_i is not None and len(K_i) != num_actuated_dofs:
            raise ValueError(f"K_i must have length {num_actuated_dofs}, but has length {len(K_i)}")
        if K_d is not None and len(K_d) != num_actuated_dofs:
            raise ValueError(f"K_d must have length {num_actuated_dofs}, but has length {len(K_d)}")
        if decimation is not None and len(decimation) != model.size.num_worlds:
            raise ValueError(f"decimation must have length {model.size.num_worlds}, but has length {len(decimation)}")

        # Override the device if provided
        if device is not None:
            self._device = device

        # Set default decimation if not provided
        if decimation is None:
            decimation = np.ones(model.size.num_worlds, dtype=np.int32)

        # Allocate the controller data
        with wp.ScopedDevice(self._device):
            self._data = PIDControllerData(
                q_j_ref=wp.zeros(num_actuated_dofs, dtype=float32),
                dq_j_ref=wp.zeros(num_actuated_dofs, dtype=float32),
                tau_j_ref=wp.zeros(num_actuated_dofs, dtype=float32),
                K_p=wp.array(K_p if K_p is not None else np.zeros(num_actuated_dofs), dtype=float32),
                K_i=wp.array(K_i if K_i is not None else np.zeros(num_actuated_dofs), dtype=float32),
                K_d=wp.array(K_d if K_d is not None else np.zeros(num_actuated_dofs), dtype=float32),
                integrator=wp.zeros(num_actuated_dofs, dtype=float32),
                decimation=wp.array(decimation, dtype=int32),
            )

    def reset(self, model: Model, state: State) -> None:
        """
        Reset the controller state.
        """

        # First reset the references to the current state
        reset_jointspace_pid_references(
            controller=self._data,
            model=model,
            state=state,
        )

        # Then zero the integrator and feedforward torques
        self._data.tau_j_ref.zero_()
        self._data.integrator.zero_()

    def set_references(
        self, q_j_ref: ArrayLike, dq_j_ref: ArrayLike | None = None, tau_j_ref: ArrayLike | None = None
    ) -> None:
        """
        Set the controller reference trajectories.

        Args:
            q_j_ref (ArrayLike): The reference actuator joint positions.
            dq_j_ref (ArrayLike | None): The reference actuator joint velocities.
            tau_j_ref (ArrayLike | None): The feedforward actuator joint torques.
        """
        if len(q_j_ref) != len(self._data.q_j_ref):
            raise ValueError(f"q_j_ref must have length {len(self._data.q_j_ref)}, but has length {len(q_j_ref)}")
        self._data.q_j_ref.assign(q_j_ref)

        if dq_j_ref is not None:
            if len(dq_j_ref) != len(self._data.dq_j_ref):
                raise ValueError(
                    f"dq_j_ref must have length {len(self._data.dq_j_ref)}, but has length {len(dq_j_ref)}"
                )
            self._data.dq_j_ref.assign(dq_j_ref)

        if tau_j_ref is not None:
            if len(tau_j_ref) != len(self._data.tau_j_ref):
                raise ValueError(
                    f"tau_j_ref must have length {len(self._data.tau_j_ref)}, but has length {len(tau_j_ref)}"
                )
            self._data.tau_j_ref.assign(tau_j_ref)

    def compute(
        self,
        model: Model,
        state: State,
        time: TimeData,
        control: Control,
    ) -> None:
        """
        Compute the control torques.

        Args:
            model (Model): The model.
            data (ModelData): The model data.
            q_j_ref (wp.array): The reference actuator joint positions.
            dq_j_ref (wp.array): The reference actuator joint velocities.
            tau_j_ref (wp.array): The feedforward actuator joint torques.
        """
        compute_jointspace_pid_control(
            controller=self._data,
            model=model,
            state=state,
            time=time,
            control=control,
        )
