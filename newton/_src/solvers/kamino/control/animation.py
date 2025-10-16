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

"""Containers and interfaces for animation reference tracking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp
from matplotlib import pyplot as plt
from warp.context import Devicelike

from ..core.model import Model
from ..core.time import TimeData
from ..core.types import float32, int32

###
# Module interface
###


__all__ = [
    "AnimationJointReference",
    "AnimationJointReferenceData",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Types
###


@dataclass
class AnimationJointReferenceData:
    """
    Container of animation references for actuated joints.

    Attributes:
        num_actuated_joint_dofs: wp.array
            The number of actuated joint DoFs per world.
        actuated_joint_dofs_offset: wp.array
            The offset indices for the actuated joint DoFs per world.
        q_j_ref: wp.array
            The reference actuator joint positions.
        dq_j_ref: wp.array
            The reference actuator joint velocities.
        loop: wp.array
            Flag indicating whether the animation should loop.
        rate: wp.array
            The rate at which to progress the animation sequence.
        length: wp.array
            The length of the animation sequence.
        step: wp.array
            The current step index in the animation sequence.

    Note:
        By default, the animation reference is allocated such that all worlds share
        the same reference data, but can progress and/or loop independently.
    """

    num_actuated_joint_dofs: wp.array | None = None
    """
    Number of actuated joint DoFs per world.\n
    Shape is ``(num_worlds,)`` and dtype is :class:`int32`.
    """

    actuated_joint_dofs_offset: wp.array | None = None
    """
    Offset indices for the actuated joint DoFs per world.\n
    Shape is ``(num_worlds,)`` and dtype is :class:`int32`.
    """

    q_j_ref: wp.array | None = None
    """
    Sequence of reference joint actuator positions.\n
    Shape is ``(max_of_num_actuated_joint_coords, sequence_length)`` and dtype is :class:`float32`.
    """

    dq_j_ref: wp.array | None = None
    """
    Sequence of reference joint actuator velocities.\n
    Shape is ``(max_of_num_actuated_joint_dofs, sequence_length)`` and dtype is :class:`float32`.
    """

    loop: wp.array | None = None
    """
    Integer flag to indicate if the animation should loop.\n
    Shape is ``(num_worlds,)`` and dtype is :class:`int32`.\n
    If `1`, the animation will restart from the beginning after reaching the end.\n
    If `0`, the animation will stop at the last frame.
    """

    rate: wp.array | None = None
    """
    Integer rate by which to progress the active frame of the animation sequence.\n
    Shape is ``(num_worlds,)`` and dtype is :class:`int32`.
    """

    length: wp.array | None = None
    """
    Integer length of the animation sequence.\n
    Shape is ``(num_worlds,)`` and dtype is :class:`int32`.
    """

    frame: wp.array | None = None
    """
    Integer index indicating the active frame of the animation sequence.\n
    Shape is ``(num_worlds,)`` and dtype is :class:`int32`.
    """


###
# Kernels
###


@wp.kernel
def _extract_initial_animation_references(
    # Inputs
    num_actuated_joint_dofs: wp.array(dtype=int32),
    actuated_joint_dofs_offset: wp.array(dtype=int32),
    animation_q_j_ref: wp.array2d(dtype=float32),
    animation_dq_j_ref: wp.array2d(dtype=float32),
    # Outputs
    q_j_ref_active: wp.array(dtype=float32),
    dq_j_ref_active: wp.array(dtype=float32),
):
    """
    A kernel to compute joint-space PID control torques for force-actuated joints.
    """
    # Retrieve the the world and DoF index from the thread indices
    wid, qid = wp.tid()

    # Retrieve the number of actuated DoFs and offset for this world
    num_ajq = num_actuated_joint_dofs[wid]
    ajq_offset = actuated_joint_dofs_offset[wid]

    # Ensure we are within the valid range of actuated DoFs for this world
    if qid >= num_ajq:
        return

    # Compute the global DoF index
    dof_index = ajq_offset + qid

    # Update the active reference arrays
    q_j_ref_active[dof_index] = animation_q_j_ref[0, qid]
    dq_j_ref_active[dof_index] = animation_dq_j_ref[0, qid]


@wp.kernel
def _advance_animation_step(
    # Inputs
    time_steps: wp.array(dtype=int32),
    animation_loop: wp.array(dtype=int32),
    animation_rate: wp.array(dtype=int32),
    animation_length: wp.array(dtype=int32),
    # Outputs
    animation_frame: wp.array(dtype=int32),
):
    """
    A kernel to compute joint-space PID control torques for force-actuated joints.
    """
    # Retrieve the the world index from the thread indices
    wid = wp.tid()

    # Retrieve the animation sequence info
    loop = animation_loop[wid]
    rate = animation_rate[wid]
    length = animation_length[wid]

    # Retrieve the current step (i.e. discrete-time index) for this world
    step = time_steps[wid]

    # Check if we need to advance the animation frame
    if step % rate != 0:
        return

    # Retrieve the current frame index for this world
    frame = animation_frame[wid]

    # Advance the frame index
    frame += 1

    # If looping is enabled, wrap the frame index around
    if loop:
        frame %= length
    # Otherwise, clamp the frame index to the last frame
    else:
        if frame >= length:
            frame = length - 1

    # Update the active reference arrays
    animation_frame[wid] = frame


# TODO: Make the 2D arrays as flattened 1D arrays to handle arbitrary layouts
@wp.kernel
def _extract_animation_references(
    # Inputs
    num_actuated_joint_dofs: wp.array(dtype=int32),
    actuated_joint_dofs_offset: wp.array(dtype=int32),
    animation_step: wp.array(dtype=int32),
    animation_q_j_ref: wp.array2d(dtype=float32),
    animation_dq_j_ref: wp.array2d(dtype=float32),
    # Outputs
    q_j_ref_active: wp.array(dtype=float32),
    dq_j_ref_active: wp.array(dtype=float32),
):
    """
    A kernel to compute joint-space PID control torques for force-actuated joints.
    """
    # Retrieve the the world and DoF index from the thread indices
    wid, qid = wp.tid()

    # Retrieve the number of actuated DoFs and offset for this world
    num_ajq = num_actuated_joint_dofs[wid]
    ajq_offset = actuated_joint_dofs_offset[wid]

    # Ensure we are within the valid range of actuated DoFs for this world
    if qid >= num_ajq:
        return

    # Retrieve the current step index for this world
    step = animation_step[wid]

    # Compute the global DoF index
    dof_index = ajq_offset + qid

    # Update the active reference arrays
    animation_step[wid] = step
    q_j_ref_active[dof_index] = animation_q_j_ref[step, qid]
    dq_j_ref_active[dof_index] = animation_dq_j_ref[step, qid]


###
# Interfaces
###


class AnimationJointReference:
    """
    Interface for joint-based animation reference tracking.

    Args:
        model: The simulation model.
        device: The device to store the reference data on.
    """

    def __init__(
        self,
        model: Model | None = None,
        input: np.ndarray | None = None,
        rate: int | list[int] = 1,
        loop: bool | list[bool] = True,
        use_fd: bool = False,
        fd_dt: float = np.inf,
        device: Devicelike = None,
    ):
        """
        A simple PID controller in joint space.

        Args:
            model (Model | None): Model used to size and allocate controller buffers.
                If None, call ``allocate()`` later.
            input (np.ndarray | None): Optional input data to initialize the animation references.
            device (Devicelike | None): Device to use for allocations and execution.
        """

        # Cache the device
        self._device: Devicelike = device

        # Declare the model dimensions meta-data
        self._num_worlds: int = 0
        self._max_of_num_actuated_dofs: int = 0
        self._sequence_length: int = 0

        # Declare the internal controller data
        self._data: AnimationJointReferenceData | None = None

        # If a model is provided, allocate the controller data
        if model is not None:
            self.allocate(
                model=model,
                input=input,
                rate=rate,
                loop=loop,
                use_fd=use_fd,
                fd_dt=fd_dt,
                device=device,
            )

    ###
    # Properties
    ###

    @property
    def device(self) -> Devicelike | None:
        """The device used for allocations and execution."""
        return self._device

    @property
    def sequence_length(self) -> int:
        """The length of the animation sequence."""
        return self._sequence_length

    @property
    def data(self) -> AnimationJointReferenceData:
        """The internal controller data."""
        if self._data is None:
            raise ValueError("Controller data is not allocated. Call allocate() first.")
        return self._data

    ###
    # Operations
    ###

    def allocate(
        self,
        model: Model,
        input: np.ndarray,
        rate: int | list[int] = 1,
        loop: bool | list[bool] = True,
        use_fd: bool = False,
        fd_dt: float = np.inf,
        device: Devicelike = None,
    ) -> None:
        """
        Allocate the animation joint reference data.

        Args:
            model (Model): The simulation model used to size the controller data.
            input (np.ndarray | None): Optional input data to initialize the animation references.
            rate (int | list[int]): Rate at which to progress the animation sequence.\n
                If a single integer is provided, it is applied to all worlds.\n
                If a list is provided, its length must match the number of worlds.
            loop (bool | list[bool]): Flag indicating whether the animation should loop.\n
                If a single boolean is provided, it is applied to all worlds.\n
                If a list is provided, its length must match the number of worlds.
            device (Devicelike | None): Device to use for allocations. If None, uses the existing device.
        Raises:
            ValueError: If the model is not valid or actuated DoFs are not properly configured.

        Note:
            The model must have only 1-DoF actuated joints for this controller to be compatible.
        """
        # Ensure the model is valid
        if model is None or model.size is None:
            raise ValueError("Model is not valid. Cannot allocate controller data.")

        # Retrieve the shape of the input data
        if input is None:
            raise ValueError("Input data must be provided for allocation.")

        # Ensure input array is valid
        if not isinstance(input, np.ndarray):
            raise ValueError("Input data must be a numpy array.")
        if input.ndim != 2:
            raise ValueError("Input data must be a 2D numpy array.")

        # Get the number of actuated coordinates and DoFs
        total_num_actuated_coords = model.size.sum_of_num_actuated_joint_coords
        total_num_actuated_dofs = model.size.sum_of_num_actuated_joint_dofs
        max_num_actuated_dofs = model.size.max_of_num_actuated_joint_dofs

        # Check if there are any actuated DoFs
        if total_num_actuated_dofs == 0:
            raise ValueError("Model has no actuated DoFs.")

        # Ensure the model has only 1-DoF actuated joints
        if total_num_actuated_coords != total_num_actuated_dofs:
            raise ValueError(
                f"Model has {total_num_actuated_coords} actuated coordinates but {total_num_actuated_dofs} actuated "
                "DoFs. AnimationJointReference is currently incompatible with multi-DoF actuated joints."
            )

        # Check that input data matches the number of actuated DoFs
        if input.shape[1] != max_num_actuated_dofs and input.shape[0] != max_num_actuated_dofs:
            raise ValueError(
                f"Input data has shape {input.shape} which does not match the "
                f"per-world number of actuated DoFs ({max_num_actuated_dofs})."
            )

        # We assume the input is organized as (sequence_length, num_actuated_dofs)
        # Transpose the input if necessary in order to match the assumed shape
        if input.shape[0] == max_num_actuated_dofs or input.shape[0] == 2 * max_num_actuated_dofs:
            input = input.T

        # Cache the model dimensions meta-data
        self._num_worlds = model.size.num_worlds
        self._max_of_num_actuated_dofs = max_num_actuated_dofs
        self._sequence_length = input.shape[0]

        # Extract the reference joint positions and velocities
        q_j_ref_np = input[:, :max_num_actuated_dofs].astype(np.float32)
        if input.shape[1] >= 2 * max_num_actuated_dofs:
            dq_j_ref_np = input[:, max_num_actuated_dofs : 2 * max_num_actuated_dofs].astype(np.float32)
        else:
            # Use finite-differences to estimate velocities if requested
            if use_fd:
                # Compute raw finite-difference velocities
                dq_j_ref_np = np.zeros_like(q_j_ref_np)
                dq_j_ref_np[1:] = np.diff(q_j_ref_np, axis=0) / fd_dt

                # Set the first velocity to match the second
                dq_j_ref_np[0] = dq_j_ref_np[1]

                # Apply a simple moving average filter to smooth out the velocities
                kernel_size = 5
                kernel = np.ones(kernel_size) / kernel_size
                for i in range(max_num_actuated_dofs):
                    dq_j_ref_np[:, i] = np.convolve(dq_j_ref_np[:, i], kernel, mode="same")

            # Otherwise, default to zero velocities
            else:
                dq_j_ref_np = np.zeros_like(q_j_ref_np)

        # Create the rate and loop arrays
        # TODO: Allow different rates/looping per world
        length_np = np.array([q_j_ref_np.shape[0]] * self._num_worlds, dtype=np.int32)
        rate_np = rate * np.ones(self._num_worlds, dtype=np.int32)
        loop_np = (1 if loop else 0) * np.ones(self._num_worlds, dtype=np.int32)

        # Override the device if provided
        if device is not None:
            self._device = device

        # Allocate the controller data
        with wp.ScopedDevice(self._device):
            self._data = AnimationJointReferenceData(
                num_actuated_joint_dofs=model.info.num_actuated_joint_dofs,
                actuated_joint_dofs_offset=model.info.joint_actuated_dofs_offset,
                q_j_ref=wp.array(q_j_ref_np, dtype=float32),
                dq_j_ref=wp.array(dq_j_ref_np, dtype=float32),
                length=wp.array(length_np, dtype=int32),
                rate=wp.array(rate_np, dtype=int32),
                loop=wp.array(loop_np, dtype=int32),
                frame=wp.zeros(self._num_worlds, dtype=int32),
            )

    def plot(self) -> None:
        # Extract numpy arrays for plotting
        q_j_ref_np = self._data.q_j_ref.numpy()
        dq_j_ref_np = self._data.dq_j_ref.numpy()

        # Plot the input data for verification
        _, axs = plt.subplots(2, 1, figsize=(10, 6))
        for i in range(self._max_of_num_actuated_dofs):
            axs[0].plot(q_j_ref_np[:, i], label=f"Joint {i}")
            axs[1].plot(dq_j_ref_np[:, i], label=f"Joint {i}")
        axs[0].set_title("Reference Joint Positions")
        axs[0].set_xlabel("Frame")
        axs[0].set_ylabel("Position")
        axs[0].legend()
        axs[1].set_title("Reference Joint Velocities")
        axs[1].set_xlabel("Frame")
        axs[1].set_ylabel("Velocity")
        axs[1].legend()
        plt.tight_layout()
        plt.show()

    # TODO: Make the enabled flag a list to allow different settings per world
    def loop(self, enabled=True) -> None:
        """
        Enable or disable looping of the animation sequence.

        Args:
            enabled (bool): If True, enable looping. If False, disable looping.
        """
        if self._data is None:
            raise ValueError("Controller data is not allocated. Call allocate() first.")
        self._data.loop.fill(1 if enabled else 0)

    def reset(self, q_j_ref_out: wp.array, dq_j_ref_out: wp.array) -> None:
        """
        Reset the animation sequence to the beginning and sets the initial references into the output arrays.
        """
        if self._data is None:
            raise ValueError("Controller data is not allocated. Call allocate() first.")
        self._data.frame.fill_(0)
        wp.launch(
            _extract_initial_animation_references,
            dim=(self._num_worlds, self._max_of_num_actuated_dofs),
            inputs=[
                self._data.num_actuated_joint_dofs,
                self._data.actuated_joint_dofs_offset,
                self._data.q_j_ref,
                self._data.dq_j_ref,
                q_j_ref_out,
                dq_j_ref_out,
            ],
            device=self._device,
        )

    def extract(self, q_j_ref_out: wp.array, dq_j_ref_out: wp.array) -> None:
        """
        Extract the reference arrays from the animation sequence at the current time-step.

        Args:
            q_j_ref_out (wp.array): Output array for the reference joint positions.
            dq_j_ref_out (wp.array): Output array for the reference joint velocities.
        """
        wp.launch(
            _extract_animation_references,
            dim=(self._num_worlds, self._max_of_num_actuated_dofs),
            inputs=[
                # Inputs:
                self._data.num_actuated_joint_dofs,
                self._data.actuated_joint_dofs_offset,
                self._data.frame,
                self._data.q_j_ref,
                self._data.dq_j_ref,
                # Outputs:
                q_j_ref_out,
                dq_j_ref_out,
            ],
            device=self._device,
        )

    def step(self, time: TimeData, q_j_ref_out: wp.array, dq_j_ref_out: wp.array) -> None:
        """
        Advance the animation sequence by the configured rate and copy the results to the output arrays.

        Args:
            q_j_ref_out (wp.array): Output array for the reference joint positions.
            dq_j_ref_out (wp.array): Output array for the reference joint velocities.
        """

        # First launch a kernel to advance the animation step index
        wp.launch(
            _advance_animation_step,
            dim=self._num_worlds,
            inputs=[
                # Inputs:
                time.steps,
                self._data.loop,
                self._data.rate,
                self._data.length,
                # Outputs:
                self._data.frame,
            ],
            device=self._device,
        )

        # Next launch a kernel to update the active reference arrays
        wp.launch(
            _extract_animation_references,
            dim=(self._num_worlds, self._max_of_num_actuated_dofs),
            inputs=[
                # Inputs:
                self._data.num_actuated_joint_dofs,
                self._data.actuated_joint_dofs_offset,
                self._data.frame,
                self._data.q_j_ref,
                self._data.dq_j_ref,
                # Outputs:
                q_j_ref_out,
                dq_j_ref_out,
            ],
            device=self._device,
        )
