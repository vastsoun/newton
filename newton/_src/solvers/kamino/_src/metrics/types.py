# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
TODO
"""

from dataclasses import dataclass

import warp as wp

###
# Module interface
###

__all__ = ["JointWrenchesData"]


###
# Types
###


@dataclass
class JointWrenchesData:
    """
    An SoA-based container to hold time-varying data of a joint system.
    """

    device: wp.DeviceLike | None = None
    """Device on which the data is allocated."""

    num_joints: int = 0
    """Total number of joints in the model (host-side)."""

    ###
    # Per-Joint Wrenches
    ###

    j_w_j_dof_act: wp.array[wp.spatial_vectorf] | None = None
    """
    Actuation wrench applied by each joint onto the corresponding follower
    body, expressed in and about the corresponding joint frame.\n
    Shape of ``(num_joints,)`` and type :class:`wp.spatial_vectorf`.
    """

    j_w_j_cts_kin: wp.array[wp.spatial_vectorf] | None = None
    """
    Kinematic constraint wrench applied by each joint onto the corresponding
    follower body, expressed in and about the corresponding joint frame.\n
    Shape of ``(num_joints,)`` and type :class:`wp.spatial_vectorf`.
    """

    j_w_j_cts_lim: wp.array[wp.spatial_vectorf] | None = None
    """
    Limit constraint wrench applied by each joint onto the corresponding
    follower body, expressed in and about the corresponding joint frame.\n
    Shape of ``(num_joints,)`` and type :class:`wp.spatial_vectorf`.
    """

    j_w_j_cts_fri: wp.array[wp.spatial_vectorf] | None = None
    """
    Friction constraint wrench applied by each joint onto the corresponding
    follower body, expressed in and about the corresponding joint frame.\n
    Shape of ``(num_joints,)`` and type :class:`wp.spatial_vectorf`.
    """

    ###
    # Operations
    ###

    def finalize(self, num_joints: int, device: wp.DeviceLike | None = None):
        """
        Finalizes the JointWrenchesData container by allocating the required data arrays on the specified device.
        """
        # First check if the container is already finalized
        if self.num_joints > 0 or self.device is not None:
            raise RuntimeError("JointWrenchesData container already finalized.")

        # Store the number of joints and device
        self.num_joints = int(num_joints)
        self.device = device

        # Proceed with allocating memory on the target device
        with wp.ScopedDevice(self.device):
            if self.j_w_j_dof_act is None:
                self.j_w_j_dof_act = wp.zeros((self.num_joints,), dtype=wp.spatial_vectorf)
            if self.j_w_j_cts_kin is None:
                self.j_w_j_cts_kin = wp.zeros((self.num_joints,), dtype=wp.spatial_vectorf)
            if self.j_w_j_cts_lim is None:
                self.j_w_j_cts_lim = wp.zeros((self.num_joints,), dtype=wp.spatial_vectorf)
            if self.j_w_j_cts_fri is None:
                self.j_w_j_cts_fri = wp.zeros((self.num_joints,), dtype=wp.spatial_vectorf)

    def clear(self):
        """
        Clears all the arrays in the JointWrenchesData container by setting them to zero.
        """
        self._assert_finalized()
        self.j_w_j_dof_act.zero_()
        self.j_w_j_cts_kin.zero_()
        self.j_w_j_cts_lim.zero_()
        self.j_w_j_cts_fri.zero_()

    ###
    # Internals
    ###

    def __post_init__(self):
        """
        Finalizes the JointWrenchesData container if it has been initialized with a number of joints.
        """
        if self.num_joints > 0:
            self.finalize(self.num_joints, self.device)

    def _assert_finalized(self):
        """
        Asserts that the JointWrenchesData container has been finalized.
        """
        if (
            self.num_joints == 0
            or self.j_w_j_dof_act is None
            or self.j_w_j_cts_kin is None
            or self.j_w_j_cts_lim is None
            or self.j_w_j_cts_fri is None
        ):
            raise RuntimeError("JointWrenchesData container not finalized. Call finalize() first.")
