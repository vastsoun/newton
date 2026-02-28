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

"""Constrained Rigid Multi-Body Model & Data Containers"""

from dataclasses import dataclass

import warp as wp

from .bodies import RigidBodiesData
from .geometry import GeometriesData
from .joints import JointsData
from .time import TimeData

###
# Module interface
###

__all__ = [
    "ModelData",
    "ModelDataInfo",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Types
###


@dataclass
class ModelDataInfo:
    """
    A container to hold the time-varying information about the set of active constraints.
    """

    ###
    # Total Constraints
    ###

    num_total_cts: wp.array | None = None
    """
    The total number of active constraints.\n
    Shape of ``(num_worlds,)`` and type :class:`int32`.
    """

    ###
    # Limits
    ###

    num_limits: wp.array | None = None
    """
    The number of active limits in each world.\n
    Shape of ``(num_worlds,)`` and type :class:`int32`.
    """

    num_limit_cts: wp.array | None = None
    """
    The number of active limit constraints.\n
    Shape of ``(num_worlds,)`` and type :class:`int32`.
    """

    limit_cts_group_offset: wp.array | None = None
    """
    The index offset of the limit constraints group within the constraints block of each world.\n
    Shape of ``(num_worlds,)`` and type :class:`int32`.
    """

    ###
    # Contacts
    ###

    num_contacts: wp.array | None = None
    """
    The number of active contacts in each world.\n
    Shape of ``(num_worlds,)`` and type :class:`int32`.
    """

    num_contact_cts: wp.array | None = None
    """
    The number of active contact constraints.\n
    Shape of ``(num_worlds,)`` and type :class:`int32`.
    """

    contact_cts_group_offset: wp.array | None = None
    """
    The index offset of the contact constraints group within the constraints block of each world.\n
    Shape of ``(num_worlds,)`` and type :class:`int32`.
    """


@dataclass
class ModelData:
    """
    A container to hold the time-varying data of the model entities.

    It includes all model-specific intermediate quantities used throughout the simulation, as needed
    to update the state of rigid bodies, joints, geometries, active constraints and time-keeping.
    """

    info: ModelDataInfo | None = None
    """The info container holding information about the set of active constraints."""

    time: TimeData | None = None
    """Time state of the model, including the current simulation step and time."""

    bodies: RigidBodiesData | None = None
    """
    Time-varying data of all rigid bodies in the model: poses, twists, wrenches,
    and moments of inertia computed in world coordinates.
    """

    joints: JointsData | None = None
    """
    Time-varying data of joints in the model: joint frames computed in world coordinates,
    constraint residuals and reactions, and generalized (DoF) quantities.
    """

    geoms: GeometriesData | None = None
    """Time-varying data of geometries in the model: poses computed in world coordinates."""
