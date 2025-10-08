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
KAMINO: Model State Containers
"""

import warp as wp


class State:
    """
    The compact time-varying state data for a :class:`Model`.
    The compact state data includes rigid body poses and twists, as well as the vector of joint constraint forces.
    The exact attributes depend on the contents of the model.
    State objects should generally be created using the :func:`Model.state()` function.
    """

    def __init__(self):
        self.q_i: wp.array | None = None
        """Array of body coordinates (7-dof transforms) in maximal coordinates with shape ``(nb,)`` and type :class:`transformf`."""
        self.u_i: wp.array | None = None
        """Array of body velocities (6-dof twists) in maximal coordinates with shape ``(nb,)`` and type :class:`vec6f`."""
        self.lambda_j: wp.array | None = None
        """Array of joint constraint forces with shape ``(njd=sum(m_j),)`` and type :class:`float32`."""
