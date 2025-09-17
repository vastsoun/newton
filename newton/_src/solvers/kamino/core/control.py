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
KAMINO: Model Control Containers
"""

import warp as wp


class Control:
    """
    The compact time-varying control data for a :class:`Model`.
    The compact control data includes controllable generalized forces of the joints.
    The exact attributes depend on the contents of the model.
    Control objects should generally be created using the :func:`Model.control()` function.
    """

    def __init__(self):
        self.tau_j: wp.array | None = None
        """Array of joint control forces with shape ``(sum(nqd_w),)`` and type :class:`float`."""
