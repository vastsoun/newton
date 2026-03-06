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

import dataclasses

Color3 = tuple[float, float, float]


@dataclasses.dataclass
class RenderConfig:
    """Viewer appearance settings for :class:`RigidBodySim`.

    All ``None`` defaults leave the standard Newton viewer appearance unchanged.
    """

    robot_color: Color3 | None = None
    """Override color for all robot shapes.  ``None`` keeps USD material colors."""

    shadow_radius: float | None = None
    """PCF shadow softness radius.  Larger = softer edges.  ``None`` keeps default (3.0)."""

    light_intensity: float | None = None
    """Diffuse light intensity multiplier.  ``None`` keeps default (1.5)."""

    spotlight_enabled: bool | None = None
    """Use cone spotlight (True) or uniform directional light (False).  ``None`` keeps default (False)."""
