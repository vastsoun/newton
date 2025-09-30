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
KAMINO: Collision Detector Types
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.kamino.core.types import (
    float32,
    vec6i,
)

###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Data Constants
###

FLOAT_MIN = -1e30
FLOAT_MAX = 1e30
EPS_BEST_COUNT = 12
MULTI_CONTACT_COUNT = 4
MULTI_POLYGON_COUNT = 8
MULTI_TILT_ANGLE = 1.0
TRIS_DIM = 3 * EPS_BEST_COUNT


###
# Primitive data Types
###


class matc3(wp.types.matrix(shape=(EPS_BEST_COUNT, 3), dtype=float32)):
    pass


class vecc3(wp.types.vector(EPS_BEST_COUNT * 3, dtype=float32)):
    pass


# Matrix definition for the `tris` scratch space which is used to store the
# triangles of the polytope. Note that the first dimension is 2, as we need
# to store the previous and current polytope. But since Warp doesn't support
# 3D matrices yet, we use 2 * 3 * EPS_BEST_COUNT as the first dimension.


class mat2c3(wp.types.matrix(shape=(2 * TRIS_DIM, 3), dtype=float32)):
    pass


class mat3p(wp.types.matrix(shape=(MULTI_POLYGON_COUNT, 3), dtype=float32)):
    pass


class mat3c(wp.types.matrix(shape=(MULTI_CONTACT_COUNT, 3), dtype=float32)):
    pass


###
# Type Constants
###

VECI1 = vec6i(0, 0, 0, 1, 1, 2)
VECI2 = vec6i(1, 2, 3, 2, 3, 3)
