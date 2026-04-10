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

"""Defines data types used by the Forward Kinematics solver."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import warp as wp

###
# Module interface
###

__all__ = [
    "ForwardKinematicsPreconditionerType",
    "ForwardKinematicsStatus",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Types
###


class ForwardKinematicsPreconditionerType(IntEnum):
    """Conjugate gradient preconditioning options of the FK solver, if sparsity is enabled."""

    NONE = 0
    """No preconditioning"""

    JACOBI_DIAGONAL = 1
    """Diagonal Jacobi preconditioner"""

    JACOBI_BLOCK_DIAGONAL = 2
    """Blockwise-diagonal Jacobi preconditioner, alternating blocks of size 3 and 4 along the diagonal,
    corresponding to the position and orientation (quaternion) of individual rigid bodies."""

    @classmethod
    def from_string(cls, s: str) -> ForwardKinematicsPreconditionerType:
        """Converts a string to a ForwardKinematicsPreconditionerType enum value."""
        try:
            return cls[s.upper()]
        except KeyError as e:
            raise ValueError(
                f"Invalid ForwardKinematicsPreconditionerType: {s}. Valid options are: {[e.name for e in cls]}"
            ) from e


@dataclass
class ForwardKinematicsStatus:
    """
    Container holding detailed information on the success/failure status of a forward kinematics solve.
    """

    success: np.ndarray(dtype=np.int32)
    """
    Solver success flag per world, as an integer array (0 = failure, 1 = success).\n
    Shape `(num_worlds,)` and type :class:`np.int32`.

    Note that in some cases the solver may fail to converge within the maximum number
    of iterations, but still produce a solution with a reasonable constraint residual.
    In such cases, the success flag will be set to 0, but the `max_constraints` field
    can be inspected to check the actual constraint residuals and determine if the
    solution is acceptable for the intended application.
    """

    iterations: np.ndarray(dtype=np.int32)
    """
    Number of Gauss-Newton iterations executed per world.\n
    Shape `(num_worlds,)` and type :class:`np.int32`.
    """

    max_constraints: np.ndarray(dtype=np.float32)
    """
    Maximal absolute kinematic constraint residual at the final solution, per world.\n
    Shape `(num_worlds,)` and type :class:`np.float32`.
    """
