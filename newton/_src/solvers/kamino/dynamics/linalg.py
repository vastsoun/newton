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
TODO
"""

from __future__ import annotations

from dataclasses import dataclass

import warp as wp

from ....core.types import override

###
# Module interface
###

__all__ = [
    "BlockSparseLinearOperators",
    "DenseLinearOperators",
    "LinearOperators",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# TODO
###


@dataclass
class DenseMatrices:
    pass


@dataclass
class BlockSparseMatrices:
    pass


# TODO: Define in linalg.core
class LinearOperators:
    pass

    def reset(self):
        raise NotImplementedError("The `zero` method is not implemented for this linear operator type.")


# TODO: Define in linalg.core
class DenseLinearOperators(LinearOperators):
    pass

    @override
    def reset(self):
        pass


# TODO: Define in linalg.core
class BlockSparseLinearOperators(LinearOperators):
    pass

    @override
    def reset(self):
        pass


# TODO: MOVE TO linalg.core
LinearOperatorsType = DenseLinearOperators | BlockSparseLinearOperators
"""A type alias for defining the supported linear operator types."""
