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
KAMINO: Linear Algebra: Core types and utilities for sparse multi-world linear systems

This module provides data structures and utilities for managing multiple
independent linear systems, including rectangular and square systems.
"""

from dataclasses import dataclass

import warp as wp
from warp.context import Devicelike

from ..core.types import FloatType, IntType, float32, int32

# from ..utils import logger as msg

###
# Module interface
###

__all__ = [
    "BlockSparseLinearOperator",
]


###
# Types
###


@dataclass
class BlockSparseLinearOperator:
    """
    A Block-Sparse Linear Operator container for representing
    and operating on multiple independent sparse linear systems.
    """

    ###
    # Host-side Metadata
    ###

    device: Devicelike | None = None
    """The device on which the data arrays are allocated."""

    dtype: FloatType = float32
    """The data type of the underlying matrix and vector data arrays."""

    itype: IntType = int32
    """The integer type used for indexing the underlying data arrays."""

    num_superblocks: int = 0
    """Host-side cache of the number of super-blocks represented in each sparse operator."""

    sum_of_nnzb: int = 0
    """Host-side cache of the sum of the number of non-zero sub-blocks over all super-blocks."""

    max_of_nnzb: int = 0
    """Host-side cache of the maximum number of non-zero sub-blocks over all super-blocks."""

    nzb_dimensions: tuple[int, int] | None = None
    """Host-side cache of the fixed non-zero sub-block dimensions contained in all sparse super-blocks."""

    ###
    # On-device Data
    ###

    maxdims: wp.array | None = None
    """
    The maximum dimensions of each sparse super-block.\n
    Shape of ``(num_superblocks,)`` and type :class:`vec2i`.
    """

    dims: wp.array | None = None
    """
    The active dimensions of each sparse super-block.\n
    Shape of ``(num_superblocks,)`` and type :class:`vec2i`.
    """

    maxnzb: wp.array | None = None
    """
    The maximum number of non-zero blocks per sparse super-block.\n
    Shape of ``(num_superblocks,)`` and type :class:`int`.
    """

    nnzb: wp.array | None = None
    """
    The active number of non-zero blocks per sparse super-block.\n
    Shape of ``(num_superblocks,)`` and type :class:`int`.
    """

    row_start: wp.array | None = None
    """
    The start index of each row vector block in flattened data arrays.\n
    Shape of ``(num_superblocks,)`` and type :class:`int`.
    """

    col_start: wp.array | None = None
    """
    The start index of each column vector block in flattened data arrays.\n
    Shape of ``(num_superblocks,)`` and type :class:`int`.
    """

    nzb_start: wp.array | None = None
    """
    The index of the first non-zero sub-block of each sparse super-block.\n
    Shape of ``(num_superblocks,)`` and type :class:`int`.
    """

    nzb_coords: wp.array | None = None
    """
    The row-column coordinates of each sparse sub-block within its corresponding super-block.\n
    Shape of ``(max_num_nonzero_subblocks,)`` and type :class:`vec2i`.
    """

    nzb_data: wp.array | None = None
    """
    The flattened array containing all sparse non-zero blocks over all super-blocks.\n
    Shape of ``(max_num_nonzero_subblocks,)`` and type :class:`float | vector | matrix`.
    """

    ###
    # Operations
    ###

    def clear(self):
        """Clears all variable sub-blocks."""
        self._assert_is_finalized()
        self.dims.zero_()
        self.nnzb.zero_()
        self.nzb_coords.zero_()

    def zero(self):
        """Sets all sub-block data to zero."""
        self._assert_is_finalized()
        self.nzb_data.zero_()

    ###
    # Internals
    ###

    def _assert_is_finalized(self):
        if self.nzb_data is None:
            raise RuntimeError("No data has been allocated. Call `finalize()` before use.")
