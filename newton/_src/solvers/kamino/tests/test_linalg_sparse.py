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

"""Unit tests for the base classes in linalg/sparse.py"""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino.core.types import float32, int32
from newton._src.solvers.kamino.linalg.sparse import BlockDType, BlockSparseLinearOperators, BlockSparseMatrices
from newton._src.solvers.kamino.tests import setup_tests, test_context
from newton._src.solvers.kamino.utils import logger as msg
from newton._src.solvers.kamino.utils.sparse import sparseview

###
# Functions
###

###
# Kernels
###

###
# Launchers
###

###
# Constructors
###


# def _build_block_matrix(
#     matrix_shape: tuple[int, int],
#     block_coords: dict,
#     block_shapes: dict,
#     dtype=np.int64,
# ) -> np.ndarray:
#     M = np.zeros(matrix_shape, dtype=dtype)
#     for block_id, coords in block_coords.items():
#         shapes = block_shapes[block_id]
#         for (br, bc), (bm, bn) in zip(coords, shapes, strict=False):
#             row_start = br * bm
#             row_end = row_start + bm
#             col_start = bc * bn
#             col_end = col_start + bn
#             M[row_start:row_end, col_start:col_end] = block_id
#     return M


# def _build_block_sparse_matrix(
#     matrix_shape: tuple[int, int],
#     block_coords: dict,
#     block_shapes: dict,
#     dtype=wp.float32,
#     itype=wp.int32,
#     device: wp.DeviceLike = None,
# ) -> BlockSparseLinearOperator:
#     spm = BlockSparseLinearOperator(dtype=dtype, itype=itype, device=device)
#     spm.dimensions = [matrix_shape]
#     spm.num_superblocks = 1
#     spm.max_num_nonzero_subblocks = sum(len(coords) for coords in block_coords.values())

#     sub_nrows = np.zeros((spm.max_num_nonzero_subblocks,), dtype=itype)
#     sub_ncols = np.zeros((spm.max_num_nonzero_subblocks,), dtype=itype)
#     sub_roffs = np.zeros((spm.max_num_nonzero_subblocks,), dtype=itype)
#     sub_coffs = np.zeros((spm.max_num_nonzero_subblocks,), dtype=itype)
#     sub_doffs = np.zeros((spm.max_num_nonzero_subblocks,), dtype=itype)

#     total_nnzb = 0
#     total_nnz = 0
#     for block_id, coords in block_coords.items():
#         shapes = block_shapes[block_id]
#         for (br, bc), (bm, bn) in zip(coords, shapes, strict=False):
#             row_start = br * bm
#             col_start = bc * bn
#             sub_nrows[total_nnzb] = bm
#             sub_ncols[total_nnzb] = bn
#             sub_roffs[total_nnzb] = row_start
#             sub_coffs[total_nnzb] = col_start
#             sub_doffs[total_nnzb] = total_nnz
#             total_nnzb += 1
#             total_nnz += bm * bn

#     with wp.ScopedDevice(device):
#         spm.superblock_nrows = wp.array([matrix_shape[0]], dtype=itype, device=device)
#         spm.superblock_ncols = wp.array([matrix_shape[1]], dtype=itype, device=device)
#         spm.superblock_nnzb = wp.array([spm.max_num_nonzero_subblocks], dtype=itype, device=device)
#         spm.superblock_nnz = wp.array([total_nnz], dtype=itype, device=device)
#         spm.subblock_nrows = wp.array(sub_nrows, dtype=itype, device=device)
#         spm.subblock_ncols = wp.array(sub_ncols, dtype=itype, device=device)
#         spm.subblock_roffs = wp.array(sub_roffs, dtype=itype, device=device)
#         spm.subblock_coffs = wp.array(sub_coffs, dtype=itype, device=device)

#         spm.subblock_data = wp.array()

#     return spm


# def make_block_sparse_matrix(dtype=np.float32) -> tuple[np.ndarray, BlockSparseLinearOperator]:
#     grid_shape = (9, 8)
#     block_dims = (5, 6)
#     matrix_shape = (block_dims[0] * grid_shape[0], block_dims[1] * grid_shape[1])

#     blue_coords = [(3, 0), (3, 3), (3, 5), (5, 0), (5, 3), (5, 5)]
#     blue_shapes = [block_dims] * len(blue_coords)
#     green_coords = [(1, 4), (1, 7), (4, 4), (4, 7), (6, 4), (6, 7), (7, 4), (7, 7)]
#     green_shapes = [block_dims] * len(green_coords)
#     orange_coords = [(0, 2), (0, 6), (2, 2), (2, 6), (8, 2), (8, 6)]
#     orange_shapes = [block_dims] * len(orange_coords)
#     block_coords = {1: blue_coords, 2: green_coords, 3: orange_coords}
#     block_shapes = {1: blue_shapes, 2: green_shapes, 3: orange_shapes}

#     A = _build_block_matrix(matrix_shape, block_coords, block_shapes, dtype)
#     A_bsm = _build_block_sparse_matrix(matrix_shape, block_coords, block_shapes, dtype)
#     return A, A_bsm


###
# Tests
###


class TestBlockDType(unittest.TestCase):
    def setUp(self):
        # Configs
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.seed = 42
        self.default_device = wp.get_device(test_context.device)
        self.verbose = True  # Set to True for verbose output

        # Set debug-level logging to print verbose test output to console
        if self.verbose:
            print("\n")  # Add newline before test output for better readability
            msg.set_log_level(msg.LogLevel.DEBUG)
        else:
            msg.reset_log_level()

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()

    ###
    # Construction Tests
    ###

    def test_00_make_block_dtype(self):
        # Default construction should fail
        self.assertRaises(TypeError, BlockDType)

        # Scalar block type, shape should be `()` to match numpy scalar behavior
        scalar_block_type_0 = BlockDType(dtype=wp.float32)
        self.assertEqual(scalar_block_type_0.dtype, wp.float32)
        self.assertEqual(scalar_block_type_0.shape, ())

        scalar_block_type_1 = BlockDType(shape=1, dtype=wp.float32)
        self.assertEqual(scalar_block_type_1.dtype, wp.float32)
        self.assertEqual(scalar_block_type_1.shape, ())

        # Vector block types
        vector_block_type_0 = BlockDType(shape=2, dtype=wp.float32)
        self.assertEqual(vector_block_type_0.dtype, wp.float32)
        self.assertEqual(vector_block_type_0.shape, (2,))

        vector_block_type_1 = BlockDType(shape=(3,), dtype=wp.float32)
        self.assertEqual(vector_block_type_1.dtype, wp.float32)
        self.assertEqual(vector_block_type_1.shape, (3,))

        # Matrix block types
        matrix_block_type_0 = BlockDType(shape=(2, 4), dtype=wp.float32)
        self.assertEqual(matrix_block_type_0.dtype, wp.float32)
        self.assertEqual(matrix_block_type_0.shape, (2, 4))

        # Invalid shape specifications should fail
        self.assertRaises(ValueError, BlockDType, shape=0, dtype=wp.float32)
        self.assertRaises(ValueError, BlockDType, shape=(-2,), dtype=wp.float32)
        self.assertRaises(ValueError, BlockDType, shape=(3, -4), dtype=wp.float32)
        self.assertRaises(ValueError, BlockDType, shape=(1, 2, 3), dtype=wp.float32)

        # Invalid dtype specifications should fail
        self.assertRaises(TypeError, BlockDType, shape=2, dtype=None)
        self.assertRaises(TypeError, BlockDType, shape=(2, 2), dtype=str)

    def test_01_block_dtype_size(self):
        # Scalar block type
        scalar_block_type = BlockDType(dtype=wp.float32)
        self.assertEqual(scalar_block_type.size, 1)

        # Vector block type
        vector_block_type = BlockDType(shape=4, dtype=wp.float32)
        self.assertEqual(vector_block_type.size, 4)

        # Matrix block type
        matrix_block_type = BlockDType(shape=(3, 5), dtype=wp.float32)
        self.assertEqual(matrix_block_type.size, 15)

    def test_02_block_dtype_warp_type(self):
        # Scalar block type
        scalar_block_type = BlockDType(dtype=wp.float32)
        warp_scalar_type = scalar_block_type.warp_type
        self.assertEqual(warp_scalar_type, wp.float32)

        # Vector block type
        vector_block_type = BlockDType(shape=4, dtype=wp.float32)
        warp_vector_type = vector_block_type.warp_type
        self.assertEqual(warp_vector_type._length_, 4)
        self.assertEqual(warp_vector_type._wp_scalar_type_, wp.float32)

        # Matrix block type
        matrix_block_type = BlockDType(shape=(3, 5), dtype=wp.float32)
        warp_matrix_type = matrix_block_type.warp_type
        self.assertEqual(warp_matrix_type._shape_, (3, 5))
        self.assertEqual(warp_matrix_type._length_, 15)
        self.assertEqual(warp_matrix_type._wp_scalar_type_, wp.float32)


# class TestBlockSparseMatrices(unittest.TestCase):
#     def setUp(self):
#         # Configs
#         if not test_context.setup_done:
#             setup_tests(clear_cache=False)
#         self.seed = 42
#         self.default_device = wp.get_device(test_context.device)
#         self.verbose = True  # Set to True for verbose output

#         # Set debug-level logging to print verbose test output to console
#         if self.verbose:
#             print("\n")  # Add newline before test output for better readability
#             msg.set_log_level(msg.LogLevel.DEBUG)
#         else:
#             msg.reset_log_level()

#     def tearDown(self):
#         self.default_device = None
#         if self.verbose:
#             msg.reset_log_level()

#     ###
#     # Construction Tests
#     ###

#     def test_00_make_single_matrix_default(self):
#         # Generate the block-sparse matrix (each colored square becomes a brxbc dense block)
#         A, A_bsm = make_block_sparse_matrix(dtype=np.float32)

#         # Create images of the sparsity patterns
#         sparseview(A, title="A (original)")


###
# Test execution
###

if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
