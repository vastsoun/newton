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

from newton._src.solvers.kamino.linalg.sparse import BlockDType, BlockSparseMatrices
from newton._src.solvers.kamino.tests import setup_tests, test_context
from newton._src.solvers.kamino.utils import logger as msg
from newton._src.solvers.kamino.utils.sparse import sparseview

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
            msg.set_log_level(msg.LogLevel.INFO)
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


class TestBlockSparseMatrices(unittest.TestCase):
    def setUp(self):
        # Configs
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.seed = 42
        self.default_device = wp.get_device(test_context.device)
        self.verbose = True  # Set to True for verbose output
        self.plot = True  # Set to True for verbose output

        # Set debug-level logging to print verbose test output to console
        if self.verbose:
            print("\n")  # Add newline before test output for better readability
            msg.set_log_level(msg.LogLevel.INFO)
        else:
            msg.reset_log_level()

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()

    ###
    # Construction Tests
    ###

    def test_00_make_default(self):
        bsm = BlockSparseMatrices()
        self.assertIsInstance(bsm, BlockSparseMatrices)

        # Host-side meta-data should be default-initialized
        self.assertIsNone(bsm.device)
        self.assertEqual(bsm.num_matrices, 0)
        self.assertEqual(bsm.sum_of_num_nzb, 0)
        self.assertEqual(bsm.max_of_num_nzb, 0)
        self.assertIsNone(bsm.nzb_dtype)
        self.assertIs(bsm.index_dtype, wp.int32)

        # On-device data should be None
        self.assertIsNone(bsm.max_dims)
        self.assertIsNone(bsm.dims)
        self.assertIsNone(bsm.max_nzb)
        self.assertIsNone(bsm.num_nzb)
        self.assertIsNone(bsm.nzb_start)
        self.assertIsNone(bsm.nzb_coords)
        self.assertIsNone(bsm.nzb_values)

        # Finalization should fail since the block size `nzb_size` is not set
        self.assertRaises(RuntimeError, bsm.finalize, capacities=[0])

    def test_01_make_single_scalar_block_sparse_matrix(self):
        bsm = BlockSparseMatrices(num_matrices=1, nzb_dtype=BlockDType(dtype=wp.float32), device=self.default_device)
        bsm.finalize(capacities=[1])

        # Check meta-data
        self.assertEqual(bsm.num_matrices, 1)
        self.assertEqual(bsm.sum_of_num_nzb, 1)
        self.assertEqual(bsm.max_of_num_nzb, 1)
        self.assertEqual(bsm.nzb_dtype.dtype, wp.float32)
        self.assertEqual(bsm.nzb_dtype.shape, ())
        self.assertIs(bsm.index_dtype, wp.int32)
        self.assertEqual(bsm.device, self.default_device)

        # Check on-device data shapes
        self.assertEqual(bsm.max_dims.shape, (1, 2))
        self.assertEqual(bsm.dims.shape, (1, 2))
        self.assertEqual(bsm.max_nzb.shape, (1,))
        self.assertEqual(bsm.num_nzb.shape, (1,))
        self.assertEqual(bsm.nzb_start.shape, (1,))
        self.assertEqual(bsm.nzb_coords.shape, (1, 2))
        self.assertEqual(bsm.nzb_values.shape, (1,))
        self.assertEqual(bsm.nzb_values.size, 1)
        self.assertEqual(bsm.nzb_values.view(dtype=wp.float32).size, 1)

    def test_02_make_single_vector_block_sparse_matrix(self):
        bsm = BlockSparseMatrices(num_matrices=1, nzb_dtype=BlockDType(shape=(6,), dtype=wp.float32))
        bsm.finalize(capacities=[1], device=self.default_device)

        # Check meta-data
        self.assertEqual(bsm.num_matrices, 1)
        self.assertEqual(bsm.sum_of_num_nzb, 1)
        self.assertEqual(bsm.max_of_num_nzb, 1)
        self.assertEqual(bsm.nzb_dtype.dtype, wp.float32)
        self.assertEqual(bsm.nzb_dtype.shape, (6,))
        self.assertIs(bsm.index_dtype, wp.int32)
        self.assertEqual(bsm.device, self.default_device)

        # Check on-device data shapes
        self.assertEqual(bsm.max_dims.shape, (1, 2))
        self.assertEqual(bsm.dims.shape, (1, 2))
        self.assertEqual(bsm.max_nzb.shape, (1,))
        self.assertEqual(bsm.num_nzb.shape, (1,))
        self.assertEqual(bsm.nzb_start.shape, (1,))
        self.assertEqual(bsm.nzb_coords.shape, (1, 2))
        self.assertEqual(bsm.nzb_values.shape, (1,))
        self.assertEqual(bsm.nzb_values.size, 1)
        self.assertEqual(bsm.nzb_values.view(dtype=wp.float32).size, 6)

    def test_03_make_single_matrix_block_sparse_matrix(self):
        bsm = BlockSparseMatrices(num_matrices=1, nzb_dtype=BlockDType(shape=(6, 5), dtype=wp.float32))
        bsm.finalize(capacities=[1], device=self.default_device)

        # Check meta-data
        self.assertEqual(bsm.num_matrices, 1)
        self.assertEqual(bsm.sum_of_num_nzb, 1)
        self.assertEqual(bsm.max_of_num_nzb, 1)
        self.assertEqual(bsm.nzb_dtype.dtype, wp.float32)
        self.assertEqual(bsm.nzb_dtype.shape, (6, 5))
        self.assertIs(bsm.index_dtype, wp.int32)
        self.assertEqual(bsm.device, self.default_device)

        # Check on-device data shapes
        self.assertEqual(bsm.max_dims.shape, (1, 2))
        self.assertEqual(bsm.dims.shape, (1, 2))
        self.assertEqual(bsm.max_nzb.shape, (1,))
        self.assertEqual(bsm.num_nzb.shape, (1,))
        self.assertEqual(bsm.nzb_start.shape, (1,))
        self.assertEqual(bsm.nzb_coords.shape, (1, 2))
        self.assertEqual(bsm.nzb_values.shape, (1,))
        self.assertEqual(bsm.nzb_values.size, 1)
        self.assertEqual(bsm.nzb_values.view(dtype=wp.float32).size, 30)

    def test_04_build_multiple_vector_block_matrices(self):
        bsm = BlockSparseMatrices(num_matrices=1, nzb_dtype=BlockDType(shape=(6,), dtype=wp.float32))
        bsm.finalize(capacities=[3, 4, 5], device=self.default_device)

        # Check meta-data
        self.assertEqual(bsm.num_matrices, 3)
        self.assertEqual(bsm.sum_of_num_nzb, 12)
        self.assertEqual(bsm.max_of_num_nzb, 5)
        self.assertEqual(bsm.nzb_dtype.dtype, wp.float32)
        self.assertEqual(bsm.nzb_dtype.shape, (6,))
        self.assertIs(bsm.index_dtype, wp.int32)
        self.assertEqual(bsm.device, self.default_device)

        # Check on-device data shapes
        self.assertEqual(bsm.max_dims.shape, (3, 2))
        self.assertEqual(bsm.dims.shape, (3, 2))
        self.assertEqual(bsm.max_nzb.shape, (3,))
        self.assertEqual(bsm.num_nzb.shape, (3,))
        self.assertEqual(bsm.nzb_start.shape, (3,))
        self.assertEqual(bsm.nzb_coords.shape, (12, 2))
        self.assertEqual(bsm.nzb_values.shape, (12,))
        self.assertEqual(bsm.nzb_values.size, 12)
        self.assertEqual(bsm.nzb_values.view(dtype=wp.float32).size, 72)

    ###
    # Building Tests
    ###

    def test_10_build_multiple_vector_block_sparse_matrices_full(self):
        """
        Tests building two fully-filled block-sparse matrices with vector-shaped blocks and same overall shape.
        """
        bsm = BlockSparseMatrices(num_matrices=2, nzb_dtype=BlockDType(shape=(6,), dtype=wp.float32))
        bsm.finalize(capacities=[2, 3], device=self.default_device)

        # Check meta-data
        self.assertEqual(bsm.num_matrices, 2)
        self.assertEqual(bsm.sum_of_num_nzb, 5)
        self.assertEqual(bsm.max_of_num_nzb, 3)
        self.assertEqual(bsm.nzb_dtype.dtype, wp.float32)
        self.assertEqual(bsm.nzb_dtype.shape, (6,))
        self.assertIs(bsm.index_dtype, wp.int32)
        self.assertEqual(bsm.device, self.default_device)

        # Check on-device data shapes
        self.assertEqual(bsm.max_dims.shape, (bsm.num_matrices, 2))
        self.assertEqual(bsm.dims.shape, (bsm.num_matrices, 2))
        self.assertEqual(bsm.max_nzb.shape, (bsm.num_matrices,))
        self.assertEqual(bsm.num_nzb.shape, (bsm.num_matrices,))
        self.assertEqual(bsm.nzb_start.shape, (bsm.num_matrices,))
        self.assertEqual(bsm.nzb_coords.shape, (bsm.sum_of_num_nzb, 2))
        self.assertEqual(bsm.nzb_values.shape, (bsm.sum_of_num_nzb,))
        self.assertEqual(bsm.nzb_values.size, bsm.sum_of_num_nzb)
        self.assertEqual(bsm.nzb_values.view(dtype=wp.float32).size, bsm.sum_of_num_nzb * bsm.nzb_dtype.size)

        # Build each matrix as follows:
        # Matrix 0: 2x12 block0diagonal with 2 non-zero blocks at on diagonals (0,0) and (1,6)
        # Matrix 1: 2x12 upper-block-triangular with 3 non-zero blocks at on at (0,0), (0,6), and (1,6)
        nzb_dims_np = np.array([[2, 12], [2, 12]], dtype=np.int32)
        num_nzb_np = np.array([[2], [3]], dtype=np.int32)
        nzb_start_np = np.array([[0], [2]], dtype=np.int32)
        nzb_coords_np = np.array([[0, 0], [1, 6], [0, 0], [0, 6], [1, 6]], dtype=np.int32)
        nzb_values_np = np.array(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
                [3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
                [9.0, 9.0, 9.0, 9.0, 9.0, 9.0],
            ],
            dtype=np.float32,
        )
        bsm.max_dims.assign(nzb_dims_np)
        bsm.dims.assign(nzb_dims_np)
        bsm.max_nzb.assign(num_nzb_np)
        bsm.num_nzb.assign(num_nzb_np)
        bsm.nzb_start.assign(nzb_start_np)
        bsm.nzb_coords.assign(nzb_coords_np)
        bsm.nzb_values.view(dtype=wp.float32).assign(nzb_values_np)
        msg.info("bsm.max_dims:\n%s", bsm.max_dims)
        msg.info("bsm.dims:\n%s", bsm.dims)
        msg.info("bsm.max_nzb:\n%s", bsm.max_nzb)
        msg.info("bsm.num_nzb:\n%s", bsm.num_nzb)
        msg.info("bsm.nzb_start:\n%s", bsm.nzb_start)
        msg.info("bsm.nzb_coords:\n%s", bsm.nzb_coords)
        msg.info("bsm.nzb_values:\n%s", bsm.nzb_values)

        # Check on-device data shapes again to ensure nothing changed during building
        self.assertEqual(bsm.max_dims.shape, (bsm.num_matrices, 2))
        self.assertEqual(bsm.dims.shape, (bsm.num_matrices, 2))
        self.assertEqual(bsm.max_nzb.shape, (bsm.num_matrices,))
        self.assertEqual(bsm.num_nzb.shape, (bsm.num_matrices,))
        self.assertEqual(bsm.nzb_start.shape, (bsm.num_matrices,))
        self.assertEqual(bsm.nzb_coords.shape, (bsm.sum_of_num_nzb, 2))
        self.assertEqual(bsm.nzb_values.shape, (bsm.sum_of_num_nzb,))
        self.assertEqual(bsm.nzb_values.size, bsm.sum_of_num_nzb)
        self.assertEqual(bsm.nzb_values.view(dtype=wp.float32).size, bsm.sum_of_num_nzb * bsm.nzb_dtype.size)

        # Convert to list of numpy arrays for easier verification
        bsm_np = bsm.numpy()
        for i in range(bsm.num_matrices):
            msg.info("bsm_np[%d]:\n%s", i, bsm_np[i])
            if self.plot:
                sparseview(bsm_np[i], title=f"bsm_np[{i}]")

        # Assign new values to the dense numpy arrays and set them back to the block-sparse matrices
        for i in range(bsm.num_matrices):
            bsm_np[i] += 1.0 * (i + 1)
        bsm.assign(bsm_np)

        # Convert again to list of numpy arrays for easier verification
        bsm_np = bsm.numpy()
        for i in range(bsm.num_matrices):
            msg.info("bsm_np[%d]:\n%s", i, bsm_np[i])
            if self.plot:
                sparseview(bsm_np[i], title=f"bsm_np[{i}]")


###
# Test execution
###

if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
