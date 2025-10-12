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

"""Unit tests for the AnimationJointReference class."""

import os
import unittest

import numpy as np
import warp as wp

import newton._src.solvers.kamino.utils.logger as msg
from newton._src.solvers.kamino.control.animation import AnimationJointReference
from newton._src.solvers.kamino.core.builder import ModelBuilder
from newton._src.solvers.kamino.core.types import float32
from newton._src.solvers.kamino.models import get_examples_usd_assets_path
from newton._src.solvers.kamino.utils.io.usd import USDImporter

###
# Tests
###


class TestAnimationJointReference(unittest.TestCase):
    def setUp(self):
        # Configs
        self.seed = 42
        self.default_device = wp.get_device()
        self.verbose = True  # Set to True for verbose output

        # Set debug-level logging to print verbose test output to console
        if self.verbose:
            print("\n")  # Add newline before test output for better readability
            msg.set_log_level(msg.LogLevel.DEBUG)
        else:
            msg.set_log_level(msg.LogLevel.WARNING)

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()

    def test_00_make_default(self):
        animation = AnimationJointReference()
        self.assertIsNotNone(animation)
        self.assertEqual(animation.device, None)
        self.assertEqual(animation._data, None)

    def test_01_make_with_walker_numpy_data(self):
        # Set paths to Walker model and animation data
        USD_MODEL_PATH = os.path.join(get_examples_usd_assets_path(), "walker/walker_floating_with_boxes.usda")
        NUMPY_ANIMATION_PATH = os.path.join(get_examples_usd_assets_path(), "walker/walker_animation_100fps.npy")

        # Import USD model of Walker
        importer = USDImporter()
        builder: ModelBuilder = importer.import_from(source=USD_MODEL_PATH)
        model = builder.finalize(device=self.default_device)

        # Retrieve the number of actuated coordinates and DoFs
        njaq = model.size.sum_of_num_actuated_joint_coords
        njad = model.size.sum_of_num_actuated_joint_dofs
        msg.info(f"number of actuated joint coordinates: {njaq}")
        msg.info(f"number of actuated joint DoFs: {njad}")
        self.assertEqual(njaq, njad)  # Ensure only 1-DoF joints

        # Load numpy animation data
        animation_np = np.load(NUMPY_ANIMATION_PATH, allow_pickle=True)
        msg.info(f"animation_np (shape={animation_np.shape}):\n{animation_np}\n")
        self.assertEqual(animation_np.shape[1], njaq)  # Ensure data matches number of joints

        # Set rate and loop parameters
        rate: int = 15  # Animation rate in Hz (frames per second)
        loop: bool = True  # Whether the animation should loop

        # Create a joint-space animation reference generator
        animation = AnimationJointReference(
            model=model, input=animation_np, rate=rate, loop=loop, use_fd=True, device=self.default_device
        )
        self.assertIsNotNone(animation)
        self.assertIsNotNone(animation.data)
        self.assertEqual(animation.device, self.default_device)
        self.assertEqual(animation.sequence_length, animation_np.shape[0])
        self.assertEqual(animation.data.q_j_ref.shape, animation_np.shape)
        self.assertEqual(animation.data.dq_j_ref.shape, animation_np.shape)
        self.assertEqual(animation.data.loop.shape, (1,))
        self.assertEqual(animation.data.loop.numpy()[0], 1 if loop else 0)
        self.assertEqual(animation.data.rate.shape, (1,))
        self.assertEqual(animation.data.rate.numpy()[0], rate)
        self.assertEqual(animation.data.step.shape, (1,))
        self.assertEqual(animation.data.step.numpy()[0], 0)

        # Check that the internal numpy arrays match the input data
        np.testing.assert_array_almost_equal(animation.data.q_j_ref.numpy(), animation_np, decimal=6)
        np.testing.assert_array_almost_equal(animation.data.dq_j_ref.numpy(), np.zeros_like(animation_np), decimal=6)

        # Allocate output arrays for joint references
        q_j_ref_out = wp.zeros(njad, dtype=float32, device=self.default_device)
        dq_j_ref_out = wp.zeros(njad, dtype=float32, device=self.default_device)

        # Retrieve the reference at the initial step (0)
        animation.reset(q_j_ref_out=q_j_ref_out, dq_j_ref_out=dq_j_ref_out)
        np.testing.assert_array_equal(animation.data.step.numpy(), np.array([0], dtype=np.int32))
        np.testing.assert_array_almost_equal(q_j_ref_out.numpy(), animation_np[0, :], decimal=6)
        np.testing.assert_array_almost_equal(dq_j_ref_out.numpy(), np.zeros(njad, dtype=np.float32), decimal=6)

        # Step through the animation and verify outputs
        num_steps = 10
        for step in range(1, num_steps + 1):
            animation.step(q_j_ref_out=q_j_ref_out, dq_j_ref_out=dq_j_ref_out)
            expected_step = (rate * step) % animation.sequence_length  # Loop around if exceeding number of frames
            np.testing.assert_array_equal(animation.data.step.numpy(), np.array([expected_step], dtype=np.int32))
            np.testing.assert_array_almost_equal(q_j_ref_out.numpy(), animation_np[expected_step, :], decimal=6)
            np.testing.assert_array_almost_equal(dq_j_ref_out.numpy(), np.zeros(njad, dtype=np.float32), decimal=6)

        # Reset the reference at the initial step (0)
        animation.reset(q_j_ref_out=q_j_ref_out, dq_j_ref_out=dq_j_ref_out)
        np.testing.assert_array_equal(animation.data.step.numpy(), np.array([0], dtype=np.int32))
        np.testing.assert_array_almost_equal(q_j_ref_out.numpy(), animation_np[0, :], decimal=6)
        np.testing.assert_array_almost_equal(dq_j_ref_out.numpy(), np.zeros(njad, dtype=np.float32), decimal=6)

        # Step through again but exceeding the number of frames to test looping
        num_steps = animation.sequence_length + 5
        for step in range(1, num_steps + 1):
            animation.step(q_j_ref_out=q_j_ref_out, dq_j_ref_out=dq_j_ref_out)
            expected_step = (rate * step) % animation.sequence_length  # Loop around if exceeding number of frames
            np.testing.assert_array_equal(animation.data.step.numpy(), np.array([expected_step], dtype=np.int32))
            np.testing.assert_array_almost_equal(q_j_ref_out.numpy(), animation_np[expected_step, :], decimal=6)
            np.testing.assert_array_almost_equal(dq_j_ref_out.numpy(), np.zeros(njad, dtype=np.float32), decimal=6)


###
# Test execution
###

if __name__ == "__main__":
    # Global numpy configurations
    np.set_printoptions(linewidth=500, precision=10, suppress=True)  # Suppress scientific notation

    # Global warp configurations
    wp.config.verbose = True
    wp.clear_kernel_cache()
    wp.clear_lto_cache()

    # Run all tests
    unittest.main(verbosity=2)
