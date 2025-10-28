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

"""Unit tests for the high-level Simulator class of Kamino"""

import unittest

import numpy as np
import warp as wp

import newton._src.solvers.kamino.utils.logger as msg
from newton._src.solvers.kamino.core.builder import ModelBuilder
from newton._src.solvers.kamino.core.types import float32
from newton._src.solvers.kamino.models.builders import build_cartpole
from newton._src.solvers.kamino.simulation.simulator import Simulator

###
# Kernels
###


@wp.kernel
def _test_control_callback(
    model_dt: wp.array(dtype=float32),
    state_t: wp.array(dtype=float32),
    control_tau_j: wp.array(dtype=float32),
):
    """
    An example control callback kernel.
    """
    # Retrieve the world index from the thread ID
    wid = wp.tid()

    # Get the fixed time-step and current time
    dt = model_dt[wid]
    t = state_t[wid]

    # Define the time window for the active external force profile
    t_start = float32(0.0)
    t_end = 10.0 * dt

    # Compute the first actuated joint index for the current world
    aid = wid * 2 + 0

    # Apply a time-dependent external force
    if t > t_start and t < t_end:
        control_tau_j[aid] = 0.1
    else:
        control_tau_j[aid] = 0.0


###
# Launchers
###


def test_control_callback(sim: Simulator):
    """
    A control callback function
    """
    wp.launch(
        _test_control_callback,
        dim=sim.model.size.num_worlds,
        inputs=[
            sim.model.time.dt,
            sim.data.solver.time.time,
            sim.data.control_n.tau_j,
        ],
    )


###
# Tests
###


class TestCartpoleSimulator(unittest.TestCase):
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

    def test_multiple_cartpoles(self):
        # Tolerance for floating point comparisons
        atol = 1e-6
        rtol = 0.0

        # Create a single-instance system
        single_builder = ModelBuilder()
        build_cartpole(single_builder)
        for i, b in enumerate(single_builder.bodies):
            msg.info(f"[single]: [builder]: body {i}: q_i: {b.q_i_0}")
            msg.info(f"[single]: [builder]: body {i}: u_i: {b.u_i_0}")

        # Create simulator and check if the initial state is consistent with the contents of the builder
        single_sim = Simulator(builder=single_builder, device=self.default_device)
        single_sim.set_control_callback(test_control_callback)
        self.assertEqual(single_sim.model.size.sum_of_num_bodies, 2)
        self.assertEqual(single_sim.model.size.sum_of_num_joints, 2)
        for i, b in enumerate(single_builder.bodies):
            np.testing.assert_allclose(single_sim.model.bodies.q_i_0.numpy()[i], b.q_i_0)
            np.testing.assert_allclose(single_sim.model.bodies.u_i_0.numpy()[i], b.u_i_0)
            np.testing.assert_allclose(single_sim.state.q_i.numpy()[i], b.q_i_0)
            np.testing.assert_allclose(single_sim.state.u_i.numpy()[i], b.u_i_0)

        # Optional verbose output - enabled globally via self.verbose
        msg.info(f"[single]: [init]: sim.model.size:\n{single_sim.model.size}\n\n")
        msg.info(f"[single]: [init]: sim.model.state.q_i:\n{single_sim.state.q_i}\n\n")
        msg.info(f"[single]: [init]: sim.model.state.u_i:\n{single_sim.state.u_i}\n\n")
        msg.info(f"[single]: [init]: sim.model.state.q_j:\n{single_sim.state.q_j}\n\n")
        msg.info(f"[single]: [init]: sim.model.state.dq_j:\n{single_sim.state.dq_j}\n\n")

        # Define the total number of sample steps to collect, and the
        # total number of execution steps from which to collect them
        num_sample_steps = 3
        num_skip_steps = 20
        num_exec_steps = 2000

        # Allocate arrays to hold the collected samples
        num_bodies = single_sim.model.size.sum_of_num_bodies
        num_joint_dofs = single_sim.model.size.sum_of_num_joint_dofs
        sample_init_q_i = np.zeros((num_sample_steps, num_bodies, 7), dtype=np.float32)
        sample_init_u_i = np.zeros((num_sample_steps, num_bodies, 6), dtype=np.float32)
        sample_next_q_i = np.zeros((num_sample_steps, num_bodies, 7), dtype=np.float32)
        sample_next_u_i = np.zeros((num_sample_steps, num_bodies, 6), dtype=np.float32)
        sample_init_q_j = np.zeros((num_sample_steps, num_joint_dofs), dtype=np.float32)
        sample_init_dq_j = np.zeros((num_sample_steps, num_joint_dofs), dtype=np.float32)
        sample_next_q_j = np.zeros((num_sample_steps, num_joint_dofs), dtype=np.float32)
        sample_next_dq_j = np.zeros((num_sample_steps, num_joint_dofs), dtype=np.float32)
        sample_ctrl_tau_j = np.zeros((num_sample_steps, num_joint_dofs), dtype=np.float32)

        # Run the simulation for the specified number of steps
        sample_freq = max(1, num_exec_steps // num_sample_steps)
        sample = 0
        msg.info(f"[sample]: sampling {num_sample_steps} transitions over {num_exec_steps} simulator steps")
        for step in range(num_skip_steps + num_exec_steps):
            # Execute a single simulation step
            single_sim.step()
            wp.synchronize()
            # Collect the initial and next state samples at the specified frequency
            if step >= num_skip_steps and step % sample_freq == 0:
                sample_init_q_i[sample, :, :] = single_sim.state_previous.q_i.numpy()
                sample_init_u_i[sample, :, :] = single_sim.state_previous.u_i.numpy()
                sample_next_q_i[sample, :, :] = single_sim.state.q_i.numpy()
                sample_next_u_i[sample, :, :] = single_sim.state.u_i.numpy()
                sample_init_q_j[sample, :] = single_sim.state_previous.q_j.numpy()
                sample_init_dq_j[sample, :] = single_sim.state_previous.dq_j.numpy()
                sample_next_q_j[sample, :] = single_sim.state.q_j.numpy()
                sample_next_dq_j[sample, :] = single_sim.state.dq_j.numpy()
                sample_ctrl_tau_j[sample, :] = single_sim.data.control_n.tau_j.numpy()
                sample += 1

        # Reshape samples for easier comparison later
        sample_init_q_i = sample_init_q_i.reshape(-1, 7)
        sample_init_u_i = sample_init_u_i.reshape(-1, 6)
        sample_next_q_i = sample_next_q_i.reshape(-1, 7)
        sample_next_u_i = sample_next_u_i.reshape(-1, 6)
        sample_init_q_j = sample_init_q_j.reshape(-1)
        sample_init_dq_j = sample_init_dq_j.reshape(-1)
        sample_next_q_j = sample_next_q_j.reshape(-1)
        sample_next_dq_j = sample_next_dq_j.reshape(-1)
        sample_ctrl_tau_j = sample_ctrl_tau_j.reshape(-1)

        # Optional verbose output
        msg.info(f"[sample]: init q_i (shape={sample_init_q_i.shape}):\n{sample_init_q_i}\n")
        msg.info(f"[sample]: init u_i (shape={sample_init_u_i.shape}):\n{sample_init_u_i}\n")
        msg.info(f"[sample]: init q_j (shape={sample_init_q_j.shape}):\n{sample_init_q_j}\n")
        msg.info(f"[sample]: init dq_j (shape={sample_init_dq_j.shape}):\n{sample_init_dq_j}\n")
        msg.info(f"[sample]: next q_i (shape={sample_next_q_i.shape}):\n{sample_next_q_i}\n")
        msg.info(f"[sample]: next u_i (shape={sample_next_u_i.shape}):\n{sample_next_u_i}\n")
        msg.info(f"[sample]: next q_j (shape={sample_next_q_j.shape}):\n{sample_next_q_j}\n")
        msg.info(f"[sample]: next dq_j (shape={sample_next_dq_j.shape}):\n{sample_next_dq_j}\n")
        msg.info(f"[sample]: control tau_j (shape={sample_ctrl_tau_j.shape}):\n{sample_ctrl_tau_j}\n")

        # Create a multi-instance system by replicating the single-instance builder
        multi_builder = ModelBuilder()
        for _i in range(num_sample_steps):
            multi_builder.add_builder(other=single_builder)
        for i, b in enumerate(multi_builder.bodies):
            msg.info(f"[multi]: [builder]: body {i}: q_i: {b.q_i_0}")
            msg.info(f"[multi]: [builder]: body {i}: u_i: {b.u_i_0}")

        # Create simulator and check if the initial state is consistent with the contents of the builder
        multi_sim = Simulator(builder=multi_builder, device=self.default_device)
        self.assertEqual(multi_sim.model.size.sum_of_num_bodies, 2 * num_sample_steps)
        self.assertEqual(multi_sim.model.size.sum_of_num_joints, 2 * num_sample_steps)
        for i, b in enumerate(multi_builder.bodies):
            np.testing.assert_allclose(multi_sim.model.bodies.q_i_0.numpy()[i], b.q_i_0)
            np.testing.assert_allclose(multi_sim.model.bodies.u_i_0.numpy()[i], b.u_i_0)
            np.testing.assert_allclose(multi_sim.state.q_i.numpy()[i], b.q_i_0)
            np.testing.assert_allclose(multi_sim.state.u_i.numpy()[i], b.u_i_0)

        # Optional verbose output - enabled globally via self.verbose
        msg.info(f"[multi]: [start]: sim.model.size:\n{multi_sim.model.size}\n\n")
        msg.info(f"[multi]: [start]: sim.model.state.q_i:\n{multi_sim.state.q_i}\n\n")
        msg.info(f"[multi]: [start]: sim.model.state.u_i:\n{multi_sim.state.u_i}\n\n")
        msg.info(f"[multi]: [start]: sim.model.state.q_j:\n{multi_sim.state.q_j}\n\n")
        msg.info(f"[multi]: [start]: sim.model.state.dq_j:\n{multi_sim.state.dq_j}\n\n")
        msg.info(f"[multi]: [start]: sim.model.control.tau_j:\n{multi_sim.control.tau_j}\n\n")

        # Set the sampled initial states into the multi-instance
        # simulator as the time-invariant initial states of the model
        multi_sim.model.bodies.q_i_0.assign(sample_init_q_i)
        multi_sim.model.bodies.u_i_0.assign(sample_init_u_i)
        np.testing.assert_allclose(multi_sim.model.bodies.q_i_0.numpy(), sample_init_q_i)
        np.testing.assert_allclose(multi_sim.model.bodies.u_i_0.numpy(), sample_init_u_i)

        # Reset the multi-instance simulator to load the new initial states
        multi_sim.reset()
        msg.info(f"[multi]: [reset]: sim.model.state_previous.q_i:\n{multi_sim.state_previous.q_i}\n\n")
        msg.info(f"[multi]: [reset]: sim.model.state_previous.u_i:\n{multi_sim.state_previous.u_i}\n\n")
        msg.info(f"[multi]: [reset]: sim.model.state_previous.q_j:\n{multi_sim.state_previous.q_j}\n\n")
        msg.info(f"[multi]: [reset]: sim.model.state_previous.dq_j:\n{multi_sim.state_previous.dq_j}\n\n")
        msg.info(f"[multi]: [reset]: sim.model.state.q_i:\n{multi_sim.state.q_i}\n\n")
        msg.info(f"[multi]: [reset]: sim.model.state.u_i:\n{multi_sim.state.u_i}\n\n")
        msg.info(f"[multi]: [reset]: sim.model.state.q_j:\n{multi_sim.state.q_j}\n\n")
        msg.info(f"[multi]: [reset]: sim.model.state.dq_j:\n{multi_sim.state.dq_j}\n\n")
        np.testing.assert_allclose(multi_sim.state_previous.q_i.numpy(), sample_init_q_i, rtol=rtol, atol=atol)
        np.testing.assert_allclose(multi_sim.state_previous.u_i.numpy(), sample_init_u_i, rtol=rtol, atol=atol)
        np.testing.assert_allclose(multi_sim.state.q_i.numpy(), sample_init_q_i, rtol=rtol, atol=atol)
        np.testing.assert_allclose(multi_sim.state.u_i.numpy(), sample_init_u_i, rtol=rtol, atol=atol)
        np.testing.assert_allclose(multi_sim.state_previous.q_j.numpy(), sample_init_q_j, rtol=rtol, atol=atol)
        np.testing.assert_allclose(multi_sim.state_previous.dq_j.numpy(), sample_init_dq_j, rtol=rtol, atol=atol)
        np.testing.assert_allclose(multi_sim.state.q_j.numpy(), sample_init_q_j, rtol=rtol, atol=atol)
        np.testing.assert_allclose(multi_sim.state.dq_j.numpy(), sample_init_dq_j, rtol=rtol, atol=atol)

        # Optional verbose output - enabled globally via self.verbose
        msg.info(f"[multi]: [init]: sim.model.state.q_i:\n{multi_sim.state.q_i}\n\n")
        msg.info(f"[multi]: [init]: sim.model.state.u_i:\n{multi_sim.state.u_i}\n\n")
        msg.info(f"[multi]: [init]: sim.model.state.q_j:\n{multi_sim.state.q_j}\n\n")
        msg.info(f"[multi]: [init]: sim.model.state.dq_j:\n{multi_sim.state.dq_j}\n\n")
        msg.info(f"[multi]: [init]: sim.model.control.tau_j:\n{multi_sim.control.tau_j}\n\n")

        # Step the multi-instance simulator once
        multi_sim.step()
        wp.synchronize()

        # Optional verbose output - enabled globally via self.verbose
        msg.info(f"[multi]: [next]: sim.model.state.q_i:\n{multi_sim.state.q_i}\n\n")
        msg.info(f"[multi]: [next]: sim.model.state.u_i:\n{multi_sim.state.u_i}\n\n")
        msg.info(f"[multi]: [next]: sim.model.state.q_j:\n{multi_sim.state.q_j}\n\n")
        msg.info(f"[multi]: [next]: sim.model.state.dq_j:\n{multi_sim.state.dq_j}\n\n")

        # DEBUG
        msg.warning(f"[multi]: multi_sim.state.q_i:\n{multi_sim.state.q_i.numpy()}\n\n")
        msg.warning(f"[multi]: sample_next_q_i:\n{sample_next_q_i}\n\n")
        msg.warning(f"[multi]: multi_sim.state.u_i:\n{multi_sim.state.u_i.numpy()}\n\n")
        msg.warning(f"[multi]: sample_next_u_i:\n{sample_next_u_i}\n\n")

        # Check that the next states match the collected samples
        np.testing.assert_allclose(multi_sim.data.solver.joints.tau_j.numpy(), sample_ctrl_tau_j, rtol=rtol, atol=atol)
        np.testing.assert_allclose(multi_sim.state_previous.q_i.numpy(), sample_init_q_i, rtol=rtol, atol=atol)
        np.testing.assert_allclose(multi_sim.state_previous.u_i.numpy(), sample_init_u_i, rtol=rtol, atol=atol)
        np.testing.assert_allclose(multi_sim.state.q_i.numpy(), sample_next_q_i, rtol=rtol, atol=atol)
        np.testing.assert_allclose(multi_sim.state.u_i.numpy(), sample_next_u_i, rtol=rtol, atol=atol)
        np.testing.assert_allclose(multi_sim.state_previous.q_j.numpy(), sample_init_q_j, rtol=rtol, atol=atol)
        np.testing.assert_allclose(multi_sim.state_previous.dq_j.numpy(), sample_init_dq_j, rtol=rtol, atol=atol)
        np.testing.assert_allclose(multi_sim.state.q_j.numpy(), sample_next_q_j, rtol=rtol, atol=atol)
        np.testing.assert_allclose(multi_sim.state.dq_j.numpy(), sample_next_dq_j, rtol=rtol, atol=atol)
        np.testing.assert_allclose(multi_sim.data.control_n.tau_j.numpy(), sample_ctrl_tau_j, rtol=rtol, atol=atol)


###
# Test execution
###

if __name__ == "__main__":
    # Global numpy configurations
    np.set_printoptions(linewidth=20000, precision=10, threshold=20000, suppress=True)  # Suppress scientific notation

    # Global warp configurations
    wp.config.enable_backward = False
    wp.config.verbose = False
    # wp.clear_kernel_cache()
    # wp.clear_lto_cache()

    # Run all tests
    unittest.main(verbosity=2)
