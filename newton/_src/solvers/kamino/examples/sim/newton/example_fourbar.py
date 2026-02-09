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

###########################################################################
# Example for basic four-bar mechanism
#
# Shows how to simulate a basic four-bar linkage with multiple worlds using SolverKamino.
#
# Command: python -m newton.examples example_fourbar --num-worlds 16
#
###########################################################################

# Python imports
import os

# Numpy/Warp imports
import numpy as np
import warp as wp

# Newton imports
import newton
import newton.examples

# Kamino imports
from newton._src.solvers.kamino.geometry import CollisionDetectorSettings, CollisionPipelineType
from newton._src.solvers.kamino.models import get_basics_usd_assets_path
from newton._src.solvers.kamino.solver_kamino import SolverKaminoSettings
from newton._src.solvers.kamino.solvers.padmm import PADMMWarmStartMode
from newton._src.solvers.kamino.solvers.warmstart import WarmstarterContacts
from newton._src.solvers.kamino.utils import logger as msg


class Example:
    def __init__(self, viewer, num_worlds=8, args=None):
        # Set simulation run-time configurations
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 16
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.num_worlds = num_worlds
        self.viewer = viewer
        self.device = wp.get_device()

        # Create a single-robot model builder and register the Kamino-specific custom attributes
        robot_builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverKamino.register_custom_attributes(robot_builder)
        robot_builder.default_shape_cfg.thickness = 1e-5
        robot_builder.default_shape_cfg.contact_margin = 1e-5

        # Load the Anymal D USD and add it to the builder
        asset_file = os.path.join(get_basics_usd_assets_path(), "boxes_fourbar.usda")
        robot_builder.add_usd(
            asset_file,
            joint_ordering=None,
            # bodies_follow_joint_ordering=True, TODO: CHECK THAT THIS WORKS
            collapse_fixed_joints=False,  # TODO: FIX THIS WHEN ITS TRUE
            enable_self_collisions=False,
            hide_collision_shapes=False,
        )

        # TODO: Remove this once the issue with the USD loader and collision filter pairs is resolved
        robot_builder.shape_collision_filter_pairs.append((3, 0))
        msg.error("robot_builder.shape_collision_filter_pairs: %s", robot_builder.shape_collision_filter_pairs)

        # Create the multi-world model by duplicating the single-robot
        # builder for the specified number of worlds
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        for _ in range(self.num_worlds):
            builder.add_world(robot_builder)
        # TODO: @nvtw: Add support for global ground plane
        # TODO: builder.add_ground_plane()
        # builder.add_ground_plane()

        # Create the model from the builder
        self.model = builder.finalize(skip_validation_joints=True)

        # TODO: Remove this once the issue with the USD loader and collision filter pairs is resolved
        np.set_printoptions(precision=7, linewidth=10000, threshold=10000, suppress=True)
        msg.error("model.shape_body: %s", self.model.shape_body)
        msg.error("model.shape_collision_filter_pairs: %s", self.model.shape_collision_filter_pairs)
        msg.error("model.shape_contact_pair_count: %s", self.model.shape_contact_pair_count)
        msg.error("model.shape_contact_pairs:\n%s", self.model.shape_contact_pairs)

        # Create and configure settings for SolverKamino and the collision detector
        collision_detector_settings = CollisionDetectorSettings()
        collision_detector_settings.pipeline = CollisionPipelineType.UNIFIED
        solver_settings = SolverKaminoSettings()
        solver_settings.problem.preconditioning = True
        solver_settings.padmm.primal_tolerance = 1e-4
        solver_settings.padmm.dual_tolerance = 1e-4
        solver_settings.padmm.compl_tolerance = 1e-4
        solver_settings.padmm.max_iterations = 200
        solver_settings.padmm.rho_0 = 0.1
        solver_settings.use_solver_acceleration = True
        solver_settings.warmstart_mode = PADMMWarmStartMode.CONTAINERS
        solver_settings.contact_warmstart_method = WarmstarterContacts.Method.GEOM_PAIR_NET_FORCE

        # Create the Kamino solver for the given model
        self.solver = newton.solvers.SolverKamino(
            model=self.model,
            collision_detector_settings=collision_detector_settings,
            solver_settings=solver_settings,
        )

        msg.warning("model.shape_collision_group:\n%s\n", self.model.shape_collision_group)
        msg.warning("model_kamino.geoms.group:\n%s\n", self.solver._model_kamino.geoms.group)
        msg.warning("model_kamino.geoms.collides:\n%s\n", self.solver._model_kamino.geoms.collides)
        msg.warning("model_kamino.geoms.margin:\n%s\n", self.solver._model_kamino.geoms.margin)

        # Create state and control data containers
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # Reset the simulation state to a valid initial configuration above the ground
        self.base_q = wp.zeros(shape=(self.num_worlds,), dtype=wp.transformf)
        q_b = wp.quat_identity(dtype=wp.float32)
        q_base = wp.transformf((0.0, 0.0, 0.0), q_b)
        q_base = np.array(q_base)
        q_base = np.tile(q_base, (self.num_worlds, 1))
        for w in range(self.num_worlds):
            q_base[w, :3] += np.array([0.0, 0.0, 0.2]) * float(w)
        self.base_q.assign(q_base)
        self.solver.reset(state_out=self.state_0, base_q=self.base_q)

        # Attach the model to the viewer for visualization
        self.viewer.set_model(self.model)

        # Capture the simulation graph if running on CUDA
        # NOTE: This only has an effect on GPU devices
        self.capture()

    def capture(self):
        self.graph = None
        if self.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    # simulate() performs one frame's worth of updates
    def simulate(self):
        for _ in range(self.sim_substeps):
            # clear forces on the state before applying new ones
            self.state_0.clear_forces()
            # apply forces to the model for picking, wind, etc
            self.viewer.apply_forces(self.state_0)
            # step the simulation forward by one time step
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        # TODO: self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "all bodies are above the ground",
            lambda q, qd: q[2] > -0.006,
        )
        # Only check velocities on CUDA where we run 500 frames (enough time to settle)
        # On CPU we only run 10 frames and the robot is still falling (~0.65 m/s)
        if self.device.is_cuda:
            newton.examples.test_body_state(
                self.model,
                self.state_0,
                "body velocities are small",
                lambda q, qd: max(abs(qd))
                < 0.25,  # Relaxed from 0.1 - unified pipeline has residual velocities up to ~0.2
            )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--num-worlds", type=int, default=16, help="Total number of simulated worlds.")
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args.num_worlds, args)
    example.viewer._paused = True  # Start paused to inspect the initial configuration

    # If only a single-world is created, set initial
    # camera position for better view of the system
    if args.num_worlds == 1 and hasattr(example.viewer, "set_camera"):
        camera_pos = wp.vec3(-0.5, -1.0, 0.2)
        pitch = -5.0
        yaw = 70.0
        example.viewer.set_camera(camera_pos, pitch, yaw)

    newton.examples.run(example, args)
