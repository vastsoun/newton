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

import argparse
import os
import time
from dataclasses import dataclass

import numpy as np
import torch
import warp as wp

import newton
import newton._src.solvers.kamino.utils.logger as msg
import newton.examples
from newton._src.solvers.kamino.core.builder import ModelBuilder
from newton._src.solvers.kamino.core.math import TWO_PI
from newton._src.solvers.kamino.core.shapes import ShapeType
from newton._src.solvers.kamino.core.types import float32, uint32, vec3f, vec6f
from newton._src.solvers.kamino.models import get_primitives_usd_assets_path
from newton._src.solvers.kamino.models.builders import build_cartpole
from newton._src.solvers.kamino.models.utils import make_homogeneous_builder
from newton._src.solvers.kamino.simulation.simulator import Simulator, SimulatorSettings
from newton._src.solvers.kamino.utils.print import print_progress_bar

###
# Constants
###

# Set the path to the external USD assets
USD_MODEL_PATH = os.path.join(get_primitives_usd_assets_path(), "cartpole.usda")


###
# RL Interfaces
###


@dataclass
class CartpoleStates:
    q_j: torch.Tensor | None = None
    dq_j: torch.Tensor | None = None


@dataclass
class CartpoleActions:
    tau_j: torch.Tensor | None = None


###
# Kernels
###


@wp.kernel
def _test_control_callback(
    model_dt: wp.array(dtype=float32),
    state_t: wp.array(dtype=float32),
    state_w_e_i: wp.array(dtype=vec6f),
    control_tau_j: wp.array(dtype=float32),
):
    """
    An example control callback kernel.
    """
    # Retrieve the world index from the thread ID
    wid = wp.tid()

    # Define the time window for the active external force profile
    t_start = float32(1.0)
    t_end = float32(1.1)

    # Get the current time
    t = state_t[wid]

    # Compute the first actuated joint index for the current world
    aid = wid * 2 + 0

    # Apply a time-dependent external force
    if t > t_start and t < t_end:
        control_tau_j[aid] = 0.1 * wp.sin(1.0 * TWO_PI * (t - t_start)) * wp.randf(uint32(wid), -1.0, 1.0)
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
            sim.data.solver.bodies.w_e_i,
            sim.data.control_n.tau_j,
        ],
    )


###
# Example class
###


class CartpoleExample:
    """ViewerGL example class for cartpole simulation."""

    def __init__(self, num_worlds: int, viewer, device, use_cuda_graph: bool = False, logging: bool = True):
        # Initialize target frames per second and corresponding time-steps
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_dt = 0.001
        self.sim_substeps = int(self.frame_dt / self.sim_dt)
        self.max_sim_steps = 10000
        msg.warning("fps: %f", self.fps)
        msg.warning("frame_dt: %f", self.frame_dt)
        msg.warning("sim_dt: %f", self.sim_dt)
        msg.warning("sim_substeps: %d", self.sim_substeps)
        msg.warning("max_sim_steps: %d", self.max_sim_steps)

        # Initialize internal time-keeping
        self.sim_time = 0.0
        self.sim_steps = 0

        # Cache a reference to the viewr
        self.viewer = viewer

        # Cache the device and other internal flags
        self.device = device
        self.use_cuda_graph: bool = use_cuda_graph
        self.logging: bool = logging

        # Create a single-instance system (always load from USD)
        msg.info("Constructing builder from imported USD ...")
        self.builder: ModelBuilder = make_homogeneous_builder(num_worlds=num_worlds, build_func=build_cartpole)[0]
        msg.warning("total mass: %f", self.builder.world.mass_total)
        msg.warning("total diag inertia: %f", self.builder.world.inertia_total)

        # Set gravity
        self.builder.gravity.enabled = True

        # Set solver settings
        settings = SimulatorSettings()
        settings.dt = self.sim_dt
        settings.problem.alpha = 0.1
        settings.solver.primal_tolerance = 1e-6
        settings.solver.dual_tolerance = 1e-6
        settings.solver.compl_tolerance = 1e-6
        settings.solver.max_iterations = 200
        settings.solver.rho_0 = 0.05

        # Retrive the number of joints per world
        nj = self.builder.world.num_joints

        # Data logging arrays
        self.log_time = np.zeros((num_worlds, self.max_sim_steps), dtype=np.float32)
        self.log_q_j = np.zeros((num_worlds, self.max_sim_steps, nj), dtype=np.float32)
        self.log_dq_j = np.zeros((num_worlds, self.max_sim_steps, nj), dtype=np.float32)
        self.log_tau_j = np.zeros((num_worlds, self.max_sim_steps, nj), dtype=np.float32)

        # Create a simulator
        msg.info("Building the simulator...")
        self.sim = Simulator(builder=self.builder, settings=settings, device=device)
        self.sim.set_control_callback(test_control_callback)

        # Don't set a newton model - we'll render everything manually using log_shapes
        self.viewer.set_model(None)

        # Extract geometry information from the kamino simulator
        self.extract_geometry_info()

        # Define colors for different parts of the system
        self.body_colors = [
            wp.array([wp.vec3(0.9, 0.1, 0.3)], dtype=wp.vec3),  # Crimson Red
            wp.array([wp.vec3(0.1, 0.7, 0.9)], dtype=wp.vec3),  # Cyan Blue
            wp.array([wp.vec3(1.0, 0.5, 0.0)], dtype=wp.vec3),  # Orange
            wp.array([wp.vec3(0.6, 0.2, 0.8)], dtype=wp.vec3),  # Purple
            wp.array([wp.vec3(0.2, 0.8, 0.2)], dtype=wp.vec3),  # Green
            wp.array([wp.vec3(0.8, 0.8, 0.2)], dtype=wp.vec3),  # Yellow
            wp.array([wp.vec3(0.8, 0.2, 0.8)], dtype=wp.vec3),  # Magenta
            wp.array([wp.vec3(0.5, 0.5, 0.5)], dtype=wp.vec3),  # Gray
        ]

        # Declare and initialize the optional computation graphs
        # NOTE: These are used for most efficient GPU runtime
        self.reset_graph = None
        self.step_graph = None
        self.simulate_graph = None

        # Capture CUDA graph if requested and available
        self.capture()

        # Warm-start the simulator before rendering
        # NOTE: This compiles and loads the warp kernels prior to execution
        msg.info("Warming up simulator...")
        self.step_once()
        self.reset()

        # Declare a PyTorch data interface for the current state and controls data
        self.states: CartpoleStates | None = None
        self.actions: CartpoleActions | None = None

        # Initialize RL interfaces
        self.make_rl_interface()

    def make_rl_interface(self):
        """
        Constructs data interfaces for batched MDP states and actions.

        Notes:
        - Each torch.Tensor wraps the underlying kamino simulator data arrays without copying.
        """
        # Retrieve the batched system dimensions
        num_worlds = self.sim.model.size.num_worlds
        num_joint_dofs = self.sim.model.size.max_of_num_actuated_joint_dofs

        # Construct state and action tensors wrapping the underlying simulator data
        self.states = CartpoleStates(
            q_j=wp.to_torch(self.sim.data.state_n.q_j).reshape(num_worlds, num_joint_dofs),
            dq_j=wp.to_torch(self.sim.data.state_n.dq_j).reshape(num_worlds, num_joint_dofs),
        )
        self.actions = CartpoleActions(
            tau_j=wp.to_torch(self.sim.data.control_n.tau_j).reshape(num_worlds, num_joint_dofs),
        )

    def extract_geometry_info(self):
        """Extract geometry information from the kamino simulator."""
        # Get collision geometry information from the simulator
        cgeom_model = self.sim.model.cgeoms

        self.geometry_info = []
        self.ground_info = None

        # Extract geometry info from collision geometries
        for i in range(cgeom_model.num_geoms):
            wid = cgeom_model.wid.numpy()[i]  # World ID
            bid = cgeom_model.bid.numpy()[i]  # Body ID (-1 for static/ground)
            sid = cgeom_model.sid.numpy()[i]  # Shape ID
            params = cgeom_model.params.numpy()[i]  # Shape parameters
            offset = cgeom_model.offset.numpy()[i]  # Geometry offset

            if bid == -1:  # Ground plane (static body)
                # Ground plane: params = [depth, width, height, 0]
                self.ground_info = {
                    "dimensions": (params[0], params[1], params[2]),  # depth, width, height
                    "offset": offset,  # position and orientation
                }
            else:  # Regular box bodies
                # Store geometry information for rendering
                geom_info = {
                    "world_id": wid,
                    "geom_id": i,
                    "body_id": bid,
                    "shape_id": sid,
                    "params": params,
                    "offset": offset,
                }
                self.geometry_info.append(geom_info)

    def capture(self):
        """Capture CUDA graph if requested and available."""
        if self.use_cuda_graph:
            msg.info("Running with CUDA graphs...")
            with wp.ScopedCapture(self.device) as reset_capture:
                self.sim.reset()
            self.reset_graph = reset_capture.graph
            with wp.ScopedCapture(self.device) as step_capture:
                self.sim.step()
            self.step_graph = step_capture.graph
            with wp.ScopedCapture(self.device) as sim_capture:
                self.simulate()
            self.simulate_graph = sim_capture.graph
        else:
            msg.info("Running with kernels...")

    def log_data(self):
        if self.sim_steps >= self.max_sim_steps:
            msg.warning("Maximum simulation steps reached, skipping data logging.")
            return
        self.log_time[:, self.sim_steps] = self.sim.data.solver.time.time.numpy()
        # self.log_q_j[self.sim_steps, :] = self.sim.data.state_n.q_j.numpy()
        # self.log_dq_j[self.sim_steps, :] = self.sim.data.state_n.dq_j.numpy()
        # self.log_tau_j[self.sim_steps, :] = self.sim.data.control_n.tau_j.numpy()

    def simulate(self):
        """Run simulation substeps."""
        for _i in range(self.sim_substeps):
            self.sim.step()
            self.sim_steps += 1
            if not self.use_cuda_graph and self.logging:
                self.log_data()

    def reset(self):
        """Reset the simulation."""
        if self.reset_graph:
            wp.capture_launch(self.reset_graph)
        else:
            self.sim.reset()
        self.sim_steps = 0
        self.sim_time = 0.0
        if not self.use_cuda_graph and self.logging:
            self.log_data()

    def step_once(self):
        """Run the simulation for a single time-step."""
        if self.step_graph:
            wp.capture_launch(self.step_graph)
        else:
            self.sim.step()
        self.sim_steps += 1
        self.sim_time += self.sim_dt
        if not self.use_cuda_graph and self.logging:
            self.log_data()

    def step(self):
        """Step the simulation."""
        if self.simulate_graph:
            wp.capture_launch(self.simulate_graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt
        msg.warning(
            "[t={%f}] q_j:\n{%s}\ndq_j:\n{%s}\ntau_j:\n{%s}\n\n",
            self.sim_time,
            self.states.q_j,
            self.states.dq_j,
            self.actions.tau_j,
        )

    def render(self):
        """Render the current frame."""
        self.viewer.begin_frame(self.sim_time)

        # Extract body poses from the kamino simulator
        try:
            # TODO: How to make this a 2D grid in XY instead of lining them all along Y?
            world_spacing = vec3f(0.0, 1.0, 0.0)
            body_poses = self.sim.model_data.bodies.q_i.numpy()

            # Render each geometry using log_shapes
            for i, geom_info in enumerate(self.geometry_info):
                wid = geom_info["world_id"]
                gid = geom_info["geom_id"]
                bid = geom_info["body_id"]
                sid = geom_info["shape_id"]
                params = geom_info["params"]
                offset = geom_info["offset"]

                # Skip static geometries (ground plane, etc.)
                if bid == -1:
                    continue

                # Get body pose if available
                if bid < len(body_poses):
                    # Convert kamino transformf to warp transform
                    pose = body_poses[bid]
                    # kamino transformf has [x, y, z, qx, qy, qz, qw] format
                    position = wp.vec3(float(pose[0]), float(pose[1]), float(pose[2])) + float(wid) * world_spacing
                    quaternion = wp.quat(float(pose[3]), float(pose[4]), float(pose[5]), float(pose[6]))
                    body_transform = wp.transform(position, quaternion)

                    # Apply geometry offset
                    offset_pos = wp.vec3(float(offset[0]), float(offset[1]), float(offset[2]))
                    offset_quat = wp.quat(float(offset[3]), float(offset[4]), float(offset[5]), float(offset[6]))
                    offset_transform = wp.transform(offset_pos, offset_quat)

                    # Combine body and offset transforms
                    final_transform = wp.transform_multiply(body_transform, offset_transform)

                    # Choose color based on body ID
                    color_idx = bid % len(self.body_colors)
                    color = self.body_colors[color_idx]

                    # Render based on shape type
                    if sid == ShapeType.BOX:  # BOX shape (SHAPE_BOX = 5)
                        # Convert kamino full dimensions to newton half-extents
                        half_extents = (params[0] / 2, params[1] / 2, params[2] / 2)
                        self.viewer.log_shapes(
                            f"/cartpole/body_{bid}_geom_{i}",
                            newton.GeoType.BOX,
                            half_extents,
                            wp.array([final_transform], dtype=wp.transform),
                            color,
                        )
                    elif sid == ShapeType.SPHERE:  # SPHERE shape (SHAPE_SPHERE = 1)
                        radius = params[0]
                        self.viewer.log_shapes(
                            f"/cartpole/body_{bid}_geom_{i}",
                            newton.GeoType.SPHERE,
                            radius,
                            wp.array([final_transform], dtype=wp.transform),
                            color,
                        )
                    elif sid == ShapeType.CAPSULE:  # CAPSULE shape (SHAPE_CAPSULE = 2)
                        radius = params[0]
                        half_height = params[1] / 2
                        self.viewer.log_shapes(
                            f"/cartpole/body_{bid}_geom_{i}",
                            newton.GeoType.CAPSULE,
                            (radius, half_height),
                            wp.array([final_transform], dtype=wp.transform),
                            color,
                        )
                    elif sid == ShapeType.CYLINDER:  # CYLINDER shape (SHAPE_CYLINDER = 2)
                        radius = params[0]
                        half_height = params[1] / 2
                        self.viewer.log_shapes(
                            f"/cartpole/body_{bid}_geom_{i}",
                            newton.GeoType.CYLINDER,
                            (radius, half_height),
                            wp.array([final_transform], dtype=wp.transform),
                            color,
                        )
                    elif sid == ShapeType.MESH:  # MESH shape (SHAPE_MESH = 9)
                        self.viewer.log_shapes(
                            name=f"/cartpole/body_{bid}_geom_{i}",
                            geo_type=newton.GeoType.MESH,
                            geo_scale=1.0,
                            xforms=wp.array([final_transform], dtype=wp.transform),
                            geo_is_solid=True,
                            colors=color,
                            geo_src=self.builder.collision_geoms[gid].shape._data,
                        )

        except Exception as e:
            print(f"Error accessing body poses: {e}")
            # print(f"Available attributes: {dir(self.sim.model_data.bodies)}")

        # Render the ground plane from kamino
        if self.ground_info:
            ground_offset = self.ground_info["offset"]
            ground_pos = wp.vec3(float(ground_offset[0]), float(ground_offset[1]), float(ground_offset[2]))
            ground_quat = wp.quat(
                float(ground_offset[3]), float(ground_offset[4]), float(ground_offset[5]), float(ground_offset[6])
            )
            ground_transform = wp.transform(ground_pos, ground_quat)

            # Convert ground plane dimensions to half-extents
            # Kamino: BoxShape(20.0, 20.0, 1.0) = full dimensions
            # Newton: expects (10.0, 10.0, 0.5) = half-extents
            ground_half_extents = (
                self.ground_info["dimensions"][0] / 2,  # 20.0 -> 10.0
                self.ground_info["dimensions"][1] / 2,  # 20.0 -> 10.0
                self.ground_info["dimensions"][2] / 2,  # 1.0 -> 0.5
            )

            # Ground plane color (gray)
            ground_color = wp.array([wp.vec3(0.7, 0.7, 0.7)], dtype=wp.vec3)

            self.viewer.log_shapes(
                "/cartpole/ground",
                newton.GeoType.BOX,
                ground_half_extents,
                wp.array([ground_transform], dtype=wp.transform),
                ground_color,
            )

        self.viewer.end_frame()

    def test(self):
        """Test function for compatibility."""
        pass


###
# Execution functions
###


def run_headless(example: CartpoleExample, num_steps: int = 25000, progress: bool = True):
    """Run the simulation in headless mode for a fixed number of steps."""
    msg.info(f"Running for {num_steps} steps...")
    start_time = time.time()
    for i in range(num_steps):
        example.step_once()
        wp.synchronize()
        if progress:
            print_progress_bar(i + 1, num_steps, start_time, prefix="Progress", suffix="")


###
# Main function
###


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cartpole simulation example")
    parser.add_argument("--viewer", choices=["gl", "usd", "rerun", "null"], default="gl", help="Viewer type")
    parser.add_argument("--device", type=str, help="The compute device to use")
    parser.add_argument("--headless", action="store_true", default=False, help="Run in headless mode")
    parser.add_argument("--num-worlds", type=int, default=3, help="Number of worlds to simulate in parallel")
    parser.add_argument("--num-frames", type=int, default=1000, help="Number of frames for null/USD viewer")
    parser.add_argument("--output-path", type=str, help="Output path for USD viewer")
    parser.add_argument("--cuda-graph", action="store_true", default=False, help="Use CUDA graphs")
    parser.add_argument("--clear-cache", action="store_true", default=False, help="Clear warp cache")
    parser.add_argument("--show-plots", action="store_true", default=False, help="Show plots after simulation")
    parser.add_argument("--test", action="store_true", default=False, help="Run tests")
    args = parser.parse_args()

    # Set global numpy configurations
    np.set_printoptions(linewidth=20000, precision=6, threshold=10000, suppress=True)  # Suppress scientific notation

    # Clear warp cache if requested
    if args.clear_cache:
        wp.clear_kernel_cache()
        wp.clear_lto_cache()

    # TODO: Make optional
    # Set the verbosity of the global message logger
    msg.set_log_level(msg.LogLevel.INFO)

    # Set device if specified, otherwise use Warp's default
    if args.device:
        device = wp.get_device(args.device)
        wp.set_device(device)
    else:
        device = wp.get_preferred_device()
        device = wp.get_device(device)

    # Determine if CUDA graphs should be used for execution
    can_use_cuda_graph = device.is_cuda and wp.is_mempool_enabled(device)
    use_cuda_graph = can_use_cuda_graph & args.cuda_graph
    msg.info(f"can_use_cuda_graph: {can_use_cuda_graph}")
    msg.info(f"use_cuda_graph: {use_cuda_graph}")

    # Run a brute-force similation loop if headless
    if args.headless:
        msg.info("Running in headless mode...")
        viewer = newton.viewer.ViewerNull(num_frames=args.num_frames)
        example = CartpoleExample(args.num_worlds, viewer, device=device, use_cuda_graph=use_cuda_graph)
        run_headless(example, num_steps=args.num_frames, progress=True)

    # Otherwise launch using a Newton viewer
    else:
        # Create viewer based on type
        if args.viewer == "gl":
            msg.info("Running in OpenGL render mode...")
            viewer = newton.viewer.ViewerGL(headless=False)
        elif args.viewer == "usd":
            if args.output_path is None:
                raise ValueError("--output-path is required when using usd viewer")
            msg.info("Running in OpenUSD render  mode...")
            viewer = newton.viewer.ViewerUSD(output_path=args.output_path, num_frames=args.num_frames)
        elif args.viewer == "rerun":
            msg.info("Running in ReRun render  mode...")
            viewer = newton.viewer.ViewerRerun()
        elif args.viewer == "null":
            msg.info("Running in Null render  mode...")
            viewer = newton.viewer.ViewerNull(num_frames=args.num_frames)
        else:
            raise ValueError(f"Invalid viewer: {args.viewer}")

        # Create and run example
        example = CartpoleExample(args.num_worlds, viewer, device=device, use_cuda_graph=use_cuda_graph)

        # Set initial camera position for better view of the cartpole
        if hasattr(viewer, "set_camera"):
            # Position camera to get a good view of the cartpole
            camera_pos = wp.vec3(2.0, 2.0, 0.3)
            pitch = -10.0
            yaw = 205.0
            viewer.set_camera(camera_pos, pitch, yaw)

        # Launch the example using Newton's built-in runtime
        newton.examples.run(example, args)
