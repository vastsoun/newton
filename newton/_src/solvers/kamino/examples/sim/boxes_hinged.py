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

import h5py
import numpy as np
import warp as wp

import newton
import newton._src.solvers.kamino.utils.logger as msg
import newton.examples
from newton._src.solvers.kamino.core.builder import ModelBuilder
from newton._src.solvers.kamino.core.types import float32, vec6f
from newton._src.solvers.kamino.examples import get_examples_data_hdf5_path, print_frame
from newton._src.solvers.kamino.models import get_primitives_usd_assets_path
from newton._src.solvers.kamino.models.builders import build_boxes_hinged
from newton._src.solvers.kamino.simulation.simulator import Simulator
from newton._src.solvers.kamino.utils.device import get_device_info
from newton._src.solvers.kamino.utils.io import hdf5
from newton._src.solvers.kamino.utils.io.usd import USDImporter
from newton._src.solvers.kamino.utils.print import print_progress_bar

###
# Kernels
###


@wp.kernel
def _control_callback(
    model_time_dt: wp.array(dtype=float32),
    state_time_t: wp.array(dtype=float32),
    state_joints_q_j: wp.array(dtype=float32),
    state_joints_dq_j: wp.array(dtype=float32),
    state_joints_tau_j: wp.array(dtype=float32),
    state_bodies_w_e_i: wp.array(dtype=vec6f),
):
    """
    An example control callback kernel.
    """
    # Set world index
    wid = int(0)
    jid = int(0)

    # Define the time window for the active external force profile
    t_start = float32(2.0)
    t_end = float32(2.5)

    # Get the current time
    t = state_time_t[wid]

    # Apply a time-dependent external force
    if t > t_start and t < t_end:
        state_joints_tau_j[jid] = -3.0
    else:
        state_joints_tau_j[jid] = 0.0


###
# Launchers
###


def control_callback(sim: Simulator):
    """
    A control callback function
    """
    wp.launch(
        _control_callback,
        dim=1,
        inputs=[
            sim.model.time.dt,
            sim.model_data.time.time,
            sim.model_data.joints.q_j,
            sim.model_data.joints.dq_j,
            sim.model_data.joints.tau_j,
            sim.model_data.bodies.w_e_i,
        ],
    )


###
# Constants
###

# Set the path to the external USD assets
USD_MODEL_PATH = os.path.join(get_primitives_usd_assets_path(), "boxes_hinged.usda")

# Set the path to the generated HDF5 dataset file
RENDER_DATASET_PATH = os.path.join(get_examples_data_hdf5_path(), "boxes_hinged.hdf5")


###
# Main function
###


def run_hdf5_mode(clear_warp_cache=True, use_cuda_graph=False, load_from_usd=True, verbose=False):
    """Run the simulation in HDF5 mode to save data to file."""
    # Application options

    # Clear the warp caches
    if clear_warp_cache:
        wp.clear_kernel_cache()
        wp.clear_lto_cache()

    # Warp configs
    # wp.config.verify_fp = True
    # wp.config.verbose = True
    # wp.config.verbose_warnings = True

    # Set global numpy configurations
    np.set_printoptions(linewidth=20000, precision=6, threshold=10000, suppress=True)  # Suppress scientific notation

    # Get the default warp device
    device = wp.get_preferred_device()
    device = wp.get_device(device)

    # Enable verbose output
    msg.set_log_level(msg.LogLevel.INFO)

    # Determine if using CUDA graphs
    can_use_cuda_graph = device.is_cuda and wp.is_mempool_enabled(device)
    msg.info(f"use_cuda_graph: {use_cuda_graph}")
    msg.info(f"can_use_cuda_graph: {can_use_cuda_graph}")

    # Create a single-instance system
    if load_from_usd:
        msg.info("Constructing builder from imported USD ...")
        importer = USDImporter()
        builder: ModelBuilder = importer.import_from(source=USD_MODEL_PATH, load_static_geometry=True)
    else:
        msg.info("Constructing builder using generator ...")
        builder = ModelBuilder()
        build_boxes_hinged(builder=builder, z_offset=0.0, ground=True)

    # Set gravity
    builder.gravity.enabled = True

    # Create a simulator
    msg.info("Building the simulator...")
    sim = Simulator(builder=builder, device=device)
    sim.set_control_callback(control_callback)

    # Capture graphs for simulator ops: reset and step
    use_cuda_graph &= can_use_cuda_graph
    reset_graph = None
    step_graph = None
    if use_cuda_graph:
        with wp.ScopedCapture(device) as reset_capture:
            sim.reset()
        reset_graph = reset_capture.graph
        with wp.ScopedCapture(device) as step_capture:
            sim.step()
        step_graph = step_capture.graph

    # Warm-start the simulator before rendering
    # NOTE: This compiles and loads the warp kernels prior to execution
    msg.info("Warming up the simulator...")
    if use_cuda_graph:
        print("Running with CUDA graphs...")
        wp.capture_launch(reset_graph)
        wp.capture_launch(step_graph)
    else:
        msg.info("Running with kernels...")
        with wp.ScopedDevice(device):
            sim.step()
            sim.reset()

    # Print application info
    msg.info("%s", get_device_info(device))

    # Construct and configure the data containers
    msg.info("Setting up HDF5 data containers...")
    sdata = hdf5.RigidBodySystemData()
    sdata.configure(simulator=sim)
    cdata = hdf5.ContactsData()
    pdata = hdf5.DualProblemData()
    pdata.configure(simulator=sim)

    # Create the output directory if it does not exist
    render_dir = os.path.dirname(RENDER_DATASET_PATH)
    if not os.path.exists(render_dir):
        os.makedirs(render_dir)

    # Create a dataset file and renderer
    msg.info("Creating the HDF5 renderer...")
    datafile = h5py.File(RENDER_DATASET_PATH, "w")
    renderer = hdf5.DatasetRenderer(sysname="boxes_hinged", datafile=datafile, dt=sim.dt)

    # Store the initial state of the system
    sdata.update_from(simulator=sim)
    cdata.update_from(simulator=sim)
    renderer.add_frame(system=sdata, contacts=cdata)
    if verbose:
        print_frame(sim, 0)

    # Step the simulation and collect frames
    ns = 10000
    msg.info(f"Collecting ns={ns} frames...")
    start_time = time.time()
    with wp.ScopedTimer("sim.step", active=True):
        with wp.ScopedDevice(device):
            for i in range(ns):
                if use_cuda_graph:
                    wp.capture_launch(step_graph)
                else:
                    with wp.ScopedDevice(device):
                        sim.step()
                wp.synchronize()

                sdata.update_from(simulator=sim)
                cdata.update_from(simulator=sim)
                pdata.update_from(simulator=sim)
                renderer.add_frame(system=sdata, contacts=cdata, problem=pdata)
                print_progress_bar(i, ns, start_time, prefix="Progress", suffix="")

    # Save the dataset
    msg.info("Saving all frames to HDF5...")
    renderer.save()


class BoxesHingedExample:
    """ViewerGL example class for boxes hinged simulation."""

    def __init__(self, viewer, load_from_usd=True, use_cuda_graph=False):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.use_cuda_graph = use_cuda_graph

        # Get the default warp device
        device = wp.get_preferred_device()
        device = wp.get_device(device)

        # Create a single-instance system
        if load_from_usd:
            msg.info("Constructing builder from imported USD ...")
            importer = USDImporter()
            builder: ModelBuilder = importer.import_from(source=USD_MODEL_PATH, load_static_geometry=True)
        else:
            msg.info("Constructing builder using generator ...")
            builder = ModelBuilder()
            build_boxes_hinged(builder=builder, z_offset=0.0, ground=True)

        # Set gravity
        builder.gravity.enabled = True

        # Create a simulator
        msg.info("Building the simulator...")
        self.sim = Simulator(builder=builder, device=device)
        self.sim.set_control_callback(control_callback)

        # Don't set a newton model - we'll render everything manually using log_shapes
        self.viewer.set_model(None)

        # Extract geometry information from the kamino simulator
        self.extract_geometry_info()

        # Define diverse colors for each box
        self.box_colors = [
            wp.array([wp.vec3(0.9, 0.1, 0.3)], dtype=wp.vec3),  # Crimson Red
            wp.array([wp.vec3(0.1, 0.7, 0.9)], dtype=wp.vec3),  # Cyan Blue
            wp.array([wp.vec3(1.0, 0.5, 0.0)], dtype=wp.vec3),  # Orange
            wp.array([wp.vec3(0.6, 0.2, 0.8)], dtype=wp.vec3),  # Purple
        ]

        # Initialize the simulator with a warm-up step
        self.sim.reset()

        # Capture CUDA graph if requested and available
        self.capture()

    def extract_geometry_info(self):
        """Extract geometry information from the kamino simulator."""
        # Get collision geometry information from the simulator
        cgeom_model = self.sim.model.cgeoms

        self.box_dimensions = []
        self.ground_info = None

        # Extract box dimensions and ground plane info from collision geometries
        for i in range(cgeom_model.num_geoms):
            bid = cgeom_model.bid.numpy()[i]  # Body ID (-1 for static/ground)
            sid = cgeom_model.sid.numpy()[i]  # Shape ID (5 = BOX from SHAPE_BOX constant)
            params = cgeom_model.params.numpy()[i]  # Shape parameters
            offset = cgeom_model.offset.numpy()[i]  # Geometry offset

            if sid == 5:  # BOX shape (SHAPE_BOX = 5)
                if bid == -1:  # Ground plane (static body)
                    # Ground plane: params = [depth, width, height, 0]
                    self.ground_info = {
                        "dimensions": (params[0], params[1], params[2]),  # depth, width, height
                        "offset": offset,  # position and orientation
                    }
                else:  # Regular box bodies
                    # Box dimensions: params = [depth, width, height, 0]
                    self.box_dimensions.append((params[0], params[1], params[2]))

    def capture(self):
        """Capture CUDA graph if requested and available."""
        if self.use_cuda_graph and wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        """Run simulation substeps."""
        for _ in range(self.sim_substeps):
            self.sim.step()

    def step(self):
        """Step the simulation."""
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def render(self):
        """Render the current frame."""
        self.viewer.begin_frame(self.sim_time)

        # Extract body poses from the kamino simulator
        try:
            body_poses = self.sim.model_data.bodies.q_i.numpy()

            # Render each box using log_shapes
            for i, (dimensions, color) in enumerate(zip(self.box_dimensions, self.box_colors, strict=False)):
                if i < len(body_poses):
                    # Convert kamino transformf to warp transform
                    pose = body_poses[i]
                    # kamino transformf has [x, y, z, qx, qy, qz, qw] format
                    position = wp.vec3(float(pose[0]), float(pose[1]), float(pose[2]))
                    quaternion = wp.quat(float(pose[3]), float(pose[4]), float(pose[5]), float(pose[6]))
                    transform = wp.transform(position, quaternion)

                    # Convert kamino full dimensions to newton half-extents
                    # Kamino: BoxShape(depth, width, height) = full dimensions
                    # Newton: expects (half_depth, half_width, half_height)
                    half_extents = (dimensions[0] / 2, dimensions[1] / 2, dimensions[2] / 2)

                    # Log the box shape
                    self.viewer.log_shapes(
                        f"/hinged/box_{i + 1}",
                        newton.GeoType.BOX,
                        half_extents,
                        wp.array([transform], dtype=wp.transform),
                        color,
                    )

        except Exception as e:
            print(f"Error accessing body poses: {e}")
            print(f"Available attributes: {dir(self.sim.model_data.bodies)}")

        # Render the ground plane from kamino
        if self.ground_info:
            ground_offset = self.ground_info["offset"]
            ground_pos = wp.vec3(float(ground_offset[0]), float(ground_offset[1]), float(ground_offset[2]))
            ground_quat = wp.quat(
                float(ground_offset[3]), float(ground_offset[4]), float(ground_offset[5]), float(ground_offset[6])
            )
            ground_transform = wp.transform(ground_pos, ground_quat)

            # Convert ground plane dimensions to half-extents
            ground_half_extents = (
                self.ground_info["dimensions"][0] / 2,
                self.ground_info["dimensions"][1] / 2,
                self.ground_info["dimensions"][2] / 2,
            )

            # Ground plane color (gray)
            ground_color = wp.array([wp.vec3(0.7, 0.7, 0.7)], dtype=wp.vec3)

            self.viewer.log_shapes(
                "/hinged/ground",
                newton.GeoType.BOX,
                ground_half_extents,
                wp.array([ground_transform], dtype=wp.transform),
                ground_color,
            )

        self.viewer.end_frame()

    def test(self):
        """Test function for compatibility."""
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Boxes hinged simulation example")
    parser.add_argument(
        "--mode",
        choices=["hdf5", "viewer"],
        default="viewer",
        help="Simulation mode: 'hdf5' for data collection, 'viewer' for live visualization",
    )
    parser.add_argument("--clear-cache", action="store_true", default=True, help="Clear warp cache")
    parser.add_argument("--cuda-graph", action="store_true", help="Use CUDA graphs")
    parser.add_argument("--load-from-usd", action="store_true", default=True, help="Load model from USD file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    # Add viewer arguments when in viewer mode
    parser.add_argument("--viewer", choices=["gl", "usd", "rerun", "null"], default="gl", help="Viewer type")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument("--device", type=str, help="Compute device")
    parser.add_argument("--output-path", type=str, help="Output path for USD viewer")
    parser.add_argument("--num-frames", type=int, default=1000, help="Number of frames for null/USD viewer")

    args = parser.parse_args()

    if args.mode == "hdf5":
        msg.info("Running in HDF5 mode...")
        run_hdf5_mode(
            clear_warp_cache=args.clear_cache,
            use_cuda_graph=args.cuda_graph,
            load_from_usd=args.load_from_usd,
            verbose=args.verbose,
        )
    elif args.mode == "viewer":
        msg.info("Running in ViewerGL mode...")

        # Set device if specified
        if args.device:
            wp.set_device(args.device)

        # Create viewer based on type
        if args.viewer == "gl":
            viewer = newton.viewer.ViewerGL(headless=args.headless)
        elif args.viewer == "usd":
            if args.output_path is None:
                raise ValueError("--output-path is required when using usd viewer")
            viewer = newton.viewer.ViewerUSD(output_path=args.output_path, num_frames=args.num_frames)
        elif args.viewer == "rerun":
            viewer = newton.viewer.ViewerRerun()
        elif args.viewer == "null":
            viewer = newton.viewer.ViewerNull(num_frames=args.num_frames)
        else:
            raise ValueError(f"Invalid viewer: {args.viewer}")

        # Create and run example
        example = BoxesHingedExample(viewer, load_from_usd=args.load_from_usd, use_cuda_graph=args.cuda_graph)

        # Set initial camera position for better view of the hinged boxes
        if hasattr(viewer, "set_camera"):
            # Position camera to get a good view of the hinged mechanism
            camera_pos = wp.vec3(0.5, -1.5, 0.8)
            pitch = -15.0
            yaw = 90.0
            viewer.set_camera(camera_pos, pitch, yaw)

        newton.examples.run(example)
