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

import numpy as np
import warp as wp

import newton
import newton._src.solvers.kamino.utils.logger as msg
import newton.examples
from newton._src.solvers.kamino.core.builder import ModelBuilder
from newton._src.solvers.kamino.core.shapes import ShapeType
from newton._src.solvers.kamino.core.types import float32, vec6f
from newton._src.solvers.kamino.models import get_primitives_usd_assets_path
from newton._src.solvers.kamino.models.builders import build_boxes_hinged
from newton._src.solvers.kamino.simulation.simulator import Simulator, SimulatorSettings
from newton._src.solvers.kamino.utils.io.usd import USDImporter
from newton._src.solvers.kamino.utils.print import print_progress_bar
from newton._src.solvers.kamino.viewer import ViewerKamino

###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Kernels
###


@wp.kernel
def _control_callback(
    model_dt: wp.array(dtype=float32),
    state_t: wp.array(dtype=float32),
    state_w_e_i: wp.array(dtype=vec6f),
    control_tau_j: wp.array(dtype=float32),
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
    t = state_t[wid]

    # Apply a time-dependent external force
    if t > t_start and t < t_end:
        control_tau_j[jid] = -3.0
    else:
        control_tau_j[jid] = 0.0


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
            sim.data.solver.time.time,
            sim.data.solver.bodies.w_e_i,
            sim.data.control_n.tau_j,
        ],
    )


###
# Constants
###

# Set the path to the external USD assets
USD_MODEL_PATH = os.path.join(get_primitives_usd_assets_path(), "boxes_hinged.usda")


###
# Main function
###


def run_headless(use_cuda_graph=False, load_from_usd=False):
    """Run the simulation in headless mode."""

    # Set global numpy configurations
    np.set_printoptions(linewidth=20000, precision=6, threshold=10000, suppress=True)  # Suppress scientific notation

    # Get the default warp device
    device = wp.get_preferred_device()
    device = wp.get_device(device)

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

    # Set solver settings
    settings = SimulatorSettings()
    settings.dt = 0.001
    settings.solver.primal_tolerance = 1e-6
    settings.solver.dual_tolerance = 1e-6
    settings.solver.compl_tolerance = 1e-6
    settings.solver.max_iterations = 200
    settings.solver.rho_0 = 1.0

    # Create a simulator
    msg.info("Building the simulator...")
    sim = Simulator(builder=builder, settings=settings, device=device)
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
        wp.capture_launch(step_graph)
        wp.capture_launch(reset_graph)
    else:
        msg.info("Running with kernels...")
        with wp.ScopedDevice(device):
            sim.step()
            sim.reset()

    # Step the simulation and collect frames
    ns = 10000
    msg.info(f"Collecting ns={ns} frames...")
    start_time = time.time()
    with wp.ScopedDevice(device):
        for i in range(ns):
            if use_cuda_graph:
                wp.capture_launch(step_graph)
            else:
                sim.step()
            wp.synchronize()
            print_progress_bar(i, ns, start_time, prefix="Progress", suffix="")


class Example:
    """ViewerGL example class for boxes hinged simulation."""

    def __init__(self, load_from_usd=True, use_cuda_graph=False):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.use_cuda_graph = use_cuda_graph

        # Get the default warp device
        device = wp.get_preferred_device()
        device = wp.get_device(device)

        # Create a single-instance system
        if load_from_usd:
            msg.info("Constructing builder from imported USD ...")
            importer = USDImporter()
            self.builder: ModelBuilder = importer.import_from(source=USD_MODEL_PATH, load_static_geometry=True)
        else:
            msg.info("Constructing builder using generator ...")
            self.builder = ModelBuilder()
            build_boxes_hinged(builder=self.builder, z_offset=0.0, ground=True)

        # Set gravity
        self.builder.gravity.enabled = True

        # Set solver settings
        settings = SimulatorSettings()
        settings.dt = 0.001
        settings.solver.primal_tolerance = 1e-6
        settings.solver.dual_tolerance = 1e-6
        settings.solver.compl_tolerance = 1e-6
        settings.solver.max_iterations = 200
        settings.solver.rho_0 = 1.0

        # Create a simulator
        msg.info("Building the simulator...")
        self.sim = Simulator(builder=self.builder, settings=settings, device=device)
        self.sim.set_control_callback(control_callback)

        # Initialize the viewer
        self.viewer = ViewerKamino(
            builder=self.builder,
            simulator=self.sim,
        )

        # Initialize the simulator with a warm-up step
        self.sim.reset()

        # Capture CUDA graph if requested and available
        self.capture()

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
        for _i in range(self.sim_substeps):
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
        self.viewer.render_frame()

    def test(self):
        """Test function for compatibility."""
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Boxes hinged simulation example")
    parser.add_argument(
        "--mode",
        choices=["headless", "viewer"],
        default="viewer",
        help="Simulation mode: 'headless' for raw simulation, 'viewer' for live visualization",
    )
    parser.add_argument("--clear-cache", action="store_true", default=False, help="Clear warp cache")
    parser.add_argument("--cuda-graph", action="store_true", default=True, help="Use CUDA graphs")
    parser.add_argument("--load-from-usd", action="store_true", default=True, help="Load model from USD file")
    parser.add_argument("--viewer", choices=["gl", "usd", "rerun", "null"], default="gl", help="Viewer type")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument("--device", type=str, help="Compute device")
    parser.add_argument("--output-path", type=str, help="Output path for USD viewer")
    parser.add_argument("--num-frames", type=int, default=1000, help="Number of frames for null/USD viewer")
    parser.add_argument("--test", action="store_true", default=False, help="Run tests")
    args = parser.parse_args()

    # Clear warp cache if requested
    if args.clear_cache:
        wp.clear_kernel_cache()
        wp.clear_lto_cache()

    # Execute based on mode
    if args.mode == "headless":
        msg.info("Running in HDF5 mode...")
        run_headless(use_cuda_graph=args.cuda_graph, load_from_usd=args.load_from_usd)

    elif args.mode == "viewer":
        msg.info("Running in ViewerGL mode...")

        # Set device if specified
        if args.device:
            wp.set_device(args.device)

        # # Create viewer based on type
        # if args.viewer == "gl":
        #     viewer = newton.viewer.ViewerGL(headless=args.headless)
        # elif args.viewer == "usd":
        #     if args.output_path is None:
        #         raise ValueError("--output-path is required when using usd viewer")
        #     viewer = newton.viewer.ViewerUSD(output_path=args.output_path, num_frames=args.num_frames)
        # elif args.viewer == "rerun":
        #     viewer = newton.viewer.ViewerRerun()
        # elif args.viewer == "null":
        #     viewer = newton.viewer.ViewerNull(num_frames=args.num_frames)
        # else:
        #     raise ValueError(f"Invalid viewer: {args.viewer}")

        # Create and run example
        example = Example(load_from_usd=args.load_from_usd, use_cuda_graph=args.cuda_graph)

        # Set initial camera position for better view of the hinged boxes
        if hasattr(example.viewer, "set_camera"):
            # Position camera to get a good view of the hinged mechanism
            camera_pos = wp.vec3(0.5, -1.5, 0.8)
            pitch = -15.0
            yaw = 90.0
            example.viewer.set_camera(camera_pos, pitch, yaw)

        newton.examples.run(example, args)
