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

import time

import warp as wp

import newton
import newton.examples

from ...core.builder import ModelBuilder
from ...examples import print_progress_bar
from ...utils import logger as msg
from ...utils.control.rand import RandomJointController
from ...utils.device import get_device_malloc_info, get_device_spec_info
from ...utils.sim import SimulationLogger, Simulator, SimulatorSettings, ViewerKamino
from .problems import CameraConfig, ControlConfig

###
# Types
###


class BenchmarkSim:
    def __init__(
        self,
        builder: ModelBuilder,
        settings: SimulatorSettings,
        control: ControlConfig | None = None,
        camera: CameraConfig | None = None,
        device: wp.DeviceLike = None,
        use_cuda_graph: bool = False,
        max_steps: int = 1000,
        seed: int = 0,
        viewer: bool = False,
        logging: bool = False,
    ):
        # Cache the device and other internal flags
        self.builder: ModelBuilder = builder
        self.device: wp.DeviceLike = device
        self.use_cuda_graph: bool = use_cuda_graph
        self.max_steps: int = max_steps

        # Create a simulator
        msg.info("Building the simulator...")
        self.sim = Simulator(builder=builder, settings=settings, device=device)

        # Create a random-action controller for the model
        self.ctlr = RandomJointController(
            model=self.sim.model,
            seed=seed,
            decimation=control.decimation if control else None,
            scale=control.scale if control else None,
        )

        # Define a callback function to wrap the execution of the controller
        def control_callback(simulator: Simulator):
            self.ctlr.compute(
                time=simulator.solver.data.time,
                control=simulator.control,
            )

        # Set the control callbacks into the simulator
        self.sim.set_control_callback(control_callback)

        # Initialize the data logger
        self.logger: SimulationLogger | None = None
        if logging:
            msg.info("Creating the sim data logger...")
            self.logger = SimulationLogger(self.max_steps, self.sim, builder)

        # Initialize the 3D viewer
        self.viewer: ViewerKamino | None = None
        if viewer:
            msg.info("Creating the 3D viewer...")
            self.viewer = ViewerKamino(
                builder=self.builder,
                simulator=self.sim,
            )
            if hasattr(self.viewer, "set_camera") and camera is not None:
                self.viewer.set_camera(wp.vec3(*camera.position), camera.pitch, camera.yaw)

        # Declare and initialize the optional computation graphs
        # NOTE: These are used for most efficient GPU runtime
        self.reset_graph = None
        self.step_graph = None

        # Capture CUDA graph if requested and available
        self._capture()

        # Warm-start the simulator before rendering
        # NOTE: This compiles and loads the warp kernels prior to execution
        msg.info("Warming up simulator...")
        self.step_once()
        self.reset()

    ###
    # Operations
    ###

    def reset(self):
        if self.reset_graph:
            wp.capture_launch(self.reset_graph)
        else:
            self.sim.reset()
        if not self.use_cuda_graph and self.logger:
            self.logger.reset()
            self.logger.log()

    def step(self):
        if self.step_graph:
            wp.capture_launch(self.step_graph)
        else:
            self.sim.step()
        if not self.use_cuda_graph and self.logger:
            self.logger.log()

    def step_once(self):
        self.step()

    def render(self):
        if self.viewer:
            self.viewer.render_frame()

    def test(self):
        pass

    def plot(self, path: str | None = None, show: bool = False):
        if self.logger:
            self.logger.plot_solver_info(path=path, show=show)
            self.logger.plot_joint_tracking(path=path, show=show)
            self.logger.plot_solution_metrics(path=path, show=show)

    ###
    # Internals
    ###

    def _capture(self):
        if self.use_cuda_graph:
            msg.info("Running with CUDA graphs...")
            with wp.ScopedCapture(self.device) as reset_capture:
                self.sim.reset()
            self.reset_graph = reset_capture.graph
            with wp.ScopedCapture(self.device) as step_capture:
                self.sim.step()
            self.step_graph = step_capture.graph
        else:
            msg.info("Running with kernels...")


###
# Functions
###


def run_single_benchmark(
    args,
    builder: ModelBuilder,
    settings: SimulatorSettings,
    control: ControlConfig | None = None,
    camera: CameraConfig | None = None,
    device: wp.DeviceLike = None,
    use_cuda_graph: bool = True,
):
    # Create example instance
    simulator = BenchmarkSim(
        builder=builder,
        settings=settings,
        control=control,
        camera=camera,
        device=device,
        use_cuda_graph=use_cuda_graph,
        max_steps=args.num_steps,
        seed=args.seed,
        viewer=args.viewer,
        logging=args.solver_metrics,
    )

    # TODO
    if simulator.viewer:
        msg.info("Running in Viewer mode...")
        newton.examples.run(simulator, args)
    else:
        msg.info(f"Running for {simulator.max_steps} steps...")
        progress = True
        start_time = time.time()
        for i in range(simulator.max_steps):
            simulator.step_once()
            wp.synchronize()
            if progress:
                print_progress_bar(i + 1, simulator.max_steps, start_time, prefix="Progress", suffix="")
        msg.info("Finished benchmark run")

    # TODO
    spec_info = get_device_spec_info(simulator.device)
    mem_info = get_device_malloc_info(simulator.device)
    msg.info("[Device spec info]: %s", spec_info)
    msg.info("[Device malloc info]: %s", mem_info)

    # # Plot logged data after the viewer is closed
    # if args.logging:
    #     OUTPUT_PLOT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), args.problems)
    #     os.makedirs(OUTPUT_PLOT_PATH, exist_ok=True)
    #     example.plot(path=OUTPUT_PLOT_PATH, show=args.show_plots)
