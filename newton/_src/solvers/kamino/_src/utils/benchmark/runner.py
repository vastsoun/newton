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
import gc
import time
from collections.abc import Callable

import warp as wp

import newton
import newton.examples

from ....examples import print_progress_bar
from ...core.builder import ModelBuilderKamino
from ...utils import logger as msg
from ...utils.control.rand import RandomJointController
from ...utils.device import _fmt_bytes, get_device_malloc_info
from ...utils.sim import SimulationLogger, Simulator, ViewerKamino
from .metrics import BenchmarkMetrics
from .problems import CameraConfig, ControlConfig, ProblemDimensions

###
# Types
###


class BenchmarkSim:
    def __init__(
        self,
        builder: ModelBuilderKamino,
        config: Simulator.Config,
        control: ControlConfig | None = None,
        camera: CameraConfig | None = None,
        device: wp.DeviceLike = None,
        use_cuda_graph: bool = False,
        max_steps: int = 1000,
        seed: int = 0,
        viewer: bool = False,
        logging: bool = False,
        physics_metrics: bool = False,
    ):
        # Cache the device and other internal flags
        self.builder: ModelBuilderKamino = builder
        self.device: wp.DeviceLike = device
        self.use_cuda_graph: bool = use_cuda_graph
        self.max_steps: int = max_steps

        # Override the default compute_metrics toggle in the
        # simulator config based on the benchmark configuration
        config.solver.compute_metrics = physics_metrics

        # Create a simulator
        msg.info("Building the simulator...")
        self.sim = Simulator(builder=builder, config=config, device=device)

        if control is None or not control.disable_controller:
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


def run_single_benchmark_with_viewer(
    args: argparse.Namespace,
    simulator: BenchmarkSim,
) -> tuple[float, float]:
    start_time = time.time()
    newton.examples.run(simulator, args)
    stop_time = time.time()
    return start_time, stop_time


def run_single_benchmark_with_progress(
    simulator: BenchmarkSim,
) -> tuple[float, float]:
    start_time = time.time()
    for step_idx in range(simulator.max_steps):
        simulator.step_once()
        wp.synchronize()
        print_progress_bar(step_idx + 1, simulator.max_steps, start_time, prefix="Progress", suffix="")
    stop_time = time.time()
    return start_time, stop_time


def run_single_benchmark_silent(
    simulator: BenchmarkSim,
) -> tuple[float, float]:
    start_time = time.time()
    for _s in range(simulator.max_steps):
        simulator.step_once()
        wp.synchronize()
    stop_time = time.time()
    return start_time, stop_time


def run_single_benchmark_with_step_metrics(
    problem_idx: int,
    config_idx: int,
    simulator: BenchmarkSim,
    metrics: BenchmarkMetrics,
) -> tuple[float, float]:
    start_time = time.time()
    step_start_time = float(start_time)
    for step_idx in range(simulator.max_steps):
        simulator.step_once()
        wp.synchronize()
        step_stop_time = time.time()
        metrics.record_step(problem_idx, config_idx, step_idx, step_stop_time - step_start_time, simulator.sim.solver)
        step_start_time = float(step_stop_time)
    stop_time = time.time()
    return start_time, stop_time


def estimate_max_num_worlds(
    args: argparse.Namespace,
    builder_fn: Callable[[int], ModelBuilderKamino],
    config: Simulator.Config,
    control: ControlConfig | None = None,
    camera: CameraConfig | None = None,
    device: wp.DeviceLike = None,
    use_cuda_graph: bool = True,
    physics_metrics: bool = False,
):
    assert device.is_cuda

    gc.collect()
    wp.synchronize()

    # Create simulator for a single world and measure memory usage
    config.collision_detector.max_contacts = 10 * 1
    builder = builder_fn(1)
    simulator = BenchmarkSim(
        builder=builder,
        config=config,
        control=control,
        camera=camera,
        device=device,
        use_cuda_graph=use_cuda_graph,
        max_steps=args.num_steps,
        seed=args.seed,
        viewer=args.viewer,
        physics_metrics=physics_metrics,
    )
    memory_1_world = float(wp.get_mempool_used_mem_current(device))

    # Deallocate memory
    del builder
    del simulator
    gc.collect()
    wp.synchronize()

    # Create simulator for a 10 worlds and measure memory usage
    config.collision_detector.max_contacts = 10 * 10
    builder = builder_fn(10)
    simulator = BenchmarkSim(
        builder=builder,
        config=config,
        control=control,
        camera=camera,
        device=device,
        use_cuda_graph=use_cuda_graph,
        max_steps=args.num_steps,
        seed=args.seed,
        viewer=args.viewer,
        physics_metrics=physics_metrics,
    )
    memory_10_worlds = float(wp.get_mempool_used_mem_current(device))

    # Deallocate memory
    del builder
    del simulator
    gc.collect()
    wp.synchronize()

    # Compute constant and variable memory cost per world
    cost_per_world = (memory_10_worlds - memory_1_world) / (10 - 1)
    constant_cost = memory_1_world - 1 * cost_per_world

    # Estimate maximal number of worlds (so as to consume 95% of the free memory)
    memory_max = 0.95 * device.free_memory
    num_worlds_max = int((memory_max - constant_cost) / cost_per_world)

    msg.notif(
        f"Memory consumption for 1 world: {_fmt_bytes(memory_1_world)}; for 10 worlds: {_fmt_bytes(memory_10_worlds)}"
    )
    msg.notif(f"Estimated constant cost: {_fmt_bytes(constant_cost)}, per-world cost: {_fmt_bytes(cost_per_world)}")
    msg.notif(f"Total free memory (with 5% margin): {_fmt_bytes(memory_max)}, estimated max worlds: {num_worlds_max}")

    return num_worlds_max


def run_single_benchmark(
    problem_idx: int,
    config_idx: int,
    metrics: BenchmarkMetrics,
    args: argparse.Namespace,
    builder: ModelBuilderKamino,
    config: Simulator.Config,
    control: ControlConfig | None = None,
    camera: CameraConfig | None = None,
    device: wp.DeviceLike = None,
    use_cuda_graph: bool = True,
    print_device_info: bool = False,
    progress: bool = False,
):
    gc.collect()

    # Create example instance
    simulator = BenchmarkSim(
        builder=builder,
        config=config,
        control=control,
        camera=camera,
        device=device,
        use_cuda_graph=use_cuda_graph,
        max_steps=args.num_steps,
        seed=args.seed,
        viewer=args.viewer,
        physics_metrics=metrics.physics_metrics is not None,
    )

    msg.info("Starting benchmark run...")
    if simulator.viewer:
        msg.info("Running in Viewer mode...")
        start_time, stop_time = run_single_benchmark_with_viewer(args, simulator)
    else:
        msg.info(f"Running for {simulator.max_steps} steps...")
        if metrics.step_time is not None:
            start_time, stop_time = run_single_benchmark_with_step_metrics(problem_idx, config_idx, simulator, metrics)
        elif progress:
            start_time, stop_time = run_single_benchmark_with_progress(simulator)
        else:
            start_time, stop_time = run_single_benchmark_silent(simulator)
    msg.info("Finished benchmark run.")

    # Record final metrics for the benchmark run
    metrics.record_total(
        problem_idx=problem_idx,
        config_idx=config_idx,
        total_time=stop_time - start_time,
        total_steps=int(simulator.sim.solver.data.time.steps.numpy()[0]),
        memory_used=float(wp.get_mempool_used_mem_current(device) if device.is_cuda else 0.0),
    )

    # Record problem dimensions
    problem_name = metrics._problem_names[problem_idx]
    if problem_name not in metrics._problem_dims:
        metrics._problem_dims[problem_name] = ProblemDimensions(
            num_body_dofs=simulator.sim.model.size.max_of_num_body_dofs,
            num_joint_dofs=simulator.sim.model.size.max_of_num_joint_dofs,
            min_delassus_dim=simulator.sim.model.size.max_of_num_kinematic_joint_cts
            + simulator.sim.model.size.max_of_num_dynamic_joint_cts,
            max_delassus_dim=simulator.sim.model.size.max_of_max_total_cts,
            num_worlds=simulator.sim.model.size.num_worlds,
        )

    # Optionally also print the total device memory allocated during the benchmark run
    if print_device_info:
        mem_info = get_device_malloc_info(simulator.device)
        msg.info("[Device malloc info]: %s", mem_info)

    # Deallocate simulator to ensure accurate memory consumption measure for the next run
    del simulator
    gc.collect()
    wp.synchronize()
