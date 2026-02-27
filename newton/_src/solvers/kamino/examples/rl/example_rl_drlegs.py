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
# Example: DR Legs RL policy play-back
#
# Runs a trained RL policy on the DR Legs robot using the Kamino solver with implicit PD joint control.
#
# Usage:
#   python example_rl_drlegs.py --policy path/to/model.pt  # trained policy
#   python example_rl_drlegs.py --headless --num-steps 200
###########################################################################

import argparse
import os

import numpy as np
import torch
import warp as wp
from warp.context import Devicelike

import newton
import newton.examples
from newton._src.solvers.kamino.core.builder import ModelBuilder
from newton._src.solvers.kamino.examples import run_headless
from newton._src.solvers.kamino.linalg.linear import SolverShorthand as LinearSolverShorthand
from newton._src.solvers.kamino.models import get_examples_usd_assets_path
from newton._src.solvers.kamino.models.builders.utils import (
    add_ground_box,
    make_homogeneous_builder,
    set_uniform_body_pose_offset,
)
from newton._src.solvers.kamino.solvers.padmm import PADMMWarmStartMode
from newton._src.solvers.kamino.solvers.warmstart import WarmstarterContacts
from newton._src.solvers.kamino.utils import logger as msg
from newton._src.solvers.kamino.utils.io.usd import USDImporter
from newton._src.solvers.kamino.utils.sim import Simulator, SimulatorSettings, ViewerKamino

from .observations import DrlegsStanceObservation

###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Example class
###


class Example:
    def __init__(
        self,
        device: Devicelike = None,
        num_worlds: int = 1,
        max_steps: int = 10000,
        headless: bool = False,
        action_scale: float = 0.25,
        control_decimation: int = 4,
    ):
        # Timing: implicit PD uses larger time-steps
        self.fps = 60
        self.sim_dt = 0.01
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = max(1, round(self.frame_dt / self.sim_dt))
        self.max_steps = max_steps
        self.control_decimation = control_decimation
        self.action_scale = action_scale

        self.device = device
        self.torch_device = "cuda" if wp.get_device(device).is_cuda else "cpu"

        # USD model loading
        EXAMPLE_ASSETS_PATH = get_examples_usd_assets_path()
        if EXAMPLE_ASSETS_PATH is None:
            raise FileNotFoundError("Failed to find USD assets path for examples: ensure `newton-assets` is installed.")
        USD_MODEL_PATH = os.path.join(EXAMPLE_ASSETS_PATH, "dr_legs/usd/dr_legs_with_meshes_and_boxes.usda")

        msg.notif("Constructing builder from imported USD ...")
        importer = USDImporter()
        self.builder: ModelBuilder = make_homogeneous_builder(
            num_worlds=num_worlds,
            build_fn=importer.import_from,
            load_static_geometry=True,
            source=USD_MODEL_PATH,
            load_drive_dynamics=True,  # implicit PD control
        )

        # Place robot above the ground
        offset = wp.transformf(0.0, 0.0, 0.265, 0.0, 0.0, 0.0, 1.0)
        set_uniform_body_pose_offset(builder=self.builder, offset=offset)

        # Ground plane
        for w in range(num_worlds):
            add_ground_box(self.builder, world_index=w, layer="world")

        # Gravity
        for w in range(self.builder.num_worlds):
            self.builder.gravity[w].enabled = True

        # Solver settings # todo final update
        settings = SimulatorSettings()
        settings.dt = self.sim_dt
        settings.solver.integrator = "moreau"
        settings.solver.problem.alpha = 0.1
        settings.solver.padmm.primal_tolerance = 1e-4
        settings.solver.padmm.dual_tolerance = 1e-4
        settings.solver.padmm.compl_tolerance = 1e-4
        settings.solver.padmm.max_iterations = 200
        settings.solver.padmm.eta = 1e-5
        settings.solver.padmm.rho_0 = 0.05
        settings.solver.use_solver_acceleration = True
        settings.solver.warmstart_mode = PADMMWarmStartMode.CONTAINERS
        settings.solver.contact_warmstart_method = WarmstarterContacts.Method.GEOM_PAIR_NET_FORCE
        settings.solver.collect_solver_info = False
        settings.solver.compute_metrics = False
        linear_solver_cls = {v: k for k, v in LinearSolverShorthand.items()}["LLTB"]
        settings.solver.linear_solver_type = linear_solver_cls
        settings.solver.linear_solver_kwargs = {}
        settings.solver.angular_velocity_damping = 0.0

        # Simulator
        msg.notif("Building the simulator...")
        self.sim = Simulator(builder=self.builder, settings=settings, device=device)

        # Extract actuated joint indices
        self.actuated_dof_indices: list[int] = []
        self.actuated_joint_names: list[str] = []
        num_joints_per_world = self.sim.model.size.max_of_num_joints
        dof_offset = 0
        for j in range(num_joints_per_world):
            joint = self.builder.joints[j]
            if joint.is_actuated:
                self.actuated_joint_names.append(joint.name)
                for dof_idx in range(joint.num_dofs):
                    self.actuated_dof_indices.append(dof_offset + dof_idx)
            dof_offset += joint.num_dofs

        self.num_actuated = len(self.actuated_dof_indices)
        msg.info(f"Actuated joints ({self.num_actuated}): {self.actuated_joint_names}")

        self.actuated_dof_indices_tensor = torch.tensor(
            self.actuated_dof_indices, device=self.torch_device, dtype=torch.long
        )

        # Observation
        self.obs_builder = DrlegsStanceObservation(
            sim=self.sim,
            num_worlds=num_worlds,
            device=self.torch_device,
            actuated_joint_indices=self.actuated_dof_indices,
            num_actions=self.num_actuated,
        )
        msg.info(f"Observation dim: {self.obs_builder.num_observations}")

        # Zero-copy torch views of control arrays
        num_joint_coords = self.sim.model.size.max_of_num_joint_coords
        num_joint_dofs = self.sim.model.size.max_of_num_joint_dofs
        self.q_j_ref_pt = wp.to_torch(self.sim.control.q_j_ref).reshape(num_worlds, num_joint_coords)
        self.dq_j_ref_pt = wp.to_torch(self.sim.control.dq_j_ref).reshape(num_worlds, num_joint_dofs)

        # Default joint positions and action buffer
        self.default_q_j = wp.to_torch(self.sim.state.q_j).reshape(num_worlds, num_joint_coords).clone()
        self.actions = torch.zeros((num_worlds, self.num_actuated), device=self.torch_device, dtype=torch.float32)

        # Policy (None = random actions)
        self.policy = None

        # Keyboard state
        self._reset_key_prev = False

        # Viewer
        self.viewer: ViewerKamino | None = None
        if not headless:
            msg.notif("Creating the 3D viewer...")
            self.viewer = ViewerKamino(
                builder=self.builder,
                simulator=self.sim,
            )

        # Warm-up
        msg.notif("Warming up simulator...")
        self.step_once()
        self.reset()

    # Simulation helpers

    def _apply_actions(self):
        """Convert policy actions to implicit PD joint position references."""
        # Start from the default pose
        self.q_j_ref_pt[:] = self.default_q_j
        # Add scaled actions at actuated joint indices only
        self.q_j_ref_pt[:, self.actuated_dof_indices_tensor] += self.action_scale * self.actions
        # Zero velocity reference
        self.dq_j_ref_pt.zero_()

    def reset(self):
        """Reset the simulation and internal state."""
        self.sim.reset()
        self.actions.zero_()
        self.obs_builder.reset()
        # Set default pose as the initial reference
        self.q_j_ref_pt[:] = self.default_q_j
        self.dq_j_ref_pt.zero_()

    def step_once(self):
        """Single physics step (used by run_headless warm-up)."""
        self.sim.step()

    def step(self):
        """One RL step: observe - infer - apply - simulate."""
        # Keyboard handling
        if self.viewer is not None and hasattr(self.viewer, "is_key_down"):
            reset_down = bool(self.viewer.is_key_down("p"))
            if reset_down and not self._reset_key_prev:
                self.reset()
            self._reset_key_prev = reset_down

        # Compute observation from current state
        obs = self.obs_builder.compute(actions=self.actions)

        # Policy inference
        with torch.no_grad():
            if self.policy is not None:
                self.actions[:] = self.policy(obs)
            else:
                # Random actions in [-1, 1] if no policy provided # todo for testing only, remove later
                self.actions[:] = 2.0 * torch.rand_like(self.actions) - 1.0

        # Write action targets to implicit PD controller
        self._apply_actions()

        # Step physics for control_decimation substeps
        for _ in range(self.control_decimation):
            for _ in range(self.sim_substeps):
                self.sim.step()

    def render(self):
        """Render the current frame."""
        if self.viewer is not None:
            self.viewer.render_frame()

    def test(self):
        """Test function for compatibility."""
        pass


###
# Main function
###

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DR Legs RL play example")
    parser.add_argument("--device", type=str, help="The compute device to use")
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run in headless mode",
    )
    parser.add_argument("--num-worlds", type=int, default=1, help="Number of worlds")
    parser.add_argument("--num-steps", type=int, default=10000, help="Steps for headless mode")
    parser.add_argument("--action-scale", type=float, default=0.25, help="Action scaling factor")
    parser.add_argument(
        "--control-decimation",
        type=int,
        default=4,
        help="Number of physics substeps per RL step",
    )
    parser.add_argument("--policy", type=str, default=None, help="Path to a TorchScript policy .pt file")
    parser.add_argument(
        "--clear-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Clear warp cache",
    )
    parser.add_argument(
        "--test",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run tests",
    )
    args = parser.parse_args()

    np.set_printoptions(linewidth=20000, precision=6, threshold=10000, suppress=True)

    if args.clear_cache:
        wp.clear_kernel_cache()
        wp.clear_lto_cache()

    msg.set_log_level(msg.LogLevel.INFO)

    if args.device:
        device = wp.get_device(args.device)
        wp.set_device(device)
    else:
        device = wp.get_preferred_device()

    msg.info(f"device: {device}")

    example = Example(
        device=device,
        num_worlds=args.num_worlds,
        max_steps=args.num_steps,
        headless=args.headless,
        action_scale=args.action_scale,
        control_decimation=args.control_decimation,
    )

    # Load trained policy if provided
    if args.policy:
        example.policy = torch.jit.load(args.policy, map_location=example.torch_device)
        msg.info(f"Loaded policy from: {args.policy}")
    else:
        msg.info("No policy provided — using random actions")

    if args.headless:
        msg.notif("Running in headless mode...")
        run_headless(example, progress=True)
    else:
        msg.notif("Running in Viewer mode...")
        if hasattr(example.viewer, "set_camera"):
            example.viewer.set_camera(wp.vec3(0.6, 0.6, 0.3), -10.0, 225.0)
        newton.examples.run(example, args)
