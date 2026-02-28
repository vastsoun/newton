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
from newton._src.solvers.kamino.examples import run_headless
from newton._src.solvers.kamino.examples.rl.observations import DrlegsStanceObservation
from newton._src.solvers.kamino.examples.rl.simulation import RigidBodySim
from newton._src.solvers.kamino.models import get_examples_usd_assets_path
from newton._src.solvers.kamino.utils import logger as msg

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
        control_decimation: int = 1,
    ):
        # Timing
        self.sim_dt = 0.01
        self.max_steps = max_steps
        self.control_decimation = control_decimation
        self.action_scale = action_scale

        # USD model path
        EXAMPLE_ASSETS_PATH = get_examples_usd_assets_path()
        if EXAMPLE_ASSETS_PATH is None:
            raise FileNotFoundError("Failed to find USD assets path for examples: ensure `newton-assets` is installed.")
        USD_MODEL_PATH = os.path.join(EXAMPLE_ASSETS_PATH, "dr_legs/usd/dr_legs_with_meshes_and_boxes.usda")

        # Create generic articulated body simulator
        self.body_sim = RigidBodySim(
            usd_model_path=USD_MODEL_PATH,
            num_worlds=num_worlds,
            sim_dt=self.sim_dt,
            device=device,
            headless=headless,
            body_pose_offset=(0.0, 0.0, 0.265, 0.0, 0.0, 0.0, 1.0),
            use_cuda_graph=True,
        )

        # Observation builder (DR Legs specific)
        self.obs_builder = DrlegsStanceObservation(
            body_sim=self.body_sim,
            action_scale=action_scale,
        )
        msg.info(f"Observation dim: {self.obs_builder.num_observations}")

        # Action buffer
        self.actions = torch.zeros(
            (num_worlds, self.body_sim.num_actuated),
            device=self.body_sim.torch_device,
            dtype=torch.float32,
        )

        # Policy (None = random actions)
        self.policy = None

        # Keyboard state
        self._reset_key_prev = False

    # Convenience accessors for the main block
    @property
    def torch_device(self) -> str:
        return self.body_sim.torch_device

    @property
    def viewer(self):
        return self.body_sim.viewer

    # Simulation helpers

    def _apply_actions(self):
        """Convert policy actions to implicit PD joint position references."""
        self.body_sim.q_j_ref.zero_()
        self.body_sim.q_j_ref[:, self.body_sim.actuated_dof_indices_tensor] = self.action_scale * self.actions
        self.body_sim.dq_j_ref.zero_()

    def reset(self):
        """Reset the simulation and internal state."""
        self.body_sim.reset()
        self.actions.zero_()
        self.obs_builder.reset()
        self.body_sim.q_j_ref.zero_()
        self.body_sim.dq_j_ref.zero_()

    def step_once(self):
        """Single physics step (used by run_headless warm-up)."""
        self.body_sim.step()

    def step(self):
        """One RL step: observe - infer - apply - simulate."""
        # Keyboard handling
        if self.body_sim.viewer is not None and hasattr(self.body_sim.viewer, "is_key_down"):
            reset_down = bool(self.body_sim.viewer.is_key_down("p"))
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
            self.body_sim.step()

    def render(self):
        """Render the current frame."""
        self.body_sim.render()

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
        default=1,
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
