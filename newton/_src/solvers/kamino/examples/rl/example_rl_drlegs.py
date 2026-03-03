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
# Example: DR Legs walk policy play-back
#
# Runs a trained walk RL policy on the DR Legs robot using the Kamino
# solver with implicit PD joint control.  Velocity commands come from an
# Xbox gamepad or keyboard via the 3-D viewer.
#
# The policy expects 72D observations:
#   base (63D) + phase encoding (4D) + cmd_vel (2D) + root_lin_vel (3D)
#
# Usage:
#   python example_rl_drlegs.py --policy path/to/model.pt
#   python example_rl_drlegs.py --policy path/to/model.pt --mode async
#   python example_rl_drlegs.py --headless --num-steps 200
###########################################################################

import argparse
import os

import numpy as np
import torch
import warp as wp
from warp.context import Devicelike

from newton._src.solvers.kamino.core.joints import JointActuationType
from newton._src.solvers.kamino.examples import run_headless
from newton._src.solvers.kamino.examples.rl.joystick import JoystickConfig, JoystickController
from newton._src.solvers.kamino.examples.rl.observations import DrlegsBaseObservation
from newton._src.solvers.kamino.examples.rl.simulation import RigidBodySim
from newton._src.solvers.kamino.examples.rl.simulation_runner import SimulationRunner
from newton._src.solvers.kamino.examples.rl.utils import _load_policy_checkpoint, periodic_encoding
from newton._src.solvers.kamino.models import get_examples_usd_assets_path
from newton._src.solvers.kamino.utils import logger as msg

###
# Module configs
###

wp.set_module_options({"enable_backward": False})

# ---------------------------------------------------------------------------
# Walk task constants
# ---------------------------------------------------------------------------
_ACTION_SCALE = 0.4  # joint position scale (rad) for walk policy
_CONTACT_DURATION = 0.35  # seconds per foot contact phase
_PHASE_RATE = 1.0 / (2.0 * _CONTACT_DURATION)  # ~1.4286 Hz gait frequency
_PHASE_EMBEDDING_K = 2  # periodic encoding order -> 4D embedding
_VEL_CMD_MAX = 0.3  # m/s max velocity command
_PD_KP = 15.0  # proportional gain (N·m/rad) for actuated joints
_PD_KD = 0.6  # derivative gain (N·m·s/rad) for actuated joints
_PD_ARMATURE = 0.01  # rotor inertia (kg·m²) for actuated joints


###
# Example class
###


class Example:
    def __init__(
        self,
        device: Devicelike = None,
        policy=None,
        headless: bool = False,
        sim_dt: float = 0.01,
        control_decimation: int = 1,
        max_steps: int = 10000,
    ):
        # Timing
        self.sim_dt = sim_dt
        self.control_decimation = control_decimation
        self.env_dt = sim_dt * control_decimation
        self.max_steps = max_steps
        num_worlds = 1

        # USD model path
        EXAMPLE_ASSETS_PATH = get_examples_usd_assets_path()
        if EXAMPLE_ASSETS_PATH is None:
            raise FileNotFoundError("Failed to find USD assets path for examples: ensure `newton-assets` is installed.")
        USD_MODEL_PATH = os.path.join(EXAMPLE_ASSETS_PATH, "dr_legs/usd/dr_legs_with_meshes_and_boxes.usda")

        # Create generic articulated body simulator
        self.sim_wrapper = RigidBodySim(
            usd_model_path=USD_MODEL_PATH,
            num_worlds=num_worlds,
            sim_dt=self.sim_dt,
            device=device,
            headless=headless,
            body_pose_offset=(0.0, 0.0, 0.265, 0.0, 0.0, 0.0, 1.0),
            use_cuda_graph=True,
        )

        # Override implicit PD gains to match training config exactly
        act_type = wp.to_torch(self.sim_wrapper.sim.model.joints.act_type)
        k_p = wp.to_torch(self.sim_wrapper.sim.model.joints.k_p_j)
        k_d = wp.to_torch(self.sim_wrapper.sim.model.joints.k_d_j)
        a_j = wp.to_torch(self.sim_wrapper.sim.model.joints.a_j)
        b_j = wp.to_torch(self.sim_wrapper.sim.model.joints.b_j)
        actuated_mask = act_type != JointActuationType.PASSIVE
        k_p[actuated_mask] = _PD_KP
        k_d[actuated_mask] = _PD_KD
        a_j[actuated_mask] = _PD_ARMATURE
        k_p[~actuated_mask] = 0.0
        k_d[~actuated_mask] = 0.0
        b_j.fill_(0.0)

        # Observation builder (63D base)
        self.obs_builder = DrlegsBaseObservation(
            body_sim=self.sim_wrapper,
            action_scale=_ACTION_SCALE,
        )

        # Phase clock for gait timing
        self._phase = torch.zeros(num_worlds, device=self.torch_device, dtype=torch.float32)
        freq_2pi, offset = periodic_encoding(k=_PHASE_EMBEDDING_K)
        self._freq_2pi = torch.from_numpy(freq_2pi).float().to(self.torch_device)
        self._offset = torch.from_numpy(offset).float().to(self.torch_device)
        self._phase_enc = torch.zeros(num_worlds, _PHASE_EMBEDDING_K * 2, device=self.torch_device, dtype=torch.float32)

        # Command velocity buffer (filled by joystick each step)
        self._cmd_vel = torch.zeros(num_worlds, 2, device=self.torch_device, dtype=torch.float32)

        # Full observation buffer (72D = 63 + 4 + 2 + 3)
        obs_dim = self.obs_builder.num_observations + _PHASE_EMBEDDING_K * 2 + 2 + 3
        self._obs_buffer = torch.zeros(num_worlds, obs_dim, device=self.torch_device, dtype=torch.float32)
        msg.info(f"Observation dim: {obs_dim}")

        # Action buffer (12 actuated joints)
        self.actions = torch.zeros(
            (num_worlds, self.sim_wrapper.num_actuated),
            device=self.torch_device,
            dtype=torch.float32,
        )

        # Joystick for velocity commands
        self.joystick = JoystickController(
            dt=self.env_dt,
            viewer=self.sim_wrapper.viewer,
            num_worlds=num_worlds,
            device=self.torch_device,
            config=JoystickConfig(
                forward_velocity_base=_VEL_CMD_MAX,
                forward_velocity_turbo=0.0,
                lateral_velocity_base=_VEL_CMD_MAX,
                lateral_velocity_turbo=0.0,
                angular_velocity_base=0.0,
                angular_velocity_turbo=0.0,
            ),
        )

        # Policy (None = random actions)
        self.policy = policy

    # Convenience accessors
    @property
    def torch_device(self) -> str:
        return self.sim_wrapper.torch_device

    @property
    def viewer(self):
        return self.sim_wrapper.viewer

    # Simulation helpers

    def _apply_actions(self):
        """Convert policy actions to implicit PD joint position references."""
        self.sim_wrapper.q_j_ref.zero_()
        self.sim_wrapper.q_j_ref[:, self.sim_wrapper.actuated_dof_indices_tensor] = _ACTION_SCALE * self.actions
        self.sim_wrapper.dq_j_ref.zero_()

    def reset(self):
        """Reset the simulation and internal state."""
        self.sim_wrapper.reset()
        self.actions.zero_()
        self.obs_builder.reset()
        self._phase.zero_()
        self._cmd_vel.zero_()
        self.sim_wrapper.q_j_ref.zero_()
        self.sim_wrapper.dq_j_ref.zero_()
        self.joystick.reset()

    def step_once(self):
        """Single physics step (used by run_headless warm-up)."""
        self.sim_wrapper.step()

    def update_input(self):
        """Transfer joystick velocity commands to the command buffer."""
        self._cmd_vel[0, 0] = self.joystick.forward_velocity
        self._cmd_vel[0, 1] = self.joystick.lateral_velocity

    def sim_step(self):
        """Observations -> policy inference -> actions -> physics step."""
        # Advance phase clock
        self._phase.add_(self.env_dt * _PHASE_RATE).remainder_(1.0)

        # Base observation (63D)
        base_obs = self.obs_builder.compute(actions=self.actions)

        # Phase encoding (4D): sin(phase * freq_2pi + offset)
        torch.sin(torch.outer(self._phase, self._freq_2pi).add_(self._offset), out=self._phase_enc)

        # Root linear velocity (3D, world frame)
        root_lin_vel = self.sim_wrapper.u_i[:, 0, :3]

        # Build full 72D observation
        d_base = base_obs.shape[1]
        d_phase = self._phase_enc.shape[1]
        self._obs_buffer[:, :d_base] = base_obs
        self._obs_buffer[:, d_base : d_base + d_phase] = self._phase_enc
        self._obs_buffer[:, d_base + d_phase : d_base + d_phase + 2] = self._cmd_vel
        self._obs_buffer[:, d_base + d_phase + 2 :] = root_lin_vel

        # Policy inference
        with torch.no_grad():
            if self.policy is not None:
                self.actions[:] = self.policy(self._obs_buffer)
            else:
                self.actions[:] = 2.0 * torch.rand_like(self.actions) - 1.0

        # Write action targets to implicit PD controller
        self._apply_actions()

        # Step physics for control_decimation substeps
        for _ in range(self.control_decimation):
            self.sim_wrapper.step()

    def step(self):
        """One RL step: check reset -> joystick -> observe -> infer -> apply -> simulate."""
        if self.joystick.check_reset():
            self.reset()
        self.joystick.update()
        self.update_input()
        self.sim_step()

    def render(self):
        """Render the current frame."""
        self.sim_wrapper.render()


###
# Main function
###

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DR Legs walk policy play example")
    parser.add_argument("--device", type=str, help="The compute device to use")
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run in headless mode",
    )
    parser.add_argument("--num-steps", type=int, default=10000, help="Steps for headless mode")
    parser.add_argument(
        "--control-decimation",
        type=int,
        default=5,
        help="Number of physics substeps per RL step",
    )
    parser.add_argument("--sim-dt", type=float, default=0.004, help="Physics substep duration in seconds")
    parser.add_argument("--policy", type=str, default=None, help="Path to an rsl_rl checkpoint .pt file")
    parser.add_argument(
        "--mode",
        choices=["sync", "async"],
        default="sync",
        help="Sim loop mode: sync (default) or async",
    )
    parser.add_argument(
        "--render-fps",
        type=float,
        default=30.0,
        help="Target render FPS for async mode (default: 30)",
    )
    args = parser.parse_args()

    np.set_printoptions(linewidth=20000, precision=6, threshold=10000, suppress=True)
    msg.set_log_level(msg.LogLevel.INFO)

    if args.device:
        device = wp.get_device(args.device)
        wp.set_device(device)
    else:
        device = wp.get_preferred_device()

    msg.info(f"device: {device}")

    # Convert warp device to torch device string
    torch_device = "cuda" if device.is_cuda else "cpu"

    # Load trained policy if provided
    policy = None
    if args.policy:
        policy = _load_policy_checkpoint(args.policy, device=torch_device)
        msg.info(f"Loaded policy from: {args.policy}")
    else:
        msg.info("No policy provided -- using random actions")

    example = Example(
        device=device,
        policy=policy,
        headless=args.headless,
        sim_dt=args.sim_dt,
        control_decimation=args.control_decimation,
        max_steps=args.num_steps,
    )

    try:
        if args.headless:
            msg.notif("Running in headless mode...")
            run_headless(example, progress=True)
        else:
            msg.notif(f"Running in Viewer mode ({args.mode})...")
            if hasattr(example.viewer, "set_camera"):
                example.viewer.set_camera(wp.vec3(0.6, 0.6, 0.3), -10.0, 225.0)
            SimulationRunner(example, mode=args.mode, render_fps=args.render_fps).run()
    except KeyboardInterrupt:
        pass
    finally:
        example.joystick.close()
