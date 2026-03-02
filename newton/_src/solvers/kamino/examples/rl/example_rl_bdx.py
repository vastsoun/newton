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
# Example: BDX RL policy play-back
#
# Runs a trained RL walking policy on the BDX humanoid robot using the
# Kamino solver with implicit PD joint control.  Velocity commands come
# from an Xbox gamepad or, when no gamepad is connected, from keyboard
# input via the 3-D viewer.
#
# Usage:
#   python example_rl_bdx.py
###########################################################################

# Python
import argparse
import os

# Thirdparty
import newton
import newton.examples
import numpy as np
import torch
import warp as wp
from newton._src.solvers.kamino.examples import run_headless
from newton._src.solvers.kamino.examples.rl.joystick import JoystickController
from newton._src.solvers.kamino.examples.rl.observations import BdxObservation
from newton._src.solvers.kamino.examples.rl.simulation import RigidBodySim
from newton._src.solvers.kamino.examples.rl.utils import _load_policy_checkpoint
from newton._src.solvers.kamino.models import get_examples_usd_assets_path
from newton._src.solvers.kamino.utils import logger as msg
from warp.context import Devicelike

# ---------------------------------------------------------------------------
# BDX joint normalization (from training config)
# ---------------------------------------------------------------------------
# Each entry maps joint name -> (position_offset, position_scale) used to
# normalise joint positions in the observation vector.

_BDX_JOINT_NORMALIZATION = {
    "NECK_FORWARD": (1.23, 0.19),
    "NECK_PITCH": (-1.09, 0.44),
    "NECK_YAW": (0.0, 0.35),
    "NECK_ROLL": (0.0, 0.11),
    "RIGHT_HIP_YAW": (0.0, 0.26),
    "RIGHT_HIP_ROLL": (0.06, 0.32),
    "RIGHT_HIP_PITCH": (0.49, 0.75),
    "RIGHT_KNEE_PITCH": (-0.91, 0.61),
    "RIGHT_ANKLE_PITCH": (0.22, 0.66),
    "LEFT_HIP_YAW": (0.0, 0.26),
    "LEFT_HIP_ROLL": (-0.06, 0.32),
    "LEFT_HIP_PITCH": (0.49, 0.75),
    "LEFT_KNEE_PITCH": (-0.91, 0.61),
    "LEFT_ANKLE_PITCH": (0.22, 0.66),
}

_BDX_JOINT_VELOCITY_SCALE = 5.0
_BDX_PATH_DEVIATION_SCALE = 0.1
_BDX_PHASE_EMBEDDING_DIM = 4


def _build_normalization(joint_names: list[str]):
    """Build ordered (offset, scale) lists from simulator joint names."""
    offsets: list[float] = []
    scales: list[float] = []
    for name in joint_names:
        if name in _BDX_JOINT_NORMALIZATION:
            o, s = _BDX_JOINT_NORMALIZATION[name]
        else:
            msg.warning(f"Joint '{name}' not in BDX normalization dict -- using identity.")
            o, s = 0.0, 1.0
        offsets.append(o)
        scales.append(s)
    return offsets, scales


###########################################################################
# Example class
###########################################################################


class Example:
    def __init__(
        self,
        device: Devicelike = None,
        policy=None,
        headless: bool = False,
    ):
        # Timing
        self.sim_dt = 0.02
        self.control_decimation = 1
        num_worlds = 1
        self.env_dt = self.sim_dt * self.control_decimation

        # USD model path
        EXAMPLE_ASSETS_PATH = get_examples_usd_assets_path()
        if EXAMPLE_ASSETS_PATH is None:
            raise FileNotFoundError("Failed to find USD assets path for examples: ensure `newton-assets` is installed.")
        USD_MODEL_PATH = os.path.join(EXAMPLE_ASSETS_PATH, "bdx/bdx.usda")

        # Create generic articulated body simulator
        self.sim_wrapper = RigidBodySim(
            usd_model_path=USD_MODEL_PATH,
            num_worlds=1,
            sim_dt=self.sim_dt,
            device=device,
            headless=headless,
            body_pose_offset=(0.0, 0.0, -0.15, 0.0, 0.0, 0.0, 1.0),
        )

        # Override PD gains (BDX-specific)
        self.sim_wrapper.sim.model.joints.k_p_j.fill_(15.0)
        self.sim_wrapper.sim.model.joints.k_d_j.fill_(0.6)
        self.sim_wrapper.sim.model.joints.a_j.fill_(0.004)
        self.sim_wrapper.sim.model.joints.b_j.fill_(0.0)

        # Build normalization from simulator joint order
        joint_pos_offset, joint_pos_scale = _build_normalization(self.sim_wrapper.joint_names)
        self.joint_pos_offset = torch.tensor(joint_pos_offset, device=self.torch_device)
        self.joint_pos_scale = torch.tensor(joint_pos_scale, device=self.torch_device)

        # Observation builder (BDX walking)
        self.obs = BdxObservation(
            body_sim=self.sim_wrapper,
            joint_position_default=joint_pos_offset,
            joint_position_range=joint_pos_scale,
            joint_velocity_scale=_BDX_JOINT_VELOCITY_SCALE,
            path_deviation_scale=_BDX_PATH_DEVIATION_SCALE,
            phase_embedding_dim=_BDX_PHASE_EMBEDDING_DIM,
            num_joints=len(self.joint_pos_offset),
        )
        msg.info(f"Observation dim: {self.obs.num_observations}")

        # Joystick / keyboard command controller
        self.joystick = JoystickController(
            obs_builder=self.obs,
            dt=self.env_dt,
            viewer=self.sim_wrapper.viewer,
        )

        # Action buffer
        self.actions = torch.zeros(
            (num_worlds, self.sim_wrapper.num_actuated),
            device=self.sim_wrapper.torch_device,
            dtype=torch.float32,
        )

        self.actions = self.sim_wrapper.q_j.clone()

        # Policy (None = zero actions)
        self.policy = policy

        # Keyboard state
        self._reset_key_prev = False

    # Convenience accessors for the main block
    @property
    def torch_device(self) -> str:
        return self.sim_wrapper.torch_device

    @property
    def viewer(self):
        return self.sim_wrapper.viewer

    def reset(self):
        """Reset the simulation and internal state."""
        self.sim_wrapper.reset()
        self.obs.reset()
        self.joystick.reset()
        self.actions = self.sim_wrapper.q_j.clone()

    def step_once(self):
        """Single physics step (used by run_headless warm-up)."""
        self.sim_wrapper.step()

    def step(self):
        """One RL step: commands -> observe -> infer -> apply -> simulate."""
        # Reset on 'p' key
        if self.sim_wrapper.viewer is not None and hasattr(self.sim_wrapper.viewer, "is_key_down"):
            reset_down = bool(self.sim_wrapper.viewer.is_key_down("p"))
            if reset_down and not self._reset_key_prev:
                self.reset()
            self._reset_key_prev = reset_down

        # Update velocity / neck commands from gamepad or keyboard
        self.joystick.update()

        # Compute observation from current state (with previous setpoints)
        obs = self.obs.compute(setpoints=self.actions)

        # Policy inference
        with torch.no_grad():
            self.actions = self.joint_pos_scale * self.policy(obs).clone() + self.joint_pos_offset

        # Write action targets to implicit PD controller
        self.sim_wrapper.q_j_ref[:] = self.actions

        # Step physics
        for _ in range(self.control_decimation):
            self.sim_wrapper.step()

    def render(self):
        """Render the current frame."""
        self.sim_wrapper.render()


###########################################################################
# Main
###########################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BDX RL play example")
    parser.add_argument("--device", type=str, help="The compute device to use")
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run in headless mode",
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

    # Convert warp device to torch device string for checkpoint loading
    torch_device = "cuda" if device.is_cuda else "cpu"

    # Load trained policy
    POLICY_PATH = os.path.join(get_examples_usd_assets_path(), "bdx/model.pt")
    policy = _load_policy_checkpoint(POLICY_PATH, device=torch_device)
    msg.info(f"Loaded policy from: {POLICY_PATH}")

    example = Example(
        device=device,
        policy=policy,
        headless=args.headless,
    )

    if args.headless:
        msg.notif("Running in headless mode...")
        run_headless(example, progress=True)
    else:
        msg.notif("Running in Viewer mode...")
        if hasattr(example.viewer, "set_camera"):
            example.viewer.set_camera(wp.vec3(0.6, 0.6, 0.3), -10.0, 225.0)
        while example.sim_wrapper.is_running():
            example.step()
            example.render()
