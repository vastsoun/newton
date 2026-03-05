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
# The policy expects 94D observations with path-frame integration:
#   ori_root_to_path (9D) + path_deviation (2D) + path_dev_heading (2D)
#   + path_cmd (3D) + cmd_linvel_in_root (3D) + cmd_angvel_in_root (3D)
#   + phase_encoding (4D) + root_linvel_in_root (3D) + root_angvel_in_root (3D)
#   + cmd_height (1D) + height_error (1D)
#   + joint_positions (36D) + action_history (24D)
#
# Usage:
#   python example_rl_drlegs.py --policy path/to/model.pt
#   python example_rl_drlegs.py --policy path/to/model.pt --mode async
#   python example_rl_drlegs.py --headless --num-steps 200
###########################################################################

import argparse
import os

import numpy as np
import torch  # noqa: TID253
import warp as wp
from warp.context import Devicelike

from newton._src.solvers.kamino.core.joints import JointActuationType
from newton._src.solvers.kamino.examples import run_headless
from newton._src.solvers.kamino.examples.rl.joystick import JoystickConfig, JoystickController
from newton._src.solvers.kamino.examples.rl.observations import DrlegsBaseObservation
from newton._src.solvers.kamino.examples.rl.simulation import RigidBodySim
from newton._src.solvers.kamino.examples.rl.simulation_runner import SimulationRunner
from newton._src.solvers.kamino.examples.rl.utils import (
    _load_policy_checkpoint,
    periodic_encoding,
    quat_inv_mul,
    quat_rotate_inv,
    quat_to_projected_yaw,
    quat_to_rotation9d,
    yaw_apply_2d,
    yaw_to_quat,
)
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
_YAW_CMD_MAX = 1.0  # rad/s max yaw rate command
_PD_KP = 15.0  # proportional gain (N·m/rad) for actuated joints
_PD_KD = 0.6  # derivative gain (N·m·s/rad) for actuated joints
_PD_ARMATURE = 0.01  # rotor inertia (kg·m²) for actuated joints
_PATH_DEVIATION_SCALE = 0.1  # scaling for path deviation observations
_LINEAR_PATH_ERROR_LIMIT = 0.1  # max path-to-root deviation before clipping (m)
_STANDING_HEIGHT = 0.265  # m, default pelvis Z from body_pose_offset
_HEIGHT_CMD_MIN = 0.13  # m, minimum pelvis height command (must match training)
_HEIGHT_CMD_MAX = 0.28  # m, maximum pelvis height command (must match training)
_HEIGHT_ERROR_SCALE = 0.05  # scaling for height error observation (must match training)


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

        # Observation builder (63D base: root_pos(3) + joints(36) + action_hist(24))
        # action_scale=_ACTION_SCALE so that history stores scale*actions = 0.4*output,
        # matching training where action_scale=1.0 but setpoints are already 0.4*output.
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

        # Path frame state
        self._path_heading = torch.zeros(num_worlds, device=self.torch_device, dtype=torch.float32)
        self._path_position = torch.zeros(num_worlds, 2, device=self.torch_device, dtype=torch.float32)

        # Command velocity buffer (filled by joystick each step)
        self._cmd_vel = torch.zeros(num_worlds, 2, device=self.torch_device, dtype=torch.float32)
        # Command yaw rate buffer (filled by joystick each step)
        self._cmd_yaw_rate = torch.zeros(num_worlds, 1, device=self.torch_device, dtype=torch.float32)

        # Height command buffer (default = standing height, adjustable via keyboard Y/N)
        self._cmd_height = torch.full((num_worlds, 1), _STANDING_HEIGHT, device=self.torch_device, dtype=torch.float32)

        # Zero column for 2D->3D padding
        self._zeros = torch.zeros(num_worlds, 1, device=self.torch_device, dtype=torch.float32)

        # Full observation buffer (94D)
        # 9 + 2 + 2 + 3 + 3 + 3 + 4 + 3 + 3 + 1 + 1 + 36 + 24 = 94
        obs_dim = 94
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
                angular_velocity_base=_YAW_CMD_MAX,
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

    def _advance_path(self):
        """Integrate path heading and position from velocity commands.
        Uses mid-point heading integration.
        """
        cmd_yaw = self._cmd_yaw_rate.squeeze(-1)  # (N,)

        # Mid-point heading for numerical accuracy
        mid_heading = self._path_heading + 0.5 * self.env_dt * cmd_yaw
        self._path_position += yaw_apply_2d(mid_heading, self._cmd_vel) * self.env_dt

        # Heading integration
        self._path_heading += cmd_yaw * self.env_dt

        # Clip path position to stay near robot (prevent drift)
        root_pos_2d = self.sim_wrapper.q_i[:, 0, :2]
        diff = self._path_position - root_pos_2d
        clipped = diff.renorm(p=2, dim=0, maxnorm=_LINEAR_PATH_ERROR_LIMIT)
        self._path_position[:] = root_pos_2d + clipped

    def reset(self):
        """Reset the simulation and internal state."""
        self.sim_wrapper.reset()
        self.actions.zero_()
        self.obs_builder.reset()
        self._phase.zero_()
        self._cmd_vel.zero_()
        self._cmd_yaw_rate.zero_()
        self._cmd_height.fill_(_STANDING_HEIGHT)
        self._path_heading.zero_()
        self._path_position[:] = self.sim_wrapper.q_i[:, 0, :2]
        self.sim_wrapper.q_j_ref.zero_()
        self.sim_wrapper.dq_j_ref.zero_()
        self.joystick.reset()

    def step_once(self):
        """Single physics step (used by run_headless warm-up)."""
        self.sim_wrapper.step()

    def update_input(self):
        """Transfer joystick velocity commands and height command to buffers."""
        self._cmd_vel[0, 0] = self.joystick.forward_velocity
        self._cmd_vel[0, 1] = self.joystick.lateral_velocity
        self._cmd_yaw_rate[0, 0] = self.joystick.angular_velocity

        # Height command via keyboard: Y = increase, N = decrease
        if self.viewer is not None and hasattr(self.viewer, "is_key_down"):
            if self.viewer.is_key_down("y"):
                self._cmd_height[0, 0] = min(self._cmd_height[0, 0].item() + 0.001, _HEIGHT_CMD_MAX)
            if self.viewer.is_key_down("n"):
                self._cmd_height[0, 0] = max(self._cmd_height[0, 0].item() - 0.001, _HEIGHT_CMD_MIN)

    def sim_step(self):
        """Observations -> policy inference -> actions -> physics step.

        Builds 94D path-frame observations matching DrlegsWalkObserver:
            ori_root_to_path(9) + path_dev(2) + path_dev_heading(2)
            + path_cmd(3) + cmd_linvel_root(3) + cmd_angvel_root(3)
            + phase_enc(4) + root_linvel_root(3) + root_angvel_root(3)
            + cmd_height(1) + height_error(1)
            + joints(36) + action_hist(24)
        """
        # Advance phase clock
        self._phase.add_(self.env_dt * _PHASE_RATE).remainder_(1.0)

        # Advance path frame
        self._advance_path()

        # Base observation (63D: root_pos(3) + joints(36) + action_hist(24))
        base_obs = self.obs_builder.compute(actions=self.actions)
        base_no_root = base_obs[:, 3:]  # 60D (joints + action_history)

        # --- Path quaternion from heading ---
        path_quat = yaw_to_quat(self._path_heading)  # (N, 4)

        # --- Root orientation relative to path frame (9D) ---
        root_quat = self.sim_wrapper.q_i[:, 0, 3:]  # (N, 4)
        root_in_path = quat_inv_mul(path_quat, root_quat)  # (N, 4)
        ori_9d = quat_to_rotation9d(root_in_path)  # (N, 9)

        # --- Path deviation in path frame (2D, scaled) ---
        diff_xy = self.sim_wrapper.q_i[:, 0, :2] - self._path_position  # (N, 2)
        diff_3d = torch.cat([diff_xy, self._zeros], dim=-1)  # (N, 3)
        dev_in_path = quat_rotate_inv(path_quat, diff_3d)[:, :2]  # (N, 2)
        inv_scale = 1.0 / _PATH_DEVIATION_SCALE
        path_dev = dev_in_path * inv_scale

        # --- Path deviation in heading frame (2D, scaled) ---
        root_heading = quat_to_projected_yaw(root_in_path)  # (N, 1)
        heading_quat = yaw_to_quat(root_heading)  # (N, 4)
        neg_dev = torch.cat([-dev_in_path, self._zeros], dim=-1)  # (N, 3)
        dev_in_heading = quat_rotate_inv(heading_quat, neg_dev)[:, :2]  # (N, 2)
        path_dev_h = dev_in_heading * inv_scale

        # --- Path command (3D, local frame) ---
        path_cmd = torch.cat([self._cmd_vel, self._cmd_yaw_rate], dim=-1)  # (N, 3)

        # --- Command velocities in root frame (3D + 3D) ---
        cmd_vel_3d = torch.cat([self._cmd_vel, self._zeros], dim=-1)  # (N, 3)
        cmd_linvel_root = quat_rotate_inv(root_in_path, cmd_vel_3d)  # (N, 3)
        cmd_angvel_3d = torch.cat([self._zeros, self._zeros, self._cmd_yaw_rate], dim=-1)  # (N, 3)
        cmd_angvel_root = quat_rotate_inv(root_in_path, cmd_angvel_3d)  # (N, 3)

        # --- Phase encoding (4D) ---
        torch.sin(torch.outer(self._phase, self._freq_2pi).add_(self._offset), out=self._phase_enc)

        # --- Actual velocities in root frame (3D + 3D) ---
        world_linvel = self.sim_wrapper.u_i[:, 0, :3]
        world_angvel = self.sim_wrapper.u_i[:, 0, 3:]
        root_linvel = quat_rotate_inv(root_quat, world_linvel)  # (N, 3)
        root_angvel = quat_rotate_inv(root_quat, world_angvel)  # (N, 3)

        # --- Pelvis height command and error (2D) ---
        actual_height = self.sim_wrapper.q_i[:, 0, 2:3]  # (N, 1)
        height_error = (actual_height - self._cmd_height) / _HEIGHT_ERROR_SCALE  # (N, 1)

        # --- Build full 94D observation ---
        i = 0
        self._obs_buffer[:, i : i + 9] = ori_9d
        i += 9
        self._obs_buffer[:, i : i + 2] = path_dev
        i += 2
        self._obs_buffer[:, i : i + 2] = path_dev_h
        i += 2
        self._obs_buffer[:, i : i + 3] = path_cmd
        i += 3
        self._obs_buffer[:, i : i + 3] = cmd_linvel_root
        i += 3
        self._obs_buffer[:, i : i + 3] = cmd_angvel_root
        i += 3
        self._obs_buffer[:, i : i + 4] = self._phase_enc
        i += 4
        self._obs_buffer[:, i : i + 3] = root_linvel
        i += 3
        self._obs_buffer[:, i : i + 3] = root_angvel
        i += 3
        self._obs_buffer[:, i : i + 1] = self._cmd_height
        i += 1
        self._obs_buffer[:, i : i + 1] = height_error
        i += 1
        self._obs_buffer[:, i : i + 60] = base_no_root
        # i += 60 → total = 94

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
