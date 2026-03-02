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

"""Xbox gamepad / keyboard controller for BDX walking-only RL inference.

Reads Xbox 360/One controller input (or keyboard via the viewer) and writes
directly to :pyattr:`BdxObservation.command` — no external motion-engine
dependency.

Input is selected automatically:

1. Xbox gamepad (if ``xbox360controller`` package is installed and a pad is
   connected).
2. Keyboard via the 3D viewer window (if a viewer is provided).
3. No-op — zero commands, robot stands still.

Keyboard layout (viewer window must have focus)::

    I / K  — forward / backward
    J / L  — strafe left / right
    U / O  — turn left / right
    T / G  — neck pitch up / down
    F / H  — neck yaw left / right
"""

from __future__ import annotations

import math

import torch

from newton._src.solvers.kamino.examples.rl.observations import BdxObservation
from newton._src.solvers.kamino.examples.rl.utils import (
    quat_to_projected_yaw,
    yaw_apply_2d,
)

# ---------------------------------------------------------------------------
# Inline utilities (avoid external deps)
# ---------------------------------------------------------------------------


def _deadband(value: float, threshold: float) -> float:
    """Remove dead zone and rescale to full range."""
    if abs(value) < threshold:
        return 0.0
    sign = 1.0 if value > 0.0 else -1.0
    return sign * (abs(value) - threshold) / (1.0 - threshold)


class _LowPassFilter:
    """Scalar backward-Euler low-pass filter."""

    def __init__(self, cutoff_hz: float, dt: float) -> None:
        omega = cutoff_hz * 2.0 * math.pi
        self.alpha = omega * dt / (omega * dt + 1.0)
        self.value: float | None = None

    def update(self, x: float) -> float:
        if self.value is None:
            self.value = x
        else:
            self.value = (1.0 - self.alpha) * self.value + self.alpha * x
        return self.value

    def reset(self) -> None:
        self.value = None


def _scale_asym(value: float, neg_scale: float, pos_scale: float) -> float:
    """Asymmetric scaling around zero."""
    return value * neg_scale if value < 0.0 else value * pos_scale


# ---------------------------------------------------------------------------
# JoystickController
# ---------------------------------------------------------------------------


class JoystickController:
    """Xbox gamepad / keyboard -> BdxObservation.command for walking-only inference.

    Gamepad mapping:
      Left stick   -> walking velocity (forward / lateral)
      Right stick  -> head look direction (neck pitch / yaw)
      Triggers     -> yaw rate (L minus R)

    Keyboard mapping (see module docstring for layout).
    """

    def __init__(
        self,
        obs_builder: BdxObservation,
        dt: float,
        viewer=None,
        forward_velocity_max: float = 0.3,
        lateral_velocity_max: float = 0.2,
        angular_velocity_max: float = 0.8,
        neck_pitch_up: float = 1.0,
        neck_pitch_down: float = 0.6,
        neck_yaw_max: float = 0.9,
        axis_deadband: float = 0.2,
        trigger_deadband: float = 0.2,
        joystick_cutoff_hz: float = 10.0,
        path_deviation_max: float = 0.1,
        phase_rate: float = 1.0,
    ) -> None:
        self._obs = obs_builder
        self._dt = dt
        self._viewer = viewer

        # Velocity limits
        self._fwd_max = forward_velocity_max
        self._lat_max = lateral_velocity_max
        self._ang_max = angular_velocity_max

        # Neck limits
        self._neck_pitch_up = neck_pitch_up
        self._neck_pitch_down = neck_pitch_down
        self._neck_yaw_max = neck_yaw_max

        # Deadband thresholds
        self._axis_db = axis_deadband
        self._trig_db = trigger_deadband

        # Path integration
        self._path_dev_max = path_deviation_max
        self._phase_rate = phase_rate

        # Internal state (tensors created on first reset)
        self._heading: torch.Tensor | None = None
        self._path_pos: torch.Tensor | None = None
        self._phase: torch.Tensor | None = None

        # Low-pass filters (one per axis)
        self._lx_f = _LowPassFilter(joystick_cutoff_hz, dt)
        self._ly_f = _LowPassFilter(joystick_cutoff_hz, dt)
        self._rx_f = _LowPassFilter(joystick_cutoff_hz, dt)
        self._ry_f = _LowPassFilter(joystick_cutoff_hz, dt)
        self._tl_f = _LowPassFilter(joystick_cutoff_hz, dt)
        self._tr_f = _LowPassFilter(joystick_cutoff_hz, dt)

        # --- Input mode detection ---
        self._controller = None
        self._mode: str | None = None  # "joystick", "keyboard", or None

        try:
            from xbox360controller import Xbox360Controller

            self._controller = Xbox360Controller(0, axis_threshold=0.015)
            self._mode = "joystick"
            print("Joystick connected.")
        except Exception:
            if viewer is not None and hasattr(viewer, "is_key_down"):
                self._mode = "keyboard"
                print(
                    "No joystick found. Using keyboard controls:\n"
                    "  I/K — forward/backward    J/L — strafe left/right\n"
                    "  U/O — turn left/right     T/G — look up/down\n"
                    "  F/H — look left/right"
                )
            else:
                print("No joystick or keyboard available. Commands will be zero.")

    # ------------------------------------------------------------------
    # Raw input reading
    # ------------------------------------------------------------------

    def _read_raw(self) -> tuple[float, float, float, float, float, float]:
        """Read raw axis values: (lx, ly, rx, ry, trigger_l, trigger_r).

        Gamepad convention: pushing stick left/up is negative.
        """
        if self._mode == "joystick":
            c = self._controller
            return (
                c.axis_l.x,
                c.axis_l.y,
                c.axis_r.x,
                c.axis_r.y,
                c.trigger_l.value,
                c.trigger_r.value,
            )

        # Keyboard — map key pairs to ±1.0 per "axis"
        v = self._viewer

        def _axis(neg_key: str, pos_key: str) -> float:
            val = 0.0
            if v.is_key_down(neg_key):
                val -= 1.0
            if v.is_key_down(pos_key):
                val += 1.0
            return val

        return (
            _axis("j", "l"),  # left stick X:  J = left(-), L = right(+)
            _axis("i", "k"),  # left stick Y:  I = up(-),   K = down(+)
            _axis("f", "h"),  # right stick X: F = left(-), H = right(+)
            _axis("t", "g"),  # right stick Y: T = up(-),   G = down(+)
            1.0 if v.is_key_down("u") else 0.0,  # left trigger  (turn left)
            1.0 if v.is_key_down("o") else 0.0,  # right trigger (turn right)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self) -> None:
        """Read input and write command tensor.  Call once per RL step."""
        if self._mode is None:
            return

        sim = self._obs._body_sim
        device = self._obs._device

        # Lazy-init state tensors on first call
        if self._heading is None:
            self._heading = torch.zeros(sim.num_worlds, 1, device=device)
            self._path_pos = torch.zeros(sim.num_worlds, 2, device=device)
            self._phase = torch.zeros(sim.num_worlds, device=device)
            self.reset()

        # --- Raw axes -> filter -> deadband ---
        raw_lx, raw_ly, raw_rx, raw_ry, raw_tl, raw_tr = self._read_raw()

        lx = _deadband(self._lx_f.update(raw_lx), self._axis_db)
        ly = _deadband(self._ly_f.update(raw_ly), self._axis_db)
        rx = _deadband(self._rx_f.update(raw_rx), self._axis_db)
        ry = _deadband(self._ry_f.update(raw_ry), self._axis_db)
        tl = _deadband(self._tl_f.update(raw_tl), self._trig_db)
        tr = _deadband(self._tr_f.update(raw_tr), self._trig_db)

        # --- Map to velocities ---
        vel_x = -ly * self._fwd_max
        vel_y = -lx * self._lat_max
        yaw_rate = (tl - tr) * self._ang_max

        # --- Map to neck (walking mode: forward=0, roll=0) ---
        neck_pitch = _scale_asym(-ry, self._neck_pitch_down, self._neck_pitch_up)
        neck_yaw = -rx * self._neck_yaw_max

        # --- Integrate path (matches joystick_commands._advance_path) ---
        dt = self._dt
        cmd_vel = torch.tensor([[vel_x, vel_y]], device=device)

        # 0. Update position with mid-point heading
        mid_heading = self._heading + 0.5 * dt * yaw_rate
        self._path_pos += yaw_apply_2d(mid_heading, cmd_vel) * dt

        # 1. Update heading
        self._heading += yaw_rate * dt

        # 2. Clip path deviation to root position (renorm per-world)
        root_pos_2d = sim.q_i[:, 0, :2]
        diff = self._path_pos - root_pos_2d
        clipped = diff.renorm(p=2, dim=0, maxnorm=self._path_dev_max)
        self._path_pos[:] = root_pos_2d + clipped

        # 3. Advance phase
        self._phase = (self._phase + self._phase_rate * dt) % 1.0

        # --- Write all 11 dims to command tensor ---
        cmd = self._obs.command
        cmd[:, BdxObservation.CMD_PATH_HEADING] = self._heading[:, 0]
        cmd[:, BdxObservation.CMD_PATH_POSITION] = self._path_pos
        cmd[:, BdxObservation.CMD_VEL] = cmd_vel
        cmd[:, BdxObservation.CMD_YAW_RATE] = yaw_rate
        cmd[:, BdxObservation.CMD_PHASE] = self._phase
        cmd[:, BdxObservation.CMD_NECK] = torch.tensor(
            [0.0, neck_pitch, neck_yaw, 0.0],
            device=device,
        )

    def reset(self) -> None:
        """Reset internal state to match current robot pose.  Call on sim reset."""
        sim = self._obs._body_sim

        if self._heading is None:
            return

        # Heading from current robot yaw
        root_quat = sim.q_i[:, 0, 3:]
        self._heading[:] = quat_to_projected_yaw(root_quat)

        # Path position from current robot XY
        self._path_pos[:] = sim.q_i[:, 0, :2]

        # Phase
        self._phase.zero_()

        # Filters
        self._lx_f.reset()
        self._ly_f.reset()
        self._rx_f.reset()
        self._ry_f.reset()
        self._tl_f.reset()
        self._tr_f.reset()
