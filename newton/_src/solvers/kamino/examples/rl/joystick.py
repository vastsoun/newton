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

"""General-purpose Xbox gamepad / keyboard controller for RL inference.

Reads Xbox 360/One controller input (or keyboard via the viewer) and
provides velocity commands and head pose deltas.  Optionally integrates
a 2-D path (heading + position) from the velocity commands.

Input is selected automatically:

1. Xbox gamepad (if ``xbox360controller`` package is installed and a pad is
   connected).
2. Keyboard via the 3D viewer window (if a viewer is provided).
3. No-op — zero commands, robot stands still.

Gamepad mapping::

    Left stick Y          forward / backward velocity
    Left stick X          yaw rate (angular velocity)
    Triggers (L minus R)  lateral velocity
    Right stick Y         head pitch (look up / down)
    Right stick X         head yaw (look left / right)
    Select / Back         reset

Keyboard layout (viewer window must have focus)::

    I / K  — forward / backward
    J / L  — strafe left / right
    U / O  — turn left / right
    T / G  — head pitch up / down
    F / H  — head yaw left / right
    P      — reset
"""

from __future__ import annotations

# Python
import math

# Thirdparty
import torch
from newton._src.solvers.kamino.examples.rl.utils import yaw_apply_2d

# ---------------------------------------------------------------------------
# Inline utilities
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
    """General-purpose Xbox gamepad / keyboard controller for RL inference.

    Reads gamepad axes or keyboard keys and exposes command outputs as
    attributes.  Optionally integrates a 2-D path (heading + position)
    from the velocity commands.

    Gamepad mapping:
      Left stick Y          -> forward velocity
      Left stick X          -> yaw rate (angular velocity)
      Triggers (L minus R)  -> lateral velocity
      Right stick Y         -> neck pitch
      Right stick X         -> neck yaw
      Select / Back         -> reset

    Keyboard mapping (see module docstring for layout).

    Output attributes (updated each :meth:`update` call):
      ``forward_velocity``  Forward velocity   (positive = forward)
      ``lateral_velocity``  Lateral velocity   (positive = strafe left)
      ``angular_velocity``  Angular velocity   (positive = turn left)
      ``head_pitch``        Head pitch command (positive = look up)
      ``head_yaw``          Head yaw command   (positive = look left)

    Path state (when ``root_pos_2d`` is passed to :meth:`update`):
      ``path_heading``      Integrated heading  ``(num_worlds, 1)``
      ``path_position``     Integrated position ``(num_worlds, 2)``
    """

    def __init__(
        self,
        dt: float,
        viewer=None,
        num_worlds: int = 1,
        device: str = "cuda:0",
        forward_velocity_max: float = 0.6,
        lateral_velocity_max: float = 0.4,
        angular_velocity_max: float = 1.7,
        head_pitch_up: float = 1.0,
        head_pitch_down: float = 0.6,
        head_yaw_max: float = 0.9,
        axis_deadband: float = 0.2,
        trigger_deadband: float = 0.2,
        cutoff_hz: float = 15.0,
        path_deviation_max: float = 0.1,
    ) -> None:
        self._dt = dt
        self._viewer = viewer
        self._num_worlds = num_worlds
        self._device = device

        # Velocity limits
        self._fwd_max = forward_velocity_max
        self._lat_max = lateral_velocity_max
        self._ang_max = angular_velocity_max

        # Head limits
        self._head_pitch_up = head_pitch_up
        self._head_pitch_down = head_pitch_down
        self._head_yaw_max = head_yaw_max

        # Deadband thresholds
        self._axis_db = axis_deadband
        self._trig_db = trigger_deadband

        # Path integration limit
        self._path_dev_max = path_deviation_max

        # Low-pass filters (named by semantic axis)
        self._forward_filter = _LowPassFilter(cutoff_hz, dt)
        self._lateral_filter = _LowPassFilter(cutoff_hz, dt)
        self._angular_filter = _LowPassFilter(cutoff_hz, dt)
        self._head_pitch_filter = _LowPassFilter(cutoff_hz, dt)
        self._head_yaw_filter = _LowPassFilter(cutoff_hz, dt)

        # Path state (per-world)
        self.path_heading = torch.zeros(num_worlds, 1, device=device)
        self.path_position = torch.zeros(num_worlds, 2, device=device)

        # Command outputs (updated by update())
        self.forward_velocity: float = 0.0
        self.lateral_velocity: float = 0.0
        self.angular_velocity: float = 0.0
        self.head_pitch: float = 0.0
        self.head_yaw: float = 0.0

        # Reset edge-detection state
        self._reset_prev = False

        # --- Input mode detection ---
        self._controller = None
        self._mode: str | None = None  # "joystick", "keyboard", or None

        try:
            # Thirdparty
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
                    "  F/H — look left/right\n"
                    "  P   — reset"
                )
            else:
                print("No joystick or keyboard available. Commands will be zero.")

    # ------------------------------------------------------------------
    # Input reading
    # ------------------------------------------------------------------

    def _read_input(self) -> tuple[float, float, float, float, float]:
        """Read controller input as semantic axes.

        Returns:
            ``(forward, lateral, angular, head_pitch, head_yaw)``

        Sign convention — positive means:
          forward   : walk forward
          lateral   : strafe left
          angular   : turn left  (CCW)
          head_pitch: look up
          head_yaw  : look left
        """
        if self._mode == "joystick":
            c = self._controller
            return (
                -c.axis_l.y,  # forward   (negate: HW up is negative)
                c.trigger_l.value - c.trigger_r.value,  # lateral   (L trigger = strafe left)
                -c.axis_l.x,  # angular   (negate: HW left is negative)
                -c.axis_r.y,  # head pitch (negate: HW up is negative)
                -c.axis_r.x,  # head yaw   (negate: HW left is negative)
            )

        # Keyboard fallback
        v = self._viewer

        def _axis(neg_key: str, pos_key: str) -> float:
            val = 0.0
            if v.is_key_down(neg_key):
                val -= 1.0
            if v.is_key_down(pos_key):
                val += 1.0
            return val

        return (
            _axis("k", "i"),  # forward:    I = forward(+), K = backward(-)
            _axis("l", "j"),  # lateral:    J = left(+),    L = right(-)
            _axis("o", "u"),  # angular:    U = left(+),    O = right(-)
            _axis("g", "t"),  # head pitch: T = up(+),      G = down(-)
            _axis("h", "f"),  # head yaw:   F = left(+),    H = right(-)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, root_pos_2d: torch.Tensor | None = None) -> None:
        """Read input, compute commands, and optionally advance the path.

        Args:
            root_pos_2d: Current robot XY position ``(num_worlds, 2)`` for
                path deviation clipping.  When ``None``, path integration
                is skipped.
        """
        if self._mode is None:
            return

        # --- Read & filter ---
        fwd_raw, lat_raw, ang_raw, npitch_raw, nyaw_raw = self._read_input()

        fwd = _deadband(self._forward_filter.update(fwd_raw), self._axis_db)
        lat = _deadband(self._lateral_filter.update(lat_raw), self._trig_db)
        ang = _deadband(self._angular_filter.update(ang_raw), self._axis_db)
        npitch = _deadband(self._head_pitch_filter.update(npitch_raw), self._axis_db)
        nyaw = _deadband(self._head_yaw_filter.update(nyaw_raw), self._axis_db)

        # --- Scale to physical units ---
        self.forward_velocity = fwd * self._fwd_max
        self.lateral_velocity = lat * self._lat_max
        self.angular_velocity = ang * self._ang_max
        self.head_pitch = _scale_asym(npitch, self._head_pitch_down, self._head_pitch_up)
        self.head_yaw = nyaw * self._head_yaw_max

        # --- Path integration ---
        if root_pos_2d is not None:
            dt = self._dt
            cmd_vel = torch.tensor(
                [[self.forward_velocity, self.lateral_velocity]],
                device=self._device,
            )

            # Mid-point heading integration
            mid_heading = self.path_heading + 0.5 * dt * self.angular_velocity
            self.path_position += yaw_apply_2d(mid_heading, cmd_vel) * dt

            # Update heading
            self.path_heading += self.angular_velocity * dt

            # Clip path deviation to root position
            diff = self.path_position - root_pos_2d
            clipped = diff.renorm(p=2, dim=0, maxnorm=self._path_dev_max)
            self.path_position[:] = root_pos_2d + clipped

    def close(self) -> None:
        """Release gamepad resources so the process can exit cleanly."""
        if self._controller is not None:
            try:
                self._controller.close()
            except Exception:
                pass
            self._controller = None

    def check_reset(self) -> bool:
        """Return True on the rising edge of the reset input.

        Gamepad: Select/Back button.  Keyboard: ``p`` key.
        """
        pressed = False
        if self._mode == "joystick":
            pressed = bool(self._controller.button_select.is_pressed)
            # Also allow keyboard 'p' when a gamepad is connected
            if not pressed and self._viewer is not None and hasattr(self._viewer, "is_key_down"):
                pressed = bool(self._viewer.is_key_down("p"))
        elif self._mode == "keyboard" and self._viewer is not None:
            pressed = bool(self._viewer.is_key_down("p"))
        triggered = pressed and not self._reset_prev
        self._reset_prev = pressed
        return triggered

    def reset(self, root_pos_2d: torch.Tensor | None = None, root_yaw: torch.Tensor | None = None) -> None:
        """Reset path state and filters.

        Args:
            root_pos_2d: Current robot XY position ``(num_worlds, 2)``.
            root_yaw: Current robot yaw angle ``(num_worlds, 1)``.
        """
        if root_yaw is not None:
            self.path_heading[:] = root_yaw
        if root_pos_2d is not None:
            self.path_position[:] = root_pos_2d

        self._forward_filter.reset()
        self._lateral_filter.reset()
        self._angular_filter.reset()
        self._head_pitch_filter.reset()
        self._head_yaw_filter.reset()
