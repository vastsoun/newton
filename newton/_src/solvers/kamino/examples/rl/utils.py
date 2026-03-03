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

"""Standalone math utilities for RL observation builders.

Provides batched torch rotation operations, stacked-index bookkeeping,
and periodic phase encoding — no external RL-framework dependency.
"""

from __future__ import annotations

# Python
import math
from typing import List, Tuple, Union

# Thirdparty
import numpy as np
import torch
import warp as wp

# ---------------------------------------------------------------------------
# Rotation helpers  (xyzw convention, warp kernels with torch tensor wrappers)
# ---------------------------------------------------------------------------

_Z_AXIS = wp.constant(wp.vec3(0.0, 0.0, 1.0))


@wp.kernel
def _quat_to_projected_yaw_kernel(
    q: wp.array(dtype=wp.float32),
    yaw: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    base = i * 4
    qx = q[base + 0]
    qy = q[base + 1]
    qz = q[base + 2]
    qw = q[base + 3]
    yaw[i] = wp.atan2(2.0 * (qz * qw + qx * qy), qw * qw + qx * qx - qy * qy - qz * qz)


def quat_to_projected_yaw(q: torch.Tensor) -> torch.Tensor:
    """Extract yaw angle from quaternion.  Returns shape ``(-1, 1)``."""
    q_flat = q.reshape(-1, 4).contiguous()
    n = q_flat.shape[0]
    yaw = torch.empty(n, dtype=torch.float32, device=q.device)
    wp.launch(
        _quat_to_projected_yaw_kernel,
        dim=n,
        inputs=[wp.from_torch(q_flat.reshape(-1)), wp.from_torch(yaw)],
        device=str(q.device),
    )
    return yaw.view(-1, 1)


@wp.kernel
def _yaw_apply_2d_kernel(
    yaw: wp.array(dtype=wp.float32),
    v: wp.array(dtype=wp.float32),
    out: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    q = wp.quat_from_axis_angle(_Z_AXIS, yaw[i])
    base = i * 2
    r = wp.quat_rotate(q, wp.vec3(v[base], v[base + 1], 0.0))
    out[base] = r[0]
    out[base + 1] = r[1]


def yaw_apply_2d(yaw: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Forward yaw rotation of a 2-D vector."""
    yaw_flat = yaw.reshape(-1).contiguous()
    v_flat = v.reshape(-1, 2).contiguous()
    n = yaw_flat.shape[0]
    out = torch.empty_like(v_flat)
    wp.launch(
        _yaw_apply_2d_kernel,
        dim=n,
        inputs=[wp.from_torch(yaw_flat), wp.from_torch(v_flat.reshape(-1)), wp.from_torch(out.reshape(-1))],
        device=str(yaw.device),
    )
    return out.view(-1, 2)


# ---------------------------------------------------------------------------
# StackedIndices — lightweight named-range bookkeeping
# ---------------------------------------------------------------------------


class StackedIndices:
    """Named ranges for indexing into a flat (stacked) vector.

    Example::

        idx = StackedIndices([("pos", 3), ("vel", 3)])
        idx.pos        # range(0, 3)
        idx.vel_idx    # 3  (scalar start, only for size-1 entries)
        len(idx)       # 6
    """

    def __init__(self, names_and_sizes: List[Tuple[str, int]]) -> None:
        start: int = 0
        self.names: List[str] = []
        for name, size in names_and_sizes:
            if hasattr(self, name):
                raise ValueError(f"Duplicate entry '{name}'.")
            if size <= 0:
                continue
            setattr(self, name, range(start, start + size))
            setattr(self, name + "_slice", slice(start, start + size))
            if size == 1:
                setattr(self, name + "_idx", start)
            start += size
            self.names.append(name)
        self.size: int = start

    def names_and_sizes(self) -> List[Tuple[str, int]]:
        return [(n, len(self[n])) for n in self.names]

    def __getitem__(self, key: Union[str, List[str], Tuple[str, ...]]) -> Union[range, List[int]]:
        if isinstance(key, str):
            return getattr(self, key)
        elif isinstance(key, (list, tuple)):
            return [i for k in key for i in getattr(self, k)]
        raise TypeError(f"Invalid key type: {type(key)}")

    def __len__(self) -> int:
        return self.size


# ---------------------------------------------------------------------------
# Phase encoding
# ---------------------------------------------------------------------------


def periodic_encoding(k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute sin/cos phase-encoding frequencies and offsets.

    Returns ``(freq_2pi, offset)`` arrays of length ``2*k``.  Each pair
    encodes ``[cos(n·2π·φ), sin(n·2π·φ)]`` for ``n = 1 … k``.
    """
    dim = k * 2
    freq_2pi = np.zeros((dim,))
    offset = np.zeros((dim,))
    for i in range(k):
        freq = 2.0 * np.pi * (1 + i)
        freq_2pi[2 * i] = freq
        freq_2pi[2 * i + 1] = freq
        offset[2 * i] = 0.5 * np.pi
    return freq_2pi, offset


# ---------------------------------------------------------------------------
# Checkpoint loading helpers
# ---------------------------------------------------------------------------


def _build_mlp_from_state_dict(sd: dict, prefix: str, activation: str = "elu") -> torch.nn.Sequential:
    """Reconstruct a Sequential MLP from a state dict with numbered layers."""
    act_map = {"elu": torch.nn.ELU, "relu": torch.nn.ReLU, "tanh": torch.nn.Tanh}
    act_cls = act_map.get(activation, torch.nn.ELU)
    # Collect linear layer indices (keys like "actor.0.weight", "actor.2.weight", ...)
    indices = sorted({int(k.split(".")[1]) for k in sd if k.startswith(prefix + ".")})
    layers: list[torch.nn.Module] = []
    for i, idx in enumerate(indices):
        w = sd[f"{prefix}.{idx}.weight"]
        b = sd[f"{prefix}.{idx}.bias"]
        lin = torch.nn.Linear(w.shape[1], w.shape[0])
        lin.weight.data.copy_(w)
        lin.bias.data.copy_(b)
        layers.append(lin)
        if i < len(indices) - 1:  # activation after every layer except the last
            layers.append(act_cls())
    return torch.nn.Sequential(*layers)


def _load_policy_checkpoint(path: str, device: str) -> callable:
    """Load a raw rsl_rl training checkpoint and return a callable policy.

    Handles both TorchScript (.pt exported via torch.jit.save) and raw
    training checkpoints (saved via torch.save with model_state_dict).

    Args:
        path: Path to the checkpoint file.
        device: Torch device string (e.g. ``"cuda"`` or ``"cpu"``).
    """
    try:
        return torch.jit.load(path, map_location=device)
    except RuntimeError:
        pass

    ckpt = torch.load(path, map_location=device, weights_only=False)
    model_sd = ckpt["model_state_dict"]

    actor = _build_mlp_from_state_dict(model_sd, "actor").to(device)
    actor.eval()

    # Observation normalizer (if present)
    obs_norm_sd = ckpt.get("obs_norm_state_dict")
    if obs_norm_sd is not None:
        mean = obs_norm_sd["_mean"].to(device)
        std = obs_norm_sd["_std"].to(device)
        eps = 1e-2

        def policy(obs: torch.Tensor) -> torch.Tensor:
            return actor((obs - mean) / (std + eps))

    else:

        def policy(obs: torch.Tensor) -> torch.Tensor:
            return actor(obs)

    return policy


# ---------------------------------------------------------------------------
# Joystick controller
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


class RateLimitedValue:
    """Scalar rate limiter — clamps the rate of change to ±rate_limit/s."""

    def __init__(self, rate_limit: float, dt: float) -> None:
        self.rate_limit = rate_limit
        self.dt = dt
        self.value: float = 0.0
        self._initialized = False

    def update(self, target: float) -> float:
        if not self._initialized:
            self._initialized = True
            self.value = target
        else:
            max_delta = self.rate_limit * self.dt
            delta = max(-max_delta, min(target - self.value, max_delta))
            self.value += delta
        return self.value

    def reset(self) -> None:
        self.value = 0.0
        self._initialized = False


def _scale_asym(value: float, neg_scale: float, pos_scale: float) -> float:
    """Asymmetric scaling around zero."""
    return value * neg_scale if value < 0.0 else value * pos_scale
