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
from typing import List, Tuple, Union

# Thirdparty
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Quaternion helpers  (xyzw convention, batched over dim 0)
# ---------------------------------------------------------------------------


@torch.jit.script
def _atan2(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    ans = torch.atan(y / x)
    ans = torch.where((x < 0.0) & (y >= 0.0), ans + torch.pi, ans)
    ans = torch.where((x < 0.0) & (y < 0.0), ans - torch.pi, ans)
    ans = torch.where(x == 0, torch.sign(y) * 0.5 * torch.pi, ans)
    return ans


@torch.jit.script
def yaw_to_quat(yaw: torch.Tensor) -> torch.Tensor:
    """Pure yaw rotation → quaternion (xyzw)."""
    yaw = yaw.reshape(-1, 1)
    half = yaw * 0.5
    z = torch.zeros_like(half)
    return torch.stack([z, z, torch.sin(half), torch.cos(half)], dim=-1).view(-1, 4)


@torch.jit.script
def quat_inv_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Quaternion product ``inv(a) * b``."""
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)
    x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (-z1 - x1) * (x2 + y2)
    yy = (w1 + y1) * (w2 + z2)
    zz = (w1 - y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (x1 - z1) * (x2 - y2))
    w = qq - ww + (y1 - z1) * (y2 - z2)
    x = qq - xx + (w1 - x1) * (x2 + w2)
    y = qq - yy + (w1 + x1) * (y2 + z2)
    z = qq - zz + (-z1 - y1) * (w2 - x2)
    return torch.stack([x, y, z, w], dim=-1).view(-1, 4)


@torch.jit.script
def quat_inv_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Inverse-quaternion rotation ``inv(q) · v``."""
    q = q.reshape(-1, 4)
    v = v.reshape(-1, 3)
    xyz = q[:, :3]
    t = xyz.cross(v, dim=-1)
    t += t
    return (v - q[:, 3:] * t + xyz.cross(t, dim=-1)).view(-1, 3)


@torch.jit.script
def quat_to_rotation9D(q: torch.Tensor) -> torch.Tensor:
    """Quaternion → flattened 3×3 rotation matrix (row-major, 9 dims)."""
    q = q.reshape(-1, 4)
    qx, qy, qz, qw = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    qxx = 2.0 * qx * qx
    qxy = 2.0 * qx * qy
    qxz = 2.0 * qx * qz
    qxw = 2.0 * qx * qw
    qyy = 2.0 * qy * qy
    qyz = 2.0 * qy * qz
    qyw = 2.0 * qy * qw
    qzz = 2.0 * qz * qz
    qzw = 2.0 * qz * qw
    return torch.stack(
        (
            1.0 - qyy - qzz,
            qxy - qzw,
            qxz + qyw,
            qxy + qzw,
            1.0 - qxx - qzz,
            qyz - qxw,
            qxz - qyw,
            qyz + qxw,
            1.0 - qxx - qyy,
        ),
        -1,
    ).view(-1, 9)


@torch.jit.script
def quat_to_projected_yaw(q: torch.Tensor) -> torch.Tensor:
    """Extract yaw angle from quaternion.  Returns shape ``(-1, 1)``."""
    q = q.reshape(-1, 4)
    qx, qy, qz, qw = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    qxx = qx * qx
    qxy = qx * qy
    qyy = qy * qy
    qzz = qz * qz
    qzw = qz * qw
    qww = qw * qw
    yaw = _atan2(2.0 * (qzw + qxy), qww + qxx - qyy - qzz)
    return yaw.view(-1, 1)


@torch.jit.script
def yaw_inv_apply_2d(yaw: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Inverse yaw rotation of a 2-D vector."""
    yaw = yaw.reshape(-1, 1)
    v = v.reshape(-1, 2)
    vx, vy = v[:, 0], v[:, 1]
    s = torch.sin(yaw[:, 0])
    c = torch.cos(yaw[:, 0])
    return torch.stack([c * vx + s * vy, -s * vx + c * vy], dim=-1).view(-1, 2)


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


def phase_encoding(
    phase: torch.Tensor,
    freq_2pi: torch.Tensor,
    offset: torch.Tensor,
) -> torch.Tensor:
    """Encode a scalar phase into a sin/cos feature vector."""
    return torch.sin(torch.outer(phase, freq_2pi) + offset)
