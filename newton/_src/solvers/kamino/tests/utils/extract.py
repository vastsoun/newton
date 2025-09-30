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

"""
KAMINO: UNIT TESTS: GENERAL UTILITIES
"""

import numpy as np

from newton._src.solvers.kamino.dynamics.delassus import DelassusOperator
from newton._src.solvers.kamino.kinematics.jacobians import DenseSystemJacobians

###
# Helper functions
###


def extract_active_constraint_dims(delassus: DelassusOperator) -> list[int]:
    # Extract the active constraint dimensions
    active_dim_np = delassus.data.dim.numpy()
    active_dims = [int(active_dim_np[i]) for i in range(len(active_dim_np))]
    return active_dims


def extract_cts_jacobians(
    jacobians: DenseSystemJacobians,
    num_bodies: list[int],
    active_dims: list[int] | None = None,
) -> list[np.ndarray]:
    # Reshape the flat Jacobian as a set of matrices
    num_body_dofs = [6 * num_bodies[i] for i in range(len(num_bodies))]
    cjmio = jacobians.data.J_cts_offsets.numpy()
    num_jacobians = int(cjmio.size)
    J_cts_flat = jacobians.data.J_cts_data.numpy()
    J_cts_flat_total_size = J_cts_flat.size
    J_cts_flat_offsets = [int(cjmio[i]) for i in range(num_jacobians)]
    J_cts_flat_sizes = [0] * num_jacobians
    J_cts_flat_offsets_ext = [*J_cts_flat_offsets, J_cts_flat_total_size]
    J_cts_flat_shapes = [(0, 0)] * num_jacobians
    for i in range(num_jacobians - 1, -1, -1):
        J_cts_flat_sizes[i] = J_cts_flat_offsets_ext[i + 1] - J_cts_flat_offsets_ext[i]
        nbd_i = num_body_dofs[i]
        J_cts_flat_shapes[i] = (J_cts_flat_sizes[i] // nbd_i, nbd_i)

    # Extract each Jacobian as a matrix
    J_cts_mat: list[np.ndarray] = []
    for i in range(num_jacobians):
        if active_dims is not None and len(active_dims) > 0:
            J_rows = active_dims[i]
        else:
            J_rows = J_cts_flat_shapes[i][0]
        J_cols = J_cts_flat_shapes[i][1]
        J_cts_mat.append(
            J_cts_flat[J_cts_flat_offsets[i] : J_cts_flat_offsets[i] + J_cts_flat_sizes[i]].reshape(
                J_cts_flat_shapes[i]
            )[:J_rows, :J_cols]
        )

    # Return the list of Jacobian matrices
    return J_cts_mat


def extract_dofs_jacobians(
    jacobians: DenseSystemJacobians,
    num_body_dofs: list[int],
    active_dims: list[int] | None = None,
) -> list[np.ndarray]:
    # Reshape the flat Jacobian as a set of matrices
    ajmio = jacobians.data.J_dofs_offsets.numpy()
    num_jacobians = int(ajmio.size)
    J_dofs_flat = jacobians.data.J_dofs_data.numpy()
    J_dofs_flat_total_size = J_dofs_flat.size
    J_dofs_flat_offsets = [int(ajmio[i]) for i in range(num_jacobians)]
    J_dofs_flat_sizes = [0] * num_jacobians
    J_dofs_flat_offsets_ext = [*J_dofs_flat_offsets, J_dofs_flat_total_size]
    J_dofs_flat_shapes = [(0, 0)] * num_jacobians
    for i in range(num_jacobians - 1, -1, -1):
        J_dofs_flat_sizes[i] = J_dofs_flat_offsets_ext[i + 1] - J_dofs_flat_offsets_ext[i]
        nbd_i = num_body_dofs[i]
        J_dofs_flat_shapes[i] = (J_dofs_flat_sizes[i] // nbd_i, nbd_i)

    # Extract each Jacobian as a matrix
    J_cts_mat: list[np.ndarray] = []
    for i in range(num_jacobians):
        if active_dims is not None and len(active_dims) > 0:
            J_rows = active_dims[i]
        else:
            J_rows = J_dofs_flat_shapes[i][0]
        J_cols = J_dofs_flat_shapes[i][1]
        J_cts_mat.append(
            J_dofs_flat[J_dofs_flat_offsets[i] : J_dofs_flat_offsets[i] + J_dofs_flat_sizes[i]].reshape(
                J_dofs_flat_shapes[i]
            )[:J_rows, :J_cols]
        )

    # Return the list of Jacobian matrices
    return J_cts_mat


def extract_delassus(delassus: DelassusOperator, only_active_dims: bool = False) -> list[np.ndarray]:
    maxdim_wp_np = delassus.data.maxdim.numpy()
    dim_wp_np = delassus.data.dim.numpy()
    mio_wp_np = delassus.data.mio.numpy()
    D_wp_np = delassus.data.D.numpy()

    # Extract each Delassus matrix for each world
    D_mat: list[np.ndarray] = []
    for i in range(delassus.num_worlds):
        D_maxdim = maxdim_wp_np[i]
        D_start = mio_wp_np[i]
        D_end = D_start + D_maxdim * D_maxdim
        if only_active_dims:
            D_dim = dim_wp_np[i]
        else:
            D_dim = D_maxdim
        D_mat.append(D_wp_np[D_start:D_end].reshape((D_maxdim, D_maxdim))[:D_dim, :D_dim])

    # Return the list of Delassus matrices
    return D_mat


def extract_problem_vector(
    delassus: DelassusOperator, vector: np.ndarray, only_active_dims: bool = False
) -> list[np.ndarray]:
    maxdim_wp_np = delassus.data.maxdim.numpy()
    dim_wp_np = delassus.data.dim.numpy()
    vio_wp_np = delassus.data.vio.numpy()

    # Extract each vector for each world
    vectors_np: list[np.ndarray] = []
    for i in range(delassus.num_worlds):
        vec_maxdim = maxdim_wp_np[i]
        vec_start = vio_wp_np[i]
        vec_end = vec_start + vec_maxdim
        if only_active_dims:
            vec_end = vec_start + dim_wp_np[i]
        else:
            vec_end = vec_start + vec_maxdim
        vectors_np.append(vector[vec_start:vec_end])

    # Return the list of Delassus matrices
    return vectors_np


def extract_info_vectors(offsets: np.ndarray, vectors: np.ndarray, dims: list[int] | None = None) -> list[np.ndarray]:
    # Determine vector sizes
    nv = offsets.size
    maxn = vectors.size // nv
    n = dims if dims is not None and len(dims) == nv else [maxn] * nv

    # Extract each vector for each world
    vectors_list: list[np.ndarray] = []
    for i in range(nv):
        vec_start = offsets[i]
        vec_end = vec_start + n[i]
        vectors_list.append(vectors[vec_start:vec_end])

    # Return the list of Delassus matrices
    return vectors_list
