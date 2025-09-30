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
KAMINO: UNIT TESTS: RANDOM DATA GENERATION
"""

import numpy as np
import scipy as sp
import warp as wp
from warp.context import Devicelike

from newton._src.solvers.kamino.core.types import float32, int32

###
# Types
###

ArrayLike = np.ndarray | list[int] | list[float] | list[list[int]] | list[list[float]] | None
"""An Array-like structure for aliasing various data types compatible with numpy."""


###
# Functions
###


def eigenvalues_from_distribution(
    size: int,
    num_pos: int | float = 0,
    num_pos_eps: int | float = 0,
    num_zero: int | float = 0,
    num_neg_eps: int | float = 0,
    num_neg: int | float = 0,
    max_pos: float = 1e2,
    min_pos: float = 1e-2,
    eps_val: float = 1e-6,
    max_neg: float = -1e-2,
    min_neg: float = -1e2,
    dtype: np.dtype = np.float64,
    shuffle: bool = False,
    seed: int | None = None,
) -> np.ndarray:
    """
    Creates an array of eigen-values based on a specified distribution.

    Notes:
        - Default max/min/eps values are set in order to generate a moderately broad spectrum.
        - The num_* arguments can be int (count) or float (percentage of size).
        - The final counts are adjusted to sum to 'size'.

    Args:
        size (int): The total size of the eigenvalue distribution.
        num_pos (int | float): The number of positive eigenvalues (count or percentage).
        num_pos_eps (int | float): The number of positive epsilon eigenvalues (count or percentage).
        num_zero (int | float): The number of zero eigenvalues (count or percentage).
        num_neg_eps (int | float): The number of negative epsilon eigenvalues (count or percentage).
        num_neg (int | float): The number of negative eigenvalues (count or percentage).
        max_pos (float): The maximum value for positive eigenvalues.
        min_pos (float): The minimum value for positive eigenvalues.
        eps_val (float): The value for epsilon eigenvalues.
        max_neg (float): The maximum value for negative eigenvalues.
        min_neg (float): The minimum value for negative eigenvalues.
        dtype (np.dtype): The data type for the eigenvalues.
        shuffle (bool): Whether to shuffle the eigenvalues.

    Returns:
        np.ndarray: The generated eigenvalue array.
    """

    # Helper to convert count/percentage to int
    def resolve_count(val):
        if isinstance(val, float):
            return int(round(val * size))
        return int(val)

    # Interpret args as either counts or percentages
    counts = {
        "num_pos": resolve_count(num_pos),
        "num_pos_eps": resolve_count(num_pos_eps),
        "num_zero": resolve_count(num_zero),
        "num_neg_eps": resolve_count(num_neg_eps),
        "num_neg": resolve_count(num_neg),
    }

    # Check total counts and correct if necessary
    total = sum(counts.values())

    # If all counts are zero, assign all eigenvalues as positive
    if total == 0:
        counts["num_pos"] = size

    # Otherwise, adjust counts to match 'size'
    elif total != size:
        # Distribute the difference to the largest group
        diff = size - total
        # Find the key with the largest count
        if counts:
            max_key = max(counts, key=lambda k: counts[k])
            counts[max_key] += diff

    # Generate the distribution of eigenvalues according to the specified counts
    eigenvalues_pos = np.linspace(max_pos, min_pos, num=counts["num_pos"]) if counts["num_pos"] > 0 else np.array([])
    eigenvalues_pos_eps = np.array([eps_val] * counts["num_pos_eps"]) if counts["num_pos_eps"] > 0 else np.array([])
    eigenvalues_zero = np.zeros(counts["num_zero"]) if counts["num_zero"] > 0 else np.array([])
    eigenvalues_neg_eps = np.array([-eps_val] * counts["num_neg_eps"]) if counts["num_neg_eps"] > 0 else np.array([])
    eigenvalues_neg = np.linspace(max_neg, min_neg, num=counts["num_neg"]) if counts["num_neg"] > 0 else np.array([])

    # Concatenate all eigenvalues into a single array of target dtype
    eigenvalues = np.concatenate(
        [
            eigenvalues_pos.astype(dtype),
            eigenvalues_pos_eps.astype(dtype),
            eigenvalues_zero.astype(dtype),
            eigenvalues_neg_eps.astype(dtype),
            eigenvalues_neg.astype(dtype),
        ]
    )

    # Optionally shuffle the eigenvalues to randomize their order
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(eigenvalues)

    # Finally return the constructed eigenvalues array
    return eigenvalues


def random_symmetric_matrix(
    dim: int,
    dtype=np.float32,
    scale: float | None = None,
    seed: int | None = None,
    rank: int | None = None,
    eigenvalues: ArrayLike = None,
    return_source: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Generate a random symmetric matrix of size (dim, dim).

    Args:
    - dim (int): The size of the matrix.
    - dtype (data-type, optional): Data type of the matrix (default is float32).
    - scale (float, optional): Scale factor for the matrix (default is 1.0).
    - seed (int, optional): Seed for the random number generator.
    - rank (int, optional): Rank of the matrix (must be <= dim).
    - eigenvalues (array-like, optional): Eigenvalues for the matrix (must be of length dim).
    - return_source (bool, optional): Whether to return the source matrix used to generate the symmetric matrix (default is False).

    Returns:
    - A (ndarray): A (dim, dim) symmetric matrix.
    """
    # Create random number generator
    rng = np.random.default_rng(seed)

    # Validate seed if provided
    if seed is not None and not isinstance(seed, int):
        raise TypeError("seed must be a int.")

    # Set a default unit scale if unspecified
    if scale is None:
        scale = 1.0
        sqrt_scale = 1.0
    # Otherwise, check if scale is a float
    else:
        if not isinstance(scale, float):
            raise TypeError("scale must be a float.")
        sqrt_scale = np.sqrt(scale)

    # Generate a symmetric matrix of random rank and eigenvalues, if unspecified
    if eigenvalues is None and rank is None:
        X = scale * rng.standard_normal((dim, dim)).astype(dtype)
        # Make a symmetric matrix from the source random matrix
        A = 0.5 * (X + X.T)

    # If eigenvalues are specified these take precedence
    elif eigenvalues is not None:
        if len(eigenvalues) != dim:
            raise ValueError("The number of eigenvalues must match the matrix dimension.")

        # Generate random square matrix
        if np.all(eigenvalues == eigenvalues[0]):
            X = rng.standard_normal((dim, dim)).astype(dtype)
        else:
            X, _ = np.linalg.qr(rng.standard_normal((dim, dim)).astype(dtype))
        # Diagonal matrix of eigenvalues
        D = np.diag(eigenvalues)
        # A = X * D * X^T
        A = scale * (X @ D @ X.T)
        # Additional step to ensure symmetry
        A = 0.5 * (A + A.T)

    # Otherwise generate a symmetric matrix of specified rank
    elif rank is not None:
        if rank > dim:
            raise ValueError("Rank must not exceed matrix dimension.")
        # Generate random rectangular matrix
        X = sqrt_scale * rng.standard_normal((dim, rank)).astype(dtype)
        # Make a rank-deficient symmetric matrix
        A = X @ X.T
        # Additional step to ensure symmetry
        A = 0.5 * (A + A.T)

    # Optionally return both final and source matrices
    if return_source:
        return A, X
    else:
        return A


def random_spd_matrix(
    dim: int,
    dtype=np.float32,
    scale: float | None = None,
    seed: int | None = None,
    return_source: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Generate a symmetric positive definite (SPD) matrix of shape (n, n).

    Args:
    - n (int): The size of the matrix.
    - dtype (data-type, optional): Data type of the matrix (default is float32).
    - scale (float, optional): Scale factor for the matrix (default is 1.0).
    - seed (int, optional): Seed for the random number generator.
    - return_source (bool, optional): Whether to return the source matrix used to generate the SPD matrix (default is False).

    Returns:
    - A (ndarray): An n x n symmetric positive definite matrix.
    """
    # Create random number generator
    rng = np.random.default_rng(seed)

    # Validate seed if provided
    if seed is not None and not isinstance(seed, int):
        raise TypeError("seed must be a int.")

    if scale is None:
        scale = 1.0
        sqrt_scale = 1.0
    else:
        if not isinstance(scale, float):
            raise TypeError("scale must be a float.")
        sqrt_scale = np.sqrt(scale)

    # Generate a random matrix
    X = sqrt_scale * rng.standard_normal((dim, dim)).astype(dtype)

    # Construct symmetric positive definite matrix: A.T @ A + dim * I
    A = X.T @ X + scale * float(dim) * np.eye(dim, dtype=dtype)

    # Ensure the matrix is symmetric
    A = 0.5 * (A + A.T)

    # Optionally return both final and source matrices
    if return_source:
        return A, X
    else:
        return A


def random_rhs_for_matrix(
    A: np.ndarray, scale: float = 1.0, return_source: bool = False
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Generate a random RHS vector b that is in the range space of A.

    Args:
        A (np.ndarray): The input matrix.
        scale (float): Scale factor for the random vector (default is 1.0).
        return_source (bool): Whether to return the source vector used to generate the RHS (default is False).

    Returns:
        np.ndarray: A random RHS vector b in the range space of A.
    """
    n = A.shape[0]
    rng = np.random.default_rng()
    x = scale * rng.standard_normal(n).astype(A.dtype)
    b = A @ x
    if return_source:
        return b, x
    return b


###
# Utilities
###


def solve_cholesky_lower_numpy(L: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve the linear system Ax = b using Cholesky decomposition.

    Args:
        A (np.ndarray): The input matrix (must be symmetric positive definite).
        b (np.ndarray): The RHS vector.

    Returns:
        np.ndarray: The solution vector x.
    """
    y = np.linalg.solve(L, b)
    x = np.linalg.solve(L.T, y)
    return y, x


def solve_cholesky_upper_numpy(U: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve the linear system Ax = b using Cholesky decomposition.

    Args:
        A (np.ndarray): The input matrix (must be symmetric positive definite).
        b (np.ndarray): The RHS vector.

    Returns:
        np.ndarray: The solution vector x.
    """
    y = np.linalg.solve(U.T, b)
    x = np.linalg.solve(U, y)
    return y, x


def solve_ldlt_lower_numpy(
    L: np.ndarray, D: np.ndarray, P: np.ndarray, b: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve the linear system Ax = b using LDL^T decomposition.

    Args:
        L (np.ndarray): The lower triangular matrix from the LDL^T decomposition.
        D (np.ndarray): The diagonal matrix from the LDL^T decomposition.
        P (np.ndarray): The permutation index array from the LDL^T decomposition.
        b (np.ndarray): The RHS vector.

    Returns:
        np.ndarray: The solution vector x.
    """
    y = np.linalg.solve(L, b)
    z = np.linalg.solve(D, y)
    x = np.linalg.solve(L.T, z)
    x = x[P]
    return y, z, x


def solve_ldlt_upper_numpy(
    U: np.ndarray, D: np.ndarray, P: np.ndarray, b: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve the linear system Ax = b using LDL^T decomposition.

    Args:
        U (np.ndarray): The upper triangular matrix from the LDL^T decomposition.
        D (np.ndarray): The diagonal matrix from the LDL^T decomposition.
        P (np.ndarray): The permutation index array from the LDL^T decomposition.
        b (np.ndarray): The RHS vector.

    Returns:
        np.ndarray: The solution vector x.
    """
    y = np.linalg.solve(U.T, b)
    z = np.linalg.solve(D, y)
    x = np.linalg.solve(U, z)
    x = x[P]
    return y, z, x


###
# Classes
###


class RandomProblemCholesky:
    def __init__(
        self,
        seed: int = 42,
        dims: list[int] | int | None = None,
        A: list[np.ndarray] | None = None,
        b: list[np.ndarray] | None = None,
        np_dtype=np.float32,
        wp_dtype=float32,
        device: Devicelike = None,
        upper: bool = False,
        verbose: bool = False,
    ):
        # Check input data to ensure they are indeed lists of numpy arrays
        if A is not None:
            if not isinstance(A, list) or not all(isinstance(a, np.ndarray) for a in A):
                raise TypeError("A must be a list of numpy arrays.")
            dims = [a.shape[0] for a in A]  # Update dims based on provided A
        if b is not None:
            if not isinstance(b, list) or not all(isinstance(b_i, np.ndarray) for b_i in b):
                raise TypeError("b must be a list of numpy arrays.")

        # Ensure the problem dimensions are valid
        if isinstance(dims, int):
            dims = [dims]
        elif isinstance(dims, list):
            if not all(isinstance(d, int) for d in dims):
                raise ValueError("All dimensions must be integers.")
        else:
            raise TypeError("Dimensions must be an integer or a list of integers.")

        # Cache the problem configurations
        self.num_blocks: int = len(dims)
        self.dims: list[int] = dims
        self.seed: int = seed
        self.np_dtype = np_dtype
        self.wp_dtype = wp_dtype
        self.device: Devicelike = device

        # Declare the lists of reference problem data
        self.A_np: list[np.ndarray] = []
        self.b_np: list[np.ndarray] = []
        self.X_np: list[np.ndarray] = []
        self.y_np: list[np.ndarray] = []
        self.x_np: list[np.ndarray] = []

        # Declare the warp arrays of contatenated problem data
        self.mio_wp: wp.array(dtype=int32) | None = None
        self.vio_wp: wp.array(dtype=int32) | None = None
        self.maxdim_wp: wp.array(dtype=int32) | None = None
        self.dim_wp: wp.array(dtype=int32) | None = None
        self.A_wp: wp.array(dtype=wp_dtype) | None = None
        self.b_wp: wp.array(dtype=wp_dtype) | None = None

        # Initialize the flattened problem data
        A_sizes = [n * n for n in self.dims]
        A_offsets = [0] + [sum(A_sizes[:i]) for i in range(1, len(A_sizes) + 1)]
        A_flat_size = sum(A_sizes)
        A_flat = np.ndarray(shape=(A_flat_size,), dtype=np_dtype)
        b_sizes = list(self.dims)
        b_offsets = [0] + [sum(b_sizes[:i]) for i in range(1, len(b_sizes) + 1)]
        b_flat_size = sum(b_sizes)
        b_flat = np.ndarray(shape=(b_flat_size,), dtype=np_dtype)

        # Generate randomized problem data
        for i, n in enumerate(self.dims):
            # Generate a random SPD matrix if not provided
            if A is None:
                A_mat = random_spd_matrix(dim=n, seed=self.seed, dtype=np_dtype)
            else:
                A_mat = A[i]
            # Generate a random RHS vector if not provided
            if b is None:
                b_vec = random_rhs_for_matrix(A_mat)
            else:
                b_vec = b[i]
            # Compute the Cholesky decomposition using numpy
            X_mat = np.linalg.cholesky(A_mat, upper=upper)
            # Compute final and intermediate reference solutions
            if upper:
                y_vec, x_vec = solve_cholesky_upper_numpy(X_mat, b_vec)
            else:
                y_vec, x_vec = solve_cholesky_lower_numpy(X_mat, b_vec)
            # Store the reference data
            self.A_np.append(A_mat)
            self.b_np.append(b_vec)
            self.X_np.append(X_mat)
            self.y_np.append(y_vec)
            self.x_np.append(x_vec)
            # Flatten the matrix and store it in the A_flat array
            A_start = A_offsets[len(self.A_np) - 1]
            A_end = A_offsets[len(self.A_np)]
            A_flat[A_start:A_end] = A_mat.flat
            # Flatten the vector and store it in the b_flat array
            b_start = b_offsets[len(self.b_np) - 1]
            b_end = b_offsets[len(self.b_np)]
            b_flat[b_start:b_end] = b_vec

        # Construct the warp arrays
        with wp.ScopedDevice(self.device):
            self.mio_wp = wp.array(A_offsets[: self.num_blocks], dtype=wp.int32)
            self.vio_wp = wp.array(b_offsets[: self.num_blocks], dtype=wp.int32)
            self.maxdim_wp = wp.array(self.dims, dtype=wp.int32)
            self.dim_wp = wp.array(self.dims, dtype=wp.int32)
            self.A_wp = wp.array(A_flat, dtype=wp.float32)
            self.b_wp = wp.array(b_flat, dtype=wp.float32)

        # Optional verbose output
        if verbose:
            print("\n")
            print(f"CholeskyProblem.blocks: {self.num_blocks}")
            print(f"CholeskyProblem.dims: {self.dims}")
            print(f"CholeskyProblem.seed: {self.seed}")
            print(f"CholeskyProblem.np_dtype: {self.np_dtype}")
            print(f"CholeskyProblem.wp_dtype: {self.wp_dtype}")
            print(f"CholeskyProblem.device: {self.device}")
            print(f"CholeskyProblem.self.A_np (shape): {[A.shape for A in self.A_np]}")
            print(f"CholeskyProblem.self.b_np (shape): {[b.shape for b in self.b_np]}")
            print(f"CholeskyProblem.self.X_np (shape): {[X.shape for X in self.X_np]}")
            print(f"CholeskyProblem.self.y_np (shape): {[y.shape for y in self.y_np]}")
            print(f"CholeskyProblem.self.x_np (shape): {[x.shape for x in self.x_np]}")
            print(f"CholeskyProblem.self.mio_wp: {self.mio_wp.numpy()}")
            print(f"CholeskyProblem.self.vio_wp: {self.vio_wp.numpy()}")
            print(f"CholeskyProblem.self.maxdim_wp: {self.maxdim_wp.numpy()}")
            print(f"CholeskyProblem.self.dim_wp: {self.dim_wp.numpy()}")
            print(f"CholeskyProblem.self.A_wp (shape): {self.A_wp.shape}")
            print(f"CholeskyProblem.self.b_wp (shape): {self.b_wp.shape}")


class RandomProblemLDLT:
    def __init__(
        self,
        seed: int = 42,
        dims: list[int] | int | None = None,
        ranks: list[int] | int | None = None,
        eigenvalues: ArrayLike = None,
        A: list[np.ndarray] | None = None,
        b: list[np.ndarray] | None = None,
        np_dtype=np.float32,
        wp_dtype=float32,
        device: Devicelike = None,
        lower: bool = True,
        verbose: bool = False,
    ):
        # Check input data to ensure they are indeed lists of numpy arrays
        if A is not None:
            if not isinstance(A, list) or not all(isinstance(a, np.ndarray) for a in A):
                raise TypeError("A must be a list of numpy arrays.")
            dims = [a.shape[0] for a in A]  # Update dims based on provided A
        if b is not None:
            if not isinstance(b, list) or not all(isinstance(b_i, np.ndarray) for b_i in b):
                raise TypeError("b must be a list of numpy arrays.")

        # Ensure the problem dimensions are valid
        if isinstance(dims, int):
            dims = [dims]
        elif isinstance(dims, list):
            if not all(isinstance(d, int) for d in dims):
                raise ValueError("All dimensions must be integers.")
        else:
            raise TypeError("Dimensions must be an integer or a list of integers.")

        # Ensure the rank dimensions are valid
        if ranks is not None:
            if isinstance(ranks, int):
                ranks = [ranks]
            elif isinstance(ranks, list):
                if not all(isinstance(r, int) for r in ranks):
                    raise ValueError("All ranks must be integers.")
            else:
                raise TypeError("Ranks must be an integer or a list of integers.")
        else:
            ranks = [None] * len(dims)

        # Ensure the eigenvalues are valid
        if eigenvalues is not None:
            if not isinstance(eigenvalues, list) or not all(isinstance(ev, int | float) for ev in eigenvalues):
                raise TypeError("Eigenvalues must be a list of numbers.")
        else:
            eigenvalues = [None] * len(dims)

        # Cache the problem configurations
        self.num_blocks: int = len(dims)
        self.dims: list[int] = dims
        self.seed: int = seed
        self.np_dtype = np_dtype
        self.wp_dtype = wp_dtype
        self.device: Devicelike = device

        # Declare the lists of reference problem data
        self.A_np: list[np.ndarray] = []
        self.b_np: list[np.ndarray] = []
        self.X_np: list[np.ndarray] = []
        self.D_np: list[np.ndarray] = []
        self.P_np: list[np.ndarray] = []
        self.y_np: list[np.ndarray] = []
        self.z_np: list[np.ndarray] = []
        self.x_np: list[np.ndarray] = []

        # Declare the warp arrays of contatenated problem data
        self.mio_wp: wp.array(dtype=int32) | None = None
        self.vio_wp: wp.array(dtype=int32) | None = None
        self.maxdim_wp: wp.array(dtype=int32) | None = None
        self.dim_wp: wp.array(dtype=int32) | None = None
        self.A_wp: wp.array(dtype=wp_dtype) | None = None
        self.b_wp: wp.array(dtype=wp_dtype) | None = None

        # Initialize the flattened problem data
        A_sizes = [n * n for n in self.dims]
        A_offsets = [0] + [sum(A_sizes[:i]) for i in range(1, len(A_sizes) + 1)]
        A_flat_size = sum(A_sizes)
        A_flat = np.ndarray(shape=(A_flat_size,), dtype=np_dtype)
        b_sizes = list(self.dims)
        b_offsets = [0] + [sum(b_sizes[:i]) for i in range(1, len(b_sizes) + 1)]
        b_flat_size = sum(b_sizes)
        b_flat = np.ndarray(shape=(b_flat_size,), dtype=np_dtype)

        # Generate randomized problem data
        for i, n in enumerate(self.dims):
            # Generate a random SPD matrix if not provided
            if A is None:
                A_mat = random_symmetric_matrix(
                    dim=n, seed=self.seed, rank=ranks[i], eigenvalues=eigenvalues[i], dtype=np_dtype
                )
            else:
                A_mat = A[i]
            # Generate a random RHS vector if not provided
            if b is None:
                b_vec = random_rhs_for_matrix(A_mat)
            else:
                b_vec = b[i]
            # Compute the LDLT decomposition using numpy
            X_mat, D_mat, P_mat = sp.linalg.ldl(A_mat, lower=lower)
            # Compute final and intermediate reference solutions
            if lower:
                y_vec, z_vec, x_vec = solve_ldlt_lower_numpy(X_mat, D_mat, P_mat, b_vec)
            else:
                y_vec, z_vec, x_vec = solve_ldlt_upper_numpy(X_mat, D_mat, P_mat, b_vec)
            # Store the reference data
            self.A_np.append(A_mat)
            self.b_np.append(b_vec)
            self.X_np.append(X_mat)
            self.D_np.append(D_mat)
            self.P_np.append(P_mat)
            self.y_np.append(y_vec)
            self.z_np.append(z_vec)
            self.x_np.append(x_vec)
            # Flatten the matrix and store it in the A_flat array
            A_start = A_offsets[len(self.A_np) - 1]
            A_end = A_offsets[len(self.A_np)]
            A_flat[A_start:A_end] = A_mat.flat
            # Flatten the vector and store it in the b_flat array
            b_start = b_offsets[len(self.b_np) - 1]
            b_end = b_offsets[len(self.b_np)]
            b_flat[b_start:b_end] = b_vec

        # Construct the warp arrays
        with wp.ScopedDevice(self.device):
            self.mio_wp = wp.array(A_offsets[: self.num_blocks], dtype=wp.int32)
            self.vio_wp = wp.array(b_offsets[: self.num_blocks], dtype=wp.int32)
            self.maxdim_wp = wp.array(self.dims, dtype=wp.int32)
            self.dim_wp = wp.array(self.dims, dtype=wp.int32)
            self.A_wp = wp.array(A_flat, dtype=wp.float32)
            self.b_wp = wp.array(b_flat, dtype=wp.float32)

        # Optional verbose output
        if verbose:
            print("\n")
            print(f"LDLTProblem.blocks: {self.num_blocks}")
            print(f"LDLTProblem.dims: {self.dims}")
            print(f"LDLTProblem.seed: {self.seed}")
            print(f"LDLTProblem.np_dtype: {self.np_dtype}")
            print(f"LDLTProblem.wp_dtype: {self.wp_dtype}")
            print(f"LDLTProblem.device: {self.device}")
            print(f"LDLTProblem.self.A_np (shape): {[A.shape for A in self.A_np]}")
            print(f"LDLTProblem.self.b_np (shape): {[b.shape for b in self.b_np]}")
            print(f"LDLTProblem.self.X_np (shape): {[X.shape for X in self.X_np]}")
            print(f"LDLTProblem.self.D_np (shape): {[D.shape for D in self.D_np]}")
            print(f"LDLTProblem.self.P_np (shape): {[P.shape for P in self.P_np]}")
            print(f"LDLTProblem.self.y_np (shape): {[y.shape for y in self.y_np]}")
            print(f"LDLTProblem.self.z_np (shape): {[z.shape for z in self.z_np]}")
            print(f"LDLTProblem.self.x_np (shape): {[x.shape for x in self.x_np]}")
            print(f"LDLTProblem.self.mio_wp: {self.mio_wp.numpy()}")
            print(f"LDLTProblem.self.vio_wp: {self.vio_wp.numpy()}")
            print(f"LDLTProblem.self.maxdim_wp: {self.maxdim_wp.numpy()}")
            print(f"LDLTProblem.self.dim_wp: {self.dim_wp.numpy()}")
            print(f"LDLTProblem.self.A_wp (shape): {self.A_wp.shape}")
            print(f"LDLTProblem.self.b_wp (shape): {self.b_wp.shape}")
