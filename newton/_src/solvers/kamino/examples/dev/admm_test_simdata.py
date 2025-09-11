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

"""TODO"""

import os

import h5py
import numpy as np
from scipy import linalg

import newton._src.solvers.kamino.utils.logger as msg
from newton._src.solvers.kamino.tests.utils.print import print_error_stats
from newton._src.solvers.kamino.utils.io import hdf5
from newton._src.solvers.kamino.utils.linalg import (
    ADMMSolver,
    RectangularMatrixProperties,
    SquareSymmetricMatrixProperties,
    compute_u_plus,
)
from newton._src.solvers.kamino.utils.sparse import sparseview

###
# Helper functions
###


def clip_below(A: np.ndarray, min: float = 0.0) -> np.ndarray:
    A_clip = np.zeros_like(A)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if np.abs(A[i, j]) < min:
                A_clip[i, j] = 0.0
            else:
                A_clip[i, j] = A[i, j]
    return A_clip


def symmetry_info(A: np.ndarray, name: str = "A", title: str = "A", eps: float = 1e-12):
    # Compute the error matrix between the kamino Delassus matrix and its transpose
    A_sym_err = A - A.T

    # Print error statistics
    print_error_stats(f"{name}_sym", A, A.T, n=A.size, show_errors=False)

    # Clip small errors to zero for visualization
    A_sym_err_clip = clip_below(A_sym_err, min=eps)

    # Visualize the error matrix as an image
    os.makedirs(PLOT_OUTPUT_PATH, exist_ok=True)
    sparseview(A_sym_err, title=f"{title} Symmetry Error", path=os.path.join(PLOT_OUTPUT_PATH, f"{name}_sym_err.png"))
    sparseview(
        A_sym_err_clip,
        title=f"{title} Symmetry Error (Clipped)",
        path=os.path.join(PLOT_OUTPUT_PATH, f"{name}_sym_err_clip.png"),
    )


def linsys_residual(A: np.ndarray, b: np.ndarray, x: np.ndarray) -> np.ndarray:
    return A @ x - b


def linsys_residual_infnorm(A: np.ndarray, b: np.ndarray, x: np.ndarray) -> float:
    return np.max(np.abs(linsys_residual(A, b, x)))


def linsys_residual_l2norm(A: np.ndarray, b: np.ndarray, x: np.ndarray) -> float:
    return np.linalg.norm(linsys_residual(A, b, x))


###
# Constants
###

# PROBLEM_NAME = "boxes_hinged"
# PROBLEM_NAME = "fourbar_free"
PROBLEM_NAME = "walker"

# Retrieve the path to the data directory
DATA_DIR_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))

# Set the path to the generated HDF5 dataset file
# HDF5_DATASET_PATH = f"{DATA_DIR_PATH}/hdf5/{PROBLEM_NAME}.hdf5"
HDF5_DATASET_PATH = f"{DATA_DIR_PATH}/hdf5/simdata.hdf5"

# Set path for generated plots
PLOT_OUTPUT_PATH = f"{DATA_DIR_PATH}/plots/{PROBLEM_NAME}"

###
# Main function
###

if __name__ == "__main__":
    # Set global numpy configurations
    np.set_printoptions(linewidth=20000, precision=6, threshold=10000, suppress=True)  # Suppress scientific notation
    msg.set_log_level(msg.LogLevel.INFO)

    # Construct and configure the data containers
    msg.info("Loading HDF5 data containers...")
    datafile_ko = h5py.File(HDF5_DATASET_PATH, "r")

    # Retrieve target data frames
    # FRAME = 1
    # FRAME = 10
    # FRAME = 100
    # FRAME = 173
    # dataframe_ko = datafile_ko[f"Worlds/{PROBLEM_NAME}/frames/{FRAME}/DualProblem"]
    dataframe_ko = datafile_ko[f"Animatronics/{PROBLEM_NAME}/DenseConstraints/0/DualProblem"]

    # Create data containers
    pdata = hdf5.DualProblemData()

    # Extract data from HDF5
    np_dtype = np.float64
    # np_dtype = np.float32
    pdata.load(dataset=dataframe_ko, dtype=np_dtype)

    # Create output directories
    os.makedirs(PLOT_OUTPUT_PATH, exist_ok=True)

    # Configure the machine-precision epsilon based in the imported dtype
    eps = float(np.finfo(np_dtype).eps)

    ###
    # Regularization
    ###

    # Regularization parameters
    # NOTE: These should be configured to those used in the simulation
    eta = np_dtype(1e-6)
    rho = np_dtype(1.0)
    epsilon = rho + eta
    r_epsilon = np_dtype(1) / epsilon
    I_epsilon = epsilon * np.eye(pdata.D.shape[0], dtype=np_dtype)
    msg.warning(
        f"Regularization:"
        f"\nrho: {rho}, {rho.dtype}"
        f"\neta: {eta}, {eta.dtype}"
        f"\nepsilon: {epsilon}, {epsilon.dtype}"
        f"\nr_epsilon: {r_epsilon}, {r_epsilon.dtype}\n"
    )

    ###
    # Dynamics Quantities
    ###

    # Retrieve the time-step from the HDF5 dataset
    dt = pdata.dt
    msg.warning(f"dt: {dt}, {dt.dtype}\n")

    # Retrieve the primal dynamics quantities
    invM = pdata.invM
    M = pdata.M
    J = pdata.J
    h = dt * pdata.h
    u_minus = pdata.u_minus
    v_star = pdata.v_i + pdata.v_b
    msg.warning(f"invM: {np.linalg.norm(invM)}, {invM.shape}, {invM.dtype}")
    msg.warning(f"M: {np.linalg.norm(M)}, {M.shape}, {M.dtype}")
    msg.warning(f"J: {np.linalg.norm(J)}, {J.shape}, {J.dtype}")
    msg.warning(f"h: {np.linalg.norm(h)}, {h.shape}, {h.dtype}")
    msg.warning(f"u_minus: {np.linalg.norm(u_minus)}, {u_minus.shape}, {u_minus.dtype}")
    msg.warning(f"v_star: {np.linalg.norm(v_star)}, {v_star.shape}, {v_star.dtype}\n")

    # Extract problem dimensions
    nbd = M.shape[0]
    ncts = J.shape[0]
    msg.warning(f"nbd: {nbd}")
    msg.warning(f"ncts: {ncts}\n")

    ###
    # Dynamics matrices
    ###

    # Compute properties of the constraint Jacobian matrix
    properties_J = RectangularMatrixProperties(J)
    print(f"J properties:\n{properties_J}\n")

    # Compute properties of the generalized mass matrix
    properties_M = SquareSymmetricMatrixProperties(M)
    print(f"M properties:\n{properties_M}\n")

    # Compute properties of the generalized mass matrix
    properties_invM = SquareSymmetricMatrixProperties(invM)
    print(f"invM properties:\n{properties_invM}\n")

    # Visualize the error matrix as an image
    sparseview(J, title="Constraint Jacobian", path=os.path.join(PLOT_OUTPUT_PATH, "J.png"))
    sparseview(M, title="Generalized Mass", path=os.path.join(PLOT_OUTPUT_PATH, "M.png"))
    sparseview(invM, title="Inverse Generalized Mass", path=os.path.join(PLOT_OUTPUT_PATH, "invM.png"))

    ###
    # KKT system
    ###

    # Assemble the KKT matrix
    kdim = nbd + ncts
    K = np.zeros((kdim, kdim), dtype=np_dtype)
    K[:nbd, :nbd] = M
    K[:nbd, nbd:] = J.T
    K[nbd:, :nbd] = J
    msg.warning(f"K: {np.linalg.norm(K)}, {K.shape}, {K.dtype}")

    # Assemble the KKT rhs
    k = np.zeros((kdim,), dtype=np_dtype)
    k[:nbd] = M @ u_minus + h
    k[nbd:] = -v_star
    msg.warning(f"k: {np.linalg.norm(k)}, {k.shape}, {k.dtype}\n")

    # Compute matrix properties
    properties_K = SquareSymmetricMatrixProperties(K)
    print(f"K properties:\n{properties_K}")

    # Render the KKT matrix and symmetry error info as images
    sparseview(K, title="KKT Matrix", path=os.path.join(PLOT_OUTPUT_PATH, "K.png"))
    symmetry_info(K, name="K", title="KKT", eps=eps)

    # # Correct symmetry of the KKT matrix
    # K_np = 0.5 * (K_np + K_np.T)

    ###
    # Dual system
    ###

    # Retrieve the dual dynamics quantities
    # NOTE: These are also the dual Schur-complement quantities
    D = pdata.D
    # D -= I_epsilon  # Remove Delassus regularization
    d = -pdata.v_f
    v_f = pdata.v_f - v_star  # Remove the free-velocity bias
    msg.warning(f"D: {np.linalg.norm(D)}, {D.shape}, {D.dtype}")
    msg.warning(f"d: {np.linalg.norm(d)}, {d.shape}, {d.dtype}\n")

    # Compute matrix properties
    properties_D = SquareSymmetricMatrixProperties(D)
    print(f"D properties:\n{properties_D}\n")

    # Render the primal schur complement matrix and symmetry error info as images
    sparseview(D, title="Delassus", path=os.path.join(PLOT_OUTPUT_PATH, "D.png"))
    symmetry_info(D, name="D", title="Dual Schur-complement", eps=eps)

    ###
    # Reference solutions using numpy & scipy
    ###

    x_np = np.linalg.solve(D, d)
    u_np = compute_u_plus(u_minus, invM, J, h, x_np)
    ux_np = np.concatenate((u_np, -x_np))
    msg.warning(f"ux_np: {np.linalg.norm(ux_np)}")
    msg.warning(f"u_np: {np.linalg.norm(u_np)}")
    msg.warning(f"x_np: {np.linalg.norm(x_np)}\n")

    x_sp = linalg.solve(D, d)
    u_sp = compute_u_plus(u_minus, invM, J, h, x_sp)
    ux_sp = np.concatenate((u_sp, -x_sp))
    msg.warning(f"ux_sp: {np.linalg.norm(ux_sp)}")
    msg.warning(f"u_sp: {np.linalg.norm(u_sp)}")
    msg.warning(f"x_sp: {np.linalg.norm(x_sp)}\n")

    ###
    # ADMM Reproduction Test
    ###

    # Configure and initialize the ADMM solver
    admm = ADMMSolver(
        dtype=np_dtype,
        primal_tolerance=1e-6,
        dual_tolerance=1e-6,
        compl_tolerance=1e-6,
        eta=1e-3,
        rho=1.0,
        omega=1.0,
        maxiter=20,
    )

    # As KKT system
    status = admm.solve_kkt(M, J, h, u_minus, v_star, use_ldlt=False)
    admm.save_info(path=PLOT_OUTPUT_PATH, suffix="_kkt")
    u_admm_kkt = admm.u_plus
    x_admm_kkt = admm.lambdas
    ux_admm_kkt = np.concatenate((u_admm_kkt, -x_admm_kkt))
    print(f"ADMM: KKT: converged: {status.converged}")
    print(f"ADMM: KKT: iterations: {status.iterations}")
    print(f"ADMM: KKT: r_p: {status.r_p}")
    print(f"ADMM: KKT: r_d: {status.r_d}")
    print(f"ADMM: KKT: r_c: {status.r_c}")
    print(f"ADMM: KKT: min(r_p): {np.min(admm.info.r_p)}")
    print(f"ADMM: KKT: min(r_d): {np.min(admm.info.r_d)}")
    print(f"ADMM: KKT: min(r_c): {np.min(admm.info.r_c)}")
    print(f"ADMM: KKT: min(r_p) at iteration: {np.argmin(admm.info.r_p)}")
    print(f"ADMM: KKT: min(r_d) at iteration: {np.argmin(admm.info.r_d)}")
    print(f"ADMM: KKT: min(r_c) at iteration: {np.argmin(admm.info.r_c)}\n")
    properties_K_admm = SquareSymmetricMatrixProperties(admm.K)
    print(f"ADMM K properties:\n{properties_K_admm}")

    # As primal Schur complement system
    status = admm.solve_schur_primal(M, J, h, u_minus, v_star, use_cholesky=False, use_ldlt=False)
    admm.save_info(path=PLOT_OUTPUT_PATH, suffix="_schur_primal")
    x_admm_schur_prim = admm.lambdas
    u_admm_schur_prim = admm.u_plus
    ux_admm_schur_prim = np.concatenate((u_admm_schur_prim, -x_admm_schur_prim))
    print(f"ADMM: Primal Schur: converged: {status.converged}")
    print(f"ADMM: Primal Schur: iterations: {status.iterations}")
    print(f"ADMM: Primal Schur: r_p: {status.r_p}")
    print(f"ADMM: Primal Schur: r_d: {status.r_d}")
    print(f"ADMM: Primal Schur: r_c: {status.r_c}")
    print(f"ADMM: Primal Schur: min(r_p): {np.min(admm.info.r_p)}")
    print(f"ADMM: Primal Schur: min(r_d): {np.min(admm.info.r_d)}")
    print(f"ADMM: Primal Schur: min(r_c): {np.min(admm.info.r_c)}")
    print(f"ADMM: Primal Schur: min(r_p) at iteration: {np.argmin(admm.info.r_p)}")
    print(f"ADMM: Primal Schur: min(r_d) at iteration: {np.argmin(admm.info.r_d)}")
    print(f"ADMM: Primal Schur: min(r_c) at iteration: {np.argmin(admm.info.r_c)}\n")
    properties_P_admm = SquareSymmetricMatrixProperties(admm.P)
    print(f"ADMM P properties:\n{properties_P_admm}")

    # As dual Schur complement system
    status = admm.solve_schur_dual(D, v_f, v_star, u_minus, invM, J, h, use_cholesky=False, use_ldlt=False, use_preconditioning=False)
    admm.save_info(path=PLOT_OUTPUT_PATH, suffix="_schur_dual")
    x_admm_schur_dual = admm.lambdas
    u_admm_schur_dual = admm.u_plus
    ux_admm_schur_dual = np.concatenate((u_admm_schur_dual, -x_admm_schur_dual))
    print(f"ADMM: Dual Schur: converged: {status.converged}")
    print(f"ADMM: Dual Schur: iterations: {status.iterations}")
    print(f"ADMM: Dual Schur: r_p: {status.r_p}")
    print(f"ADMM: Dual Schur: r_d: {status.r_d}")
    print(f"ADMM: Dual Schur: r_c: {status.r_c}")
    print(f"ADMM: Dual Schur: min(r_p): {np.min(admm.info.r_p)}")
    print(f"ADMM: Dual Schur: min(r_d): {np.min(admm.info.r_d)}")
    print(f"ADMM: Dual Schur: min(r_c): {np.min(admm.info.r_c)}")
    print(f"ADMM: Dual Schur: min(r_p) at iteration: {np.argmin(admm.info.r_p)}")
    print(f"ADMM: Dual Schur: min(r_d) at iteration: {np.argmin(admm.info.r_d)}")
    print(f"ADMM: Dual Schur: min(r_c) at iteration: {np.argmin(admm.info.r_c)}\n")
    properties_D_admm = SquareSymmetricMatrixProperties(admm.D)
    print(f"ADMM D properties:\n{properties_D_admm}")

    # As dual Schur complement system w/ preconditioning
    status = admm.solve_schur_dual(D, v_f, v_star, u_minus, invM, J, h, use_cholesky=False, use_ldlt=False, use_preconditioning=True)
    admm.save_info(path=PLOT_OUTPUT_PATH, suffix="_schur_dual_prec")
    x_admm_schur_dual_prec = admm.lambdas
    u_admm_schur_dual_prec = admm.u_plus
    ux_admm_schur_dual_prec = np.concatenate((u_admm_schur_dual_prec, -x_admm_schur_dual_prec))
    print(f"ADMM: Prec. Dual Schur: converged: {status.converged}")
    print(f"ADMM: Prec. Dual Schur: iterations: {status.iterations}")
    print(f"ADMM: Prec. Dual Schur: r_p: {status.r_p}")
    print(f"ADMM: Prec. Dual Schur: r_d: {status.r_d}")
    print(f"ADMM: Prec. Dual Schur: r_c: {status.r_c}")
    print(f"ADMM: Prec. Dual Schur: min(r_p): {np.min(admm.info.r_p)}")
    print(f"ADMM: Prec. Dual Schur: min(r_d): {np.min(admm.info.r_d)}")
    print(f"ADMM: Prec. Dual Schur: min(r_c): {np.min(admm.info.r_c)}")
    print(f"ADMM: Prec. Dual Schur: min(r_p) at iteration: {np.argmin(admm.info.r_p)}")
    print(f"ADMM: Prec. Dual Schur: min(r_d) at iteration: {np.argmin(admm.info.r_d)}")
    print(f"ADMM: Prec. Dual Schur: min(r_c) at iteration: {np.argmin(admm.info.r_c)}\n")
    properties_D_prec_admm = SquareSymmetricMatrixProperties(admm.D)
    print(f"ADMM Precond. D properties:\n{properties_D_prec_admm}")

    ###
    # Performance Metrics
    ###

    k_norm_l2 = np.linalg.norm(k)
    k_norm_inf = np.max(np.abs(k))
    d_norm_l2 = np.linalg.norm(d)
    d_norm_inf = np.max(np.abs(d))

    ux_np_norm = np.linalg.norm(ux_np)
    ux_sp_norm = np.linalg.norm(ux_sp)
    ux_admm_kkt_norm = np.linalg.norm(ux_admm_kkt)
    ux_admm_schur_prim_norm = np.linalg.norm(ux_admm_schur_prim)
    ux_admm_schur_dual_norm = np.linalg.norm(ux_admm_schur_dual)
    ux_admm_schur_dual_prec_norm = np.linalg.norm(ux_admm_schur_dual_prec)

    x_np_norm = np.linalg.norm(x_np)
    x_sp_norm = np.linalg.norm(x_sp)
    x_admm_kkt_norm = np.linalg.norm(x_admm_kkt)
    x_admm_schur_prim_norm = np.linalg.norm(x_admm_schur_prim)
    x_admm_schur_dual_norm = np.linalg.norm(x_admm_schur_dual)
    x_admm_schur_dual_prec_norm = np.linalg.norm(x_admm_schur_dual_prec)

    res_kkt_ux_np_l2 = linsys_residual_l2norm(K, k, ux_np)
    res_kkt_ux_sp_l2 = linsys_residual_l2norm(K, k, ux_sp)
    res_kkt_ux_admm_kkt_l2 = linsys_residual_l2norm(K, k, ux_admm_kkt)
    res_kkt_ux_admm_schur_prim_l2 = linsys_residual_l2norm(K, k, ux_admm_schur_prim)
    res_kkt_ux_admm_schur_dual_l2 = linsys_residual_l2norm(K, k, ux_admm_schur_dual)
    res_kkt_ux_admm_schur_dual_prec_l2 = linsys_residual_l2norm(K, k, ux_admm_schur_dual_prec)

    res_kkt_ux_np_infnorm = linsys_residual_infnorm(K, k, ux_np)
    res_kkt_ux_sp_infnorm = linsys_residual_infnorm(K, k, ux_sp)
    res_kkt_ux_admm_kkt_infnorm = linsys_residual_infnorm(K, k, ux_admm_kkt)
    res_kkt_ux_admm_schur_prim_infnorm = linsys_residual_infnorm(K, k, ux_admm_schur_prim)
    res_kkt_ux_admm_schur_dual_infnorm = linsys_residual_infnorm(K, k, ux_admm_schur_dual)
    res_kkt_ux_admm_schur_dual_prec_infnorm = linsys_residual_infnorm(K, k, ux_admm_schur_dual_prec)

    res_dual_x_np_l2 = linsys_residual_l2norm(D, d, x_np)
    res_dual_x_sp_l2 = linsys_residual_l2norm(D, d, x_sp)
    res_dual_x_admm_kkt_l2 = linsys_residual_l2norm(D, d, x_admm_kkt)
    res_dual_x_admm_schur_prim_l2 = linsys_residual_l2norm(D, d, x_admm_schur_prim)
    res_dual_x_admm_schur_dual_l2 = linsys_residual_l2norm(D, d, x_admm_schur_dual)
    res_dual_x_admm_schur_dual_prec_l2 = linsys_residual_l2norm(D, d, x_admm_schur_dual_prec)

    res_dual_x_np_infnorm = linsys_residual_infnorm(D, d, x_np)
    res_dual_x_sp_infnorm = linsys_residual_infnorm(D, d, x_sp)
    res_dual_x_admm_kkt_infnorm = linsys_residual_infnorm(D, d, x_admm_kkt)
    res_dual_x_admm_schur_prim_infnorm = linsys_residual_infnorm(D, d, x_admm_schur_prim)
    res_dual_x_admm_schur_dual_infnorm = linsys_residual_infnorm(D, d, x_admm_schur_dual)
    res_dual_x_admm_schur_dual_prec_infnorm = linsys_residual_infnorm(D, d, x_admm_schur_dual_prec)

    rel_res_kkt_ux_np_l2 = res_kkt_ux_np_l2 / k_norm_l2 if k_norm_l2 > 0 else res_kkt_ux_np_l2
    rel_res_kkt_ux_sp_l2 = res_kkt_ux_sp_l2 / k_norm_l2 if k_norm_l2 > 0 else res_kkt_ux_sp_l2
    rel_res_kkt_ux_admm_kkt_l2 = res_kkt_ux_admm_kkt_l2 / k_norm_l2 if k_norm_l2 > 0 else res_kkt_ux_admm_kkt_l2
    rel_res_kkt_ux_admm_schur_prim_l2 = res_kkt_ux_admm_schur_prim_l2 / k_norm_l2 if k_norm_l2 > 0 else res_kkt_ux_admm_schur_prim_l2
    rel_res_kkt_ux_admm_schur_dual_l2 = res_kkt_ux_admm_schur_dual_l2 / k_norm_l2 if k_norm_l2 > 0 else res_kkt_ux_admm_schur_dual_l2
    rel_res_kkt_ux_admm_schur_dual_prec_l2 = res_kkt_ux_admm_schur_dual_prec_l2 / k_norm_l2 if k_norm_l2 > 0 else res_kkt_ux_admm_schur_dual_prec_l2

    rel_res_kkt_ux_np_infnorm = res_kkt_ux_np_infnorm / k_norm_inf if k_norm_inf > 0 else res_kkt_ux_np_infnorm
    rel_res_kkt_ux_sp_infnorm = res_kkt_ux_sp_infnorm / k_norm_inf if k_norm_inf > 0 else res_kkt_ux_sp_infnorm
    rel_res_kkt_ux_admm_kkt_infnorm = res_kkt_ux_admm_kkt_infnorm / k_norm_inf if k_norm_inf > 0 else res_kkt_ux_admm_kkt_infnorm
    rel_res_kkt_ux_admm_schur_prim_infnorm = res_kkt_ux_admm_schur_prim_infnorm / k_norm_inf if k_norm_inf > 0 else res_kkt_ux_admm_schur_prim_infnorm
    rel_res_kkt_ux_admm_schur_dual_infnorm = res_kkt_ux_admm_schur_dual_infnorm / k_norm_inf if k_norm_inf > 0 else res_kkt_ux_admm_schur_dual_infnorm
    rel_res_kkt_ux_admm_schur_dual_prec_infnorm = res_kkt_ux_admm_schur_dual_prec_infnorm / k_norm_inf if k_norm_inf > 0 else res_kkt_ux_admm_schur_dual_prec_infnorm

    rel_res_dual_x_np_l2 = res_dual_x_np_l2 / d_norm_l2 if d_norm_l2 > 0 else res_dual_x_np_l2
    rel_res_dual_x_sp_l2 = res_dual_x_sp_l2 / d_norm_l2 if d_norm_l2 > 0 else res_dual_x_sp_l2
    rel_res_dual_x_admm_kkt_l2 = res_dual_x_admm_kkt_l2 / d_norm_l2 if d_norm_l2 > 0 else res_dual_x_admm_kkt_l2
    rel_res_dual_x_admm_schur_prim_l2 = res_dual_x_admm_schur_prim_l2 / d_norm_l2 if d_norm_l2 > 0 else res_dual_x_admm_schur_prim_l2
    rel_res_dual_x_admm_schur_dual_l2 = res_dual_x_admm_schur_dual_l2 / d_norm_l2 if d_norm_l2 > 0 else res_dual_x_admm_schur_dual_l2
    rel_res_dual_x_admm_schur_dual_prec_l2 = res_dual_x_admm_schur_dual_prec_l2 / d_norm_l2 if d_norm_l2 > 0 else res_dual_x_admm_schur_dual_prec_l2

    rel_res_dual_x_np_infnorm = res_dual_x_np_infnorm / d_norm_inf if d_norm_inf > 0 else res_dual_x_np_infnorm
    rel_res_dual_x_sp_infnorm = res_dual_x_sp_infnorm / d_norm_inf if d_norm_inf > 0 else res_dual_x_sp_infnorm
    rel_res_dual_x_admm_kkt_infnorm = res_dual_x_admm_kkt_infnorm / d_norm_inf if d_norm_inf > 0 else res_dual_x_admm_kkt_infnorm
    rel_res_dual_x_admm_schur_prim_infnorm = res_dual_x_admm_schur_prim_infnorm / d_norm_inf if d_norm_inf > 0 else res_dual_x_admm_schur_prim_infnorm
    rel_res_dual_x_admm_schur_dual_infnorm = res_dual_x_admm_schur_dual_infnorm / d_norm_inf if d_norm_inf > 0 else res_dual_x_admm_schur_dual_infnorm
    rel_res_dual_x_admm_schur_dual_prec_infnorm = res_dual_x_admm_schur_dual_prec_infnorm / d_norm_inf if d_norm_inf > 0 else res_dual_x_admm_schur_dual_prec_infnorm

    ###
    # Summary of solving the KKT system
    ###

    # Compare solution norms
    print("\nux NORMS:")
    print(f"ux_np                   : {ux_np_norm}")
    print(f"ux_sp                   : {ux_sp_norm}")
    print(f"ux_admm_kkt             : {ux_admm_kkt_norm}")
    print(f"ux_admm_schur_prim      : {ux_admm_schur_prim_norm}")
    print(f"ux_admm_schur_dual      : {ux_admm_schur_dual_norm}")
    print(f"ux_admm_schur_dual_prec : {ux_admm_schur_dual_prec_norm}")

    # Compare absolute errors
    print("\n(K @ ux - k) ABSOLUTE ERRORS (L2):")
    print(f"ux_np                   : {res_kkt_ux_np_l2}")
    print(f"ux_sp                   : {res_kkt_ux_sp_l2}")
    print(f"ux_admm_kkt             : {res_kkt_ux_admm_kkt_l2}")
    print(f"ux_admm_schur_prim      : {res_kkt_ux_admm_schur_prim_l2}")
    print(f"ux_admm_schur_dual      : {res_kkt_ux_admm_schur_dual_l2}")
    print(f"ux_admm_schur_dual_prec : {res_kkt_ux_admm_schur_dual_prec_l2}")
    print("\n(K @ ux - k) ABSOLUTE ERRORS (INF):")
    print(f"ux_np                   : {res_kkt_ux_np_infnorm}")
    print(f"ux_sp                   : {res_kkt_ux_sp_infnorm}")
    print(f"ux_admm_kkt             : {res_kkt_ux_admm_kkt_infnorm}")
    print(f"ux_admm_schur_prim      : {res_kkt_ux_admm_schur_prim_infnorm}")
    print(f"ux_admm_schur_dual      : {res_kkt_ux_admm_schur_dual_infnorm}")
    print(f"ux_admm_schur_dual_prec : {res_kkt_ux_admm_schur_dual_prec_infnorm}")

    # Compare relative errors
    print("\n(K @ ux - k) RELATIVE ERRORS (L2):")
    print(f"ux_np                   : {rel_res_kkt_ux_np_l2}")
    print(f"ux_sp                   : {rel_res_kkt_ux_sp_l2}")
    print(f"ux_admm_kkt             : {rel_res_kkt_ux_admm_kkt_l2}")
    print(f"ux_admm_schur_prim      : {rel_res_kkt_ux_admm_schur_prim_l2}")
    print(f"ux_admm_schur_dual      : {rel_res_kkt_ux_admm_schur_dual_l2}")
    print(f"ux_admm_schur_dual_prec : {rel_res_kkt_ux_admm_schur_dual_prec_l2}")
    print("\n(K @ ux - k) RELATIVE ERRORS (INF):")
    print(f"ux_np                   : {rel_res_kkt_ux_np_infnorm}")
    print(f"ux_sp                   : {rel_res_kkt_ux_sp_infnorm}")
    print(f"ux_admm_kkt             : {rel_res_kkt_ux_admm_kkt_infnorm}")
    print(f"ux_admm_schur_prim      : {rel_res_kkt_ux_admm_schur_prim_infnorm}")
    print(f"ux_admm_schur_dual      : {rel_res_kkt_ux_admm_schur_dual_infnorm}")
    print(f"ux_admm_schur_dual_prec : {rel_res_kkt_ux_admm_schur_dual_prec_infnorm}")

    ###
    # Summary of solving the dual system
    ###

    # Compare solution norms
    print("\nx NORMS:")
    print(f"x_np                   : {x_np_norm}")
    print(f"x_sp                   : {x_sp_norm}")
    print(f"x_admm_kkt             : {x_admm_kkt_norm}")
    print(f"x_admm_schur_prim      : {x_admm_schur_prim_norm}")
    print(f"x_admm_schur_dual      : {x_admm_schur_dual_norm}")
    print(f"x_admm_schur_dual_prec : {x_admm_schur_dual_prec_norm}")

    # Compare absolute errors
    print("\n(D @ x + v_f) ABSOLUTE ERRORS (L2):")
    print(f"x_np                   : {res_dual_x_np_l2}")
    print(f"x_sp                   : {res_dual_x_sp_l2}")
    print(f"x_admm_kkt             : {res_dual_x_admm_kkt_l2}")
    print(f"x_admm_schur_prim      : {res_dual_x_admm_schur_prim_l2}")
    print(f"x_admm_schur_dual      : {res_dual_x_admm_schur_dual_l2}")
    print(f"x_admm_schur_dual_prec : {res_dual_x_admm_schur_dual_prec_l2}")
    print("\n(D @ x + v_f) ABSOLUTE ERRORS (INF):")
    print(f"x_np                   : {res_dual_x_np_infnorm}")
    print(f"x_sp                   : {res_dual_x_sp_infnorm}")
    print(f"x_admm_kkt             : {res_dual_x_admm_kkt_infnorm}")
    print(f"x_admm_schur_prim      : {res_dual_x_admm_schur_prim_infnorm}")
    print(f"x_admm_schur_dual      : {res_dual_x_admm_schur_dual_infnorm}")
    print(f"x_admm_schur_dual_prec : {res_dual_x_admm_schur_dual_prec_infnorm}")

    # Compare relative errors
    print("\n(D @ x + v_f) RELATIVE ERRORS (L2):")
    print(f"x_np                   : {rel_res_dual_x_np_l2}")
    print(f"x_sp                   : {rel_res_dual_x_sp_l2}")
    print(f"x_admm_kkt             : {rel_res_dual_x_admm_kkt_l2}")
    print(f"x_admm_schur_prim      : {rel_res_dual_x_admm_schur_prim_l2}")
    print(f"x_admm_schur_dual      : {rel_res_dual_x_admm_schur_dual_l2}")
    print(f"x_admm_schur_dual_prec : {rel_res_dual_x_admm_schur_dual_prec_l2}")
    print("\n(D @ x + v_f) RELATIVE ERRORS (INF):")
    print(f"x_np                   : {rel_res_dual_x_np_infnorm}")
    print(f"x_sp                   : {rel_res_dual_x_sp_infnorm}")
    print(f"x_admm_kkt             : {rel_res_dual_x_admm_kkt_infnorm}")
    print(f"x_admm_schur_prim      : {rel_res_dual_x_admm_schur_prim_infnorm}")
    print(f"x_admm_schur_dual      : {rel_res_dual_x_admm_schur_dual_infnorm}")
    print(f"x_admm_schur_dual_prec : {rel_res_dual_x_admm_schur_dual_prec_infnorm}")
