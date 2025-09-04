import os
import h5py
import numpy as np

import newton._src.solvers.kamino.utils.logger as msg
from newton._src.solvers.kamino.utils.io import hdf5
from newton._src.solvers.kamino.utils.sparse import sparseview
from newton._src.solvers.kamino.tests.utils.print import print_error_stats
from newton._src.solvers.kamino.utils.linalg import (
    SquareSymmetricMatrixProperties,
    ADMMSolver,
)


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


def linsys_residual(A: np.ndarray, b: np.ndarray, x: np.ndarray) -> np.ndarray:
    return A @ x - b


def linsys_residual_infnorm(A: np.ndarray, b: np.ndarray, x: np.ndarray) -> float:
    return np.max(np.abs(linsys_residual(A, b, x)))


def linsys_residual_l2norm(A: np.ndarray, b: np.ndarray, x: np.ndarray) -> float:
    return np.linalg.norm(linsys_residual(A, b, x))


def compute_u_next(
    u_p: np.ndarray,
    invM: np.ndarray,
    J: np.ndarray,
    h: np.ndarray,
    lambdas: np.ndarray,
    dt: float
) -> np.ndarray:
    """
    Compute the next-step generalized velocity vector.
    """
    return u_p + invM @ ((J.T @ lambdas) + h)


###
# Constants
###

# PROBLEM_NAME = "boxes_hinged"
# PROBLEM_NAME = "fourbar_free"
PROBLEM_NAME = "walker"

# Retrieve the path to the data directory
DATA_DIR_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))

# Set the path to the generated HDF5 dataset file
HDF5_DATASET_PATH = f"{DATA_DIR_PATH}/hdf5/{PROBLEM_NAME}.hdf5"

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
    datafile_ko = h5py.File(HDF5_DATASET_PATH, 'r')

    # Retrieve target data frames
    # FRAME = 1
    # FRAME = 100
    FRAME = 173
    dataframe_ko = datafile_ko[f'Worlds/{PROBLEM_NAME}/frames/{FRAME}/DualProblem']

    # Create data containers
    pdata = hdf5.DualProblemData()

    # Extract data from HDF5
    # np_dtype = np.float64
    np_dtype = np.float32
    pdata.load(dataset=dataframe_ko, dtype=np_dtype)

    # Create output directories
    os.makedirs(PLOT_OUTPUT_PATH, exist_ok=True)

    # Configure the machine-precision epsilon based in the imported dtype
    eps = float(np.finfo(np_dtype).eps)

    ###
    # Dynamics
    ###

    # Retrieve the time-step from the HDF5 dataset
    dt = pdata.dt
    msg.warning(f"dt: {dt}")

    # Retrieve the EoM dynamics quantities
    invM = pdata.invM
    M = pdata.M
    J = pdata.J
    h = pdata.h
    u_p = pdata.u_minus
    v_star = pdata.v_i + pdata.v_b
    msg.warning(f"invM: {np.linalg.norm(invM)}, {invM.shape}, {invM.dtype}")
    msg.warning(f"M: {np.linalg.norm(M)}, {M.shape}, {M.dtype}")
    msg.warning(f"J: {np.linalg.norm(J)}, {J.shape}, {J.dtype}")
    msg.warning(f"h: {np.linalg.norm(h)}, {h.shape}, {h.dtype}")
    msg.warning(f"u_p: {np.linalg.norm(u_p)}, {u_p.shape}, {u_p.dtype}")
    msg.warning(f"v_star: {np.linalg.norm(v_star)}, {v_star.shape}, {v_star.dtype}\n")

    # Extract problem dimensions
    nbd = M.shape[0]
    ncts = J.shape[0]
    msg.warning(f"nbd: {nbd}")
    msg.warning(f"ncts: {ncts}\n")

    ###
    # Jacobians
    ###

    # Visualize the error matrix as an image
    sparseview(J, title="Jacobian", path=os.path.join(PLOT_OUTPUT_PATH, "J_np.png"))

    ###
    # Delassus
    ###

    # Extract Delassus matrices
    D_np = pdata.D
    msg.warning(f"D_np: {D_np.shape}, {D_np.dtype}")
    # msg.info(f"D_np: {D_np.shape}:\n{D_np}")

    # Optionally remove existing regularization
    D_np -= (1.0 + 1e-3) * np.eye(D_np.shape[0])

    # Compute matrix properties
    properties_D = SquareSymmetricMatrixProperties(D_np)
    print(f"Delassus properties:\n{properties_D}\n")
    # print(f"Delassus eigenvalues:\n{properties_D.lambdas}\n")

    # Visualize the error matrix as an image
    sparseview(D_np, title="Delassus", path=os.path.join(PLOT_OUTPUT_PATH, "D_np.png"))

    # Compute the error matrix between the kamino Delassus matrix and its transpose
    D_np_err = D_np - D_np.T

    # Print error statistics
    print_error_stats("D_DT_ko", D_np, D_np.T, n=D_np.size, show_errors=False)

    # Clip small errors to zero for visualization
    eps_limit = eps
    D_np_err_clip = clip_below(D_np_err, min=eps_limit)

    # Visualize the error matrix as an image
    os.makedirs(PLOT_OUTPUT_PATH, exist_ok=True)
    sparseview(D_np_err, title="Delassus Symmetry Error", path=os.path.join(PLOT_OUTPUT_PATH, "D_np_err.png"))
    sparseview(D_np_err_clip, title="Delassus Symmetry Error (Clipped)", path=os.path.join(PLOT_OUTPUT_PATH, "D_np_err_clip.png"))

    ###
    # Dual linear system
    ###

    v_np = pdata.v_f
    msg.warning(f"v_np: {v_np.shape}, {v_np.dtype}")
    msg.warning(f"D_np: {np.linalg.norm(D_np)}")
    msg.warning(f"v_np: {np.linalg.norm(v_np)}")

    x_np = np.linalg.solve(D_np, -v_np)
    u_n_np = compute_u_next(u_p, invM, J, h, x_np, dt)
    msg.warning(f"x_np: {np.linalg.norm(x_np)}")
    msg.warning(f"u_n_np: {np.linalg.norm(u_n_np)}\n")

    ###
    # KKT matrix
    ###

    kdim = nbd + ncts
    K_np = np.zeros((kdim, kdim), dtype=M.dtype)
    K_np[:nbd, :nbd] = M
    K_np[:nbd, nbd:] = J.T
    K_np[nbd:, :nbd] = J
    K_np[nbd:, nbd:] = np.zeros(ncts, dtype=M.dtype)
    msg.warning(f"K_np: {K_np.shape}")

    # Compute matrix properties
    properties_K = SquareSymmetricMatrixProperties(K_np)
    print(f"K_np properties:\n{properties_K}")

    # Visualize the error matrix as an image
    sparseview(K_np, title="KKT Matrix", path=os.path.join(PLOT_OUTPUT_PATH, "K_np.png"))

    # Compute the error matrix between the kamino Delassus matrix and its transpose
    K_np_err = K_np - K_np.T

    # Print error statistics
    print_error_stats("K_KT_np", K_np, K_np.T, n=K_np.size, show_errors=False)

    # Clip small errors to zero for visualization
    eps_limit = eps
    K_np_err_clip = clip_below(K_np_err, min=eps_limit)

    # Visualize the error matrix as an image
    os.makedirs(PLOT_OUTPUT_PATH, exist_ok=True)
    sparseview(K_np_err, title="KKT Symmetry Error", path=os.path.join(PLOT_OUTPUT_PATH, "K_np_err.png"))
    sparseview(K_np_err_clip, title="KKT Symmetry Error (Clipped)", path=os.path.join(PLOT_OUTPUT_PATH, "K_np_err_clip.png"))

    # Correct symmetry of the KKT matrix
    K_np = 0.5 * (K_np + K_np.T)

    ###
    # KKT linear system
    ###

    k_np = np.zeros((kdim,), dtype=M.dtype)
    k_np[:nbd] = M @ u_p + h
    k_np[nbd:] = - v_star
    msg.warning(f"k_np: {k_np.shape}")
    msg.warning(f"K_np: {np.linalg.norm(K_np)}")
    msg.warning(f"k_np: {np.linalg.norm(k_np)}")

    ux_np = np.concatenate((u_n_np, x_np))
    msg.warning(f"ux_np: {np.linalg.norm(ux_np)}\n")

    ###
    # Solve w/ naive preconditioning
    ###

    D_cp = np.copy(D_np)
    D_cp_scale = 1.0 / np.max(np.abs(D_cp))
    D_cp_scaled = D_cp_scale * D_cp
    v_np_scaled = D_cp_scale * v_np
    x_np_scaled = np.linalg.solve(D_cp_scaled, -v_np_scaled)
    u_n_np_scaled = compute_u_next(u_p, invM, J, h, x_np_scaled, dt)
    msg.warning(f"D_cp_scale: {D_cp_scale}")
    msg.warning(f"D_cp_scaled: {np.linalg.norm(D_cp_scaled)}, {D_cp_scaled.shape}, {D_cp_scaled.dtype}")
    msg.warning(f"v_np_scaled: {np.linalg.norm(v_np_scaled)}, {v_np_scaled.shape}, {v_np_scaled.dtype}")
    msg.warning(f"x_np_scaled: {np.linalg.norm(x_np_scaled)}, {x_np_scaled.shape}, {x_np_scaled.dtype}")
    msg.warning(f"u_n_np_scaled: {np.linalg.norm(u_n_np_scaled)}, {u_n_np_scaled.shape}, {u_n_np_scaled.dtype}\n")

    ux_np_scaled = np.concatenate((u_n_np_scaled, x_np_scaled))
    msg.warning(f"ux_np_scaled: {np.linalg.norm(ux_np_scaled)}, {ux_np_scaled.shape}, {ux_np_scaled.dtype}\n")

    ###
    # ADMM Reproduction Test
    ###

    admm = ADMMSolver(
        primal_tolerance=1e-6,
        dual_tolerance=1e-6,
        compl_tolerance=1e-6,
        eta=1e-3,
        rho=1.0,
        omega=1.0,
        maxiter=200,
    )

    # As Schur (i.e. dual) system
    status = admm.solve_schur(D_np, v_np, use_cholesky=False, use_ldlt=True)
    admm.save_info(path=PLOT_OUTPUT_PATH, suffix="_schur")
    x_admm_schur = admm.lambdas
    u_n_admm_schur = compute_u_next(u_p, invM, J, h, x_admm_schur, dt)
    ux_admm_schur = np.concatenate((u_n_admm_schur, x_admm_schur))
    print(f"ADMM: Schur: converged: {status.converged}")
    print(f"ADMM: Schur: iterations: {status.iterations}")
    print(f"ADMM: Schur: r_p: {status.r_p}")
    print(f"ADMM: Schur: r_d: {status.r_d}")
    print(f"ADMM: Schur: r_c: {status.r_c}")
    print("\n")

    min_r_p_idx = np.argmin(admm.info.r_p)
    min_r_d_idx = np.argmin(admm.info.r_d)
    min_r_c_idx = np.argmin(admm.info.r_c)
    print(f"ADMM: Schur: min(r_p): {np.min(admm.info.r_p)}")
    print(f"ADMM: Schur: min(r_d): {np.min(admm.info.r_d)}")
    print(f"ADMM: Schur: min(r_c): {np.min(admm.info.r_c)}")
    print(f"ADMM: Schur: min(r_p) at iteration: {min_r_p_idx}")
    print(f"ADMM: Schur: min(r_d) at iteration: {min_r_d_idx}")
    print(f"ADMM: Schur: min(r_c) at iteration: {min_r_c_idx}")
    print("\n")

    properties_D_schur = SquareSymmetricMatrixProperties(admm.D)
    print(f"ADMM Schur properties:\n{properties_D_schur}")

    # As Schur (i.e. dual) system with preconditioning
    status = admm.solve_schur_preconditioned(D_np, v_np, use_cholesky=False, use_ldlt=True)
    admm.save_info(path=PLOT_OUTPUT_PATH, suffix="_schur_prec")
    x_admm_schur_prec = admm.lambdas
    u_n_admm_schur_prec = compute_u_next(u_p, invM, J, h, x_admm_schur_prec, dt)
    ux_admm_schur_prec = np.concatenate((u_n_admm_schur_prec, x_admm_schur_prec))
    print(f"ADMM: Schur Prec.: converged: {status.converged}")
    print(f"ADMM: Schur Prec.: iterations: {status.iterations}")
    print(f"ADMM: Schur Prec.: r_p: {status.r_p}")
    print(f"ADMM: Schur Prec.: r_d: {status.r_d}")
    print(f"ADMM: Schur Prec.: r_c: {status.r_c}")
    print("\n")

    min_r_p_idx = np.argmin(admm.info.r_p)
    min_r_d_idx = np.argmin(admm.info.r_d)
    min_r_c_idx = np.argmin(admm.info.r_c)
    print(f"ADMM: Schur Prec.: min(r_p): {np.min(admm.info.r_p)}")
    print(f"ADMM: Schur Prec.: min(r_d): {np.min(admm.info.r_d)}")
    print(f"ADMM: Schur Prec.: min(r_c): {np.min(admm.info.r_c)}")
    print(f"ADMM: Schur Prec.: min(r_p) at iteration: {min_r_p_idx}")
    print(f"ADMM: Schur Prec.: min(r_d) at iteration: {min_r_d_idx}")
    print(f"ADMM: Schur Prec.: min(r_c) at iteration: {min_r_c_idx}")
    print("\n")

    properties_D_schur_prec = SquareSymmetricMatrixProperties(admm.Dp)
    print(f"ADMM Schur Preconditioned properties:\n{properties_D_schur_prec}")

    # As KKT system
    status = admm.solve_kkt(M, J, h, u_p, v_star, use_ldlt=True)
    admm.save_info(path=PLOT_OUTPUT_PATH, suffix="_kkt")
    x_admm_kkt = admm.lambdas
    u_n_admm_kkt = admm.u_plus
    ux_admm_kkt = np.concatenate((u_n_admm_kkt, x_admm_kkt))

    print(f"ADMM: KKT: converged: {status.converged}")
    print(f"ADMM: KKT: iterations: {status.iterations}")
    print(f"ADMM: KKT: r_p: {status.r_p}")
    print(f"ADMM: KKT: r_d: {status.r_d}")
    print(f"ADMM: KKT: r_c: {status.r_c}")
    print("\n")

    min_r_p_idx = np.argmin(admm.info.r_p)
    min_r_d_idx = np.argmin(admm.info.r_d)
    min_r_c_idx = np.argmin(admm.info.r_c)
    print(f"ADMM: KKT: min(r_p): {np.min(admm.info.r_p)}")
    print(f"ADMM: KKT: min(r_d): {np.min(admm.info.r_d)}")
    print(f"ADMM: KKT: min(r_c): {np.min(admm.info.r_c)}")
    print(f"ADMM: KKT: min(r_p) at iteration: {min_r_p_idx}")
    print(f"ADMM: KKT: min(r_d) at iteration: {min_r_d_idx}")
    print(f"ADMM: KKT: min(r_c) at iteration: {min_r_c_idx}")
    print("\n")

    properties_K_admm = SquareSymmetricMatrixProperties(admm.K)
    print(f"ADMM KKT properties:\n{properties_K_admm}")

    ###
    # Metrics
    ###

    v_np_norm_inf = np.max(np.abs(v_np))
    v_np_norm_l2 = np.linalg.norm(v_np)

    k_np_norm_inf = np.max(np.abs(k_np))
    k_np_norm_l2 = np.linalg.norm(k_np)

    x_np_norm = np.linalg.norm(x_np)
    x_np_scaled_norm = np.linalg.norm(x_np_scaled)
    x_admm_schur_norm = np.linalg.norm(x_admm_schur)
    x_admm_schur_prec_norm = np.linalg.norm(x_admm_schur_prec)
    x_admm_kkt_norm = np.linalg.norm(x_admm_kkt)

    ux_np_norm = np.linalg.norm(ux_np)
    ux_np_scaled_norm = np.linalg.norm(ux_np_scaled)
    ux_admm_schur_norm = np.linalg.norm(ux_admm_schur)
    ux_admm_schur_prec_norm = np.linalg.norm(ux_admm_schur_prec)
    ux_admm_kkt_norm = np.linalg.norm(ux_admm_kkt)

    dx_np_scaled = x_np_scaled - x_np
    dx_admm_schur = x_admm_schur - x_np
    dx_admm_schur_prec = x_admm_schur_prec - x_np
    dx_admm_kkt = x_admm_kkt - x_np

    dux_np_scaled = ux_np_scaled - ux_np
    dux_admm_schur = ux_admm_schur - ux_np
    dux_admm_schur_prec = ux_admm_schur_prec - ux_np
    dux_admm_kkt = ux_admm_kkt - ux_np

    dx_np_scaled_infnorm = np.max(np.abs(dx_np_scaled))
    dx_admm_schur_infnorm = np.max(np.abs(dx_admm_schur))
    dx_admm_schur_prec_infnorm = np.max(np.abs(dx_admm_schur_prec))
    dx_admm_kkt_infnorm = np.max(np.abs(dx_admm_kkt))

    dx_np_scaled_l2 = np.linalg.norm(dx_np_scaled)
    dx_admm_schur_l2 = np.linalg.norm(dx_admm_schur)
    dx_admm_schur_prec_l2 = np.linalg.norm(dx_admm_schur_prec)
    dx_admm_kkt_l2 = np.linalg.norm(dx_admm_kkt)

    dux_np_scaled_infnorm = np.max(np.abs(dux_np_scaled))
    dux_admm_schur_infnorm = np.max(np.abs(dux_admm_schur))
    dux_admm_schur_prec_infnorm = np.max(np.abs(dux_admm_schur_prec))
    dux_admm_kkt_infnorm = np.max(np.abs(dux_admm_kkt))

    dux_np_scaled_l2 = np.linalg.norm(dux_np_scaled)
    dux_admm_schur_l2 = np.linalg.norm(dux_admm_schur)
    dux_admm_schur_prec_l2 = np.linalg.norm(dux_admm_schur_prec)
    dux_admm_kkt_l2 = np.linalg.norm(dux_admm_kkt)

    res_dual_x_np_infnorm = linsys_residual_infnorm(D_np, -v_np, x_np)
    res_dual_x_np_scaled_infnorm = linsys_residual_infnorm(D_np, -v_np, x_np_scaled)
    res_dual_x_admm_schur_infnorm = linsys_residual_infnorm(D_np, -v_np, x_admm_schur)
    res_dual_x_admm_schur_prec_infnorm = linsys_residual_infnorm(D_np, -v_np, x_admm_schur_prec)
    res_dual_x_admm_kkt_infnorm = linsys_residual_infnorm(D_np, -v_np, x_admm_kkt)

    rel_res_dual_x_np_infnorm = res_dual_x_np_infnorm / v_np_norm_inf if v_np_norm_inf > 0 else res_dual_x_np_infnorm
    rel_res_dual_x_np_scaled_infnorm = res_dual_x_np_scaled_infnorm / v_np_norm_inf if v_np_norm_inf > 0 else res_dual_x_np_scaled_infnorm
    rel_res_dual_x_admm_schur_infnorm = res_dual_x_admm_schur_infnorm / v_np_norm_inf if v_np_norm_inf > 0 else res_dual_x_admm_schur_infnorm
    rel_res_dual_x_admm_schur_prec_infnorm = res_dual_x_admm_schur_prec_infnorm / v_np_norm_inf if v_np_norm_inf > 0 else res_dual_x_admm_schur_prec_infnorm
    rel_res_dual_x_admm_kkt_infnorm = res_dual_x_admm_kkt_infnorm / v_np_norm_inf if v_np_norm_inf > 0 else res_dual_x_admm_kkt_infnorm

    res_dual_x_np_l2 = linsys_residual_l2norm(D_np, -v_np, x_np)
    res_dual_x_np_scaled_l2 = linsys_residual_l2norm(D_np, -v_np, x_np_scaled)
    res_dual_x_admm_schur_l2 = linsys_residual_l2norm(D_np, -v_np, x_admm_schur)
    res_dual_x_admm_schur_prec_l2 = linsys_residual_l2norm(D_np, -v_np, x_admm_schur_prec)
    res_dual_x_admm_kkt_l2 = linsys_residual_l2norm(D_np, -v_np, x_admm_kkt)

    rel_res_dual_x_np_l2 = res_dual_x_np_l2 / v_np_norm_l2 if v_np_norm_l2 > 0 else res_dual_x_np_l2
    rel_res_dual_x_np_scaled_l2 = res_dual_x_np_scaled_l2 / v_np_norm_l2 if v_np_norm_l2 > 0 else res_dual_x_np_scaled_l2
    rel_res_dual_x_admm_schur_l2 = res_dual_x_admm_schur_l2 / v_np_norm_l2 if v_np_norm_l2 > 0 else res_dual_x_admm_schur_l2
    rel_res_dual_x_admm_schur_prec_l2 = res_dual_x_admm_schur_prec_l2 / v_np_norm_l2 if v_np_norm_l2 > 0 else res_dual_x_admm_schur_prec_l2
    rel_res_dual_x_admm_kkt_l2 = res_dual_x_admm_kkt_l2 / v_np_norm_l2 if v_np_norm_l2 > 0 else res_dual_x_admm_kkt_l2

    # TODO: errors w.r.t solving the KKT linear system (better conditioned, and actually what we want)
    res_kkt_ux_np_infnorm = linsys_residual_infnorm(K_np, k_np, ux_np)
    res_kkt_ux_np_scaled_infnorm = linsys_residual_infnorm(K_np, k_np, ux_np_scaled)
    res_kkt_ux_admm_schur_infnorm = linsys_residual_infnorm(K_np, k_np, ux_admm_schur)
    res_kkt_ux_admm_schur_prec_infnorm = linsys_residual_infnorm(K_np, k_np, ux_admm_schur_prec)
    res_kkt_ux_admm_kkt_infnorm = linsys_residual_infnorm(K_np, k_np, ux_admm_kkt)

    rel_res_kkt_ux_np_infnorm = res_kkt_ux_np_infnorm / k_np_norm_inf if k_np_norm_inf > 0 else res_kkt_ux_np_infnorm
    rel_res_kkt_ux_np_scaled_infnorm = res_kkt_ux_np_scaled_infnorm / k_np_norm_inf if k_np_norm_inf > 0 else res_kkt_ux_np_scaled_infnorm
    rel_res_kkt_ux_admm_schur_infnorm = res_kkt_ux_admm_schur_infnorm / k_np_norm_inf if k_np_norm_inf > 0 else res_kkt_ux_admm_schur_infnorm
    rel_res_kkt_ux_admm_schur_prec_infnorm = res_kkt_ux_admm_schur_prec_infnorm / k_np_norm_inf if k_np_norm_inf > 0 else res_kkt_ux_admm_schur_prec_infnorm
    rel_res_kkt_ux_admm_kkt_infnorm = res_kkt_ux_admm_kkt_infnorm / k_np_norm_inf if k_np_norm_inf > 0 else res_kkt_ux_admm_kkt_infnorm

    res_kkt_ux_np_l2 = linsys_residual_l2norm(K_np, k_np, ux_np)
    res_kkt_ux_np_scaled_l2 = linsys_residual_l2norm(K_np, k_np, ux_np_scaled)
    res_kkt_ux_admm_schur_l2 = linsys_residual_l2norm(K_np, k_np, ux_admm_schur)
    res_kkt_ux_admm_schur_prec_l2 = linsys_residual_l2norm(K_np, k_np, ux_admm_schur_prec)
    res_kkt_ux_admm_kkt_l2 = linsys_residual_l2norm(K_np, k_np, ux_admm_kkt)

    rel_res_kkt_ux_np_l2 = res_kkt_ux_np_l2 / k_np_norm_l2 if k_np_norm_l2 > 0 else res_kkt_ux_np_l2
    rel_res_kkt_ux_np_scaled_l2 = res_kkt_ux_np_scaled_l2 / k_np_norm_l2 if k_np_norm_l2 > 0 else res_kkt_ux_np_scaled_l2
    rel_res_kkt_ux_admm_schur_l2 = res_kkt_ux_admm_schur_l2 / k_np_norm_l2 if k_np_norm_l2 > 0 else res_kkt_ux_admm_schur_l2
    rel_res_kkt_ux_admm_schur_prec_l2 = res_kkt_ux_admm_schur_prec_l2 / k_np_norm_l2 if k_np_norm_l2 > 0 else res_kkt_ux_admm_schur_prec_l2
    rel_res_kkt_ux_admm_kkt_l2 = res_kkt_ux_admm_kkt_l2 / k_np_norm_l2 if k_np_norm_l2 > 0 else res_kkt_ux_admm_kkt_l2

    ###
    # Summary of solving the dual system
    ###

    # Compare solution norms

    print("\nx NORMS:")
    print(f"x_np              : {x_np_norm}")
    print(f"x_np_scaled       : {x_np_scaled_norm}")
    print(f"x_admm_schur      : {x_admm_schur_norm}")
    print(f"x_admm_schur_prec : {x_admm_schur_prec_norm}")
    print(f"x_admm_kkt        : {x_admm_kkt_norm}")

    # Compare solution differences

    print("\n\n(x - x_np) DIFFERENCES (INF):")
    print(f"x_np_scaled       : {dx_np_scaled_infnorm}")
    print(f"x_admm_schur      : {dx_admm_schur_infnorm}")
    print(f"x_admm_schur_prec : {dx_admm_schur_prec_infnorm}")
    print(f"x_admm_kkt        : {dx_admm_kkt_infnorm}")

    print("\n\n(x - x_np) DIFFERENCES (L2):")
    print(f"x_np_scaled       : {dx_np_scaled_l2}")
    print(f"x_admm_schur      : {dx_admm_schur_l2}")
    print(f"x_admm_schur_prec : {dx_admm_schur_prec_l2}")
    print(f"x_admm_kkt        : {dx_admm_kkt_l2}")

    # Compare absolute errors

    print("\n\n(D @ x + v_f) ABSOLUTE ERRORS (INF):")
    print(f"x_np              : {res_dual_x_np_infnorm}")
    print(f"x_np_scaled       : {res_dual_x_np_scaled_infnorm}")
    print(f"x_admm_schur      : {res_dual_x_admm_schur_infnorm}")
    print(f"x_admm_schur_prec : {res_dual_x_admm_schur_prec_infnorm}")
    print(f"x_admm_kkt        : {res_dual_x_admm_kkt_infnorm}")

    print("\n\n(D @ x + v_f) ABSOLUTE ERRORS (L2):")
    print(f"x_np              : {res_dual_x_np_l2}")
    print(f"x_np_scaled       : {res_dual_x_np_scaled_l2}")
    print(f"x_admm_schur      : {res_dual_x_admm_schur_l2}")
    print(f"x_admm_schur_prec : {res_dual_x_admm_schur_prec_l2}")
    print(f"x_admm_kkt        : {res_dual_x_admm_kkt_l2}")

    # Compare relative errors

    print("\n\n(D @ x + v_f) RELATIVE ERRORS (INF):")
    print(f"x_np              : {rel_res_dual_x_np_infnorm}")
    print(f"x_np_scaled       : {rel_res_dual_x_np_scaled_infnorm}")
    print(f"x_admm_schur      : {rel_res_dual_x_admm_schur_infnorm}")
    print(f"x_admm_schur_prec : {rel_res_dual_x_admm_schur_prec_infnorm}")
    print(f"x_admm_kkt        : {rel_res_dual_x_admm_kkt_infnorm}")

    print("\n\n(D @ x + v_f) RELATIVE ERRORS (L2):")
    print(f"x_np              : {rel_res_dual_x_np_l2}")
    print(f"x_np_scaled       : {rel_res_dual_x_np_scaled_l2}")
    print(f"x_admm_schur      : {rel_res_dual_x_admm_schur_l2}")
    print(f"x_admm_schur_prec : {rel_res_dual_x_admm_schur_prec_l2}")
    print(f"x_admm_kkt        : {rel_res_dual_x_admm_kkt_l2}")

    ###
    # Summary of solving the KKT system
    ###

    # Compare solution norms

    print("\nux NORMS:")
    print(f"ux_np              : {ux_np_norm}")
    print(f"ux_np_scaled       : {ux_np_scaled_norm}")
    print(f"ux_admm_schur      : {ux_admm_schur_norm}")
    print(f"ux_admm_schur_prec : {ux_admm_schur_prec_norm}")
    print(f"ux_admm_kkt        : {ux_admm_kkt_norm}")

    # Compare solution differences

    print("\n\n(ux - ux_np) DIFFERENCES (INF):")
    print(f"ux_np_scaled       : {dux_np_scaled_infnorm}")
    print(f"ux_admm_schur      : {dux_admm_schur_infnorm}")
    print(f"ux_admm_schur_prec : {dux_admm_schur_prec_infnorm}")
    print(f"ux_admm_kkt        : {dux_admm_kkt_infnorm}")

    print("\n\n(ux - ux_np) DIFFERENCES (L2):")
    print(f"ux_np_scaled       : {dux_np_scaled_l2}")
    print(f"ux_admm_schur      : {dux_admm_schur_l2}")
    print(f"ux_admm_schur_prec : {dux_admm_schur_prec_l2}")
    print(f"ux_admm_kkt        : {dux_admm_kkt_l2}")

    # Compare absolute errors

    print("\n\n(K @ ux - k) ABSOLUTE ERRORS (INF):")
    print(f"ux_np              : {res_kkt_ux_np_infnorm}")
    print(f"ux_np_scaled       : {res_kkt_ux_np_scaled_infnorm}")
    print(f"ux_admm_schur      : {res_kkt_ux_admm_schur_infnorm}")
    print(f"ux_admm_schur_prec : {res_kkt_ux_admm_schur_prec_infnorm}")
    print(f"ux_admm_kkt        : {res_kkt_ux_admm_kkt_infnorm}")

    print("\n\n(K @ ux - k) ABSOLUTE ERRORS (L2):")
    print(f"ux_np              : {res_kkt_ux_np_l2}")
    print(f"ux_np_scaled       : {res_kkt_ux_np_scaled_l2}")
    print(f"ux_admm_schur      : {res_kkt_ux_admm_schur_l2}")
    print(f"ux_admm_schur_prec : {res_kkt_ux_admm_schur_prec_l2}")
    print(f"ux_admm_kkt        : {res_kkt_ux_admm_kkt_l2}")

    # Compare relative errors

    print("\n\n(K @ ux - k) RELATIVE ERRORS (INF):")
    print(f"ux_np              : {rel_res_kkt_ux_np_infnorm}")
    print(f"ux_np_scaled       : {rel_res_kkt_ux_np_scaled_infnorm}")
    print(f"ux_admm_schur      : {rel_res_kkt_ux_admm_schur_infnorm}")
    print(f"ux_admm_schur_prec : {rel_res_kkt_ux_admm_schur_prec_infnorm}")
    print(f"ux_admm_kkt        : {rel_res_kkt_ux_admm_kkt_infnorm}")

    print("\n\n(K @ ux - k) RELATIVE ERRORS (L2):")
    print(f"ux_np              : {rel_res_kkt_ux_np_l2}")
    print(f"ux_np_scaled       : {rel_res_kkt_ux_np_scaled_l2}")
    print(f"ux_admm_schur      : {rel_res_kkt_ux_admm_schur_l2}")
    print(f"ux_admm_schur_prec : {rel_res_kkt_ux_admm_schur_prec_l2}")
    print(f"ux_admm_kkt        : {rel_res_kkt_ux_admm_kkt_l2}")
