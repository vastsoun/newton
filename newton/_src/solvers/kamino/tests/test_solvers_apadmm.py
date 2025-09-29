###########################################################################
# KAMINO: UNIT TESTS: SOLVERS: Proximal ADMM Dual Solver
###########################################################################

import os
import unittest
import numpy as np
import warp as wp

import matplotlib.pyplot as plt

from newton._src.solvers.kamino.core.math import vec3f, screw
from newton._src.solvers.kamino.linalg.cholesky import SequentialCholeskyFactorizer
from newton._src.solvers.kamino.dynamics.dual import DualProblem
from newton._src.solvers.kamino.simulation.simulator import compute_constraint_body_wrenches
from newton._src.solvers.kamino.models.builders import (
    build_box_on_plane,
    build_box_pendulum,
    build_boxes_hinged,
    build_boxes_nunchaku,
    build_boxes_fourbar,
)
from newton._src.solvers.kamino.models.utils import (
    make_single_builder,
    make_homogeneous_builder,
    make_heterogeneous_builder
)
from newton._src.solvers.kamino.models.builders import (
    add_ground_geom,
    add_velocity_bias,
    offset_builder,
)

# Test utilities
from newton._src.solvers.kamino.tests.utils.make import make_containers, update_containers
from newton._src.solvers.kamino.tests.utils.extract import extract_delassus, extract_problem_vector, extract_info_vectors
from newton._src.solvers.kamino.tests.utils.print import print_model_info


# Module to be tested
from newton._src.solvers.kamino.solvers.apadmm import APADMMSettings, APADMMDualSolver


###
# Helper functions
###

def print_problem_summary(problem: DualProblem):
    print("Dual Problem Summary:")
    print(f"problem.data.num_worlds: {problem.data.num_worlds}")
    print(f"problem.data.num_maxdims: {problem.data.num_maxdims}")
    print(f"problem.data.nl:{problem.data.nl}")
    print(f"problem.data.nc:{problem.data.nc}")
    print(f"problem.data.config: {problem.data.config}")
    print(f"problem.data.maxdim: {problem.data.maxdim}")
    print(f"problem.data.dim: {problem.data.dim}")
    print(f"problem.data.mio: {problem.data.mio}")
    print(f"problem.data.vio: {problem.data.vio}")
    print(f"problem.data.u_f (shape): {problem.data.u_f.shape}")
    print(f"problem.data.v_b (shape): {problem.data.v_b.shape}")
    print(f"problem.data.v_i (shape): {problem.data.v_i.shape}")
    print(f"problem.data.v_f (shape): {problem.data.v_f.shape}")
    print(f"problem.data.mu (shape): {problem.data.mu.shape}")
    print(f"problem.data.D (shape): {problem.data.D.shape}")


def print_solver_summary(solver: APADMMDualSolver):
    print("PADMM Solver Summary:")
    print("solver.size.num_worlds: ", solver.size.num_worlds)
    print("solver.size.max_limits: ", solver.size.sum_of_max_limits)
    print("solver.size.max_contacts: ", solver.size.sum_of_max_contacts)
    print("solver.size.max_unilaterals: ", solver.size.sum_of_max_unilaterals)
    print("solver.size.max_total_cts: ", solver.size.sum_of_max_total_cts)
    print("solver.size.max_iters: ", solver._max_iters)
    print("solver.data.config: ", solver.data.config)
    print("solver.data.status: ", solver.data.status)
    print("solver.data.penalty: ", solver.data.penalty)
    print("solver.data.info.offsets: ", solver.data.info.offsets)
    print("solver.data.info.r_primal: ", solver.data.info.r_primal.shape)
    print("solver.data.info.r_dual: ", solver.data.info.r_dual.shape)
    print("solver.data.info.r_compl: ", solver.data.info.r_compl.shape)
    print("solver.data.state.s:", solver.data.state.s.shape)
    print("solver.data.state.v:", solver.data.state.v.shape)
    print("solver.data.state.x:", solver.data.state.x.shape)
    print("solver.data.state.x_p:", solver.data.state.x_p.shape)
    print("solver.data.state.y:", solver.data.state.y.shape)
    print("solver.data.state.y_p:", solver.data.state.y_p.shape)
    print("solver.data.state.z:", solver.data.state.z.shape)
    print("solver.data.state.z_p:", solver.data.state.z_p.shape)
    print("solver.data.solution.v_plus:", solver.data.solution.v_plus.shape)
    print("solver.data.solution.lambdas:", solver.data.solution.lambdas.shape)


def save_solver_info(solver: APADMMDualSolver, path: str | None = None, verbose: bool = False):
    nw = solver.size.num_worlds
    status = solver.data.status.numpy()
    iterations = [status[w][1] for w in range(nw)]
    offsets_np = solver.data.info.offsets.numpy()
    norm_s_np = extract_info_vectors(offsets_np, solver.data.info.norm_s.numpy(), iterations)
    norm_x_np = extract_info_vectors(offsets_np, solver.data.info.norm_x.numpy(), iterations)
    norm_y_np = extract_info_vectors(offsets_np, solver.data.info.norm_y.numpy(), iterations)
    norm_z_np = extract_info_vectors(offsets_np, solver.data.info.norm_z.numpy(), iterations)
    f_ccp_np = extract_info_vectors(offsets_np, solver.data.info.f_ccp.numpy(), iterations)
    f_ncp_np = extract_info_vectors(offsets_np, solver.data.info.f_ncp.numpy(), iterations)
    r_dx_np = extract_info_vectors(offsets_np, solver.data.info.r_dx.numpy(), iterations)
    r_dy_np = extract_info_vectors(offsets_np, solver.data.info.r_dy.numpy(), iterations)
    r_dz_np = extract_info_vectors(offsets_np, solver.data.info.r_dz.numpy(), iterations)
    r_primal_np = extract_info_vectors(offsets_np, solver.data.info.r_primal.numpy(), iterations)
    r_dual_np = extract_info_vectors(offsets_np, solver.data.info.r_dual.numpy(), iterations)
    r_compl_np = extract_info_vectors(offsets_np, solver.data.info.r_compl.numpy(), iterations)
    r_pd_np = extract_info_vectors(offsets_np, solver.data.info.r_pd.numpy(), iterations)
    r_dp_np = extract_info_vectors(offsets_np, solver.data.info.r_dp.numpy(), iterations)
    r_ncp_primal_np = extract_info_vectors(offsets_np, solver.data.info.r_ncp_primal.numpy(), iterations)
    r_ncp_dual_np = extract_info_vectors(offsets_np, solver.data.info.r_ncp_dual.numpy(), iterations)
    r_ncp_compl_np = extract_info_vectors(offsets_np, solver.data.info.r_ncp_compl.numpy(), iterations)
    r_ncp_natmap_np = extract_info_vectors(offsets_np, solver.data.info.r_ncp_natmap.numpy(), iterations)

    if verbose:
        for w in range(nw):
            print(f"[World {w}] =======================================================================")
            print(f"solver.info.norm_s: {norm_s_np[w]}")
            print(f"solver.info.norm_x: {norm_x_np[w]}")
            print(f"solver.info.norm_y: {norm_y_np[w]}")
            print(f"solver.info.norm_z: {norm_z_np[w]}")
            print(f"solver.info.f_ccp: {f_ccp_np[w]}")
            print(f"solver.info.f_ncp: {f_ncp_np[w]}")
            print(f"solver.info.r_dx: {r_dx_np[w]}")
            print(f"solver.info.r_dy: {r_dy_np[w]}")
            print(f"solver.info.r_dz: {r_dz_np[w]}")
            print(f"solver.info.r_primal: {r_primal_np[w]}")
            print(f"solver.info.r_dual: {r_dual_np[w]}")
            print(f"solver.info.r_compl: {r_compl_np[w]}")
            print(f"solver.info.r_pd: {r_pd_np[w]}")
            print(f"solver.info.r_dp: {r_dp_np[w]}")
            print(f"solver.info.r_ncp_primal: {r_ncp_primal_np[w]}")
            print(f"solver.info.r_ncp_dual: {r_ncp_dual_np[w]}")
            print(f"solver.info.r_ncp_compl: {r_ncp_compl_np[w]}")
            print(f"solver.info.r_ncp_natmap: {r_ncp_natmap_np[w]}")

    # List of (label, data) for plotting
    info_list = [
        ("norm_s", norm_s_np),
        ("norm_x", norm_x_np),
        ("norm_y", norm_y_np),
        ("norm_z", norm_z_np),
        ("f_ccp", f_ccp_np),
        ("f_ncp", f_ncp_np),
        ("r_dx", r_dx_np),
        ("r_dy", r_dy_np),
        ("r_dz", r_dz_np),
        ("r_primal", r_primal_np),
        ("r_dual", r_dual_np),
        ("r_compl", r_compl_np),
        ("r_pd", r_pd_np),
        ("r_dp", r_dp_np),
        ("r_ncp_primal", r_ncp_primal_np),
        ("r_ncp_dual", r_ncp_dual_np),
        ("r_ncp_compl", r_ncp_compl_np),
        ("r_ncp_natmap", r_ncp_natmap_np),
    ]

    # Plot all info as subplots: rows=info_list, cols=worlds
    n_rows = len(info_list)
    n_cols = nw
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 2.5 * n_rows), squeeze=False)
    for row, (label, arr) in enumerate(info_list):
        for col in range(nw):
            ax = axes[row, col]
            ax.plot(arr[col], label=f'{label}')
            ax.set_xlabel('Iteration')
            ax.set_ylabel(label)
            if row == 0:
                ax.set_title(f'World {col}')
            if col == 0:
                ax.set_ylabel(label)
            else:
                ax.set_ylabel("")
            ax.grid(True)
    plt.tight_layout()
    if path is None:
        plt.savefig(os.path.dirname(os.path.realpath(__file__)) + "/data/apadmm_solver_info.pdf", format="pdf", dpi=300, bbox_inches="tight")
    else:
        plt.savefig(path, format="pdf", dpi=300, bbox_inches="tight")


###
# Tests
###

class TestPADMMDualSolver(unittest.TestCase):

    def setUp(self):
        self.verbose = True  # Set to True for detailed output
        self.savefig = True  # Set to True to generate solver info plots
        self.default_device = wp.get_device()

    def tearDown(self):
        self.default_device = None

    def test_01_padmm_solve_box_on_plane(self):
        """
        Tests the Proximal ADMM solver on a simple box-on-plane problem with four contacts and the body at rest and undisturbed.
        """
        # Model constants
        max_world_contacts = 12

        # Construct the model description using model builders for different systems
        builder, _, _ = make_single_builder(build_func=build_box_on_plane)
        # builder, _, _ = make_single_builder(build_func=build_boxes_hinged, ground=True)
        # builder, _, _ = make_single_builder(build_func=build_boxes_nunchaku)
        # builder, _, _ = make_single_builder(build_func=build_boxes_fourbar)
        # builder, _, _ = make_homogeneous_builder(num_worlds=4, build_func=build_box_on_plane)
        # builder, _, _ = make_homogeneous_builder(num_worlds=4, build_func=build_boxes_hinged)
        # builder, _, _ = make_homogeneous_builder(num_worlds=4, build_func=build_boxes_nunchaku)
        # builder, _, _ = make_homogeneous_builder(num_worlds=4, build_func=build_boxes_fourbar)
        # builder, _, _ = make_heterogeneous_builder()

        # Set ad-hoc configurations
        builder.gravity.enabled = True
        u_0 = screw(vec3f(+10.0, 0.0, 0.0), vec3f(0.0, 0.0, 0.0))
        for body in builder.bodies:
            body.u_i_0 = u_0

        # Create the model and containers from the builder
        model, state, limits, detector, jacobians = make_containers(
            builder=builder,
            max_world_contacts=max_world_contacts,
            device=self.default_device
        )

        # Update the containers
        update_containers(model=model, state=state, limits=limits, detector=detector, jacobians=jacobians)
        if self.verbose:
            print("\n")  # Print a newline for better readability
            print_model_info(model)
            print("\n")  # Print a newline for better readability

        # Create the Delassus operator
        problem = DualProblem(
            model=model,
            state=state,
            limits=limits,
            contacts=detector.contacts,
            factorizer=SequentialCholeskyFactorizer,
            device=self.default_device
        )

        # Build the dual problem
        problem.build(
            model=model,
            state=state,
            limits=limits.data,
            contacts=detector.contacts.data,
            jacobians=jacobians.data
        )

        # Optional verbose output
        if self.verbose:
            print("\n")  # Print a newline for better readability
            print_problem_summary(problem)
            print("\n")  # Print a newline for better readability

        # Define custom solver settings
        settings = APADMMSettings()
        settings.primal_tolerance = 1e-6
        settings.dual_tolerance = 1e-6
        settings.compl_tolerance = 1e-6
        settings.restart_tolerance = 0.999
        settings.eta = 1e-5
        settings.rho_0 = 0.01  # 9.7  # 2.7
        settings.omega = 1.0  # 1.99
        settings.max_iterations = 500

        # Create the ADMM solver
        solver = APADMMDualSolver(
            model=model,
            limits=limits,
            contacts=detector.contacts,
            settings=settings,
            collect_info=True,
            device=self.default_device
        )

        # # Optional verbose output
        # if self.verbose:
        #     print("\n")  # Print a newline for better readability
        #     print_solver_summary(solver)
        #     print("\n")  # Print a newline for better readability

        # TODO:
        # 1. Interface to set the solver parameters
        # 2. How to make running solver only when constraints are active?

        # Solve the example problem
        solver.solve(problem=problem)

        # Extract numpy arrays from the solver state and solution
        only_active_dims = True
        D_wp_np = extract_delassus(problem.delassus, only_active_dims=only_active_dims)
        v_i_wp_np = extract_problem_vector(problem.delassus, problem.data.v_i.numpy(), only_active_dims=only_active_dims)
        v_b_wp_np = extract_problem_vector(problem.delassus, problem.data.v_b.numpy(), only_active_dims=only_active_dims)
        v_f_wp_np = extract_problem_vector(problem.delassus, problem.data.v_f.numpy(), only_active_dims=only_active_dims)
        P_wp_np = extract_problem_vector(problem.delassus, problem.data.P.numpy(), only_active_dims=only_active_dims)
        s_wp_np = extract_problem_vector(problem.delassus, solver.data.state.s.numpy(), only_active_dims=only_active_dims)
        v_wp_np = extract_problem_vector(problem.delassus, solver.data.state.v.numpy(), only_active_dims=only_active_dims)
        x_wp_np = extract_problem_vector(problem.delassus, solver.data.state.x.numpy(), only_active_dims=only_active_dims)
        y_wp_np = extract_problem_vector(problem.delassus, solver.data.state.y.numpy(), only_active_dims=only_active_dims)
        z_wp_np = extract_problem_vector(problem.delassus, solver.data.state.z.numpy(), only_active_dims=only_active_dims)
        v_plus_wp_np = extract_problem_vector(problem.delassus, solver.data.solution.v_plus.numpy(), only_active_dims=only_active_dims)
        lambdas_wp_np = extract_problem_vector(problem.delassus, solver.data.solution.lambdas.numpy(), only_active_dims=only_active_dims)
        r_primal_wp_np = extract_problem_vector(problem.delassus, solver.data.residuals.r_primal.numpy(), only_active_dims=only_active_dims)
        r_dual_wp_np = extract_problem_vector(problem.delassus, solver.data.residuals.r_dual.numpy(), only_active_dims=only_active_dims)
        r_compl_wp_np = extract_problem_vector(problem.delassus, solver.data.residuals.r_compl.numpy(), only_active_dims=only_active_dims)

        # Retrieve the number of worlds in the model
        nw = model.info.num_worlds

        # Optional verbose output
        if self.verbose:
            status = solver.data.status.numpy()
            print("\n")  # Print a newline for better readability
            for w in range(nw):
                # Reconstruct the ADMM regularization matrix
                dtype = D_wp_np[w].dtype

                # Recover D from D_reg
                I_np = dtype.type(settings.eta + settings.rho_0) * np.eye(D_wp_np[w].shape[0], dtype=dtype)
                D = D_wp_np[w] - I_np

                # eigevalues of preconditioned D
                norm_v_f = np.linalg.norm(v_f_wp_np[w])
                eigvals = np.linalg.eigvalsh(D)
                eig_min = np.min(eigvals)
                eig_max = np.max(eigvals)
                L = eig_max + settings.eta
                m = max(eig_min, 0.0) + settings.eta
                kappa_D = L / m
                rho_0 = np.pow(kappa_D, 1.0)
                rho_1 = np.sqrt(L * m)
                rho_opt = rho_1 * rho_0

                # # Compute the preconditioned Delassus and v_f
                # D_p = np.diag(P_wp_np[w]) @ D @ np.diag(P_wp_np[w])
                # v_p = np.diag(P_wp_np[w]) @ v_f_wp_np[w]

                # # Regularized preconditioned Delassus
                # D_p_reg = D_p + I_np

                # TODO
                min_P = np.min(P_wp_np[w])
                max_P = np.max(P_wp_np[w])
                mean_P = np.mean(P_wp_np[w])
                norm_P = np.linalg.norm(P_wp_np[w])
                rho_p_min = settings.rho_0 / (min_P * min_P)
                rho_p_max = settings.rho_0 / (max_P * max_P)
                rho_p_mean = settings.rho_0 / (mean_P * mean_P)
                rho_p_norm = settings.rho_0 / (norm_P * norm_P)
                # Print all solver data
                print(f"[World {w}] =======================================================================")
                print(f"D_reg:\n{D_wp_np[w]}")
                print(f"D:\n{D}")
                print(f"v_i:\n{v_i_wp_np[w]}")
                print(f"v_b:\n{v_b_wp_np[w]}")
                print(f"v_f:\n{v_f_wp_np[w]}")
                print(f"P:\n{P_wp_np[w]}")
                print(f"norm_v_f: {norm_v_f}")
                print(f"min(P): {min_P}")
                print(f"max(P): {max_P}")
                print(f"mean(P): {mean_P}")
                print(f"norm(P): {norm_P}")
                print(f"eig_max: {eig_max}")
                print(f"eig_min: {eig_min}")
                print(f"L: {L}")
                print(f"m: {m}")
                print(f"kappa_D: {kappa_D}")
                # print(f"magic: {magic}")
                # print(f"rmagic: {rmagic}")
                print(f"rho_0: {rho_0}")
                print(f"rho_1: {rho_1}")
                print(f"rho_opt: {rho_opt}")
                print(f"rho_p_min: {rho_p_min}")
                print(f"rho_p_max: {rho_p_max}")
                print(f"rho_p_mean: {rho_p_mean}")
                print(f"rho_p_norm: {rho_p_norm}")
                # print(f"D_p:\n{D_p}")
                # print(f"D_p_reg:\n{D_p_reg}")
                # print(f"v_p:\n{v_p}")
                print("---------")
                print(f"s: {s_wp_np[w]}")
                print(f"v: {v_wp_np[w]}")
                print(f"x: {x_wp_np[w]}")
                print(f"y: {y_wp_np[w]}")
                print(f"z: {z_wp_np[w]}")
                print("---------")
                print(f" v_plus: {v_plus_wp_np[w]}")
                print(f"lambdas: {lambdas_wp_np[w]}")
                print("---------")
                print(f"r_primal: {r_primal_wp_np[w]}")
                print(f"  r_dual: {r_dual_wp_np[w]}")
                print(f" r_compl: {r_compl_wp_np[w]}")
                print("---------")
                print(f"iterations: {status[w][1]}")
                print(f"converged: {status[w][0]}")
                print(f"r_p: {status[w][2]}")
                print(f"r_d: {status[w][3]}")
                print(f"r_c: {status[w][4]}")
            print("\n")  # Print a newline for better readability

        # Recover original Delassus matrix and v_f from preconditioned versions
        D_true = np.diag(np.reciprocal(P_wp_np[w])) @ D @ np.diag(np.reciprocal(P_wp_np[w]))
        v_f_true = np.diag(np.reciprocal(P_wp_np[w])) @ v_f_wp_np[w]

        # Extract solver explicit solution
        v_plus_np = extract_problem_vector(problem.delassus, solver.data.info.v_plus.numpy(), only_active_dims=only_active_dims)
        v_aug_np = extract_problem_vector(problem.delassus, solver.data.info.v_aug.numpy(), only_active_dims=only_active_dims)
        s_np = extract_problem_vector(problem.delassus, solver.data.info.s.numpy(), only_active_dims=only_active_dims)
        # Compute the true dual solution and error
        v_plus_true = np.matmul(D_true, lambdas_wp_np[w]) + v_f_true
        # error_dual_abs = np.linalg.norm(v_plus_true - v_plus_wp_np[w], ord=np.inf)

        # Print solution/info/true values
        if self.verbose:
            print("\n")  # Print a newline for better readability
            for w in range(nw):
                print(f"[World {w}] =======================================================================")
                print(f"   state:      s: {s_wp_np[w]}")
                print(f"    info:      s: {s_np[w]}")
                print(f"   state:      y: {y_wp_np[w]}")
                print(f"solution: lambda: {lambdas_wp_np[w]}")
                print(f"   state:      z: {z_wp_np[w]}")
                print(f"    info:  v_aug: {v_aug_np[w]}")
                print(f"solution: v_plus: {v_plus_wp_np[w]}")
                print(f"    info: v_plus: {v_plus_np[w]}")
                print(f"    true: v_plus: {v_plus_true}")
            print("\n")  # Print a newline for better readability

        # Extract solver info
        if self.savefig:
            print("Generating solver info plots...")
            save_solver_info(solver)

        # # Apply the solution to the model state
        # with wp.ScopedTimer("compute_constraint_body_wrenches", active=self.verbose):
        #     compute_constraint_body_wrenches(
        #         model=model,
        #         state=state,
        #         limits=limits,
        #         contacts=detector.contacts._data,
        #         jacobians=jacobians._data,
        #         lambdas_offsets=problem.data.vio,
        #         lambdas_data=solver.data.solution.lambdas,
        #     )
        # if self.verbose:
        #     print("PADMM: model_data.bodies.w_j_i:\n", state.bodies.w_j_i.numpy())
        #     print("PADMM: model_data.bodies.w_l_i:\n", state.bodies.w_l_i.numpy())
        #     print("PADMM: model_data.bodies.w_c_i:\n", state.bodies.w_c_i.numpy())

    # def test_02_padmm_solve_boxes_hinged(self):
    #     """
    #     Tests the Proximal ADMM solver on a simple box-on-plane problem with four contacts and the body at rest and undisturbed.
    #     """
    #     # Model constants
    #     max_world_contacts = 12

    #     # Construct the model description using model builders for different systems
    #     builder, _, _ = make_single_builder(build_func=build_box_on_plane)
    #     # builder, _, _ = make_single_builder(build_func=build_boxes_hinged)
    #     # builder, _, _ = make_single_builder(build_func=build_boxes_nunchaku)
    #     # builder, _, _ = make_single_builder(build_func=build_boxes_fourbar)
    #     # builder, _, _ = make_homogeneous_builder(num_worlds=4, build_func=build_box_on_plane)
    #     # builder, _, _ = make_homogeneous_builder(num_worlds=4, build_func=build_boxes_hinged)
    #     # builder, _, _ = make_homogeneous_builder(num_worlds=4, build_func=build_boxes_nunchaku)
    #     # builder, _, _ = make_homogeneous_builder(num_worlds=4, build_func=build_boxes_fourbar)
    #     # builder, _, _ = make_heterogeneous_builder()

    #     # Define parameter ranges
    #     # u_0_range = np.linspace(0.1, 10.0, 10, dtype=float)
    #     # rho_0_range = np.linspace(0.1, 10.0, 10, dtype=float)
    #     u_0_range = np.linspace(0.1, 5.0, int((5.0 - 0.1)/0.1) + 1, dtype=float)
    #     rho_0_range = np.linspace(0.01, 5.0, int((5.0 - 0.01)/0.01) + 1, dtype=float)
    #     # n_u_0 = 100
    #     # n_u_0 = 50
    #     # u_0_range = np.linspace(1.0, 50.0, n_u_0, dtype=float)
    #     # rho_0_range = np.linspace(1.0, 20.0, int((20.0 - 1.0)/0.1) + 1, dtype=float)
    #     # n_u_0 = 100
    #     # u_0_range = np.linspace(1.0, 100.0, n_u_0, dtype=float)
    #     # rho_0_range = np.linspace(0.01, 20.0, int((20.0 - 0.01)/0.01) + 1, dtype=float)

    #     # Prepare storage for results
    #     n_u_0 = u_0_range.shape[0]
    #     n_rho = rho_0_range.shape[0]
    #     v_f_norm = np.zeros((n_u_0,), dtype=float)
    #     r_pd_min_norm = np.zeros((n_u_0, n_rho), dtype=float)
    #     r_pd_max_norm = np.zeros((n_u_0, n_rho), dtype=float)
    #     r_pd_mean_norm = np.zeros((n_u_0, n_rho), dtype=float)
    #     r_p_final_norm = np.zeros((n_u_0, n_rho), dtype=float)
    #     r_d_final_norm = np.zeros((n_u_0, n_rho), dtype=float)
    #     r_c_final_norm = np.zeros((n_u_0, n_rho), dtype=float)
    #     iterations = np.zeros((n_u_0, n_rho), dtype=int)
    #     print("u_0_range:\n", u_0_range)
    #     print("rho_0_range:\n", rho_0_range)

    #     #
    #     for i in range(n_u_0):
    #         for j in range(n_rho):
    #             # Set ad-hoc configurations
    #             builder.gravity.enabled = True
    #             for b in builder.bodies:
    #                 b.u_i_0 = screw(vec3f(u_0_range[i], 0.0, 0.0), vec3f(0.0, 0.0, 0.0))

    #             # Create the model and containers from the builder
    #             model, state, limits, detector, jacobians = make_containers(
    #                 builder=builder,
    #                 max_world_contacts=max_world_contacts,
    #                 device=self.default_device
    #             )

    #             # Update the containers
    #             update_containers(model=model, state=state, limits=limits, detector=detector, jacobians=jacobians)

    #             # Create the Delassus operator
    #             problem = DualProblem(
    #                 model=model,
    #                 state=state,
    #                 limits=limits,
    #                 contacts=detector.contacts,
    #                 factorizer=SequentialCholeskyFactorizer,
    #                 device=self.default_device
    #             )

    #             # Build the dual problem
    #             problem.build(
    #                 model=model,
    #                 state=state,
    #                 limits=limits.data,
    #                 contacts=detector.contacts.data,
    #                 jacobians=jacobians.data
    #             )

    #             # Define custom solver settings
    #             settings = PADMMSettings()
    #             settings.primal_tolerance = 1e-6
    #             settings.dual_tolerance = 1e-6
    #             settings.compl_tolerance = 1e-6
    #             settings.eta = 1e-5
    #             settings.rho_0 = rho_0_range[j]
    #             settings.max_iterations = 500

    #             # Create the ADMM solver
    #             solver = PADMMDualSolver(
    #                 model=model,
    #                 limits=limits,
    #                 contacts=detector.contacts,
    #                 settings=settings,
    #                 collect_info=True,
    #                 device=self.default_device
    #             )

    #             # Solve the example problem
    #             solver.solve(problem=problem)

    #             # Extract numpy arrays from the solver state and solution
    #             only_active_dims = True
    #             status = solver.data.status.numpy()
    #             iters_ij = status[0][1]
    #             v_f_wp_np = extract_problem_vector(problem.delassus, problem.data.v_f.numpy(), only_active_dims=only_active_dims)
    #             r_p_info_np = solver.data.info.r_primal.numpy()[:iters_ij]
    #             r_d_info_np = solver.data.info.r_dual.numpy()[:iters_ij]
    #             # print(f"[i={i}, j={j}] iters_ij: {iters_ij}")
    #             # print(f"[i={i}, j={j}] r_p_info_np: {r_p_info_np}")
    #             # print(f"[i={i}, j={j}] r_d_info_np: {r_d_info_np}")
    #             r_pd_info_np = np.divide(
    #                 r_p_info_np,
    #                 r_d_info_np,
    #                 out=np.full_like(r_p_info_np, np.inf, dtype=float),
    #                 where=r_d_info_np > np.finfo(float).eps
    #             )
    #             # print(f"[i={i}, j={j}] r_pd_info_np: {r_pd_info_np}\n\n")

    #             # Store results
    #             r_pd_min_norm[i][j] = np.min(r_pd_info_np)
    #             r_pd_max_norm[i][j] = np.max(r_pd_info_np)
    #             r_pd_mean_norm[i][j] = np.mean(r_pd_info_np)
    #             r_p_final_norm[i][j] = status[0][2]
    #             r_d_final_norm[i][j] = status[0][3]
    #             r_c_final_norm[i][j] = status[0][4]
    #             iterations[i][j] = iters_ij

    #             # Capture v_f norm (same for all j)
    #             v_f_norm[i] = np.linalg.norm(v_f_wp_np[0])

    #     # Print matrices
    #     print("v_f_norm:\n", v_f_norm)
    #     # print("iterations:\n", iterations)
    #     # print("r_pd_min_norm:\n", r_pd_min_norm)
    #     # print("r_pd_max_norm:\n", r_pd_max_norm)
    #     # print("r_pd_mean_norm:\n", r_pd_mean_norm)

    #     # Find the smallest value iterations for each value of rho_0
    #     best_iterations = np.min(iterations, axis=1)
    #     best_indices = np.argmin(iterations, axis=1)
    #     print(f"best_iterations:\n{best_iterations}")
    #     print(f"best_indices:\n{best_indices}")

    #     # rho_0 value (column) that yielded the minimal iterations for each u_0 (row)
    #     best_rho_0 = np.take(rho_0_range, best_indices)
    #     best_r_pd_min_norm = np.take(r_pd_min_norm, best_indices)
    #     best_r_pd_max_norm = np.take(r_pd_max_norm, best_indices)
    #     print(f"best_rho_0:\n{best_rho_0}\n")

    #     print(f"Best avg rho_0: {np.mean(best_rho_0)}")
    #     print(f"Best rho_0 in: [{np.min(best_rho_0)}, {np.max(best_rho_0)}]")
    #     print(f"Best rho_0 has r_pd in: [{np.min(best_r_pd_min_norm)}, {np.max(best_r_pd_max_norm)}]")

    #     # print(f"iterations:\n{iterations}")
    #     # print(f"r_p_final_norm:\n{r_p_final_norm}")
    #     # print(f"r_d_final_norm:\n{r_d_final_norm}")
    #     # print(f"r_c_final_norm:\n{r_c_final_norm}")
    #     # print(f"r_pd_min_norm:\n{r_pd_min_norm}")
    #     # print(f"r_pd_max_norm:\n{r_pd_max_norm}")
    #     # print(f"r_pd_mean_norm:\n{r_pd_mean_norm}")

    #     # Create 2D heatmap plot with X=v_f_norm, Y=rho_0, Z=iterations
    #     plt.figure(figsize=(10, 8))
    #     plt.imshow(iterations.T, origin='lower', aspect='auto', extent=[v_f_norm[0], v_f_norm[-1], rho_0_range[0], rho_0_range[-1]], cmap='viridis')
    #     plt.colorbar(label='Iterations to Converge')
    #     plt.scatter(v_f_norm, best_rho_0, color='red', label='Best rho_0 per v_f', marker='x')
    #     plt.xlabel('Free-velocity L2 Norm (v_f)')
    #     plt.ylabel('ADMM Penalty Parameter (rho_0)')
    #     plt.title('PADMM Solver Convergence Analysis')
    #     plt.legend()
    #     plt.grid(False)
    #     if self.savefig:
    #         plt.savefig(os.path.dirname(os.path.realpath(__file__)) + "/data/padmm_solver_rho0_v_f_heatmap_iters.pdf", format="pdf", dpi=300, bbox_inches="tight")
    #     if self.verbose:
    #         plt.show()
    #     plt.close()

    #     # Create 2D heatmap plot with X=v_f_norm, Y=rho_0, Z=r_p_norm
    #     plt.figure(figsize=(10, 8))
    #     plt.imshow(np.log10(r_p_final_norm).T, origin='lower', aspect='auto', extent=[v_f_norm[0], v_f_norm[-1], rho_0_range[0], rho_0_range[-1]], cmap='viridis')
    #     plt.colorbar(label='Final Primal Residual Norm')
    #     plt.scatter(v_f_norm, best_rho_0, color='red', label='Best rho_0 per v_f', marker='x')
    #     plt.xlabel('Free-velocity L2 Norm (v_f)')
    #     plt.ylabel('ADMM Penalty Parameter (rho_0)')
    #     plt.title('PADMM Solver Primal Residual Analysis')
    #     plt.legend()
    #     plt.grid(False)
    #     if self.savefig:
    #         plt.savefig(os.path.dirname(os.path.realpath(__file__)) + "/data/padmm_solver_rho0_v_f_r_p_heatmap.pdf", format="pdf", dpi=300, bbox_inches="tight")
    #     if self.verbose:
    #         plt.show()
    #     plt.close()

    #     # Create 2D heatmap plot with X=v_f_norm, Y=rho_0, Z=r_d_norm
    #     plt.figure(figsize=(10, 8))
    #     plt.imshow(np.log10(r_d_final_norm).T, origin='lower', aspect='auto', extent=[v_f_norm[0], v_f_norm[-1], rho_0_range[0], rho_0_range[-1]], cmap='viridis')
    #     plt.colorbar(label='Final Dual Residual Norm')
    #     plt.scatter(v_f_norm, best_rho_0, color='red', label='Best rho_0 per v_f', marker='x')
    #     plt.xlabel('Free-velocity L2 Norm (v_f)')
    #     plt.ylabel('ADMM Penalty Parameter (rho_0)')
    #     plt.title('PADMM Solver Dual Residual Analysis')
    #     plt.legend()
    #     plt.grid(False)
    #     if self.savefig:
    #         plt.savefig(os.path.dirname(os.path.realpath(__file__)) + "/data/padmm_solver_rho0_v_f_r_d_heatmap.pdf", format="pdf", dpi=300, bbox_inches="tight")
    #     if self.verbose:
    #         plt.show()
    #     plt.close()

    #     # Create 2D heatmap plot with X=v_f_norm, Y=rho_0, Z=r_pd_min_norm
    #     plt.figure(figsize=(10, 8))
    #     plt.imshow(np.log10(r_pd_min_norm).T, origin='lower', aspect='auto', extent=[v_f_norm[0], v_f_norm[-1], rho_0_range[0], rho_0_range[-1]], cmap='viridis')
    #     plt.colorbar(label='Min Primal/Dual Residual Ratio')
    #     plt.scatter(v_f_norm, best_rho_0, color='red', label='Best rho_0 per v_f', marker='x')
    #     plt.xlabel('Free-velocity L2 Norm (v_f)')
    #     plt.ylabel('ADMM Penalty Parameter (rho_0)')
    #     plt.title('PADMM Solver Min Primal-Dual Residual Ratio Analysis')
    #     plt.legend()
    #     plt.grid(False)
    #     if self.savefig:
    #         plt.savefig(os.path.dirname(os.path.realpath(__file__)) + "/data/padmm_solver_rho0_v_f_r_pd_min_heatmap.pdf", format="pdf", dpi=300, bbox_inches="tight")
    #     if self.verbose:
    #         plt.show()
    #     plt.close()

    #     # Create 2D heatmap plot with X=v_f_norm, Y=rho_0, Z=r_pd_max_norm
    #     plt.figure(figsize=(10, 8))
    #     plt.imshow(np.log10(r_pd_max_norm).T, origin='lower', aspect='auto', extent=[v_f_norm[0], v_f_norm[-1], rho_0_range[0], rho_0_range[-1]], cmap='viridis')
    #     plt.colorbar(label='Max Primal/Dual Residual Ratio')
    #     plt.scatter(v_f_norm, best_rho_0, color='red', label='Best rho_0 per v_f', marker='x')
    #     plt.xlabel('Free-velocity L2 Norm (v_f)')
    #     plt.ylabel('ADMM Penalty Parameter (rho_0)')
    #     plt.title('PADMM Solver Max Primal-Dual Residual Ratio Analysis')
    #     plt.legend()
    #     plt.grid(False)
    #     if self.savefig:
    #         plt.savefig(os.path.dirname(os.path.realpath(__file__)) + "/data/padmm_solver_rho0_v_f_r_pd_max_heatmap.pdf", format="pdf", dpi=300, bbox_inches="tight")
    #     if self.verbose:
    #         plt.show()
    #     plt.close()

    #     # Create 2D heatmap plot with X=v_f_norm, Y=rho_0, Z=r_pd_mean_norm
    #     plt.figure(figsize=(10, 8))
    #     plt.imshow(np.log10(r_pd_mean_norm).T, origin='lower', aspect='auto', extent=[v_f_norm[0], v_f_norm[-1], rho_0_range[0], rho_0_range[-1]], cmap='viridis')
    #     plt.colorbar(label='Mean Primal/Dual Residual Ratio')
    #     plt.scatter(v_f_norm, best_rho_0, color='red', label='Best rho_0 per v_f', marker='x')
    #     plt.xlabel('Free-velocity L2 Norm (v_f)')
    #     plt.ylabel('ADMM Penalty Parameter (rho_0)')
    #     plt.title('PADMM Solver Mean Primal-Dual Residual Ratio Analysis')
    #     plt.legend()
    #     plt.grid(False)
    #     if self.savefig:
    #         plt.savefig(os.path.dirname(os.path.realpath(__file__)) + "/data/padmm_solver_rho0_v_f_r_pd_mean_heatmap.pdf", format="pdf", dpi=300, bbox_inches="tight")
    #     if self.verbose:
    #         plt.show()
    #     plt.close()


###
# Test execution
###

if __name__ == "__main__":
    # Global numpy configurations
    np.set_printoptions(linewidth=1000, precision=10, threshold=10000, suppress=True)  # Suppress scientific notation

    # Global warp configurations
    wp.config.enable_backward = False
    wp.config.verbose = False
    wp.clear_kernel_cache()
    wp.clear_lto_cache()

    # Run all tests
    unittest.main(verbosity=2)
