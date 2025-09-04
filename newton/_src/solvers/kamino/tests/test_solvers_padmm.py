###########################################################################
# KAMINO: UNIT TESTS: SOLVERS: Proximal ADMM Dual Solver
###########################################################################

import os
import unittest
import numpy as np
import warp as wp

import matplotlib.pyplot as plt

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

# Test utilities
from newton._src.solvers.kamino.tests.utils.make import make_containers, update_containers
from newton._src.solvers.kamino.tests.utils.extract import extract_problem_vector, extract_info_vectors
from newton._src.solvers.kamino.tests.utils.print import print_model_info


# Module to be tested
from newton._src.solvers.kamino.solvers.padmm import PADMMSettings, PADMMDualSolver


###
# Helper functions
###

def print_problem_summary(problem: DualProblem):
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


def print_solver_summary(solver: PADMMDualSolver):
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


def save_solver_info(solver: PADMMDualSolver, path: str | None = None, verbose: bool = False):
    nw = solver.size.num_worlds
    status = solver.data.status.numpy()
    iterations = [status[w][1] for w in range(nw)]
    offsets_np = solver.data.info.offsets.numpy()
    f_ccp_np = extract_info_vectors(offsets_np, solver.data.info.f_ccp.numpy(), iterations)
    f_ncp_np = extract_info_vectors(offsets_np, solver.data.info.f_ncp.numpy(), iterations)
    r_iter_np = extract_info_vectors(offsets_np, solver.data.info.r_iter.numpy(), iterations)
    r_primal_np = extract_info_vectors(offsets_np, solver.data.info.r_primal.numpy(), iterations)
    r_dual_np = extract_info_vectors(offsets_np, solver.data.info.r_dual.numpy(), iterations)
    r_compl_np = extract_info_vectors(offsets_np, solver.data.info.r_compl.numpy(), iterations)
    r_ncp_primal_np = extract_info_vectors(offsets_np, solver.data.info.r_ncp_primal.numpy(), iterations)
    r_ncp_dual_np = extract_info_vectors(offsets_np, solver.data.info.r_ncp_dual.numpy(), iterations)
    r_ncp_compl_np = extract_info_vectors(offsets_np, solver.data.info.r_ncp_compl.numpy(), iterations)
    r_ncp_natmap_np = extract_info_vectors(offsets_np, solver.data.info.r_ncp_natmap.numpy(), iterations)

    if verbose:
        for w in range(nw):
            print(f"[World {w}] =======================================================================")
            print(f"solver.info.f_ccp: {f_ccp_np[w]}")
            print(f"solver.info.f_ncp: {f_ncp_np[w]}")
            print(f"solver.info.r_iter: {r_iter_np[w]}")
            print(f"solver.info.r_primal: {r_primal_np[w]}")
            print(f"solver.info.r_dual: {r_dual_np[w]}")
            print(f"solver.info.r_compl: {r_compl_np[w]}")
            print(f"solver.info.r_ncp_primal: {r_ncp_primal_np[w]}")
            print(f"solver.info.r_ncp_dual: {r_ncp_dual_np[w]}")
            print(f"solver.info.r_ncp_compl: {r_ncp_compl_np[w]}")
            print(f"solver.info.r_ncp_natmap: {r_ncp_natmap_np[w]}")

    # List of (label, data) for plotting
    info_list = [
        ("f_ccp", f_ccp_np),
        ("f_ncp", f_ncp_np),
        ("r_iter", r_iter_np),
        ("r_primal", r_primal_np),
        ("r_dual", r_dual_np),
        ("r_compl", r_compl_np),
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
        plt.savefig(os.path.dirname(os.path.realpath(__file__)) + "/data/padmm_solver_info.pdf", format="pdf", dpi=300, bbox_inches="tight")
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
        # builder, _, _ = make_single_builder(build_func=build_box_on_plane)
        # builder, _, _ = make_single_builder(build_func=build_boxes_nunchaku)
        # builder, _, _ = make_homogeneous_builder(num_worlds=4, build_func=build_box_on_plane)
        # builder, _, _ = make_homogeneous_builder(num_worlds=4, build_func=build_boxes_hinged)
        # builder, _, _ = make_homogeneous_builder(num_worlds=4, build_func=build_boxes_nunchaku)
        # builder, _, _ = make_homogeneous_builder(num_worlds=4, build_func=build_boxes_fourbar)
        builder, _, _ = make_heterogeneous_builder()

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
        settings = PADMMSettings()
        settings.primal_tolerance = 1e-8
        settings.dual_tolerance = 1e-8
        settings.compl_tolerance = 1e-8
        # settings.eta = 1e-5
        # settings.max_iterations = 1000

        # Create the ADMM solver
        solver = PADMMDualSolver(
            model=model,
            limits=limits,
            contacts=detector.contacts,
            settings=settings,
            collect_info=True,
            device=self.default_device
        )

        # Optional verbose output
        if self.verbose:
            print("\n")  # Print a newline for better readability
            print_solver_summary(solver)
            print("\n")  # Print a newline for better readability

        # TODO:
        # 1. Interface to set the solver parameters
        # 2. How to make running solver only when constraints are active?

        # Solve the example problem
        with wp.ScopedTimer("solver.solve", active=self.verbose):
            with wp.ScopedDevice(self.default_device):
                solver.solve(problem=problem)

        # Extract numpy arrays from the solver state and solution
        only_active_dims = True
        # D_wp_np = extract_delassus(problem.delassus, only_active_dims=only_active_dims)
        s_wp_np = extract_problem_vector(problem.delassus, solver.data.state.s.numpy(), only_active_dims=only_active_dims)
        v_wp_np = extract_problem_vector(problem.delassus, solver.data.state.v.numpy(), only_active_dims=only_active_dims)
        x_wp_np = extract_problem_vector(problem.delassus, solver.data.state.x.numpy(), only_active_dims=only_active_dims)
        y_wp_np = extract_problem_vector(problem.delassus, solver.data.state.y.numpy(), only_active_dims=only_active_dims)
        z_wp_np = extract_problem_vector(problem.delassus, solver.data.state.z.numpy(), only_active_dims=only_active_dims)
        v_plus_wp_np = extract_problem_vector(problem.delassus, solver.data.solution.v_plus.numpy(), only_active_dims=only_active_dims)
        lambdas_wp_np = extract_problem_vector(problem.delassus, solver.data.solution.lambdas.numpy(), only_active_dims=only_active_dims)
        r_primal_wp_np = extract_problem_vector(problem.delassus, solver.data.residuals.r_primal.numpy(), only_active_dims=only_active_dims)
        r_dual_wp_np = extract_problem_vector(problem.delassus, solver.data.residuals.r_dual.numpy(), only_active_dims=only_active_dims)

        # Retrieve the number of worlds in the model
        nw = model.info.num_worlds

        # Optional verbose output
        if self.verbose:
            print("\n")  # Print a newline for better readability
            # Print the solver state vectors
            for w in range(nw):
                print(f"[World {w}] =======================================================================")
                print(f"      s[{w}]: {s_wp_np[w]}")
                print(f"      v[{w}]: {v_wp_np[w]}")
                print(f"      x[{w}]: {x_wp_np[w]}")
                print(f"      y[{w}]: {y_wp_np[w]}")
                print(f"      z[{w}]: {z_wp_np[w]}")

            # Print the solver solution vectors
            for w in range(nw):
                print(f"[World {w}] =======================================================================")
                print(f" v_plus[{w}]: {v_plus_wp_np[w]}")
                print(f"lambdas[{w}]: {lambdas_wp_np[w]}")

            # Print the solver residuals
            for w in range(nw):
                print(f"[World {w}] =======================================================================")
                print(f"r_primal[{w}]: {r_primal_wp_np[w]}")
                print(f"  r_dual[{w}]: {r_dual_wp_np[w]}")
                # print(f"r_compl[{w}]: {r_compl_wp_np[w]}")

            # Print the solver status
            status = solver.data.status.numpy()
            for w in range(nw):
                print(f"[World {w}] =======================================================================")
                print(f"solver.status: iterations: {status[w][1]}")
                print(f"solver.status: converged: {status[w][0]}")
                print(f"solver.status: r_p: {status[w][2]}")
                print(f"solver.status: r_d: {status[w][3]}")
                print(f"solver.status: r_c: {status[w][4]}")
            print("\n")  # Print a newline for better readability

        # Extract solver explicit solution
        v_plus_np = extract_problem_vector(problem.delassus, solver.data.info.v_plus.numpy(), only_active_dims=only_active_dims)
        v_aug_np = extract_problem_vector(problem.delassus, solver.data.info.v_aug.numpy(), only_active_dims=only_active_dims)
        s_np = extract_problem_vector(problem.delassus, solver.data.info.s.numpy(), only_active_dims=only_active_dims)
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
