###########################################################################
# KAMINO: UNIT TESTS: DYNAMICS: DUAL PROBLEM
###########################################################################

import unittest
import numpy as np
import warp as wp

from newton._src.solvers.kamino.linalg.cholesky import SequentialCholeskyFactorizer
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
from newton._src.solvers.kamino.tests.utils.extract import extract_problem_vector
from newton._src.solvers.kamino.tests.utils.print import print_model_info

# Module to be tested
from newton._src.solvers.kamino.dynamics.dual import DualProblem


###
# Tests
###

class TestDualProblem(unittest.TestCase):

    def setUp(self):
        self.verbose = False  # Set to True for detailed output
        self.default_device = wp.get_device()

    def tearDown(self):
        self.default_device = None

    def test_01_allocate_dual_problem(self):
        """
        Tests the allocation of a DualProblem data members.
        """
        # Model constants
        max_world_contacts = 12

        # Construct the model description using model builders for different systems
        # builder, _, _ = make_single_builder(build_func=build_box_on_plane)
        builder, _, _ = make_heterogeneous_builder()

        # Create the model and containers from the builder
        model, state, limits, detector, jacobians = make_containers(
            builder=builder,
            max_world_contacts=max_world_contacts,
            device=self.default_device
        )

        # Create the Delassus operator
        problem = DualProblem(
            model=model,
            state=state,
            limits=limits,
            contacts=detector.contacts,
            factorizer=SequentialCholeskyFactorizer,
            device=self.default_device
        )

        # Optional verbose output
        if self.verbose:
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

        # Extract expected allocation sizes
        nw = model.info.num_worlds
        nb = model.size.sum_of_num_bodies
        maxnl = limits.num_model_max_limits
        maxnc = detector.contacts.num_model_max_contacts
        maxdims = model.size.sum_of_num_joint_cts + maxnl + 3 * maxnc

        # Check allocations
        self.assertEqual(problem.data.config.size, nw)
        self.assertEqual(problem.data.maxdim.size, nw)
        self.assertEqual(problem.data.dim.size, nw)
        self.assertEqual(problem.data.mio.size, nw)
        self.assertEqual(problem.data.vio.size, nw)
        self.assertEqual(problem.data.u_f.size, nb)
        self.assertEqual(problem.data.v_b.size, maxdims)
        self.assertEqual(problem.data.v_i.size, maxdims)
        self.assertEqual(problem.data.v_f.size, maxdims)
        self.assertEqual(problem.data.mu.size, maxnc)
        maxdim_np = problem.data.maxdim.numpy()
        self.assertEqual(np.sum(maxdim_np), maxdims)
        dim_np = problem.data.dim.numpy()
        self.assertEqual(np.sum(dim_np), 0)

    def test_02_dual_problem_build(self):
        """
        Tests building the dual problem from time-varying data.
        """
        # Model constants
        max_world_contacts = 12

        # Construct the model description using model builders for different systems
        # builder, _, _ = make_single_builder(build_func=build_box_on_plane)
        builder, _, _ = make_heterogeneous_builder()
        num_worlds = builder.num_worlds

        # Create the model and containers from the builder
        model, state, limits, detector, jacobians = make_containers(
            builder=builder,
            max_world_contacts=max_world_contacts,
            device=self.default_device
        )

        # Update the containers
        update_containers(model=model, state=state, limits=limits, detector=detector, jacobians=jacobians)
        if self.verbose:
            print_model_info(model)

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

        # Extract numpy arrays from the problem data
        v_b_wp_np = problem.data.v_b.numpy()
        v_i_wp_np = problem.data.v_i.numpy()
        v_f_wp_np = problem.data.v_f.numpy()

        # Extract free-velocity and solution vectors lists of numpy arrays
        v_b_np = extract_problem_vector(problem.delassus, vector=v_b_wp_np, only_active_dims=True)
        v_i_np = extract_problem_vector(problem.delassus, vector=v_i_wp_np, only_active_dims=True)
        v_f_np = extract_problem_vector(problem.delassus, vector=v_f_wp_np, only_active_dims=True)

        # Optional verbose output
        if self.verbose:
            print("")  # Print a blank line for better readability
            print(f"problem.data.maxdim: {problem.data.maxdim}")
            print(f"problem.data.dim: {problem.data.dim}")
            print(f"problem.data.mio: {problem.data.mio}")
            print(f"problem.data.vio: {problem.data.vio}")
            print(f"problem.data.D: {problem.data.D.shape}")
            print(f"problem.data.u_f:\n{problem.data.u_f}")
            print(f"problem.data.v_b:\n{problem.data.v_b}")
            print(f"problem.data.v_i:\n{problem.data.v_i}")
            print(f"problem.data.v_f:\n{problem.data.v_f}")
            print(f"problem.data.mu:\n{problem.data.mu}")
            for w in range(num_worlds):
                print(f"problem.data.v_b[{w}]:\n{v_b_np[w]}")
            for w in range(num_worlds):
                print(f"problem.data.v_i[{w}]:\n{v_i_np[w]}")
            for w in range(num_worlds):
                print(f"problem.data.v_f[{w}]:\n{v_f_np[w]}")

        # Check the problem data
        # TODO


###
# Test execution
###

if __name__ == "__main__":
    # Global numpy configurations
    np.set_printoptions(linewidth=2000, precision=10, suppress=True)  # Suppress scientific notation

    # Global warp configurations
    wp.config.enable_backward = False
    wp.config.verbose = False
    wp.clear_kernel_cache()
    wp.clear_lto_cache()

    # Run all tests
    unittest.main(verbosity=2)
