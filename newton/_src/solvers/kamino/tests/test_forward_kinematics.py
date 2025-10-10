###########################################################################
# KAMINO: UNIT TESTS: KINEMATICS: FORWARD KINEMATICS
###########################################################################

import hashlib
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino.kinematics.forward_kinematics import ForwardKinematicsSolver
from newton._src.solvers.kamino.tests.test_utils_diff_check import diff_check, run_test_single_joint_examples

###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Tests
###


class JacobianCheckForwardKinematics(unittest.TestCase):
    def setUp(self):
        self.default_device = wp.get_device()

    def tearDown(self):
        self.default_device = None

    def test_Jacobian_check(self):
        # Initialize RNG
        test_name = "Forward Kinematics Jacobian check"
        seed = int(hashlib.sha256(test_name.encode("utf8")).hexdigest(), 16)
        rng = np.random.default_rng(seed)

        def test_function(model):
            assert model.size.num_worlds == 1  # For simplicity we assume a single world

            # Generate (random) state
            state = model.data(device=self.default_device)
            num_bodies_q_i = state.bodies.q_i.shape[0] * 7
            random_state = rng.uniform(-1.0, 1.0, num_bodies_q_i).astype("float32")
            state.bodies.q_i.assign(random_state)

            # Generate (random) controls
            num_joints_q_j = state.joints.q_j.shape[0]
            state.joints.q_j.assign(rng.uniform(-1.0, 1.0, num_joints_q_j).astype("float32"))

            # Evaluate analytic Jacobian
            solver = ForwardKinematicsSolver(model)
            pos_control_transforms = solver.eval_position_control_transformations(state)
            jacobian = solver.eval_kinematic_constraints_jacobian(state, pos_control_transforms)

            # Check against finite differences Jacobian
            rb_state_init_np = state.bodies.q_i.numpy().flatten()  # Save current state of bodies

            def eval_constraints(rb_state_np):
                state.bodies.q_i.assign(rb_state_np)
                constraints = solver.eval_kinematic_constraints(state, pos_control_transforms)
                state.bodies.q_i.assign(rb_state_init_np)  # Reset state
                return constraints.numpy()[0]

            return diff_check(
                eval_constraints,
                jacobian.numpy()[0],
                rb_state_init_np,
                epsilon=1e-4,
                tolerance_abs=5e-3,
                tolerance_rel=5e-3,
            )

        success = run_test_single_joint_examples(test_function, test_name, device=self.default_device)
        self.assertTrue(success)


###
# Test execution
###

if __name__ == "__main__":
    # Global numpy configurations
    np.set_printoptions(
        linewidth=999999, edgeitems=999999, threshold=999999, precision=10, suppress=True
    )  # Suppress scientific notation

    # Global warp configurations
    wp.config.enable_backward = False
    wp.config.verbose = True
    # wp.clear_kernel_cache()
    # wp.clear_lto_cache()

    # Run all tests
    unittest.main(verbosity=2)
