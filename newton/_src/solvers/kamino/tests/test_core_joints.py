###########################################################################
# KAMINO: UNIT TESTS: CORE: BUILDER
###########################################################################

import unittest

import numpy as np
import warp as wp

# Moduel to be tested
from newton._src.solvers.kamino.core.joints import JOINT_REVOLUTE, JointDoFType

###
# Tests
###


class TestCoreJoints(unittest.TestCase):
    def setUp(self):
        self.verbose = False  # Set to True to enable verbose output
        self.default_device = wp.get_device()

    def tearDown(self):
        self.default_device = None

    def test_joint_dof_type_enum(self):
        doftype = JointDoFType.REVOLUTE

        # Optional verbose output
        if self.verbose:
            print("")  # Add a newline for better readability
            print(f"doftype: {doftype}")
            print(f"doftype.value: {doftype.value}")
            print(f"doftype.name: {doftype.name}")
            print(f"doftype.num_cts: {doftype.num_cts}")
            print(f"doftype.num_dofs: {doftype.num_dofs}")

        # Check the enum values
        self.assertEqual(doftype.value, JOINT_REVOLUTE)
        self.assertEqual(doftype.name, "REVOLUTE")
        self.assertEqual(doftype.num_cts, 5)
        self.assertEqual(doftype.num_dofs, 1)


###
# Test execution
###

if __name__ == "__main__":
    # Global numpy configurations
    np.set_printoptions(linewidth=500, precision=10, suppress=True)  # Suppress scientific notation

    # Global warp configurations
    wp.config.verbose = True
    wp.clear_kernel_cache()
    wp.clear_lto_cache()

    # Run all tests
    unittest.main(verbosity=2)
