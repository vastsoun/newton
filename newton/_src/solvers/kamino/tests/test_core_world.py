###########################################################################
# KAMINO: UNIT TESTS: WORLD DESCRIPTOR
###########################################################################

import unittest

import numpy as np
import warp as wp

# Moduel to be tested

###
# Tests
###


class TestWorldDescriptor(unittest.TestCase):
    def setUp(self):
        self.default_device = wp.get_device()

    def tearDown(self):
        self.default_device = None

    def test_single_model(self):
        # TODO
        pass


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
