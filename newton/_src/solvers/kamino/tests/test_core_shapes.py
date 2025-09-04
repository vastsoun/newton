###########################################################################
# KAMINO: UNIT TESTS: CORE: SHAPES
###########################################################################

import unittest
import numpy as np
import warp as wp

# Module to be tested
from newton._src.solvers.kamino.core.shapes import ShapeType, ShapeDescriptor, SphereShape


###
# Tests
###

class TestShapes(unittest.TestCase):

    def test_sphere_type(self):
        # Create a default-constructed surface material
        sid = ShapeType.SPHERE
        # Check default values
        self.assertEqual(sid, 1)
        self.assertEqual(ShapeDescriptor._num_params_of(sid), 1)

    def test_sphere_shape(self):
        # Create a sphere shape
        radius = 1.0
        sphere = SphereShape(radius)
        # Check default values
        self.assertEqual(sphere.name, "sphere")
        self.assertEqual(sphere.typeid, ShapeType.SPHERE)
        self.assertEqual(sphere.nparams, 1)
        self.assertEqual(sphere.params[0], radius)


###
# Test execution
###

if __name__ == "__main__":
    # Global numpy configurations
    np.set_printoptions(linewidth=200, precision=3, suppress=True)  # Suppress scientific notation

    # Global warp configurations
    wp.config.verbose = True
    wp.clear_kernel_cache()
    wp.clear_lto_cache()

    # Run all tests
    unittest.main(verbosity=2)
