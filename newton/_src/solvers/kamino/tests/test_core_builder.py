###########################################################################
# KAMINO: UNIT TESTS: CORE: BUILDER
###########################################################################

import unittest
import numpy as np
import warp as wp

# Test utilities
from newton._src.solvers.kamino.models.builders import (
    build_box_on_plane,
    build_box_pendulum,
    build_boxes_hinged,
    build_boxes_nunchaku,
    build_boxes_fourbar
)

# Moduel to be tested
from newton._src.solvers.kamino.core.builder import ModelBuilder


###
# Tests
###

class TestModelBuilder(unittest.TestCase):

    def setUp(self):
        self.verbose = False  # Set to True to enable verbose output
        self.default_device = wp.get_device()

    def tearDown(self):
        self.default_device = None

    def test_make_builder_box_on_plane(self):
        builder = ModelBuilder()

        # Construct a first model
        bids, jids, gids = build_box_on_plane(builder)

        # Optional verbose output
        if self.verbose:
            print("")  # Add a newline for better readability
            print(f"builder.num_bodies: {builder.num_bodies}")
            print(f"builder.num_joints: {builder.num_joints}")
            print(f"builder.num_collision_geoms: {builder.num_collision_geoms}")
            print(f"builder.num_physical_geoms: {builder.num_physical_geoms}")
            for i, body in enumerate(builder.bodies):
                print(f"body {i}: {body}")
            for j, joint in enumerate(builder.joints):
                print(f"joint {j}: {joint}")
            for g, geom in enumerate(builder.collision_geoms):
                print(f"geom {g}: {geom}")

        # Check the number of bodies, joints, and collision geometries
        for i, bid in enumerate(bids):
            self.assertEqual(bid, i)
            self.assertEqual(bid, builder.bodies[i].bid)
        for i, jid in enumerate(jids):
            self.assertEqual(jid, i)
            self.assertEqual(jid, builder.joints[i].jid)
        for i, gid in enumerate(gids):
            self.assertEqual(gid, i)
            self.assertEqual(gid, builder.collision_geoms[i].gid)

        # Build the model
        model = builder.finalize(self.default_device)
        self.assertEqual(model.size.sum_of_num_bodies, 1)
        self.assertEqual(model.size.sum_of_num_joints, 0)
        self.assertEqual(model.size.sum_of_num_collision_geoms, 2)
        self.assertEqual(model.size.sum_of_num_physical_geoms, 0)
        self.assertEqual(model.device, self.default_device)

    def test_make_builder_box_pendulum(self):
        builder = ModelBuilder()

        # Construct a first model
        bids, jids, gids = build_box_pendulum(builder)

        # Optional verbose output
        if self.verbose:
            print("")  # Add a newline for better readability
            print(f"builder.num_bodies: {builder.num_bodies}")
            print(f"builder.num_joints: {builder.num_joints}")
            print(f"builder.num_collision_geoms: {builder.num_collision_geoms}")
            print(f"builder.num_physical_geoms: {builder.num_physical_geoms}")
            for i, body in enumerate(builder.bodies):
                print(f"body {i}: {body}")
            for j, joint in enumerate(builder.joints):
                print(f"joint {j}: {joint}")
            for g, geom in enumerate(builder.collision_geoms):
                print(f"geom {g}: {geom}")

        # Check the number of bodies, joints, and collision geometries
        for i, bid in enumerate(bids):
            self.assertEqual(bid, i)
            self.assertEqual(bid, builder.bodies[i].bid)
        for i, jid in enumerate(jids):
            self.assertEqual(jid, i)
            self.assertEqual(jid, builder.joints[i].jid)
        for i, gid in enumerate(gids):
            self.assertEqual(gid, i)
            self.assertEqual(gid, builder.collision_geoms[i].gid)

        # Build the model
        model = builder.finalize(self.default_device)
        self.assertEqual(model.size.sum_of_num_bodies, 1)
        self.assertEqual(model.size.sum_of_num_joints, 1)
        self.assertEqual(model.size.sum_of_num_collision_geoms, 2)
        self.assertEqual(model.size.sum_of_num_physical_geoms, 0)
        self.assertEqual(model.device, self.default_device)

    def test_make_builder_boxes_hinged(self):
        builder = ModelBuilder()

        # Construct a first model
        bids, jids, gids = build_boxes_hinged(builder)

        # Optional verbose output
        if self.verbose:
            print("")  # Add a newline for better readability
            print(f"builder.num_bodies: {builder.num_bodies}")
            print(f"builder.num_joints: {builder.num_joints}")
            print(f"builder.num_collision_geoms: {builder.num_collision_geoms}")
            print(f"builder.num_physical_geoms: {builder.num_physical_geoms}")
            for i, body in enumerate(builder.bodies):
                print(f"body {i}: {body}")
            for j, joint in enumerate(builder.joints):
                print(f"joint {j}: {joint}")
            for g, geom in enumerate(builder.collision_geoms):
                print(f"geom {g}: {geom}")

        # Check the number of bodies, joints, and collision geometries
        for i, bid in enumerate(bids):
            self.assertEqual(bid, i)
            self.assertEqual(bid, builder.bodies[i].bid)
        for i, jid in enumerate(jids):
            self.assertEqual(jid, i)
            self.assertEqual(jid, builder.joints[i].jid)
        for i, gid in enumerate(gids):
            self.assertEqual(gid, i)
            self.assertEqual(gid, builder.collision_geoms[i].gid)

        # Build the model
        model = builder.finalize(self.default_device)
        self.assertEqual(model.size.sum_of_num_bodies, 2)
        self.assertEqual(model.size.sum_of_num_joints, 1)
        self.assertEqual(model.size.sum_of_num_collision_geoms, 3)
        self.assertEqual(model.size.sum_of_num_physical_geoms, 0)
        self.assertEqual(model.device, self.default_device)

    def test_make_builder_boxes_nunchaku(self):
        builder = ModelBuilder()

        # Construct a first model
        bids, jids, gids = build_boxes_nunchaku(builder)

        # Optional verbose output
        if self.verbose:
            print("")  # Add a newline for better readability
            print(f"builder.num_bodies: {builder.num_bodies}")
            print(f"builder.num_joints: {builder.num_joints}")
            print(f"builder.num_collision_geoms: {builder.num_collision_geoms}")
            print(f"builder.num_physical_geoms: {builder.num_physical_geoms}")
            for i, body in enumerate(builder.bodies):
                print(f"body {i}: {body}")
            for j, joint in enumerate(builder.joints):
                print(f"joint {j}: {joint}")
            for g, geom in enumerate(builder.collision_geoms):
                print(f"geom {g}: {geom}")

        # Check the number of bodies, joints, and collision geometries
        for i, bid in enumerate(bids):
            self.assertEqual(bid, i)
            self.assertEqual(bid, builder.bodies[i].bid)
        for i, jid in enumerate(jids):
            self.assertEqual(jid, i)
            self.assertEqual(jid, builder.joints[i].jid)
        for i, gid in enumerate(gids):
            self.assertEqual(gid, i)
            self.assertEqual(gid, builder.collision_geoms[i].gid)

        # Build the model
        model = builder.finalize(self.default_device)
        self.assertEqual(model.size.sum_of_num_bodies, 3)
        self.assertEqual(model.size.sum_of_num_joints, 2)
        self.assertEqual(model.size.sum_of_num_collision_geoms, 4)
        self.assertEqual(model.size.sum_of_num_physical_geoms, 0)
        self.assertEqual(model.device, self.default_device)

    def test_make_builder_boxes_fourbar(self):
        builder = ModelBuilder()

        # Construct a first model
        bids, jids, gids = build_boxes_fourbar(builder)

        # Optional verbose output
        if self.verbose:
            print("")  # Add a newline for better readability
            print(f"builder.num_bodies: {builder.num_bodies}")
            print(f"builder.num_joints: {builder.num_joints}")
            print(f"builder.num_collision_geoms: {builder.num_collision_geoms}")
            print(f"builder.num_physical_geoms: {builder.num_physical_geoms}")
            for i, body in enumerate(builder.bodies):
                print(f"body {i}: {body}")
            for j, joint in enumerate(builder.joints):
                print(f"joint {j}: {joint}")
            for g, geom in enumerate(builder.collision_geoms):
                print(f"geom {g}: {geom}")

        # Check the number of bodies, joints, and collision geometries
        for i, bid in enumerate(bids):
            self.assertEqual(bid, i)
            self.assertEqual(bid, builder.bodies[i].bid)
        for i, jid in enumerate(jids):
            self.assertEqual(jid, i)
            self.assertEqual(jid, builder.joints[i].jid)
        for i, gid in enumerate(gids):
            self.assertEqual(gid, i)
            self.assertEqual(gid, builder.collision_geoms[i].gid)

        # Build the model
        model = builder.finalize(self.default_device)
        self.assertEqual(model.size.sum_of_num_bodies, 4)
        self.assertEqual(model.size.sum_of_num_joints, 4)
        self.assertEqual(model.size.sum_of_num_collision_geoms, 5)
        self.assertEqual(model.size.sum_of_num_physical_geoms, 0)
        self.assertEqual(model.device, self.default_device)


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
