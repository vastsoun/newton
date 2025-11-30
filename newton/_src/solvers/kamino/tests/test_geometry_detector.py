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

"""Unit tests for `geometry/detector.py`"""

import unittest

import numpy as np
import warp as wp

###
# Tests
###


# class TestGeometryCollisionDetector(unittest.TestCase):
#     def setUp(self):
#         self.default_device = wp.get_device()
#         self.verbose = False  # Set to True for detailed output

#         # Set debug-level logging to print verbose test output to console
#         if self.verbose:
#             msg.info("\n")  # Add newline before test output for better readability
#             msg.set_log_level(msg.LogLevel.DEBUG)
#         else:
#             msg.reset_log_level()

#         # Set the common build function and geometry parameters
#         self.build_func = build_boxes_nunchaku
#         self.num_collisions = 3  # NOTE: specialized to build_boxes_nunchaku
#         self.num_contacts = 9  # NOTE: specialized to build_boxes_nunchaku
#         self.max_contacts = 12  # NOTE: This is specialized to the nunchaku model
#         msg.info(f"build_func: {self.build_func.__name__}")
#         msg.info(f"num_collisions: {self.num_collisions}")
#         msg.info(f"num_contacts: {self.num_contacts}")
#         msg.info(f"max_contacts: {self.max_contacts}")

#     def tearDown(self):
#         self.default_device = None
#         if self.verbose:
#             msg.reset_log_level()

#     def test_01_update_collision_geometry_state(self):
#         # Create and set up a model builder
#         builder = make_homogeneous_builder(num_worlds=3, build_fn=self.build_func)

#         # Finalize the model
#         model = builder.finalize(self.default_device)

#         # Create a state container
#         state = model.data()

#         # Update the absolute poses (i.e. computed in world coordinates) of the collision geometries
#         update_collision_geometries_state(state.bodies.q_i, model.cgeoms, state.cgeoms)

#         # Optional verbose output
#         msg.info("state.bodies.q_i:\n%s", state.bodies.q_i)
#         msg.info("state.cgeoms.pose:\n%s", state.cgeoms.pose)
#         msg.info("state.cgeoms.aabb:\n%s", state.cgeoms.aabb)
#         msg.info("state.cgeoms.radius:\n%s", state.cgeoms.radius)

#     def test_02_nxn_broadphase(self):
#         # Create and set up a model builder
#         builder = make_homogeneous_builder(num_worlds=4, build_fn=self.build_func)
#         num_worlds = builder.num_worlds

#         # Finalize the model
#         model = builder.finalize(self.default_device)

#         # Create a state container
#         state = model.data()

#         # Update the state of the collision geometries
#         update_collision_geometries_state(state.bodies.q_i, model.cgeoms, state.cgeoms)

#         # Create collisions container
#         collisions = Collisions(builder=builder, device=self.default_device)

#         # Execute brute-force (NxN) broadphase
#         with wp.ScopedTimer("nxn_broadphase"):
#             nxn_broadphase(model.cgeoms, state.cgeoms, collisions.model, collisions.data)

#         # Check collision output
#         self.assertEqual(collisions.data.model_num_collisions.numpy()[0], num_worlds * self.num_collisions)
#         for i in range(num_worlds):
#             self.assertEqual(collisions.data.world_num_collisions.numpy()[i], self.num_collisions)

#         # Optional verbose output
#         msg.info("collisions.model.num_model_geom_pairs: %s", collisions.model.num_model_geom_pairs)
#         msg.info("collisions.model.num_world_geom_pairs: %s", collisions.model.num_world_geom_pairs)
#         msg.info("collisions.model.model_num_pairs: %s", collisions.model.model_num_pairs)
#         msg.info("collisions.model.world_num_pairs: %s", collisions.model.world_num_pairs)
#         msg.info("collisions.model.wid: %s", collisions.model.wid)
#         msg.info("collisions.model.pairid: %s", collisions.model.pairid)
#         msg.info("collisions.model.geom_pair:\n%s", collisions.model.geom_pair)
#         msg.info("collisions.data.model_num_collisions: %s", collisions.data.model_num_collisions)
#         msg.info("collisions.data.world_num_collisions: %s", collisions.data.world_num_collisions)
#         msg.info("collisions.data.wid: %s", collisions.data.wid)
#         msg.info("collisions.data.geom_pair:\n%s", collisions.data.geom_pair)

#     def test_03_primitive_narrowphase(self):
#         # Create and set up a model builder
#         builder = make_homogeneous_builder(num_worlds=4, build_fn=self.build_func)
#         num_worlds = builder.num_worlds

#         # Finalize the model
#         model = builder.finalize(self.default_device)

#         # Create a state container
#         state = model.data()

#         # Update the state of the collision geometries
#         update_collision_geometries_state(state.bodies.q_i, model.cgeoms, state.cgeoms)

#         # Create collisions container
#         collisions = Collisions(builder=builder, device=self.default_device)

#         # Execute brute-force (NxN) broadphase
#         nxn_broadphase(model.cgeoms, state.cgeoms, collisions.model, collisions.data)

#         # Create a contacts container
#         capacity = [self.max_contacts] * num_worlds  # Custom capacity for each world
#         contacts = Contacts(capacity=capacity, device=self.default_device)

#         # Execute narrowphase for primitive shapes
#         with wp.ScopedTimer("primitive_narrowphase"):
#             primitive_narrowphase(model, state, collisions, contacts)

#         # Optional verbose output
#         msg.info("contacts.num_model_max_contacts: %s", contacts.data.num_model_max_contacts)
#         msg.info("contacts.num_world_max_contacts: %s", contacts.data.num_world_max_contacts)
#         msg.info("contacts.model_max_contacts: %s", contacts.data.model_max_contacts)
#         msg.info("contacts.model_num_contacts: %s", contacts.data.model_num_contacts)
#         msg.info("contacts.world_max_contacts: %s", contacts.data.world_max_contacts)
#         msg.info("contacts.world_num_contacts: %s", contacts.data.world_num_contacts)
#         msg.info("contacts.wid: %s", contacts.data.wid)
#         msg.info("contacts.cid: %s", contacts.data.cid)
#         msg.info("contacts.gid_AB:\n%s", contacts.data.gid_AB)
#         msg.info("contacts.bid_AB:\n%s", contacts.data.bid_AB)
#         msg.info("contacts.position_A:\n%s", contacts.data.position_A)
#         msg.info("contacts.position_B:\n%s", contacts.data.position_B)
#         msg.info("contacts.gapfunc:\n%s", contacts.data.gapfunc)
#         msg.info("contacts.frame:\n%s", contacts.data.frame)
#         msg.info("contacts.material:\n%s", contacts.data.material)

#     def test_04_collision_detector(self):
#         # Create and set up a model builder
#         builder = make_homogeneous_builder(num_worlds=10, build_fn=self.build_func)

#         # Finalize the model
#         model = builder.finalize(self.default_device)

#         # Create a state container
#         state = model.data()

#         # Create a collision detector
#         detector = CollisionDetector(
#             builder=builder, default_max_contacts=self.max_contacts, device=self.default_device
#         )

#         # Peroform collision detection
#         with wp.ScopedTimer("detector.collide"):
#             detector.collide(model, state)

#         # Optional verbose output
#         msg.info("detector.contacts.num_model_max_contacts: %s", detector.contacts.data.num_model_max_contacts)
#         msg.info("detector.contacts.num_world_max_contacts: %s", detector.contacts.data.num_world_max_contacts)
#         msg.info("detector.contacts.model_max_contacts: %s", detector.contacts.data.model_max_contacts)
#         msg.info("detector.contacts.model_num_contacts: %s", detector.contacts.data.model_num_contacts)
#         msg.info("detector.contacts.world_max_contacts: %s", detector.contacts.data.world_max_contacts)
#         msg.info("detector.contacts.world_num_contacts: %s", detector.contacts.data.world_num_contacts)
#         msg.info("detector.contacts.wid: %s", detector.contacts.data.wid)
#         msg.info("detector.contacts.cid: %s", detector.contacts.data.cid)
#         msg.info("detector.contacts.gid_AB:\n%s", detector.contacts.data.gid_AB)
#         msg.info("detector.contacts.bid_AB:\n%s", detector.contacts.data.bid_AB)
#         msg.info("detector.contacts.position_A:\n%s", detector.contacts.data.position_A)
#         msg.info("detector.contacts.position_B:\n%s", detector.contacts.data.position_B)
#         msg.info("detector.contacts.gapfunc:\n%s", detector.contacts.data.gapfunc)
#         msg.info("detector.contacts.frame:\n%s", detector.contacts.data.frame)
#         msg.info("detector.contacts.material:\n%s", detector.contacts.data.material)


###
# Test execution
###

if __name__ == "__main__":
    # Global numpy configurations
    np.set_printoptions(linewidth=500, precision=10, suppress=True)  # Suppress scientific notation

    # Global warp configurations
    wp.config.enable_backward = False
    wp.config.verbose = False
    wp.clear_kernel_cache()
    wp.clear_lto_cache()

    # Run all tests
    unittest.main(verbosity=2)
