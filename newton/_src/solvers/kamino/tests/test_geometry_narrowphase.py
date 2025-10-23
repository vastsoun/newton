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

"""Unit tests for the collider finctions of narrow-phase collision detection"""

import unittest

import numpy as np
import warp as wp

import newton._src.solvers.kamino.utils.logger as msg
from newton._src.solvers.kamino.core import ModelBuilder
from newton._src.solvers.kamino.core.geometry import update_collision_geometries_state
from newton._src.solvers.kamino.core.shapes import (
    BoxShape,
    SphereShape,
)
from newton._src.solvers.kamino.core.types import mat33f, transformf, vec6f
from newton._src.solvers.kamino.geometry.broadphase import nxn_broadphase
from newton._src.solvers.kamino.geometry.collisions import Collisions
from newton._src.solvers.kamino.geometry.contacts import Contacts
from newton._src.solvers.kamino.geometry.primitives import primitive_narrowphase

###
# Builders
###


def build_sphere_on_sphere(dz_0: float = 0.0) -> ModelBuilder:
    builder: ModelBuilder = ModelBuilder()
    bid0 = builder.add_body(
        m_i=1.0,
        i_I_i=mat33f(np.eye(3, dtype=np.float32)),
        q_i_0=transformf(0.0, 0.0, 0.5 + 0.5 * dz_0, 0.0, 0.0, 0.0, 1.0),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    bid1 = builder.add_body(
        m_i=1.0,
        i_I_i=mat33f(np.eye(3, dtype=np.float32)),
        q_i_0=transformf(0.0, 0.0, -0.5 - 0.5 * dz_0, 0.0, 0.0, 0.0, 1.0),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    builder.add_collision_layer("default")
    builder.add_collision_geometry(body_id=bid0, shape=SphereShape(0.5))
    builder.add_collision_geometry(body_id=bid1, shape=SphereShape(0.5))
    return builder


def build_box_on_box(dz_0: float = 0.0) -> ModelBuilder:
    builder: ModelBuilder = ModelBuilder()
    bid0 = builder.add_body(
        m_i=1.0,
        i_I_i=mat33f(np.eye(3, dtype=np.float32)),
        q_i_0=transformf(0.0, 0.0, 0.5 + 0.5 * dz_0, 0.0, 0.0, 0.0, 1.0),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    bid1 = builder.add_body(
        m_i=1.0,
        i_I_i=mat33f(np.eye(3, dtype=np.float32)),
        q_i_0=transformf(0.0, 0.0, -0.5 - 0.5 * dz_0, 0.0, 0.0, 0.0, 1.0),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    builder.add_collision_layer("default")
    builder.add_collision_geometry(body_id=bid0, shape=BoxShape(1.0, 1.0, 1.0))
    builder.add_collision_geometry(body_id=bid1, shape=BoxShape(2.0, 2.0, 1.0))
    return builder


###
# Tests
###


class TestGeometryContacts(unittest.TestCase):
    def setUp(self):
        self.default_device = wp.get_device()
        self.max_contacts = 16  # Maximum number of contacts for the test
        self.verbose = True  # Set to True for verbose output

        # Set debug-level logging to print verbose test output to console
        if self.verbose:
            print("\n")  # Add newline before test output for better readability
            msg.set_log_level(msg.LogLevel.DEBUG)
        else:
            msg.set_log_level(msg.LogLevel.WARNING)

    def tearDown(self):
        self.default_device = None

    def test_01_sphere_on_sphere(self):
        # Define the initial body/geom separation distance
        dz_0 = -0.1

        # Create and set up a model builder
        builder = build_sphere_on_sphere(dz_0=dz_0)
        num_worlds = builder.num_worlds

        # Finalize the model and data
        model = builder.finalize(self.default_device)
        data = model.data()

        # Update the state of the collision geometries
        update_collision_geometries_state(data.bodies.q_i, model.cgeoms, data.cgeoms)

        # Create collisions container
        collisions = Collisions(builder=builder, device=self.default_device)

        # Execute brute-force (NxN) broadphase
        nxn_broadphase(model.cgeoms, data.cgeoms, collisions.cmodel, collisions.cdata)

        # Create a contacts container
        capacity = [self.max_contacts] * num_worlds  # Custom capacity for each world
        contacts = Contacts(capacity=capacity, device=self.default_device)

        # Execute narrowphase for primitive shapes
        primitive_narrowphase(model, data, collisions, contacts)

        # Optional verbose output
        msg.warning(f"bodies.q_i:\n{data.bodies.q_i}")
        msg.warning(f"contacts.model_num_contacts: {contacts.model_num_contacts}")
        msg.warning(f"contacts.world_num_contacts: {contacts.world_num_contacts}")
        msg.warning(f"contacts.wid: {contacts.wid}")
        msg.warning(f"contacts.cid: {contacts.cid}")
        msg.warning(f"contacts.body_A:\n{contacts.body_A}")
        msg.warning(f"contacts.body_B:\n{contacts.body_B}")
        msg.warning(f"contacts.gapfunc:\n{contacts.gapfunc}")
        msg.warning(f"contacts.frame:\n{contacts.frame}")
        msg.warning(f"contacts.material:\n{contacts.material}")

    def test_02_box_on_box(self):
        # Define the initial body/geom separation distance
        dz_0 = -0.1

        # Create and set up a model builder
        builder = build_box_on_box(dz_0=dz_0)
        num_worlds = builder.num_worlds

        # Finalize the model and data
        model = builder.finalize(self.default_device)
        data = model.data()

        # Update the state of the collision geometries
        update_collision_geometries_state(data.bodies.q_i, model.cgeoms, data.cgeoms)

        # Create collisions container
        collisions = Collisions(builder=builder, device=self.default_device)

        # Execute brute-force (NxN) broadphase
        nxn_broadphase(model.cgeoms, data.cgeoms, collisions.cmodel, collisions.cdata)

        # Create a contacts container
        capacity = [self.max_contacts] * num_worlds  # Custom capacity for each world
        contacts = Contacts(capacity=capacity, device=self.default_device)

        # Execute narrowphase for primitive shapes
        primitive_narrowphase(model, data, collisions, contacts)

        # Optional verbose output
        msg.warning(f"bodies.q_i:\n{data.bodies.q_i}")
        msg.warning(f"contacts.model_num_contacts: {contacts.model_num_contacts}")
        msg.warning(f"contacts.world_num_contacts: {contacts.world_num_contacts}")
        msg.warning(f"contacts.wid: {contacts.wid}")
        msg.warning(f"contacts.cid: {contacts.cid}")
        msg.warning(f"contacts.body_A:\n{contacts.body_A}")
        msg.warning(f"contacts.body_B:\n{contacts.body_B}")
        msg.warning(f"contacts.gapfunc:\n{contacts.gapfunc}")
        msg.warning(f"contacts.frame:\n{contacts.frame}")
        msg.warning(f"contacts.material:\n{contacts.material}")


###
# Test execution
###

if __name__ == "__main__":
    # Global numpy configurations
    np.set_printoptions(linewidth=20000, threshold=20000, precision=10, suppress=True)

    # Global warp configurations
    wp.config.verbose = False
    wp.clear_kernel_cache()
    wp.clear_lto_cache()

    # Run all tests
    unittest.main(verbosity=2)
