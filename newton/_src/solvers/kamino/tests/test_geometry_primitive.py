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

"""Unit tests for the collider functions of narrow-phase collision detection"""

import unittest

import numpy as np
import warp as wp
from warp.context import Devicelike

from newton._src.solvers.kamino.core.builder import ModelBuilder
from newton._src.solvers.kamino.core.model import Model, ModelData
from newton._src.solvers.kamino.core.types import float32, int32, mat83f, vec2i
from newton._src.solvers.kamino.geometry.contacts import Contacts
from newton._src.solvers.kamino.geometry.primitive.broadphase import (
    BoundingVolumesData,
    CollisionCandidatesData,
    CollisionCandidatesModel,
    nxn_broadphase_aabb,
    nxn_broadphase_bs,
    update_geoms_aabb,
    update_geoms_bs,
)
from newton._src.solvers.kamino.geometry.primitive.narrowphase import primitive_narrowphase
from newton._src.solvers.kamino.models.builders import build_boxes_nunchaku
from newton._src.solvers.kamino.models.utils import make_homogeneous_builder
from newton._src.solvers.kamino.tests.utils import models as test_models
from newton._src.solvers.kamino.utils import logger as msg

###
# Utilities
###


class PrimitiveBroadPhaseTestBS:
    def __init__(self, builder: ModelBuilder, device: Devicelike = None):
        # Retrieve the number of world
        num_worlds = builder.num_worlds
        num_geoms = len(builder.collision_geoms)
        # Construct collision pairs
        world_num_geom_pairs, model_geom_pair, model_pairid, model_wid = builder.make_collision_candidate_pairs()
        model_num_geom_pairs = len(model_geom_pair)
        # Allocate the collision model data
        with wp.ScopedDevice(device):
            # Allocate the bounding volumes data
            self.bvdata = BoundingVolumesData(radius=wp.zeros(shape=(num_geoms,), dtype=float32))
            # Allocate the time-invariant collision candidates model
            self._cmodel = CollisionCandidatesModel(
                num_model_geom_pairs=model_num_geom_pairs,
                num_world_geom_pairs=world_num_geom_pairs,
                model_num_pairs=wp.array([model_num_geom_pairs], dtype=int32),
                world_num_pairs=wp.array(world_num_geom_pairs, dtype=int32),
                wid=wp.array(model_wid, dtype=int32),
                pairid=wp.array(model_pairid, dtype=int32),
                geom_pair=wp.array(model_geom_pair, dtype=vec2i),
            )
            # Allocate the time-varying collision candidates data
            self._cdata = CollisionCandidatesData(
                num_model_geom_pairs=model_num_geom_pairs,
                model_num_collisions=wp.zeros(shape=(1,), dtype=int32),
                world_num_collisions=wp.zeros(shape=(num_worlds,), dtype=int32),
                wid=wp.zeros(shape=(model_num_geom_pairs,), dtype=int32),
                geom_pair=wp.zeros(shape=(model_num_geom_pairs,), dtype=vec2i),
            )

    def collide(self, model: Model, data: ModelData, default_margin: float = 0.0):
        self._cdata.clear()
        update_geoms_bs(data.bodies.q_i, model.cgeoms, data.cgeoms, self.bvdata, default_margin)
        nxn_broadphase_bs(model.cgeoms, data.cgeoms, self.bvdata, self._cmodel, self._cdata)


class PrimitiveBroadPhaseTestAABB:
    def __init__(self, builder: ModelBuilder, device: Devicelike = None):
        # Retrieve the number of world
        num_worlds = builder.num_worlds
        num_geoms = len(builder.collision_geoms)
        # Construct collision pairs
        world_num_geom_pairs, model_geom_pair, model_pairid, model_wid = builder.make_collision_candidate_pairs()
        model_num_geom_pairs = len(model_geom_pair)
        # Allocate the collision model data
        with wp.ScopedDevice(device):
            # Allocate the bounding volumes data
            self.bvdata = BoundingVolumesData(aabb=wp.zeros(shape=(num_geoms,), dtype=mat83f))
            # Allocate the time-invariant collision candidates model
            self._cmodel = CollisionCandidatesModel(
                num_model_geom_pairs=model_num_geom_pairs,
                num_world_geom_pairs=world_num_geom_pairs,
                model_num_pairs=wp.array([model_num_geom_pairs], dtype=int32),
                world_num_pairs=wp.array(world_num_geom_pairs, dtype=int32),
                wid=wp.array(model_wid, dtype=int32),
                pairid=wp.array(model_pairid, dtype=int32),
                geom_pair=wp.array(model_geom_pair, dtype=vec2i),
            )
            # Allocate the time-varying collision candidates data
            self._cdata = CollisionCandidatesData(
                num_model_geom_pairs=model_num_geom_pairs,
                model_num_collisions=wp.zeros(shape=(1,), dtype=int32),
                world_num_collisions=wp.zeros(shape=(num_worlds,), dtype=int32),
                wid=wp.zeros(shape=(model_num_geom_pairs,), dtype=int32),
                geom_pair=wp.zeros(shape=(model_num_geom_pairs,), dtype=vec2i),
            )

    def collide(self, model: Model, data: ModelData, default_margin: float = 0.0):
        self._cdata.clear()
        update_geoms_aabb(data.bodies.q_i, model.cgeoms, data.cgeoms, self.bvdata, default_margin)
        nxn_broadphase_aabb(model.cgeoms, self.bvdata, self._cmodel, self._cdata)


def check_broadphase_allocations(
    testcase: unittest.TestCase,
    builder: ModelBuilder,
    broadphase: PrimitiveBroadPhaseTestBS | PrimitiveBroadPhaseTestAABB,
):
    # Calculate the maximum number of geometry pairs
    _, model_geom_pairs, *_ = builder.make_collision_candidate_pairs()
    num_geom_pairs = len(model_geom_pairs)
    # Construct a broad-phase
    testcase.assertEqual(broadphase._cmodel.num_model_geom_pairs, num_geom_pairs)
    testcase.assertEqual(sum(broadphase._cmodel.num_world_geom_pairs), num_geom_pairs)
    testcase.assertEqual(broadphase._cmodel.model_num_pairs.size, 1)
    testcase.assertEqual(broadphase._cmodel.world_num_pairs.size, builder.num_worlds)
    testcase.assertEqual(broadphase._cmodel.wid.size, num_geom_pairs)
    testcase.assertEqual(broadphase._cmodel.pairid.size, num_geom_pairs)
    testcase.assertEqual(broadphase._cmodel.geom_pair.size, num_geom_pairs)
    np.testing.assert_array_equal(broadphase._cmodel.geom_pair.numpy(), model_geom_pairs)
    testcase.assertEqual(broadphase._cdata.model_num_collisions.size, 1)
    testcase.assertEqual(broadphase._cdata.world_num_collisions.size, builder.num_worlds)
    testcase.assertEqual(broadphase._cdata.wid.size, num_geom_pairs)
    testcase.assertEqual(broadphase._cdata.geom_pair.size, num_geom_pairs)


def test_broadphases_on_primitive_combinations(
    testcase: unittest.TestCase, num_worlds: int = 1, device: Devicelike = None
):
    """
    Tests all primitive broad-phase implementations over all shape combinations.
    """
    broadphase_types = [PrimitiveBroadPhaseTestAABB, PrimitiveBroadPhaseTestBS]
    for bp_type in broadphase_types:
        for shape_combo in test_models.shape_combinations:
            # Create a test model builder, model, and data
            build_fn = test_models.shape_combination_to_builder[shape_combo]
            builder = make_homogeneous_builder(num_worlds=num_worlds, build_fn=build_fn)
            model = builder.finalize(device)
            data = model.data()

            # Calculate expected model geom pairs
            _, model_geom_pairs, *_ = builder.make_collision_candidate_pairs()
            num_model_geom_pairs = len(model_geom_pairs)

            # Create a broad-phase backend
            broadphase = bp_type(builder=builder, device=device)
            check_broadphase_allocations(testcase, builder, broadphase)

            # Perform broad-phase collision detection and check results
            broadphase.collide(model, data, default_margin=0.0)

            # Check results
            np.testing.assert_array_equal(
                actual=broadphase._cdata.model_num_collisions.numpy()[0],
                desired=builder.num_collision_geoms // 2,
                err_msg=f"{bp_type.__name__}: Failed `model_num_collisions` check for shape combination: {shape_combo}",
            )
            np.testing.assert_array_equal(
                actual=broadphase._cdata.world_num_collisions.numpy(),
                desired=np.ones(shape=(num_model_geom_pairs,)),
                err_msg=f"{bp_type.__name__}: Failed `world_num_collisions` check for shape combination: {shape_combo}",
            )
            np.testing.assert_array_equal(
                actual=broadphase._cdata.wid.numpy(),
                desired=np.arange(builder.num_worlds),
                err_msg=f"{bp_type.__name__}: Failed `wid` check for shape combination: {shape_combo}",
            )


def test_narrowphase_on_primitive_combinations(
    testcase: unittest.TestCase,
    expected_contacts_per_combination,
    num_worlds: int = 1,
    max_contacts_per_pair: int = 8,
    default_margin: float = 0.0,
    dz_0: float = 0.0,
    device: Devicelike = None,
):
    """
    Tests all primitive broad-phase implementations over all shape combinations.
    """
    broadphase_types = [PrimitiveBroadPhaseTestAABB, PrimitiveBroadPhaseTestBS]
    for bp_type in broadphase_types:
        for shape_combo in test_models.shape_combinations:
            # Run the test only for shape combinations with defined expected contacts
            if shape_combo not in expected_contacts_per_combination:
                msg.info("narrow-phase test: skipping `%s`", shape_combo)
                continue
            msg.info("narrow-phase test: testing `%s`", shape_combo)

            # Create a test model builder, model, and data
            build_fn = test_models.shape_combination_to_builder[shape_combo]
            builder = make_homogeneous_builder(num_worlds=num_worlds, build_fn=build_fn, dz_0=dz_0)
            model = builder.finalize(device)
            data = model.data()

            # Calculate expected model geom pairs
            _, model_geom_pairs, *_ = builder.make_collision_candidate_pairs()
            num_model_geom_pairs = len(model_geom_pairs)
            geom_pairs_per_world = num_model_geom_pairs // num_worlds

            # Create a broad-phase backend
            broadphase = bp_type(builder=builder, device=device)
            check_broadphase_allocations(testcase, builder, broadphase)

            # Perform broad-phase collision detection and check results
            broadphase.collide(model, data, default_margin=default_margin)

            # Create a contacts container
            capacity = [geom_pairs_per_world * max_contacts_per_pair] * num_worlds  # Custom capacity for each world
            contacts = Contacts(capacity=capacity, device=device)
            contacts.clear()

            shape_combo_expected_contacts_per_world = [
                expected_contacts_per_combination[shape_combo]
            ] * builder.num_worlds
            shape_combo_expected_contacts_total = sum(shape_combo_expected_contacts_per_world)

            # Execute narrowphase for primitive shapes
            primitive_narrowphase(model, data, broadphase._cdata, contacts, default_margin)

            # Check results
            np.testing.assert_array_equal(
                actual=contacts.model_num_contacts.numpy()[0],
                desired=shape_combo_expected_contacts_total,
                err_msg=f"{bp_type.__name__}: Failed `model_num_contacts` check for shape combination: {shape_combo}",
            )
            np.testing.assert_array_equal(
                actual=contacts.world_num_contacts.numpy(),
                desired=shape_combo_expected_contacts_per_world,
                err_msg=f"{bp_type.__name__}: Failed `world_num_contacts` check for shape combination: {shape_combo}",
            )


def test_narrowphase_specialized(testcase: unittest.TestCase, test_config: dict, expected: dict):
    """
    Tests a special case of primitive shape combination/configuration
    specified in `test_config` and `expected` dictionaries.
    """

    # Create a test model builder, model, and data
    builder: ModelBuilder = test_config["builder_fn"](dz_0=test_config["dz_0"])
    model = builder.finalize(testcase.default_device)
    data = model.data()

    # Calculate expected model geom pairs
    _, model_geom_pairs, *_ = builder.make_collision_candidate_pairs()
    num_model_geom_pairs = len(model_geom_pairs)
    geom_pairs_per_world = num_model_geom_pairs // builder.num_worlds

    # Create a contacts container
    capacity = [geom_pairs_per_world * test_config["max_contacts_per_pair"]] * builder.num_worlds
    contacts = Contacts(capacity=capacity, device=testcase.default_device)

    broadphase_types = [PrimitiveBroadPhaseTestAABB, PrimitiveBroadPhaseTestBS]
    for bp_type in broadphase_types:
        # Create a broad-phase backend
        broadphase = bp_type(builder=builder, device=testcase.default_device)
        check_broadphase_allocations(testcase, builder, broadphase)

        # Perform broad-phase collision detection and check results
        broadphase.collide(model, data, default_margin=0.0)

        # Execute narrowphase for primitive shapes
        contacts.clear()
        primitive_narrowphase(model, data, broadphase._cdata, contacts, test_config["default_margin"])

        # Optional verbose output
        testname = test_config["name"]
        bvname = bp_type.__name__
        msg.info("[%s][%s]: bodies.q_i:\n%s", testname, bvname, data.bodies.q_i)
        msg.info("[%s][%s]: contacts.model_num_contacts: %s", testname, bvname, contacts.model_num_contacts)
        msg.info("[%s][%s]: contacts.world_num_contacts: %s", testname, bvname, contacts.world_num_contacts)
        msg.info("[%s][%s]: contacts.wid: %s", testname, bvname, contacts.wid)
        msg.info("[%s][%s]: contacts.cid: %s", testname, bvname, contacts.cid)
        msg.info("[%s][%s]: contacts.gid_AB: %s", testname, bvname, contacts.gid_AB)
        msg.info("[%s][%s]: contacts.bid_AB: %s", testname, bvname, contacts.bid_AB)
        msg.info("[%s][%s]: contacts.position_A:\n%s", testname, bvname, contacts.position_A)
        msg.info("[%s][%s]: contacts.position_B:\n%s", testname, bvname, contacts.position_B)
        msg.info("[%s][%s]: contacts.gapfunc:\n%s", testname, bvname, contacts.gapfunc)
        msg.info("[%s][%s]: contacts.frame:\n%s", testname, bvname, contacts.frame)
        msg.info("[%s][%s]: contacts.material:\n%s", testname, bvname, contacts.material)

        # Check results: We expect exactly 1 contact per world
        testcase.assertEqual(
            first=contacts.model_num_contacts.numpy()[0],
            second=expected["num_contacts"],
            msg=f"{bvname}: Failed `model_num_collisions` check for `{testname}`",
        )
        testcase.assertEqual(
            first=contacts.world_num_contacts.numpy()[0],
            second=expected["num_contacts"],
            msg=f"{bvname}: Failed `world_num_collisions` check for `{testname}`",
        )
        testcase.assertEqual(
            first=contacts.wid.numpy()[0],
            second=0,
            msg=f"{bvname}: Failed `wid` check for `{testname}`",
        )
        # Per-contact checks
        num_active = contacts.model_num_contacts.numpy()[0]
        np.testing.assert_array_equal(
            actual=contacts.cid.numpy()[:num_active],
            desired=np.arange(expected["num_contacts"], dtype=np.int32),
            err_msg=f"{bvname}: Failed `cid` check for `boxes_nunchaku`",
        )
        np.testing.assert_almost_equal(
            actual=contacts.gid_AB.numpy()[:num_active],
            desired=expected["gid_AB"],
            err_msg=f"{bvname}: Failed `gid_AB` check for `{testname}`",
        )
        np.testing.assert_almost_equal(
            actual=contacts.bid_AB.numpy()[:num_active],
            desired=expected["bid_AB"],
            err_msg=f"{bvname}: Failed `bid_AB` check for `{testname}`",
        )
        np.testing.assert_almost_equal(
            actual=contacts.position_A.numpy()[:num_active],
            desired=expected["position_A"],
            decimal=6,
            err_msg=f"{bvname}: Failed `position_A` check for `{testname}`",
        )
        np.testing.assert_almost_equal(
            actual=contacts.position_B.numpy()[:num_active],
            desired=expected["position_B"],
            decimal=6,
            err_msg=f"{bvname}: Failed `position_B` check for `{testname}`",
        )
        np.testing.assert_almost_equal(
            actual=contacts.gapfunc.numpy()[:num_active],
            desired=expected["gapfunc"],
            decimal=6,
            err_msg=f"{bvname}: Failed `gapfunc` check for `{testname}`",
        )
        np.testing.assert_almost_equal(
            actual=contacts.frame.numpy()[:num_active],
            desired=expected["frame"],
            decimal=6,
            err_msg=f"{bvname}: Failed `frame` check for `{testname}`",
        )


###
# Constants
###


# Define the expected number of contacts for each supported shape combination
# NOTE: This is based on "median" contact configurations as described in the
# docstring of `test_01_narrowphase_on_primitive_combinations_singleworld`.
expected_median_contacts_per_combination = {
    ("sphere", "sphere"): 1,
    ("sphere", "box"): 1,
    ("sphere", "capsule"): 1,
    ("sphere", "cylinder"): 1,
    # ("sphere", "cone"): TODO,
    # ("sphere", "ellipsoid"): 1,
    ("box", "box"): 4,
    ("box", "capsule"): 1,
    # ("box", "cylinder"): TODO,
    # ("box", "cone"): TODO,
    # ("box", "ellipsoid"): 1,
    ("capsule", "capsule"): 1,
    # ("capsule", "cylinder"): TODO,
    # ("capsule", "cone"): TODO,
    # ("capsule", "ellipsoid"): 1,
    # ("cylinder", "cylinder"): TODO,
    # ("cylinder", "cone"): TODO,
    # ("cylinder", "ellipsoid"): TODO,
    # ("cone", "cone"): TODO,
    # ("cone", "ellipsoid"): 1,
    # ("ellipsoid", "ellipsoid"): 1,
}

###
# Tests
###


class TestGeometryPrimitiveBroadPhase(unittest.TestCase):
    def setUp(self):
        self.default_device = wp.get_device()
        self.verbose = False  # Set to True for verbose output

        # Set debug-level logging to print verbose test output to console
        if self.verbose:
            print("\n")  # Add newline before test output for better readability
            msg.set_log_level(msg.LogLevel.INFO)
        else:
            msg.reset_log_level()

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()

    def test_01_broadphase_on_primitive_combinations_singleworld(self):
        test_broadphases_on_primitive_combinations(self, num_worlds=1, device=self.default_device)

    def test_02_broadphase_on_primitive_combinations_multiworld(self):
        test_broadphases_on_primitive_combinations(self, num_worlds=11, device=self.default_device)

    def test_03_broadphase_on_boxes_nunchaku_singleworld(self):
        broadphase_types = [PrimitiveBroadPhaseTestAABB, PrimitiveBroadPhaseTestBS]
        for bp_type in broadphase_types:
            # Create a test model builder, model, and data
            builder = build_boxes_nunchaku()
            model = builder.finalize(self.default_device)
            data = model.data()

            # Create a broad-phase backend
            broadphase = bp_type(builder=builder, device=self.default_device)
            check_broadphase_allocations(self, builder, broadphase)

            # Perform broad-phase collision detection and check results
            broadphase.collide(model, data, default_margin=0.0)

            # Check results
            # 2x boxes, with 1 from each box, +1 from the sphere (=3, each colliding with the ground)
            np.testing.assert_array_equal(
                actual=broadphase._cdata.model_num_collisions.numpy()[0],
                desired=3,
                err_msg=f"{bp_type.__name__}: Failed `model_num_collisions` check for `boxes_nunchaku`",
            )
            np.testing.assert_array_equal(
                actual=broadphase._cdata.world_num_collisions.numpy()[0],
                desired=3,
                err_msg=f"{bp_type.__name__}: Failed `world_num_collisions` check for `boxes_nunchaku`",
            )
            np.testing.assert_array_equal(
                actual=broadphase._cdata.wid.numpy()[0],
                desired=0,
                err_msg=f"{bp_type.__name__}: Failed `wid` check for `boxes_nunchaku`",
            )

    def test_04_broadphase_on_boxes_nunchaku_multiworld(self):
        broadphase_types = [PrimitiveBroadPhaseTestAABB, PrimitiveBroadPhaseTestBS]
        for bp_type in broadphase_types:
            # Create a test model builder, model, and data
            builder = make_homogeneous_builder(num_worlds=11, build_fn=build_boxes_nunchaku)
            model = builder.finalize(self.default_device)
            data = model.data()

            # Create a broad-phase backend
            broadphase = bp_type(builder=builder, device=self.default_device)
            check_broadphase_allocations(self, builder, broadphase)

            # Perform broad-phase collision detection and check results
            broadphase.collide(model, data, default_margin=0.0)

            # Check results
            # 2x boxes, with 1 from each box, +1 from the sphere (=3, each colliding with the ground)
            np.testing.assert_array_equal(
                actual=broadphase._cdata.model_num_collisions.numpy()[0],
                desired=3 * builder.num_worlds,
                err_msg=f"{bp_type.__name__}: Failed `model_num_collisions` check for `boxes_nunchaku`",
            )
            np.testing.assert_array_equal(
                actual=broadphase._cdata.world_num_collisions.numpy(),
                desired=3 * np.ones(shape=(builder.num_worlds,)),
                err_msg=f"{bp_type.__name__}: Failed `world_num_collisions` check for `boxes_nunchaku`",
            )
            np.testing.assert_array_equal(
                actual=broadphase._cdata.wid.numpy()[: broadphase._cdata.model_num_collisions.numpy()[0]],
                desired=np.tile(np.arange(builder.num_worlds), (3, 1)).T.flatten(),
                err_msg=f"{bp_type.__name__}: Failed `wid` check for `boxes_nunchaku`",
            )


class TestGeometryPrimitiveNarrowPhase(unittest.TestCase):
    def setUp(self):
        self.default_device = wp.get_device()
        self.verbose = True  # Set to True for verbose output

        # Set debug-level logging to print verbose test output to console
        if self.verbose:
            print("\n")  # Add newline before test output for better readability
            msg.set_log_level(msg.LogLevel.INFO)
        else:
            msg.reset_log_level()

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()

    def test_01_narrowphase_on_primitive_combinations_singleworld(self):
        """
        Tests the primitive narrow-phase implementation over all
        shape combinations under the following idealized conditions:
        - all shapes are perfectly stacked along the vertical (z) axis
        - all shapes are centered at the origin in the (x,y) plane
        - all shapes are positioned and oriented in "nominal" configurations
        that would would generate "median" number of contacts per shape combination
        - the geoms are perfectly touching (i.e. penetration is exactly zero)
        - all contact margins are set to zero

        Notes:
        - We refer to these "median" expected contacts as those that are neither the worst-case
        (i.e. maximum possible contacts) nor the best-case (i.e. minimum possible contacts).
        - An example of a "median" configuration is a box-on-box arrangement where two boxes are
        perfectly aligned and touching face-to-face, generating 4 contact points. The worst-case
        would be if the boxes were slightly offset, generating 8 contact points (i.e. full face-face
        contact with 4 points on each face). The best-case would be if the boxes were touching at a
        single edge or corner, generating only 1 contact point.
        - Specialized configurations such as the ones mentioned above are tested in separate test cases.
        """
        # Configure for zero-valued contact margins and initial penetrations
        default_margin: float = 0.0  # No contact margin
        dz_0: float = 0.0  # No initial penetration

        # Run the narrowphase tests
        test_narrowphase_on_primitive_combinations(
            self,
            num_worlds=1,
            expected_contacts_per_combination=expected_median_contacts_per_combination,
            default_margin=default_margin,
            dz_0=dz_0,
            device=self.default_device,
        )

    def test_02_narrowphase_on_primitive_combinations_multiworld(self):
        """
        Tests the primitive narrow-phase implementation over all shape
        combinations under the idealized conditions, with multiple worlds.

        See `test_01_narrowphase_on_primitive_combinations_singleworld` for details.
        """
        # Configure for zero-valued contact margins and initial penetrations
        default_margin: float = 0.0  # No contact margin
        dz_0: float = 0.0  # No initial penetration

        # Run the narrowphase tests
        test_narrowphase_on_primitive_combinations(
            self,
            num_worlds=11,
            expected_contacts_per_combination=expected_median_contacts_per_combination,
            default_margin=default_margin,
            dz_0=dz_0,
            device=self.default_device,
        )

    ###
    # Tests for special cases of shape combinations/configurations
    ###

    def test_03_narrowphase_on_sphere_on_sphere(self):
        """
        Tests the narrow-phase collision detection for a special case of
        two spheres stacked along the vertical (z) axis, centered at the origin
        in the (x,y) plane, and slightly penetrating each other.
        """
        # NOTE: We set to negative value to move the geoms into each other,
        # i.e. move the bottom geom upwards and the top geom downwards.
        dz_0 = -0.01

        # Test meta-data
        test_config = {
            "name": "sphere_on_sphere",
            "builder_fn": test_models.build_sphere_on_sphere,
            "max_contacts_per_pair": 2,
            "default_margin": 0.0,
            "dz_0": dz_0,
        }
        expected = {
            "num_contacts": 1,
            "gid_AB": np.array([[0, 1]], dtype=np.int32),
            "bid_AB": np.array([[0, 1]], dtype=np.int32),
            "position_A": np.array([[0.0, 0.0, 0.5 * abs(dz_0)]], dtype=np.float32),
            "position_B": np.array([[0.0, 0.0, -0.5 * abs(dz_0)]], dtype=np.float32),
            "gapfunc": np.array([[0.0, 0.0, 1.0, dz_0]], dtype=np.float32),
            "frame": np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
        }
        test_narrowphase_specialized(self, test_config, expected)

    def test_04_narrowphase_on_box_on_box(self):
        """
        Tests the narrow-phase collision detection for a special case of
        two boxes stacked along the vertical (z) axis, centered at the origin
        in the (x,y) plane, and slightly penetrating each other.
        """
        # NOTE: We set to negative value to move the geoms into each other,
        # i.e. move the bottom geom upwards and the top geom downwards.
        dz_0 = -0.01

        # Test meta-data
        test_config = {
            "name": "box_on_box",
            "builder_fn": test_models.build_box_on_box,
            "max_contacts_per_pair": 8,
            "default_margin": 0.0,
            "dz_0": dz_0,
        }
        expected = {
            "num_contacts": 4,
            "gid_AB": np.tile(np.array([0, 1], dtype=np.int32), reps=(4, 1)),
            "bid_AB": np.tile(np.array([0, 1], dtype=np.int32), reps=(4, 1)),
            "position_A": np.array(
                [
                    [-0.5, -0.5, 0.5 * abs(dz_0)],
                    [0.5, -0.5, 0.5 * abs(dz_0)],
                    [-0.5, 0.5, 0.5 * abs(dz_0)],
                    [0.5, 0.5, 0.5 * abs(dz_0)],
                ],
                dtype=np.float32,
            ),
            "position_B": np.array(
                [
                    [-0.5, -0.5, -0.5 * abs(dz_0)],
                    [0.5, -0.5, -0.5 * abs(dz_0)],
                    [-0.5, 0.5, -0.5 * abs(dz_0)],
                    [0.5, 0.5, -0.5 * abs(dz_0)],
                ],
                dtype=np.float32,
            ),
            "gapfunc": np.tile(np.array([0.0, 0.0, 1.0, dz_0], dtype=np.float32), reps=(4, 1)),
            "frame": np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), reps=(4, 1)),
        }
        test_narrowphase_specialized(self, test_config, expected)

    def test_05_narrowphase_on_sphere_on_box(self):
        """
        Tests the narrow-phase collision detection for a special case of
        a sphere on top of a box, stacked along the vertical (z) axis,
        centered at the origin in the (x,y) plane, and slightly penetrating each other.
        """
        # NOTE: We set to negative value to move the geoms into each other,
        # i.e. move the bottomo geom upwards and the top geom downwards.
        dz_0 = -0.01

        # Test meta-data
        test_config = {
            "name": "sphere_on_box",
            "builder_fn": test_models.build_sphere_on_box,
            "max_contacts_per_pair": 1,
            "default_margin": 0.0,
            "dz_0": dz_0,
        }
        expected = {
            "num_contacts": 1,
            "gid_AB": np.array([[0, 1]], dtype=np.int32),
            "bid_AB": np.array([[0, 1]], dtype=np.int32),
            "position_A": np.array([[0.0, 0.0, 0.5 * abs(dz_0)]], dtype=np.float32),
            "position_B": np.array([[0.0, 0.0, -0.5 * abs(dz_0)]], dtype=np.float32),
            "gapfunc": np.array([[0.0, 0.0, 1.0, dz_0]], dtype=np.float32),
            "frame": np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
        }
        test_narrowphase_specialized(self, test_config, expected)


###
# Test execution
###

if __name__ == "__main__":
    # Global numpy configurations
    np.set_printoptions(linewidth=20000, threshold=20000, precision=7, suppress=True)

    # Global warp configurations
    wp.config.verbose = False
    wp.clear_kernel_cache()
    wp.clear_lto_cache()

    # Run all tests
    unittest.main(verbosity=2)
