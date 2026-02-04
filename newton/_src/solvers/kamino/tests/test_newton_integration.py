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

"""TODO"""

import copy
import math
import os
import unittest

import numpy as np
import warp as wp

from newton._src.core import Axis
from newton._src.sim import (
    ActuatorMode,
    Contacts,
    Control,
    Model,
    ModelBuilder,
    State,
)
from newton._src.solvers.kamino.core import inertia
from newton._src.solvers.kamino.core.builder import ModelBuilderKamino
from newton._src.solvers.kamino.core.control import ControlKamino
from newton._src.solvers.kamino.core.data import DataKamino
from newton._src.solvers.kamino.core.joints import JOINT_QMAX, JOINT_QMIN
from newton._src.solvers.kamino.core.model import ModelKamino
from newton._src.solvers.kamino.core.state import StateKamino
from newton._src.solvers.kamino.geometry.contacts import DEFAULT_GEOM_PAIR_MAX_CONTACTS, ContactsKamino
from newton._src.solvers.kamino.models import basics, get_basics_usd_assets_path
from newton._src.solvers.kamino.tests import setup_tests, test_context
from newton._src.solvers.kamino.utils import logger as msg

###
# Utilities
###


def register_solver_attributes(builder: ModelBuilder) -> None:
    """
    TODO
    """
    builder.add_custom_attribute(
        ModelBuilder.CustomAttribute(
            name="num_body_dofs",
            frequency=Model.AttributeFrequency.WORLD,
            assignment=Model.AttributeAssignment.MODEL,
            dtype=wp.int32,
            default=0,
            namespace="info",
        )
    )


def build_boxes_fourbar_newton(
    builder: ModelBuilder | None = None,
    z_offset: float = 0.0,
    fixedbase: bool = False,
    floatingbase: bool = True,
    limits: bool = True,
    ground: bool = True,
    verbose: bool = False,
    new_world: bool = True,
    actuator_ids: list[int] | None = None,
) -> ModelBuilder:
    """
    Constructs a basic model of a four-bar linkage.

    Args:
        builder (ModelBuilder | None):
            An optional existing model builder to populate.\n
            If `None`, a new builder is created.
        z_offset (float):
            A vertical offset to apply to the initial position of the box.
        ground (bool):
            Whether to add a static ground plane to the model.
        new_world (bool):
            Whether to create a new world in the builder for this model.\n
            If `True`, a new world is created and added to the builder.

    Returns:
        ModelBuilder: A model builder containing the four-bar linkage.
    """
    # Create a new builder if none is provided
    if builder is None:
        _builder = ModelBuilder()
    else:
        _builder = builder

    # Create a new world in the builder if requested or if a new builder was created
    if new_world or builder is None:
        _builder.begin_world()

    # Set default actuator IDs if none are provided
    if actuator_ids is None:
        actuator_ids = [1, 3]
    elif not isinstance(actuator_ids, list):
        raise TypeError("actuator_ids, if specified, must be provided as a list of integers.")

    ###
    # Base Parameters
    ###

    # Constant to set an initial z offset for the bodies
    # NOTE: for testing purposes, recommend values are {0.0, -0.001}
    z_0 = z_offset

    # Box dimensions
    d = 0.01
    w = 0.01
    h = 0.1

    # Margins
    mj = 0.001
    dj = 0.5 * d + mj

    ###
    # Body parameters
    ###

    # Box dimensions
    d_1 = h
    w_1 = w
    h_1 = d
    d_2 = d
    w_2 = w
    h_2 = h
    d_3 = h
    w_3 = w
    h_3 = d
    d_4 = d
    w_4 = w
    h_4 = h

    # Inertial properties
    m_i = 1.0
    i_I_i_1 = inertia.solid_cuboid_body_moment_of_inertia(m_i, d_1, w_1, h_1)
    i_I_i_2 = inertia.solid_cuboid_body_moment_of_inertia(m_i, d_2, w_2, h_2)
    i_I_i_3 = inertia.solid_cuboid_body_moment_of_inertia(m_i, d_3, w_3, h_3)
    i_I_i_4 = inertia.solid_cuboid_body_moment_of_inertia(m_i, d_4, w_4, h_4)
    if verbose:
        print(f"i_I_i_1:\n{i_I_i_1}")
        print(f"i_I_i_2:\n{i_I_i_2}")
        print(f"i_I_i_3:\n{i_I_i_3}")
        print(f"i_I_i_4:\n{i_I_i_4}")

    # Initial body positions
    r_0 = wp.vec3f(0.0, 0.0, z_0)
    dr_b1 = wp.vec3f(0.0, 0.0, 0.5 * d)
    dr_b2 = wp.vec3f(0.5 * h + dj, 0.0, 0.5 * h + dj)
    dr_b3 = wp.vec3f(0.0, 0.0, 0.5 * d + h + dj + mj)
    dr_b4 = wp.vec3f(-0.5 * h - dj, 0.0, 0.5 * h + dj)

    # Initial positions of the bodies
    r_b1 = r_0 + dr_b1
    r_b2 = r_b1 + dr_b2
    r_b3 = r_b1 + dr_b3
    r_b4 = r_b1 + dr_b4
    if verbose:
        print(f"r_b1: {r_b1}")
        print(f"r_b2: {r_b2}")
        print(f"r_b3: {r_b3}")
        print(f"r_b4: {r_b4}")

    # Initial body poses
    q_i_1 = wp.transformf(r_b1, wp.quat_identity(dtype=wp.float32))
    q_i_2 = wp.transformf(r_b2, wp.quat_identity(dtype=wp.float32))
    q_i_3 = wp.transformf(r_b3, wp.quat_identity(dtype=wp.float32))
    q_i_4 = wp.transformf(r_b4, wp.quat_identity(dtype=wp.float32))

    # Initial joint positions
    r_j1 = wp.vec3f(r_b2.x, 0.0, r_b1.z)
    r_j2 = wp.vec3f(r_b2.x, 0.0, r_b3.z)
    r_j3 = wp.vec3f(r_b4.x, 0.0, r_b3.z)
    r_j4 = wp.vec3f(r_b4.x, 0.0, r_b1.z)
    if verbose:
        print(f"r_j1: {r_j1}")
        print(f"r_j2: {r_j2}")
        print(f"r_j3: {r_j3}")
        print(f"r_j4: {r_j4}")

    ###
    # Bodies
    ###

    bid1 = _builder.add_link(
        key="link_1",
        mass=m_i,
        I_m=i_I_i_1,
        xform=q_i_1,
        lock_inertia=True,
    )

    bid2 = _builder.add_link(
        key="link_2",
        mass=m_i,
        I_m=i_I_i_2,
        xform=q_i_2,
        lock_inertia=True,
    )

    bid3 = _builder.add_link(
        key="link_3",
        mass=m_i,
        I_m=i_I_i_3,
        xform=q_i_3,
        lock_inertia=True,
    )

    bid4 = _builder.add_link(
        key="link_4",
        mass=m_i,
        I_m=i_I_i_4,
        xform=q_i_4,
        lock_inertia=True,
    )

    ###
    # Geometries
    ###

    _builder.add_shape_box(
        key="box_1",
        body=bid1,
        hx=0.5 * d_1,
        hy=0.5 * w_1,
        hz=0.5 * h_1,
        cfg=ModelBuilder.ShapeConfig(contact_margin=0.0),
    )
    _builder.add_shape_box(
        key="box_2",
        body=bid2,
        hx=0.5 * d_2,
        hy=0.5 * w_2,
        hz=0.5 * h_2,
        cfg=ModelBuilder.ShapeConfig(contact_margin=0.0),
    )

    _builder.add_shape_box(
        key="box_3",
        body=bid3,
        hx=0.5 * d_3,
        hy=0.5 * w_3,
        hz=0.5 * h_3,
        cfg=ModelBuilder.ShapeConfig(contact_margin=0.0),
    )

    _builder.add_shape_box(
        key="box_4",
        body=bid4,
        hx=0.5 * d_4,
        hy=0.5 * w_4,
        hz=0.5 * h_4,
        cfg=ModelBuilder.ShapeConfig(contact_margin=0.0),
    )

    # Add a static collision layer and geometry for the plane
    if ground:
        _builder.add_shape_box(
            key="ground",
            body=-1,
            hx=10.0,
            hy=10.0,
            hz=0.5,
            xform=wp.transformf(0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 1.0),
            cfg=ModelBuilder.ShapeConfig(contact_margin=0.0),
        )

    ###
    # Joints
    ###

    if limits:
        qmin = -0.25 * math.pi
        qmax = 0.25 * math.pi
    else:
        qmin = float(JOINT_QMIN)
        qmax = float(JOINT_QMAX)

    if fixedbase:
        _builder.add_joint_fixed(
            key="world_to_link1",
            parent=-1,
            child=bid1,
            parent_xform=wp.transform_identity(dtype=wp.float32),
            child_xform=wp.transformf(-r_b1, wp.quat_identity(dtype=wp.float32)),
        )

    if floatingbase:
        _builder.add_joint_free(
            key="world_to_link1",
            parent=-1,
            child=bid1,
            parent_xform=wp.transform_identity(dtype=wp.float32),
            child_xform=wp.transform_identity(dtype=wp.float32),
        )

    passive_joint_dof_config = ModelBuilder.JointDofConfig(
        axis=Axis.Y,
        actuator_mode=ActuatorMode.NONE,
        limit_lower=qmin,
        limit_upper=qmax,
    )
    actuated_joint_dof_config = ModelBuilder.JointDofConfig(
        axis=Axis.Y,
        actuator_mode=ActuatorMode.EFFORT,
        limit_lower=qmin,
        limit_upper=qmax,
    )

    _builder.add_joint_revolute(
        key="link1_to_link2",
        parent=bid1,
        child=bid2,
        axis=actuated_joint_dof_config if 1 in actuator_ids else passive_joint_dof_config,
        parent_xform=wp.transformf(r_j1 - r_b1, wp.quat_identity(dtype=wp.float32)),
        child_xform=wp.transformf(r_j1 - r_b2, wp.quat_identity(dtype=wp.float32)),
    )

    _builder.add_joint_revolute(
        key="link2_to_link3",
        parent=bid2,
        child=bid3,
        axis=actuated_joint_dof_config if 2 in actuator_ids else passive_joint_dof_config,
        parent_xform=wp.transformf(r_j2 - r_b2, wp.quat_identity(dtype=wp.float32)),
        child_xform=wp.transformf(r_j2 - r_b3, wp.quat_identity(dtype=wp.float32)),
    )

    _builder.add_joint_revolute(
        key="link3_to_link4",
        parent=bid3,
        child=bid4,
        axis=actuated_joint_dof_config if 3 in actuator_ids else passive_joint_dof_config,
        parent_xform=wp.transformf(r_j3 - r_b3, wp.quat_identity(dtype=wp.float32)),
        child_xform=wp.transformf(r_j3 - r_b4, wp.quat_identity(dtype=wp.float32)),
    )

    _builder.add_joint_revolute(
        key="link4_to_link1",
        parent=bid4,
        child=bid1,
        axis=actuated_joint_dof_config if 4 in actuator_ids else passive_joint_dof_config,
        parent_xform=wp.transformf(r_j4 - r_b4, wp.quat_identity(dtype=wp.float32)),
        child_xform=wp.transformf(r_j4 - r_b1, wp.quat_identity(dtype=wp.float32)),
    )

    # Signal the end of setting-up the new world
    if new_world or builder is None:
        _builder.end_world()

    # Return the lists of element indices
    return _builder


def assert_model_size_equal(test: unittest.TestCase, model0: ModelKamino, model1: ModelKamino) -> None:
    test.assertEqual(
        first=model0.size.num_worlds, second=model1.size.num_worlds, msg="`model.size.num_worlds` are not equal."
    )
    test.assertEqual(
        first=model0.size.sum_of_num_bodies,
        second=model1.size.sum_of_num_bodies,
        msg="`model.size.sum_of_num_bodies` are not equal.",
    )
    test.assertEqual(
        first=model0.size.max_of_num_bodies,
        second=model1.size.max_of_num_bodies,
        msg="`model.size.max_of_num_bodies` are not equal.",
    )
    test.assertEqual(
        first=model0.size.sum_of_num_joints,
        second=model1.size.sum_of_num_joints,
        msg="`model.size.sum_of_num_joints` are not equal.",
    )
    test.assertEqual(
        first=model0.size.max_of_num_joints,
        second=model1.size.max_of_num_joints,
        msg="`model.size.max_of_num_joints` are not equal.",
    )
    test.assertEqual(
        first=model0.size.sum_of_num_passive_joints,
        second=model1.size.sum_of_num_passive_joints,
        msg="`model.size.sum_of_num_passive_joints` are not equal.",
    )
    test.assertEqual(
        first=model0.size.max_of_num_passive_joints,
        second=model1.size.max_of_num_passive_joints,
        msg="`model.size.max_of_num_passive_joints` are not equal.",
    )
    test.assertEqual(
        first=model0.size.sum_of_num_actuated_joints,
        second=model1.size.sum_of_num_actuated_joints,
        msg="`model.size.sum_of_num_actuated_joints` are not equal.",
    )
    test.assertEqual(
        first=model0.size.max_of_num_actuated_joints,
        second=model1.size.max_of_num_actuated_joints,
        msg="`model.size.max_of_num_actuated_joints` are not equal.",
    )
    test.assertEqual(
        first=model0.size.sum_of_num_geoms,
        second=model1.size.sum_of_num_geoms,
        msg="`model.size.sum_of_num_geoms` are not equal.",
    )
    test.assertEqual(
        first=model0.size.max_of_num_geoms,
        second=model1.size.max_of_num_geoms,
        msg="`model.size.max_of_num_geoms` are not equal.",
    )
    test.assertEqual(
        first=model0.size.sum_of_num_material_pairs,
        second=model1.size.sum_of_num_material_pairs,
        msg="`model.size.sum_of_num_material_pairs` are not equal.",
    )
    test.assertEqual(
        first=model0.size.max_of_num_material_pairs,
        second=model1.size.max_of_num_material_pairs,
        msg="`model.size.max_of_num_material_pairs` are not equal.",
    )
    test.assertEqual(
        first=model0.size.sum_of_num_body_dofs,
        second=model1.size.sum_of_num_body_dofs,
        msg="`model.size.sum_of_num_body_dofs` are not equal.",
    )
    test.assertEqual(
        first=model0.size.max_of_num_body_dofs,
        second=model1.size.max_of_num_body_dofs,
        msg="`model.size.max_of_num_body_dofs` are not equal.",
    )
    test.assertEqual(
        first=model0.size.sum_of_num_joint_coords,
        second=model1.size.sum_of_num_joint_coords,
        msg="`model.size.sum_of_num_joint_coords` are not equal.",
    )
    test.assertEqual(
        first=model0.size.max_of_num_joint_coords,
        second=model1.size.max_of_num_joint_coords,
        msg="`model.size.max_of_num_joint_coords` are not equal.",
    )
    test.assertEqual(
        first=model0.size.sum_of_num_joint_dofs,
        second=model1.size.sum_of_num_joint_dofs,
        msg="`model.size.sum_of_num_joint_dofs` are not equal.",
    )
    test.assertEqual(
        first=model0.size.max_of_num_joint_dofs,
        second=model1.size.max_of_num_joint_dofs,
        msg="`model.size.max_of_num_joint_dofs` are not equal.",
    )
    test.assertEqual(
        first=model0.size.sum_of_num_passive_joint_coords,
        second=model1.size.sum_of_num_passive_joint_coords,
        msg="`model.size.sum_of_num_passive_joint_coords` are not equal.",
    )
    test.assertEqual(
        first=model0.size.max_of_num_passive_joint_coords,
        second=model1.size.max_of_num_passive_joint_coords,
        msg="`model.size.max_of_num_passive_joint_coords` are not equal.",
    )
    test.assertEqual(
        first=model0.size.sum_of_num_passive_joint_dofs,
        second=model1.size.sum_of_num_passive_joint_dofs,
        msg="`model.size.sum_of_num_passive_joint_dofs` are not equal.",
    )
    test.assertEqual(
        first=model0.size.max_of_num_passive_joint_dofs,
        second=model1.size.max_of_num_passive_joint_dofs,
        msg="`model.size.max_of_num_passive_joint_dofs` are not equal.",
    )
    test.assertEqual(
        first=model0.size.sum_of_num_actuated_joint_coords,
        second=model1.size.sum_of_num_actuated_joint_coords,
        msg="`model.size.sum_of_num_actuated_joint_coords` are not equal.",
    )
    test.assertEqual(
        first=model0.size.max_of_num_actuated_joint_coords,
        second=model1.size.max_of_num_actuated_joint_coords,
        msg="`model.size.max_of_num_actuated_joint_coords` are not equal.",
    )
    test.assertEqual(
        first=model0.size.sum_of_num_actuated_joint_dofs,
        second=model1.size.sum_of_num_actuated_joint_dofs,
        msg="`model.size.sum_of_num_actuated_joint_dofs` are not equal.",
    )
    test.assertEqual(
        first=model0.size.max_of_num_actuated_joint_dofs,
        second=model1.size.max_of_num_actuated_joint_dofs,
        msg="`model.size.max_of_num_actuated_joint_dofs` are not equal.",
    )
    test.assertEqual(
        first=model0.size.sum_of_num_joint_cts,
        second=model1.size.sum_of_num_joint_cts,
        msg="`model.size.sum_of_num_joint_cts` are not equal.",
    )
    test.assertEqual(
        first=model0.size.max_of_num_joint_cts,
        second=model1.size.max_of_num_joint_cts,
        msg="`model.size.max_of_num_joint_cts` are not equal.",
    )
    test.assertEqual(
        first=model0.size.sum_of_max_limits,
        second=model1.size.sum_of_max_limits,
        msg="`model.size.sum_of_max_limits` are not equal.",
    )
    test.assertEqual(
        first=model0.size.max_of_max_limits,
        second=model1.size.max_of_max_limits,
        msg="`model.size.max_of_max_limits` are not equal.",
    )
    test.assertEqual(
        first=model0.size.sum_of_max_contacts,
        second=model1.size.sum_of_max_contacts,
        msg="`model.size.sum_of_max_contacts` are not equal.",
    )
    test.assertEqual(
        first=model0.size.max_of_max_contacts,
        second=model1.size.max_of_max_contacts,
        msg="`model.size.max_of_max_contacts` are not equal.",
    )
    test.assertEqual(
        first=model0.size.sum_of_max_unilaterals,
        second=model1.size.sum_of_max_unilaterals,
        msg="`model.size.sum_of_max_unilaterals` are not equal.",
    )
    test.assertEqual(
        first=model0.size.max_of_max_unilaterals,
        second=model1.size.max_of_max_unilaterals,
        msg="`model.size.max_of_max_unilaterals` are not equal.",
    )
    test.assertEqual(
        first=model0.size.sum_of_max_total_cts,
        second=model1.size.sum_of_max_total_cts,
        msg="`model.size.sum_of_max_total_cts` are not equal.",
    )
    test.assertEqual(
        first=model0.size.max_of_max_total_cts,
        second=model1.size.max_of_max_total_cts,
        msg="`model.size.max_of_max_total_cts` are not equal.",
    )


def assert_model_info_equal(test: unittest.TestCase, model0: ModelKamino, model1: ModelKamino) -> None:
    test.assertEqual(model0.info.num_worlds, model1.info.num_worlds, "num_worlds are not equal.")
    np.testing.assert_allclose(
        actual=model0.info.num_bodies.numpy(),
        desired=model1.info.num_bodies.numpy(),
        err_msg="model.info.num_bodies are not equal.",
    )
    np.testing.assert_allclose(
        actual=model0.info.num_joints.numpy(),
        desired=model1.info.num_joints.numpy(),
        err_msg="model.info.num_joints are not equal.",
    )
    np.testing.assert_allclose(
        actual=model0.info.num_passive_joints.numpy(),
        desired=model1.info.num_passive_joints.numpy(),
        err_msg="model.info.num_passive_joints are not equal.",
    )
    np.testing.assert_allclose(
        actual=model0.info.num_actuated_joints.numpy(),
        desired=model1.info.num_actuated_joints.numpy(),
        err_msg="model.info.num_actuated_joints are not equal.",
    )
    np.testing.assert_allclose(
        actual=model0.info.num_geoms.numpy(),
        desired=model1.info.num_geoms.numpy(),
        err_msg="model.info.num_geoms are not equal.",
    )
    np.testing.assert_allclose(
        actual=model0.info.num_body_dofs.numpy(),
        desired=model1.info.num_body_dofs.numpy(),
        err_msg="model.info.num_body_dofs are not equal.",
    )
    np.testing.assert_allclose(
        actual=model0.info.num_joint_coords.numpy(),
        desired=model1.info.num_joint_coords.numpy(),
        err_msg="model.info.num_joint_coords are not equal.",
    )
    np.testing.assert_allclose(
        actual=model0.info.num_joint_dofs.numpy(),
        desired=model1.info.num_joint_dofs.numpy(),
        err_msg="model.info.num_joint_dofs are not equal.",
    )
    np.testing.assert_allclose(
        actual=model0.info.num_passive_joint_coords.numpy(),
        desired=model1.info.num_passive_joint_coords.numpy(),
        err_msg="model.info.num_passive_joint_coords are not equal.",
    )
    np.testing.assert_allclose(
        actual=model0.info.num_passive_joint_dofs.numpy(),
        desired=model1.info.num_passive_joint_dofs.numpy(),
        err_msg="model.info.num_passive_joint_dofs are not equal.",
    )
    np.testing.assert_allclose(
        actual=model0.info.num_actuated_joint_coords.numpy(),
        desired=model1.info.num_actuated_joint_coords.numpy(),
        err_msg="model.info.num_actuated_joint_coords are not equal.",
    )
    np.testing.assert_allclose(
        actual=model0.info.num_actuated_joint_dofs.numpy(),
        desired=model1.info.num_actuated_joint_dofs.numpy(),
        err_msg="model.info.num_actuated_joint_dofs are not equal.",
    )
    np.testing.assert_allclose(
        actual=model0.info.num_joint_cts.numpy(),
        desired=model1.info.num_joint_cts.numpy(),
        err_msg="model.info.num_joint_cts are not equal.",
    )
    np.testing.assert_allclose(
        actual=model0.info.bodies_offset.numpy(),
        desired=model1.info.bodies_offset.numpy(),
        err_msg="model.info.bodies_offset are not equal.",
    )
    np.testing.assert_allclose(
        actual=model0.info.joints_offset.numpy(),
        desired=model1.info.joints_offset.numpy(),
        err_msg="model.info.joints_offset are not equal.",
    )
    np.testing.assert_allclose(
        actual=model0.info.body_dofs_offset.numpy(),
        desired=model1.info.body_dofs_offset.numpy(),
        err_msg="model.info.body_dofs_offset are not equal.",
    )
    np.testing.assert_allclose(
        actual=model0.info.joint_coords_offset.numpy(),
        desired=model1.info.joint_coords_offset.numpy(),
        err_msg="model.info.joint_coords_offset are not equal.",
    )
    np.testing.assert_allclose(
        actual=model0.info.joint_dofs_offset.numpy(),
        desired=model1.info.joint_dofs_offset.numpy(),
        err_msg="model.info.joint_dofs_offset are not equal.",
    )
    np.testing.assert_allclose(
        actual=model0.info.joint_passive_coords_offset.numpy(),
        desired=model1.info.joint_passive_coords_offset.numpy(),
        err_msg="model.info.joint_passive_coords_offset are not equal.",
    )
    np.testing.assert_allclose(
        actual=model0.info.joint_passive_dofs_offset.numpy(),
        desired=model1.info.joint_passive_dofs_offset.numpy(),
        err_msg="model.info.joint_passive_dofs_offset are not equal.",
    )
    np.testing.assert_allclose(
        actual=model0.info.joint_actuated_coords_offset.numpy(),
        desired=model1.info.joint_actuated_coords_offset.numpy(),
        err_msg="model.info.joint_actuated_coords_offset are not equal.",
    )
    np.testing.assert_allclose(
        actual=model0.info.joint_actuated_dofs_offset.numpy(),
        desired=model1.info.joint_actuated_dofs_offset.numpy(),
        err_msg="model.info.joint_actuated_dofs_offset are not equal.",
    )
    np.testing.assert_allclose(
        actual=model0.info.joint_cts_offset.numpy(),
        desired=model1.info.joint_cts_offset.numpy(),
        err_msg="model.info.joint_cts_offset are not equal.",
    )
    np.testing.assert_allclose(
        actual=model0.info.base_body_index.numpy(),
        desired=model1.info.base_body_index.numpy(),
        err_msg="model.info.base_body_index are not equal.",
    )
    np.testing.assert_allclose(
        actual=model0.info.base_joint_index.numpy(),
        desired=model1.info.base_joint_index.numpy(),
        err_msg="model.info.base_joint_index are not equal.",
    )
    np.testing.assert_allclose(
        actual=model0.info.mass_min.numpy(),
        desired=model1.info.mass_min.numpy(),
        err_msg="model.info.mass_min are not equal.",
    )
    np.testing.assert_allclose(
        actual=model0.info.mass_max.numpy(),
        desired=model1.info.mass_max.numpy(),
        err_msg="model.info.mass_max are not equal.",
    )
    np.testing.assert_allclose(
        actual=model0.info.mass_total.numpy(),
        desired=model1.info.mass_total.numpy(),
        err_msg="model.info.mass_total are not equal.",
    )
    np.testing.assert_allclose(
        actual=model0.info.inertia_total.numpy(),
        desired=model1.info.inertia_total.numpy(),
        err_msg="model.info.inertia_total are not equal.",
    )


def assert_model_bodies_equal(test: unittest.TestCase, model0: ModelKamino, model1: ModelKamino) -> None:
    test.assertEqual(
        first=model0.bodies.num_bodies,
        second=model1.bodies.num_bodies,
        msg="model.bodies.num_bodies are not equal.",
    )
    np.testing.assert_allclose(
        actual=model0.bodies.wid.numpy(),
        desired=model1.bodies.wid.numpy(),
        err_msg="model.bodies.wid are not equal.",
    )
    np.testing.assert_allclose(
        actual=model0.bodies.bid.numpy(),
        desired=model1.bodies.bid.numpy(),
        err_msg="model.bodies.bid are not equal.",
    )
    np.testing.assert_allclose(
        actual=model0.bodies.i_r_com_i.numpy(),
        desired=model1.bodies.i_r_com_i.numpy(),
        err_msg="model.bodies.i_r_com_i are not equal.",
    )
    np.testing.assert_allclose(
        actual=model0.bodies.m_i.numpy(),
        desired=model1.bodies.m_i.numpy(),
        err_msg="model.bodies.m_i are not equal.",
    )
    np.testing.assert_allclose(
        actual=model0.bodies.inv_m_i.numpy(),
        desired=model1.bodies.inv_m_i.numpy(),
        err_msg="model.bodies.inv_m_i are not equal.",
    )
    np.testing.assert_allclose(
        actual=model0.bodies.i_I_i.numpy(),
        desired=model1.bodies.i_I_i.numpy(),
        err_msg="model.bodies.i_I_i are not equal.",
    )
    np.testing.assert_allclose(
        actual=model0.bodies.inv_i_I_i.numpy(),
        desired=model1.bodies.inv_i_I_i.numpy(),
        err_msg="model.bodies.inv_i_I_i are not equal.",
    )
    np.testing.assert_allclose(
        actual=model0.bodies.q_i_0.numpy(),
        desired=model1.bodies.q_i_0.numpy(),
        err_msg="model.bodies.q_i_0 are not equal.",
    )
    np.testing.assert_allclose(
        actual=model0.bodies.u_i_0.numpy(),
        desired=model1.bodies.u_i_0.numpy(),
        err_msg="model.bodies.u_i_0 are not equal.",
    )


def assert_model_joints_equal(test: unittest.TestCase, model0: ModelKamino, model1: ModelKamino) -> None:
    test.assertEqual(
        first=model0.joints.num_joints,
        second=model1.joints.num_joints,
        msg="model.joints.num_joints are not equal.",
    )
    np.testing.assert_allclose(
        actual=model0.joints.wid.numpy(),
        desired=model1.joints.wid.numpy(),
        err_msg="model.joints.wid are not equal.",
    )
    np.testing.assert_allclose(
        actual=model0.joints.jid.numpy(),
        desired=model1.joints.jid.numpy(),
        err_msg="model.joints.jid are not equal.",
    )
    np.testing.assert_allclose(
        actual=model0.joints.dof_type.numpy(),
        desired=model1.joints.dof_type.numpy(),
        err_msg="model.joints.dof_type are not equal.",
    )
    np.testing.assert_allclose(
        actual=model0.joints.act_type.numpy(),
        desired=model1.joints.act_type.numpy(),
        err_msg="model.joints.act_type are not equal.",
    )
    np.testing.assert_allclose(
        actual=model0.joints.bid_B.numpy(),
        desired=model1.joints.bid_B.numpy(),
        err_msg="model.joints.bid_B are not equal.",
    )
    np.testing.assert_allclose(
        actual=model0.joints.bid_F.numpy(),
        desired=model1.joints.bid_F.numpy(),
        err_msg="model.joints.bid_F are not equal.",
    )
    np.testing.assert_allclose(
        actual=model0.joints.B_r_Bj.numpy(),
        desired=model1.joints.B_r_Bj.numpy(),
        err_msg="model.joints.B_r_Bj are not equal.",
    )
    np.testing.assert_allclose(
        actual=model0.joints.act_type.numpy(),
        desired=model1.joints.act_type.numpy(),
        err_msg="model.joints.act_type are not equal.",
    )
    np.testing.assert_allclose(
        actual=model0.joints.F_r_Fj.numpy(),
        desired=model1.joints.F_r_Fj.numpy(),
        err_msg="model.joints.F_r_Fj are not equal.",
    )
    np.testing.assert_allclose(
        actual=model0.joints.X_j.numpy(),
        desired=model1.joints.X_j.numpy(),
        err_msg="model.joints.X_j are not equal.",
    )
    np.testing.assert_allclose(
        actual=model0.joints.q_j_min.numpy(),
        desired=model1.joints.q_j_min.numpy(),
        err_msg="model.joints.q_j_min are not equal.",
    )
    np.testing.assert_allclose(
        actual=model0.joints.q_j_max.numpy(),
        desired=model1.joints.q_j_max.numpy(),
        err_msg="model.joints.q_j_max are not equal.",
    )
    np.testing.assert_allclose(
        actual=model0.joints.dq_j_max.numpy(),
        desired=model1.joints.dq_j_max.numpy(),
        err_msg="model.joints.dq_j_max are not equal.",
    )
    np.testing.assert_allclose(
        actual=model0.joints.tau_j_max.numpy(),
        desired=model1.joints.tau_j_max.numpy(),
        err_msg="model.joints.tau_j_max are not equal.",
    )
    np.testing.assert_allclose(
        actual=model0.joints.q_j_0.numpy(),
        desired=model1.joints.q_j_0.numpy(),
        err_msg="model.joints.q_j_0 are not equal.",
    )
    np.testing.assert_allclose(
        actual=model0.joints.dq_j_0.numpy(),
        desired=model1.joints.dq_j_0.numpy(),
        err_msg="model.joints.dq_j_0 are not equal.",
    )
    np.testing.assert_allclose(
        actual=model0.joints.num_coords.numpy(),
        desired=model1.joints.num_coords.numpy(),
        err_msg="model.joints.num_coords are not equal.",
    )
    np.testing.assert_allclose(
        actual=model0.joints.num_dofs.numpy(),
        desired=model1.joints.num_dofs.numpy(),
        err_msg="model.joints.num_dofs are not equal.",
    )
    np.testing.assert_allclose(
        actual=model0.joints.num_cts.numpy(),
        desired=model1.joints.num_cts.numpy(),
        err_msg="model.joints.num_cts are not equal.",
    )
    np.testing.assert_allclose(
        actual=model0.joints.coords_offset.numpy(),
        desired=model1.joints.coords_offset.numpy(),
        err_msg="model.joints.coords_offset are not equal.",
    )
    np.testing.assert_allclose(
        actual=model0.joints.dofs_offset.numpy(),
        desired=model1.joints.dofs_offset.numpy(),
        err_msg="model.joints.dofs_offset are not equal.",
    )
    np.testing.assert_allclose(
        actual=model0.joints.passive_coords_offset.numpy(),
        desired=model1.joints.passive_coords_offset.numpy(),
        err_msg="model.joints.passive_coords_offset are not equal.",
    )
    np.testing.assert_allclose(
        actual=model0.joints.passive_dofs_offset.numpy(),
        desired=model1.joints.passive_dofs_offset.numpy(),
        err_msg="model.joints.passive_dofs_offset are not equal.",
    )
    np.testing.assert_allclose(
        actual=model0.joints.actuated_coords_offset.numpy(),
        desired=model1.joints.actuated_coords_offset.numpy(),
        err_msg="model.joints.actuated_coords_offset are not equal.",
    )
    np.testing.assert_allclose(
        actual=model0.joints.actuated_dofs_offset.numpy(),
        desired=model1.joints.actuated_dofs_offset.numpy(),
        err_msg="model.joints.actuated_dofs_offset are not equal.",
    )
    np.testing.assert_allclose(
        actual=model0.joints.cts_offset.numpy(),
        desired=model1.joints.cts_offset.numpy(),
        err_msg="model.joints.cts_offset are not equal.",
    )


def assert_model_geoms_equal(test: unittest.TestCase, model0: ModelKamino, model1: ModelKamino) -> None:
    test.assertEqual(
        first=model0.geoms.num_geoms,
        second=model1.geoms.num_geoms,
        msg="model.geoms.num_geoms are not equal.",
    )
    test.assertEqual(
        first=model0.geoms.num_collidable_geoms,
        second=model1.geoms.num_collidable_geoms,
        msg="model.geoms.num_collidable_geoms are not equal.",
    )
    test.assertEqual(
        first=model0.geoms.num_collidable_geom_pairs,
        second=model1.geoms.num_collidable_geom_pairs,
        msg="model.geoms.num_collidable_geom_pairs are not equal.",
    )
    test.assertLessEqual(
        a=model0.geoms.model_max_contacts,
        b=model1.geoms.model_max_contacts,
        msg="model.geoms.model_max_contacts are not equal.",
    )
    for w in range(model0.size.num_worlds):
        test.assertLessEqual(
            a=model0.geoms.world_max_contacts[w],
            b=model1.geoms.world_max_contacts[w],
            msg=f"model.geoms.world_max_contacts[{w}] are not less-than-or-equal.",
        )
    np.testing.assert_allclose(
        actual=model0.geoms.wid.numpy(),
        desired=model1.geoms.wid.numpy(),
        err_msg="model.geoms.wid are not equal.",
    )
    np.testing.assert_allclose(
        actual=model0.geoms.gid.numpy(),
        desired=model1.geoms.gid.numpy(),
        err_msg="model.geoms.gid are not equal.",
    )
    np.testing.assert_allclose(
        actual=model0.geoms.bid.numpy(),
        desired=model1.geoms.bid.numpy(),
        err_msg="model.geoms.bid are not equal.",
    )
    np.testing.assert_allclose(
        actual=model0.geoms.sid.numpy(),
        desired=model1.geoms.sid.numpy(),
        err_msg="model.geoms.sid are not equal.",
    )
    np.testing.assert_allclose(
        actual=model0.geoms.mid.numpy(),
        desired=model1.geoms.mid.numpy(),
        err_msg="model.geoms.mid are not equal.",
    )
    np.testing.assert_allclose(
        actual=model0.geoms.ptr.numpy(),
        desired=model1.geoms.ptr.numpy(),
        err_msg="model.geoms.ptr are not equal.",
    )
    np.testing.assert_allclose(
        actual=model0.geoms.params.numpy(),
        desired=model1.geoms.params.numpy(),
        err_msg="model.geoms.params are not equal.",
    )
    np.testing.assert_allclose(
        actual=model0.geoms.offset.numpy(),
        desired=model1.geoms.offset.numpy(),
        err_msg="model.geoms.offset are not equal.",
    )
    np.testing.assert_allclose(
        actual=model0.geoms.group.numpy(),
        desired=model1.geoms.group.numpy(),
        err_msg="model.geoms.group are not equal.",
    )
    np.testing.assert_allclose(
        actual=model0.geoms.collides.numpy(),
        desired=model1.geoms.collides.numpy(),
        err_msg="model.geoms.collides are not equal.",
    )
    np.testing.assert_allclose(
        actual=model0.geoms.margin.numpy(),
        desired=model1.geoms.margin.numpy(),
        err_msg="model.geoms.margin are not equal.",
    )
    np.testing.assert_allclose(
        actual=model0.geoms.collidable_pairs.numpy(),
        desired=model1.geoms.collidable_pairs.numpy(),
        err_msg="model.geoms.collidable_pairs are not equal.",
    )


def assert_model_materials_equal(test: unittest.TestCase, model0: ModelKamino, model1: ModelKamino) -> None:
    np.testing.assert_allclose(
        actual=model0.materials.num_materials,
        desired=model1.materials.num_materials,
        err_msg="model.materials.num_materials are not equal.",
    )
    # TODO: Re-enable this check once density info is properly populated
    # np.testing.assert_allclose(
    #     actual=model0.materials.density.numpy(),
    #     desired=model1.materials.density.numpy(),
    #     err_msg="model.materials.density are not equal.",
    # )
    np.testing.assert_allclose(
        actual=model0.materials.restitution.numpy(),
        desired=model1.materials.restitution.numpy(),
        err_msg="model.materials.restitution are not equal.",
    )
    np.testing.assert_allclose(
        actual=model0.materials.static_friction.numpy(),
        desired=model1.materials.static_friction.numpy(),
        err_msg="model.materials.static_friction are not equal.",
    )
    np.testing.assert_allclose(
        actual=model0.materials.dynamic_friction.numpy(),
        desired=model1.materials.dynamic_friction.numpy(),
        err_msg="model.materials.dynamic_friction are not equal.",
    )


def assert_model_material_pairs_equal(test: unittest.TestCase, model0: ModelKamino, model1: ModelKamino) -> None:
    np.testing.assert_allclose(
        actual=model0.material_pairs.num_material_pairs,
        desired=model1.material_pairs.num_material_pairs,
        err_msg="model.material_pairs.num_material_pairs are not equal.",
    )
    np.testing.assert_allclose(
        actual=model0.material_pairs.restitution.numpy(),
        desired=model1.material_pairs.restitution.numpy(),
        err_msg="model.material_pairs.restitution are not equal.",
    )
    np.testing.assert_allclose(
        actual=model0.material_pairs.static_friction.numpy(),
        desired=model1.material_pairs.static_friction.numpy(),
        err_msg="model.material_pairs.static_friction are not equal.",
    )
    np.testing.assert_allclose(
        actual=model0.material_pairs.dynamic_friction.numpy(),
        desired=model1.material_pairs.dynamic_friction.numpy(),
        err_msg="model.material_pairs.dynamic_friction are not equal.",
    )


def assert_model_equal(test: unittest.TestCase, model0: ModelKamino, model1: ModelKamino) -> None:
    assert_model_size_equal(test, model0, model1)
    assert_model_info_equal(test, model0, model1)
    assert_model_bodies_equal(test, model0, model1)
    assert_model_joints_equal(test, model0, model1)
    assert_model_geoms_equal(test, model0, model1)
    assert_model_materials_equal(test, model0, model1)
    assert_model_material_pairs_equal(test, model0, model1)


def assert_state_equal(test: unittest.TestCase, state0: StateKamino, state1: StateKamino) -> None:
    test.assertEqual(state0.q_i.shape, state1.q_i.shape, "state.q_i shapes are not equal.")
    test.assertEqual(state0.u_i.shape, state1.u_i.shape, "state.u_i shapes are not equal.")
    test.assertEqual(state0.w_i.shape, state1.w_i.shape, "state.w_i shapes are not equal.")
    test.assertEqual(state0.q_j.shape, state1.q_j.shape, "state.q_j shapes are not equal.")
    test.assertEqual(state0.q_j_p.shape, state1.q_j_p.shape, "state.q_j_p shapes are not equal.")
    test.assertEqual(state0.dq_j.shape, state1.dq_j.shape, "state.dq_j shapes are not equal.")
    test.assertEqual(state0.lambda_j.shape, state1.lambda_j.shape, "state.lambda_j shapes are not equal.")
    np.testing.assert_allclose(
        actual=state0.q_i.numpy(),
        desired=state1.q_i.numpy(),
        err_msg="state.q_i are not equal.",
    )
    np.testing.assert_allclose(
        actual=state0.u_i.numpy(),
        desired=state1.u_i.numpy(),
        err_msg="state.u_i are not equal.",
    )
    np.testing.assert_allclose(
        actual=state0.w_i.numpy(),
        desired=state1.w_i.numpy(),
        err_msg="state.w_i are not equal.",
    )
    np.testing.assert_allclose(
        actual=state0.q_j.numpy(),
        desired=state1.q_j.numpy(),
        err_msg="state.q_j are not equal.",
    )
    np.testing.assert_allclose(
        actual=state0.q_j_p.numpy(),
        desired=state1.q_j_p.numpy(),
        err_msg="state.q_j_p are not equal.",
    )
    np.testing.assert_allclose(
        actual=state0.dq_j.numpy(),
        desired=state1.dq_j.numpy(),
        err_msg="state.dq_j are not equal.",
    )
    np.testing.assert_allclose(
        actual=state0.lambda_j.numpy(),
        desired=state1.lambda_j.numpy(),
        err_msg="state.lambda_j are not equal.",
    )


def assert_control_equal(test: unittest.TestCase, control0: ControlKamino, control1: ControlKamino) -> None:
    test.assertEqual(control0.tau_j.shape, control1.tau_j.shape, "control.tau_j shapes are not equal.")
    np.testing.assert_allclose(
        actual=control0.tau_j.numpy(),
        desired=control1.tau_j.numpy(),
        err_msg="control.tau_j are not equal.",
    )


###
# Tests
###


class TestKaminoNewtonIntegration(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        # self.verbose = test_context.verbose  # Set to True to enable verbose output
        self.verbose = True  # Set to True to enable verbose output

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

    def test_usd_boxes_fourbar_newton(self):
        """
        TODO
        """
        builder = ModelBuilder()

        # TODO
        USD_MODEL_PATH = os.path.join(get_basics_usd_assets_path(), "boxes_fourbar.usda")
        builder.begin_world()
        builder.add_usd(source=USD_MODEL_PATH, joint_ordering=None)
        builder.end_world()

        # TODO
        msg.info("builder.particle_count: %s", builder.particle_count)
        msg.info("builder.body_count: %s", builder.body_count)
        msg.info("builder.body_world: %s", builder.body_world)
        msg.info("builder.body_key: %s", builder.body_key)
        msg.info("builder.shape_count: %s", builder.shape_count)
        msg.info("builder.shape_world: %s", builder.shape_world)
        msg.info("builder.shape_key: %s", builder.shape_key)
        msg.info("builder.joint_count: %s", builder.joint_count)
        msg.info("builder.joint_coord_count: %s", builder.joint_coord_count)
        msg.info("builder.joint_dof_count: %s", builder.joint_dof_count)
        msg.info("builder.joint_key: %s", builder.joint_key)
        msg.info("builder.joint_world: %s", builder.joint_world)
        msg.info("builder.joint_parent: %s", builder.joint_parent)
        msg.info("builder.joint_child: %s", builder.joint_child)
        msg.info("builder.joint_q_start: %s", builder.joint_q_start)
        msg.info("builder.joint_qd_start: %s", builder.joint_qd_start)
        msg.info("builder.articulation_count: %s", builder.articulation_count)
        msg.info("builder.articulation_start: %s", builder.articulation_start)
        msg.info("builder.articulation_world: %s", builder.articulation_world)

    def test_boxes_fourbar_newton(self):
        """
        TODO
        """
        ###
        # Builders
        ###

        # TODO
        builder_0: ModelBuilder = build_boxes_fourbar_newton(
            builder=None,
            z_offset=0.0,
            fixedbase=False,
            floatingbase=True,
            limits=True,
            ground=True,
            new_world=True,
            actuator_ids=[0, 1, 2, 3, 4],
        )

        # Duplicate the world to test multi-world handling
        builder_0.begin_world()
        builder_0.add_builder(copy.deepcopy(builder_0))
        builder_0.end_world()

        # TODO
        builder_1: ModelBuilderKamino = basics.build_boxes_fourbar(
            builder=None,
            z_offset=0.0,
            fixedbase=False,
            floatingbase=True,
            limits=True,
            ground=True,
            new_world=True,
            actuator_ids=[0, 1, 2, 3, 4],
        )

        # Duplicate the world to test multi-world handling
        builder_1.add_builder(copy.deepcopy(builder_1))

        # TODO
        msg.info("builder_0.num_worlds: %s", builder_0.num_worlds)
        msg.info("builder_0.body_count: %s", builder_0.body_count)
        msg.info("builder_0.body_world: %s", builder_0.body_world)
        msg.info("builder_0.body_key: %s", builder_0.body_key)
        msg.info("builder_0.body_mass: %s", builder_0.body_mass)
        msg.info("builder_0.body_com: %s", builder_0.body_com)
        msg.info("builder_0.body_q: %s", builder_0.body_q)
        msg.info("builder_0.body_qd: %s", builder_0.body_qd)
        msg.info("builder_0.body_inertia: %s", builder_0.body_inertia)
        msg.info("builder_0.joint_count: %s", builder_0.joint_count)
        msg.info("builder_0.joint_coord_count: %s", builder_0.joint_coord_count)
        msg.info("builder_0.joint_dof_count: %s", builder_0.joint_dof_count)
        msg.info("builder_0.joint_world: %s", builder_0.joint_world)
        msg.info("builder_0.joint_key: %s", builder_0.joint_key)
        msg.info("builder_0.joint_parent: %s", builder_0.joint_parent)
        msg.info("builder_0.joint_child: %s", builder_0.joint_child)
        msg.info("builder_0.shape_count: %s", builder_0.shape_count)
        msg.info("builder_0.shape_world: %s", builder_0.shape_world)
        msg.info("builder_0.shape_key: %s", builder_0.shape_key)
        msg.info("builder_0.shape_type: %s", builder_0.shape_type)
        msg.info("builder_0.articulation_count: %s\n", builder_0.articulation_count)
        msg.info("builder_1.num_worlds: %s", builder_1.num_worlds)
        msg.info("builder_1.num_bodies: %s", builder_1.num_bodies)
        msg.info("builder_1.num_body_dofs: %s", builder_1.num_body_dofs)
        msg.info("builder_1.num_joints: %s", builder_1.num_joints)
        msg.info("builder_1.num_joint_coords: %s", builder_1.num_joint_coords)
        msg.info("builder_1.num_joint_dofs: %s", builder_1.num_joint_dofs)
        msg.info("builder_1.num_joint_cts: %s", builder_1.num_joint_cts)
        msg.info("builder_1.num_geoms: %s", builder_1.num_geoms)

        ###
        # Models
        ###

        # TODO
        model_0_nwt: Model = builder_0.finalize(skip_validation_joints=True)
        model_0: ModelKamino = ModelKamino.from_newton(model_0_nwt)
        model_1: ModelKamino = builder_1.finalize()
        msg.info("model_0_nwt.body_key (type: %s): %s", type(model_0_nwt.body_key), model_0_nwt.body_key)
        msg.info("model_0_nwt.body_mass (type: %s): %s", type(model_0_nwt.body_mass), model_0_nwt.body_mass)
        msg.info("model_0_nwt.body_com (type: %s):\n%s", type(model_0_nwt.body_com), model_0_nwt.body_com)
        msg.info("model_0_nwt.body_inertia (type: %s):\n%s", type(model_0_nwt.body_inertia), model_0_nwt.body_inertia)
        msg.info("model_0_nwt.joint_key (type: %s): %s", type(model_0_nwt.joint_key), model_0_nwt.joint_key)
        msg.info("model_0_nwt.joint_type (type: %s): %s", type(model_0_nwt.joint_type), model_0_nwt.joint_type)
        msg.info("model_0_nwt.joint_parent (type: %s): %s", type(model_0_nwt.joint_parent), model_0_nwt.joint_parent)
        msg.info("model_0_nwt.joint_child (type: %s): %s", type(model_0_nwt.joint_child), model_0_nwt.joint_child)
        msg.info("model_0_nwt.joint_q_start (type: %s): %s", type(model_0_nwt.joint_q_start), model_0_nwt.joint_q_start)
        msg.info(
            "model_0_nwt.joint_qd_start (type: %s): %s", type(model_0_nwt.joint_qd_start), model_0_nwt.joint_qd_start
        )
        msg.info("model_0_nwt.joint_q (type: %s): %s", type(model_0_nwt.joint_q), model_0_nwt.joint_q)
        msg.info("model_0_nwt.joint_qd (type: %s): %s", type(model_0_nwt.joint_qd), model_0_nwt.joint_qd)
        msg.info(
            "model_0_nwt.joint_dof_dim (type: %s):\n%s", type(model_0_nwt.joint_dof_dim), model_0_nwt.joint_dof_dim
        )
        msg.info("model_0_nwt.joint_world (type: %s): %s", type(model_0_nwt.joint_world), model_0_nwt.joint_world)
        msg.info(
            "model_0_nwt.articulation_count (type: %s): %s",
            type(model_0_nwt.articulation_count),
            model_0_nwt.articulation_count,
        )
        msg.info(
            "model_0_nwt.articulation_key (type: %s): %s",
            type(model_0_nwt.articulation_key),
            model_0_nwt.articulation_key,
        )
        msg.info(
            "model_0_nwt.articulation_start (type: %s): %s",
            type(model_0_nwt.articulation_start),
            model_0_nwt.articulation_start,
        )
        msg.info(
            "model_0_nwt.articulation_world (type: %s): %s\n",
            type(model_0_nwt.articulation_world),
            model_0_nwt.articulation_world,
        )
        msg.info("model_0.size:\n%s", model_0.size)
        msg.info("model_1.size:\n%s", model_1.size)
        msg.error("model_0_nwt.joint_act_mode:\n%s\n\n", model_0_nwt.joint_act_mode)

        # Checks
        assert_model_equal(self, model_0, model_1)

        ###
        # State
        ###

        # TODO
        state_0_nwt: State = model_0_nwt.state()
        state_0: StateKamino = model_0.state()
        state_1: StateKamino = model_1.state()
        state_2 = StateKamino.from_newton(model_0_nwt, state_0_nwt)

        # TODO
        msg.info("state_0_nwt.body_count (type: %s): %s", type(state_0_nwt.body_count), state_0_nwt.body_count)
        msg.info(
            "state_0_nwt.particle_count (type: %s): %s", type(state_0_nwt.particle_count), state_0_nwt.particle_count
        )
        msg.info(
            "state_0_nwt.joint_coord_count (type: %s): %s",
            type(state_0_nwt.joint_coord_count),
            state_0_nwt.joint_coord_count,
        )
        msg.info(
            "state_0_nwt.joint_dof_count (type: %s): %s", type(state_0_nwt.joint_dof_count), state_0_nwt.joint_dof_count
        )
        msg.info("state_0_nwt.particle_q (type: %s):\n%s", type(state_0_nwt.particle_q), state_0_nwt.particle_q)
        msg.info("state_0_nwt.particle_qd (type: %s):\n%s", type(state_0_nwt.particle_qd), state_0_nwt.particle_qd)
        msg.info("state_0_nwt.particle_f (type: %s):\n%s", type(state_0_nwt.particle_f), state_0_nwt.particle_f)
        msg.info("state_0_nwt.body_q (type: %s):\n%s", type(state_0_nwt.body_q), state_0_nwt.body_q)
        msg.info("state_0_nwt.body_qd (type: %s):\n%s", type(state_0_nwt.body_qd), state_0_nwt.body_qd)
        msg.info("state_0_nwt.body_q_prev (type: %s):\n%s", type(state_0_nwt.body_q_prev), state_0_nwt.body_q_prev)
        msg.info("state_0_nwt.body_qdd (type: %s):\n%s", type(state_0_nwt.body_qdd), state_0_nwt.body_qdd)
        msg.info("state_0_nwt.body_f (type: %s):\n%s", type(state_0_nwt.body_f), state_0_nwt.body_f)
        msg.info("state_0_nwt.joint_q (type: %s):\n%s", type(state_0_nwt.joint_q), state_0_nwt.joint_q)
        msg.info("state_0_nwt.joint_qd (type: %s):\n%s\n\n", type(state_0_nwt.joint_qd), state_0_nwt.joint_qd)
        msg.info("state_0.q_i (shape=%s):\n%s", state_0.q_i.shape, state_0.q_i)
        msg.info("state_0.u_i (shape=%s):\n%s", state_0.u_i.shape, state_0.u_i)
        msg.info("state_0.w_i (shape=%s):\n%s", state_0.w_i.shape, state_0.w_i)
        msg.info("state_0.q_j (shape=%s): %s", state_0.q_j.shape, state_0.q_j)
        msg.info("state_0.q_j_p (shape=%s): %s", state_0.q_j_p.shape, state_0.q_j_p)
        msg.info("state_0.dq_j (shape=%s): %s", state_0.dq_j.shape, state_0.dq_j)
        msg.info("state_0.lambda_j (shape=%s): %s\n\n", state_0.lambda_j.shape, state_0.lambda_j)
        msg.info("state_1.q_i (shape=%s):\n%s", state_1.q_i.shape, state_1.q_i)
        msg.info("state_1.u_i (shape=%s):\n%s", state_1.u_i.shape, state_1.u_i)
        msg.info("state_1.w_i (shape=%s):\n%s", state_1.w_i.shape, state_1.w_i)
        msg.info("state_1.q_j (shape=%s): %s", state_1.q_j.shape, state_1.q_j)
        msg.info("state_1.q_j_p (shape=%s): %s", state_1.q_j_p.shape, state_1.q_j_p)
        msg.info("state_1.dq_j (shape=%s): %s", state_1.dq_j.shape, state_1.dq_j)
        msg.info("state_1.lambda_j (shape=%s): %s\n\n", state_1.lambda_j.shape, state_1.lambda_j)
        msg.info("state_2.q_i (shape=%s):\n%s", state_2.q_i.shape, state_2.q_i)
        msg.info("state_2.u_i (shape=%s):\n%s", state_2.u_i.shape, state_2.u_i)
        msg.info("state_2.w_i (shape=%s):\n%s", state_2.w_i.shape, state_2.w_i)
        msg.info("state_2.q_j (shape=%s): %s", state_2.q_j.shape, state_2.q_j)
        msg.info("state_2.q_j_p (shape=%s): %s", state_2.q_j_p.shape, state_2.q_j_p)
        msg.info("state_2.dq_j (shape=%s): %s", state_2.dq_j.shape, state_2.dq_j)
        msg.info("state_2.lambda_j (shape=%s): %s\n\n", state_2.lambda_j.shape, state_2.lambda_j)

        # Checks
        assert_state_equal(self, state_0, state_1)
        assert_state_equal(self, state_1, state_2)

        ###
        # Control
        ###

        # TODO
        control_0_nwt: Control = model_0_nwt.control()
        control_0: ControlKamino = model_0.control()
        control_1: ControlKamino = model_1.control()
        control_2 = ControlKamino.from_newton(control_0_nwt)

        # TODO
        msg.info("control_0_nwt.joint_f (shape=%s): %s\n\n", control_0_nwt.joint_f.shape, control_0_nwt.joint_f)
        msg.info("control_0.tau_j (shape=%s): %s\n\n", control_0.tau_j.shape, control_0.tau_j)
        msg.info("control_1.tau_j (shape=%s): %s\n\n", control_1.tau_j.shape, control_1.tau_j)
        msg.info("control_2.tau_j (shape=%s): %s\n\n", control_2.tau_j.shape, control_2.tau_j)

        # Checks
        assert_control_equal(self, control_0, control_1)
        assert_control_equal(self, control_1, control_2)

        ###
        # Data
        ###

        # TODO
        data_0: DataKamino = model_0.data()
        # data_1: DataKamino = model_1.data()

        # TODO
        msg.info("data_0.bodies.q_i (shape=%s):\n%s\n\n", data_0.bodies.q_i.shape, data_0.bodies.q_i)
        msg.info("data_0.bodies.u_i (shape=%s):\n%s\n\n", data_0.bodies.u_i.shape, data_0.bodies.u_i)
        msg.info("data_0.bodies.I_i (shape=%s):\n%s\n\n", data_0.bodies.I_i.shape, data_0.bodies.I_i)
        msg.info("data_0.bodies.inv_I_i (shape=%s):\n%s\n\n", data_0.bodies.inv_I_i.shape, data_0.bodies.inv_I_i)
        msg.info("data_0.bodies.w_i (shape=%s):\n%s\n\n", data_0.bodies.w_i.shape, data_0.bodies.w_i)
        msg.info("data_0.bodies.w_a_i (shape=%s):\n%s\n\n", data_0.bodies.w_a_i.shape, data_0.bodies.w_a_i)
        msg.info("data_0.bodies.w_j_i (shape=%s):\n%s\n\n", data_0.bodies.w_j_i.shape, data_0.bodies.w_j_i)
        msg.info("data_0.bodies.w_l_i (shape=%s):\n%s\n\n", data_0.bodies.w_l_i.shape, data_0.bodies.w_l_i)
        msg.info("data_0.bodies.w_c_i (shape=%s):\n%s\n\n", data_0.bodies.w_c_i.shape, data_0.bodies.w_c_i)
        msg.info("data_0.bodies.w_e_i (shape=%s):\n%s\n\n", data_0.bodies.w_e_i.shape, data_0.bodies.w_e_i)

        ###
        # Contacts
        ###

        # TODO
        model_max_contacts_1, world_max_contacts_1 = builder_1.compute_required_contact_capacity(
            max_contacts_per_pair=DEFAULT_GEOM_PAIR_MAX_CONTACTS,
            max_contacts_per_world=None,  # Let the builder compute this value from the number of geoms/shapes
        )

        # TODO
        msg.info("model_0_nwt.shape_collision_filter_pairs: %s", model_0_nwt.shape_collision_filter_pairs)
        msg.info("model_0_nwt.shape_contact_pair_count: %s", model_0_nwt.shape_contact_pair_count)
        msg.info("model_0_nwt.shape_contact_pairs:\n%s\n\n", model_0_nwt.shape_contact_pairs)

        # TODO
        msg.info("model_0_nwt.rigid_contact_max: %s", model_0_nwt.rigid_contact_max)
        msg.info("builder_1.model_max_contacts_1: %s", model_max_contacts_1)
        msg.info("builder_1.world_max_contacts_1: %s\n\n", world_max_contacts_1)

        # TODO
        contacts_0_nwt: Contacts = Contacts(
            rigid_contact_max=model_0_nwt.rigid_contact_max,
            soft_contact_max=0,
            requires_grad=model_0_nwt.requires_grad,
            device=model_0_nwt.device,
            per_contact_shape_properties=False,
            clear_buffers=True,
        )
        contacts_1: ContactsKamino = ContactsKamino(capacity=world_max_contacts_1, device=model_1.device)

        # TODO
        msg.info("contacts_0_nwt.rigid_contact_max: %s", contacts_0_nwt.rigid_contact_max)
        msg.info("contacts_1.model_max_contacts_host: %s", contacts_1.model_max_contacts_host)
        msg.info("contacts_1.world_max_contacts_host: %s\n\n", contacts_1.world_max_contacts_host)


###
# Test execution
###

if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
