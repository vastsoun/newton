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

"""
KAMINO: UNIT TESTS: COMPARISON UTILITIES
"""

import unittest

import numpy as np

from newton._src.solvers.kamino.core.builder import ModelBuilder

###
# Array-like comparisons
###


def lists_equal(list1, list2) -> bool:
    return np.array_equal(list1, list2)


def arrays_equal(arr1, arr2, tolerance=1e-6) -> bool:
    return np.allclose(arr1, arr2, atol=tolerance)


def matrices_equal(m1, m2, tolerance=1e-6) -> bool:
    return np.allclose(m1, m2, atol=tolerance)


def vectors_equal(v1, v2, tolerance=1e-6) -> bool:
    return np.allclose(v1, v2, atol=tolerance)


###
# Container comparisons
###


def assert_builders_equal(
    fixture: unittest.TestCase,
    builder1: ModelBuilder,
    builder2: ModelBuilder,
    skip_colliders: bool = False,
    skip_materials: bool = False,
):
    """
    Compares two ModelBuilder instances for equality.
    """
    fixture.assertEqual(builder1.num_bodies, builder2.num_bodies)
    fixture.assertEqual(builder1.num_joints, builder2.num_joints)
    fixture.assertEqual(builder1.num_geoms, builder2.num_geoms)
    fixture.assertEqual(builder1.num_materials, builder2.num_materials)

    for i in range(builder1.num_bodies):
        fixture.assertEqual(builder1.bodies[i].wid, builder2.bodies[i].wid)
        fixture.assertEqual(builder1.bodies[i].bid, builder2.bodies[i].bid)
        fixture.assertAlmostEqual(builder1.bodies[i].m_i, builder2.bodies[i].m_i)
        fixture.assertTrue(matrices_equal(builder1.bodies[i].i_I_i, builder2.bodies[i].i_I_i))
        fixture.assertTrue(vectors_equal(builder1.bodies[i].q_i_0, builder2.bodies[i].q_i_0))
        fixture.assertTrue(vectors_equal(builder1.bodies[i].u_i_0, builder2.bodies[i].u_i_0))

    for j in range(builder1.num_joints):
        fixture.assertEqual(builder1.joints[j].wid, builder2.joints[j].wid)
        fixture.assertEqual(builder1.joints[j].jid, builder2.joints[j].jid)
        fixture.assertEqual(builder1.joints[j].act_type, builder2.joints[j].act_type)
        fixture.assertEqual(builder1.joints[j].dof_type, builder2.joints[j].dof_type)
        fixture.assertEqual(builder1.joints[j].bid_B, builder2.joints[j].bid_B)
        fixture.assertEqual(builder1.joints[j].bid_F, builder2.joints[j].bid_F)
        fixture.assertTrue(
            vectors_equal(builder1.joints[j].B_r_Bj, builder2.joints[j].B_r_Bj),
            f"Joint {j} B_r_Bj:\nleft:\n{builder1.joints[j].B_r_Bj}\nright:\n{builder2.joints[j].B_r_Bj}",
        )
        fixture.assertTrue(
            vectors_equal(builder1.joints[j].F_r_Fj, builder2.joints[j].F_r_Fj),
            f"Joint {j} F_r_Fj:\nleft:\n{builder1.joints[j].F_r_Fj}\nright:\n{builder2.joints[j].F_r_Fj}",
        )
        fixture.assertTrue(
            matrices_equal(builder1.joints[j].X_j, builder2.joints[j].X_j),
            f"Joint {j} X_j:\nleft:\n{builder1.joints[j].X_j}\nright:\n{builder2.joints[j].X_j}",
        )
        fixture.assertTrue(
            arrays_equal(builder1.joints[j].q_j_min, builder2.joints[j].q_j_min),
            f"Joint {j} q_j_min:\nleft:\n{builder1.joints[j].q_j_min}\nright:\n{builder2.joints[j].q_j_min}",
        )
        fixture.assertTrue(
            arrays_equal(builder1.joints[j].q_j_max, builder2.joints[j].q_j_max),
            f"Joint {j} q_j_max:\nleft:\n{builder1.joints[j].q_j_max}\nright:\n{builder2.joints[j].q_j_max}",
        )
        fixture.assertTrue(
            arrays_equal(builder1.joints[j].dq_j_max, builder2.joints[j].dq_j_max),
            f"Joint {j} dq_j_max:\nleft:\n{builder1.joints[j].dq_j_max}\nright:\n{builder2.joints[j].dq_j_max}",
        )
        fixture.assertTrue(
            arrays_equal(builder1.joints[j].tau_j_max, builder2.joints[j].tau_j_max),
            f"Joint {j} tau_j_max:\nleft:\n{builder1.joints[j].tau_j_max}\nright:\n{builder2.joints[j].tau_j_max}",
        )
        fixture.assertTrue(
            arrays_equal(builder1.joints[j].a_j, builder2.joints[j].a_j),
            f"Joint {j} a_j:\nleft:\n{builder1.joints[j].a_j}\nright:\n{builder2.joints[j].a_j}",
        )
        fixture.assertTrue(
            arrays_equal(builder1.joints[j].b_j, builder2.joints[j].b_j),
            f"Joint {j} b_j:\nleft:\n{builder1.joints[j].b_j}\nright:\n{builder2.joints[j].b_j}",
        )
        fixture.assertTrue(
            arrays_equal(builder1.joints[j].k_p_j, builder2.joints[j].k_p_j),
            f"Joint {j} k_p_j:\nleft:\n{builder1.joints[j].k_p_j}\nright:\n{builder2.joints[j].k_p_j}",
        )
        fixture.assertTrue(
            arrays_equal(builder1.joints[j].k_d_j, builder2.joints[j].k_d_j),
            f"Joint {j} k_d_j:\nleft:\n{builder1.joints[j].k_d_j}\nright:\n{builder2.joints[j].k_d_j}",
        )
        fixture.assertEqual(builder1.joints[j].num_coords, builder2.joints[j].num_coords)
        fixture.assertEqual(builder1.joints[j].num_dofs, builder2.joints[j].num_dofs)
        fixture.assertEqual(builder1.joints[j].num_passive_coords, builder2.joints[j].num_passive_coords)
        fixture.assertEqual(builder1.joints[j].num_passive_dofs, builder2.joints[j].num_passive_dofs)
        fixture.assertEqual(builder1.joints[j].num_actuated_coords, builder2.joints[j].num_actuated_coords)
        fixture.assertEqual(builder1.joints[j].num_actuated_dofs, builder2.joints[j].num_actuated_dofs)
        fixture.assertEqual(builder1.joints[j].num_actuated_dofs, builder2.joints[j].num_actuated_dofs)
        fixture.assertEqual(builder1.joints[j].num_cts, builder2.joints[j].num_cts)
        fixture.assertEqual(builder1.joints[j].num_dynamic_cts, builder2.joints[j].num_dynamic_cts)
        fixture.assertEqual(builder1.joints[j].num_kinematic_cts, builder2.joints[j].num_kinematic_cts)
        fixture.assertEqual(builder1.joints[j].coords_offset, builder2.joints[j].coords_offset)
        fixture.assertEqual(builder1.joints[j].dofs_offset, builder2.joints[j].dofs_offset)
        fixture.assertEqual(builder1.joints[j].passive_coords_offset, builder2.joints[j].passive_coords_offset)
        fixture.assertEqual(builder1.joints[j].passive_dofs_offset, builder2.joints[j].passive_dofs_offset)
        fixture.assertEqual(builder1.joints[j].actuated_coords_offset, builder2.joints[j].actuated_coords_offset)
        fixture.assertEqual(builder1.joints[j].actuated_dofs_offset, builder2.joints[j].actuated_dofs_offset)
        fixture.assertEqual(builder1.joints[j].cts_offset, builder2.joints[j].cts_offset)
        fixture.assertEqual(builder1.joints[j].dynamic_cts_offset, builder2.joints[j].dynamic_cts_offset)
        fixture.assertEqual(builder1.joints[j].kinematic_cts_offset, builder2.joints[j].kinematic_cts_offset)
        fixture.assertEqual(builder1.joints[j].is_binary, builder2.joints[j].is_binary)
        fixture.assertEqual(builder1.joints[j].is_passive, builder2.joints[j].is_passive)
        fixture.assertEqual(builder1.joints[j].is_actuated, builder2.joints[j].is_actuated)
        fixture.assertEqual(builder1.joints[j].is_dynamic, builder2.joints[j].is_dynamic)
        fixture.assertEqual(builder1.joints[j].is_implicit_pd, builder2.joints[j].is_implicit_pd)

    for k in range(builder1.num_geoms):
        fixture.assertEqual(builder1.geoms[k].wid, builder2.geoms[k].wid)
        fixture.assertEqual(builder1.geoms[k].gid, builder2.geoms[k].gid)
        fixture.assertEqual(builder1.geoms[k].mid, builder2.geoms[k].mid)
        fixture.assertEqual(builder1.geoms[k].body, builder2.geoms[k].body)
        fixture.assertEqual(builder1.geoms[k].shape.type, builder2.geoms[k].shape.type)
        fixture.assertEqual(builder1.geoms[k].shape.num_params, builder2.geoms[k].shape.num_params)
        fixture.assertTrue(lists_equal(builder1.geoms[k].shape.paramsvec, builder2.geoms[k].shape.paramsvec))
        if not skip_materials:
            fixture.assertEqual(builder1.geoms[k].material, builder2.geoms[k].material)
        if not skip_colliders:
            fixture.assertEqual(builder1.geoms[k].group, builder2.geoms[k].group)
            fixture.assertEqual(builder1.geoms[k].collides, builder2.geoms[k].collides)
            fixture.assertEqual(builder1.geoms[k].max_contacts, builder2.geoms[k].max_contacts)
            fixture.assertEqual(builder1.geoms[k].gap, builder2.geoms[k].gap)
            fixture.assertEqual(builder1.geoms[k].margin, builder2.geoms[k].margin)

    if not skip_materials:
        for m in range(builder1.num_materials):
            fixture.assertEqual(builder1.materials[m].wid, builder2.materials[m].wid)
            fixture.assertEqual(builder1.materials[m].mid, builder2.materials[m].mid)
            fixture.assertEqual(builder1.materials[m].density, builder2.materials[m].density)
            fixture.assertEqual(builder1.materials[m].restitution, builder2.materials[m].restitution)
            fixture.assertEqual(builder1.materials[m].static_friction, builder2.materials[m].static_friction)
            fixture.assertEqual(builder1.materials[m].dynamic_friction, builder2.materials[m].dynamic_friction)
