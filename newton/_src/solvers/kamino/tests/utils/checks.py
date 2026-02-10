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

from ...core.builder import ModelBuilderKamino
from ...core.control import ControlKamino
from ...core.model import ModelKamino
from ...core.state import StateKamino

###
# Module interface
###

__all__ = [
    "arrays_equal",
    "assert_builders_equal",
    "assert_control_equal",
    "assert_model_bodies_equal",
    "assert_model_equal",
    "assert_model_geoms_equal",
    "assert_model_info_equal",
    "assert_model_joints_equal",
    "assert_model_material_pairs_equal",
    "assert_model_materials_equal",
    "assert_model_size_equal",
    "assert_state_equal",
    "lists_equal",
    "matrices_equal",
    "vectors_equal",
]

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
    builder1: ModelBuilderKamino,
    builder2: ModelBuilderKamino,
    skip_colliders: bool = False,
):
    """
    Compares two ModelBuilderKamino instances for equality.
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
        fixture.assertEqual(builder1.joints[j].cts_offset, builder2.joints[j].cts_offset)
        fixture.assertEqual(builder1.joints[j].dofs_offset, builder2.joints[j].dofs_offset)
        fixture.assertEqual(builder1.joints[j].bid_B, builder2.joints[j].bid_B)
        fixture.assertEqual(builder1.joints[j].bid_F, builder2.joints[j].bid_F)
        fixture.assertTrue(
            vectors_equal(builder1.joints[j].B_r_Bj, builder2.joints[j].B_r_Bj),
            f"Joint {j} B_r_Bj mismatch:\nleft:\n{builder1.joints[j].B_r_Bj}\nright:\n{builder2.joints[j].B_r_Bj}",
        )
        fixture.assertTrue(
            vectors_equal(builder1.joints[j].F_r_Fj, builder2.joints[j].F_r_Fj),
            f"Joint {j} F_r_Fj mismatch:\nleft:\n{builder1.joints[j].F_r_Fj}\nright:\n{builder2.joints[j].F_r_Fj}",
        )
        fixture.assertTrue(
            matrices_equal(builder1.joints[j].X_j, builder2.joints[j].X_j),
            f"Joint {j} X_j mismatch:\nleft:\n{builder1.joints[j].X_j}\nright:\n{builder2.joints[j].X_j}",
        )
        fixture.assertTrue(
            arrays_equal(builder1.joints[j].q_j_min, builder2.joints[j].q_j_min),
            f"Joint {j} q_j_min mismatch:\nleft:\n{builder1.joints[j].q_j_min}\nright:\n{builder2.joints[j].q_j_min}",
        )
        fixture.assertTrue(
            arrays_equal(builder1.joints[j].q_j_max, builder2.joints[j].q_j_max),
            f"Joint {j} q_j_max mismatch:\nleft:\n{builder1.joints[j].q_j_max}\nright:\n{builder2.joints[j].q_j_max}",
        )
        fixture.assertTrue(
            arrays_equal(builder1.joints[j].dq_j_max, builder2.joints[j].dq_j_max),
            f"Joint {j} dq_j_max mismatch:\nleft:\n{builder1.joints[j].dq_j_max}\n"
            f"right:\n{builder2.joints[j].dq_j_max}",
        )
        fixture.assertTrue(
            arrays_equal(builder1.joints[j].tau_j_max, builder2.joints[j].tau_j_max),
            f"Joint {j} tau_j_max mismatch:\nleft:\n{builder1.joints[j].tau_j_max}\n"
            f"right:\n{builder2.joints[j].tau_j_max}",
        )

    for k in range(builder1.num_geoms):
        fixture.assertEqual(builder1.geoms[k].wid, builder2.geoms[k].wid)
        fixture.assertEqual(builder1.geoms[k].gid, builder2.geoms[k].gid)
        fixture.assertEqual(builder1.geoms[k].lid, builder2.geoms[k].lid)
        fixture.assertEqual(builder1.geoms[k].bid, builder2.geoms[k].bid)
        fixture.assertEqual(builder1.geoms[k].shape.type, builder2.geoms[k].shape.type)
        fixture.assertEqual(builder1.geoms[k].shape.num_params, builder2.geoms[k].shape.num_params)
        fixture.assertTrue(lists_equal(builder1.geoms[k].shape.paramsvec, builder2.geoms[k].shape.paramsvec))
        fixture.assertEqual(builder1.geoms[k].mid, builder2.geoms[k].mid)
        fixture.assertEqual(builder1.geoms[k].max_contacts, builder2.geoms[k].max_contacts)
        if not skip_colliders:
            fixture.assertEqual(builder1.geoms[k].group, builder2.geoms[k].group)
            fixture.assertEqual(builder1.geoms[k].collides, builder2.geoms[k].collides)

    for m in range(builder1.num_materials):
        fixture.assertEqual(builder1.materials[m].wid, builder2.materials[m].wid)
        fixture.assertEqual(builder1.materials[m].mid, builder2.materials[m].mid)
        fixture.assertEqual(builder1.materials[m].density, builder2.materials[m].density)
        fixture.assertEqual(builder1.materials[m].restitution, builder2.materials[m].restitution)
        fixture.assertEqual(builder1.materials[m].static_friction, builder2.materials[m].static_friction)
        fixture.assertEqual(builder1.materials[m].dynamic_friction, builder2.materials[m].dynamic_friction)


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
        rtol=1e-7,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        actual=model0.bodies.inv_i_I_i.numpy(),
        desired=model1.bodies.inv_i_I_i.numpy(),
        err_msg="model.bodies.inv_i_I_i are not equal.",
        atol=1e-5,
        rtol=1e-5,
    )
    # np.testing.assert_allclose(
    #     actual=model0.bodies.q_i_0.numpy(),
    #     desired=model1.bodies.q_i_0.numpy(),
    #     err_msg="model.bodies.q_i_0 are not equal.",
    # )
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


def assert_model_geoms_equal(
    test: unittest.TestCase,
    model0: ModelKamino,
    model1: ModelKamino,
    check_ptr: bool = True,
    check_group_and_collides: bool = True,
) -> None:
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
    # test.assertEqual(
    #     first=model0.geoms.num_collidable_geom_pairs,
    #     second=model1.geoms.num_collidable_geom_pairs,
    #     msg="model.geoms.num_collidable_geom_pairs are not equal.",
    # )
    # test.assertLessEqual(
    #     a=model0.geoms.model_max_contacts,
    #     b=model1.geoms.model_max_contacts,
    #     msg="model.geoms.model_max_contacts are not equal.",
    # )
    # for w in range(model0.size.num_worlds):
    #     test.assertLessEqual(
    #         a=model0.geoms.world_max_contacts[w],
    #         b=model1.geoms.world_max_contacts[w],
    #         msg=f"model.geoms.world_max_contacts[{w}] are not less-than-or-equal.",
    #     )
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
    if check_ptr:
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
    if check_group_and_collides:
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
    # np.testing.assert_allclose(
    #     actual=model0.geoms.margin.numpy(),
    #     desired=model1.geoms.margin.numpy(),
    #     err_msg="model.geoms.margin are not equal.",
    # )
    # np.testing.assert_allclose(
    #     actual=model0.geoms.collidable_pairs.numpy(),
    #     desired=model1.geoms.collidable_pairs.numpy(),
    #     err_msg="model.geoms.collidable_pairs are not equal.",
    # )


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


def assert_model_equal(
    test: unittest.TestCase,
    model0: ModelKamino,
    model1: ModelKamino,
    check_geom_source_ptr: bool = True,
    check_geom_group_and_collides: bool = True,
) -> None:
    assert_model_size_equal(test, model0, model1)
    assert_model_info_equal(test, model0, model1)
    assert_model_bodies_equal(test, model0, model1)
    assert_model_joints_equal(test, model0, model1)
    assert_model_geoms_equal(
        test, model0, model1, check_ptr=check_geom_source_ptr, check_group_and_collides=check_geom_group_and_collides
    )
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
