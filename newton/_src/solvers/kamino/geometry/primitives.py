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

"""Narrow-phase collision detection operations on geometric primitives"""

from __future__ import annotations

from typing import Any

import warp as wp

from ....geometry.collision_primitive import (
    collide_box_box,
    collide_capsule_box,
    collide_capsule_capsule,
    collide_plane_box,
    collide_plane_capsule,
    collide_plane_cylinder,
    collide_plane_ellipsoid,
    collide_plane_sphere,
    collide_sphere_box,
    collide_sphere_capsule,
    collide_sphere_cylinder,
    collide_sphere_sphere,
)
from ..core.model import Model, ModelData
from ..core.shapes import ShapeType
from ..core.types import (
    float32,
    int32,
    mat33f,
    transformf,
    vec2f,
    vec2i,
    vec3f,
    vec4f,
)
from ..geometry.collisions import Collisions
from ..geometry.contacts import Contacts
from ..geometry.math import make_contact_frame_znorm

###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Constants
###

DEFAULT_MARGIN = wp.constant(float32(1e-5))


###
# Geometry helper Types
###


@wp.struct
class Box:
    bid: int32
    gid: int32
    pos: vec3f
    rot: mat33f
    size: vec3f


@wp.struct
class Sphere:
    bid: int32
    gid: int32
    pos: vec3f
    rot: mat33f
    radius: float32


@wp.struct
class Capsule:
    bid: int32
    gid: int32
    pos: vec3f
    rot: mat33f
    axis: vec3f
    radius: float32
    half_length: float32


@wp.struct
class Cylinder:
    bid: int32
    gid: int32
    pos: vec3f
    rot: mat33f
    axis: vec3f
    radius: float32
    half_height: float32


@wp.struct
class Plane:
    bid: int32
    gid: int32
    normal: vec3f
    pos: vec3f


@wp.struct
class Ellipsoid:
    bid: int32
    gid: int32
    pos: vec3f
    rot: mat33f
    size: vec3f


@wp.func
def make_box(pose: transformf, params: vec4f, gid: int32, bid: int32) -> Box:
    box = Box()
    box.bid = bid
    box.gid = gid
    box.pos = wp.transform_get_translation(pose)
    box.rot = wp.quat_to_matrix(wp.transform_get_rotation(pose))
    box.size = vec3f(0.5 * params[0], 0.5 * params[1], 0.5 * params[2])
    return box


@wp.func
def make_sphere(pose: transformf, params: vec4f, gid: int32, bid: int32) -> Sphere:
    sphere = Sphere()
    sphere.bid = bid
    sphere.gid = gid
    sphere.pos = wp.transform_get_translation(pose)
    sphere.rot = wp.quat_to_matrix(wp.transform_get_rotation(pose))
    sphere.radius = params[0]
    return sphere


@wp.func
def make_capsule(pose: transformf, params: vec4f, gid: int32, bid: int32) -> Capsule:
    capsule = Capsule()
    capsule.bid = bid
    capsule.gid = gid
    capsule.pos = wp.transform_get_translation(pose)
    rot_mat = wp.quat_to_matrix(wp.transform_get_rotation(pose))
    capsule.rot = rot_mat
    # Capsule axis is along the local Z-axis
    capsule.axis = vec3f(rot_mat[0, 2], rot_mat[1, 2], rot_mat[2, 2])
    capsule.radius = params[0]
    capsule.half_length = params[1] * 0.5
    return capsule


@wp.func
def make_cylinder(pose: transformf, params: vec4f, gid: int32, bid: int32) -> Cylinder:
    cylinder = Cylinder()
    cylinder.bid = bid
    cylinder.gid = gid
    cylinder.pos = wp.transform_get_translation(pose)
    rot_mat = wp.quat_to_matrix(wp.transform_get_rotation(pose))
    cylinder.rot = rot_mat
    # Cylinder axis is along the local Z-axis
    cylinder.axis = vec3f(rot_mat[0, 2], rot_mat[1, 2], rot_mat[2, 2])
    cylinder.radius = params[0]
    cylinder.half_height = params[1] * 0.5
    return cylinder


@wp.func
def make_plane(pose: transformf, params: vec4f, gid: int32, bid: int32) -> Plane:
    plane = Plane()
    plane.bid = bid
    plane.gid = gid
    # Plane normal is stored in params[0:3]
    plane.normal = vec3f(params[0], params[1], params[2])
    # Plane position is the transform translation
    plane.pos = wp.transform_get_translation(pose)
    return plane


@wp.func
def make_ellipsoid(pose: transformf, params: vec4f, gid: int32, bid: int32) -> Ellipsoid:
    ellipsoid = Ellipsoid()
    ellipsoid.bid = bid
    ellipsoid.gid = gid
    ellipsoid.pos = wp.transform_get_translation(pose)
    ellipsoid.rot = wp.quat_to_matrix(wp.transform_get_rotation(pose))
    # Ellipsoid size (radii) stored in params[0:3]
    ellipsoid.size = vec3f(params[0], params[1], params[2])
    return ellipsoid


###
# Utility Functions
###


@wp.func
def add_single_contact(
    # Inputs:
    model_max_contacts_in: int32,
    world_max_contacts_in: int32,
    wid_in: int32,
    bid_1_in: int32,
    bid_2_in: int32,
    margin_in: float32,
    distance_in: float32,
    position_in: vec3f,
    normal_in: vec3f,
    friction_in: float32,
    restitution_in: float32,
    # Outputs:
    contacts_model_num_out: wp.array(dtype=int32),
    contacts_world_num_out: wp.array(dtype=int32),
    contact_wid_out: wp.array(dtype=int32),
    contact_cid_out: wp.array(dtype=int32),
    contact_body_A_out: wp.array(dtype=vec4f),
    contact_body_B_out: wp.array(dtype=vec4f),
    contact_gapfunc_out: wp.array(dtype=vec4f),
    contact_frame_out: wp.array(dtype=mat33f),
    contact_material_out: wp.array(dtype=vec2f),
):
    # Skip if the contact distance exceeds the specified margin
    if (distance_in - margin_in) > 0.0:
        return

    # TODO: This will cause problems if the number of contacts exceeds the maximum as we are
    # incrementing the contact counters and do not decrement if we've exceeded the maximum
    mcid = wp.atomic_add(contacts_model_num_out, 0, 1)
    wcid = wp.atomic_add(contacts_world_num_out, wid_in, 1)
    if mcid < model_max_contacts_in and wcid < world_max_contacts_in:
        # Perform body assignment
        if bid_2_in < 0:
            bid_A_in = bid_2_in
            bid_B_in = bid_1_in
            normal_in = -normal_in
        else:
            bid_A_in = bid_1_in
            bid_B_in = bid_2_in

        # Compute absolute penetration distance
        distance_abs = wp.abs(distance_in)

        # The colliders compute the contact point in the middle, and thus to get the
        # per-geom contact points we need to offset by the penetration depth along the normal
        position_A_in = position_in + 0.5 * distance_abs * normal_in
        position_B_in = position_in - 0.5 * distance_abs * normal_in

        # Store the active contact output data
        contact_wid_out[mcid] = wid_in
        contact_cid_out[mcid] = wcid
        contact_body_A_out[mcid] = vec4f(position_A_in[0], position_A_in[1], position_A_in[2], float32(bid_A_in))
        contact_body_B_out[mcid] = vec4f(position_B_in[0], position_B_in[1], position_B_in[2], float32(bid_B_in))
        contact_gapfunc_out[mcid] = vec4f(normal_in[0], normal_in[1], normal_in[2], distance_in)
        contact_frame_out[mcid] = make_contact_frame_znorm(normal_in)
        contact_material_out[mcid][0] = friction_in
        contact_material_out[mcid][1] = restitution_in


def make_add_multiple_contacts(MAX_CONTACTS: int, SHARED_NORMAL: bool):
    # Define the function to add multiple contacts
    @wp.func
    def add_multiple_contacts(
        # Inputs:
        model_max_contacts_in: int32,
        world_max_contacts_in: int32,
        wid_in: int32,
        bid_1_in: int32,
        bid_2_in: int32,
        margin_in: float32,
        distances_in: wp.types.vector(MAX_CONTACTS, wp.float32),
        positions_in: wp.types.matrix((MAX_CONTACTS, 3), wp.float32),
        normals_in: Any,
        friction_in: float32,
        restitution_in: float32,
        # Outputs:
        contacts_model_num_out: wp.array(dtype=int32),
        contacts_world_num_out: wp.array(dtype=int32),
        contact_wid_out: wp.array(dtype=int32),
        contact_cid_out: wp.array(dtype=int32),
        contact_body_A_out: wp.array(dtype=vec4f),
        contact_body_B_out: wp.array(dtype=vec4f),
        contact_gapfunc_out: wp.array(dtype=vec4f),
        contact_frame_out: wp.array(dtype=mat33f),
        contact_material_out: wp.array(dtype=vec2f),
    ):
        # Count valid contacts (those with finite distance)
        num_contacts = wp.int32(0)
        for i in range(MAX_CONTACTS):
            if distances_in[i] != wp.inf and distances_in[i] <= margin_in:
                num_contacts += 1

        # Skip operation if no contacts were detected
        if num_contacts == 0:
            return

        # Assign body indices
        if bid_2_in < 0:
            bid_A = bid_2_in
            bid_B = bid_1_in
        else:
            bid_A = bid_1_in
            bid_B = bid_2_in

        # Increment the active contact counter
        mcio = wp.atomic_add(contacts_model_num_out, 0, num_contacts)
        wcio = wp.atomic_add(contacts_world_num_out, wid_in, num_contacts)

        # Retrieve the maximum number of contacts that can be stored for this geom pair
        max_num_contacts = wp.min(wp.min(model_max_contacts_in - mcio, world_max_contacts_in - wcio), num_contacts)

        # Add generated contacts data to the output arrays
        contact_idx = wp.int32(0)
        for i in range(8):
            # Break if we've reached the maximum number of contacts for this geom pair
            if contact_idx >= max_num_contacts:
                break

            # If contact is valid, store it
            if distances_in[i] != wp.inf and distances_in[i] <= margin_in:
                # Compute the global contact index
                mcid = mcio + contact_idx

                # Get contact data
                dist = distances_in[i]
                pos = vec3f(positions_in[i, 0], positions_in[i, 1], positions_in[i, 2])
                if wp.static(SHARED_NORMAL):
                    normal = normals_in
                else:
                    normal = vec3f(normals_in[i, 0], normals_in[i, 1], normals_in[i, 2])
                dist_abs = wp.abs(dist)

                # Adjust normal direction based on body assignment
                if bid_2_in < 0:
                    normal = -normal

                # Generate a contact frame
                frame = make_contact_frame_znorm(normal)

                # This collider computes the contact point in the middle, and thus to get the
                # per-geom contact we need to offset the contact point by the penetration depth
                pos_A = pos + 0.5 * normal * dist_abs
                pos_B = pos - 0.5 * normal * dist_abs

                # Store contact data
                contact_wid_out[mcid] = wid_in
                contact_cid_out[mcid] = wcio + contact_idx
                contact_body_A_out[mcid] = vec4f(pos_A[0], pos_A[1], pos_A[2], float32(bid_A))
                contact_body_B_out[mcid] = vec4f(pos_B[0], pos_B[1], pos_B[2], float32(bid_B))
                contact_gapfunc_out[mcid] = vec4f(normal[0], normal[1], normal[2], dist)
                contact_frame_out[mcid] = frame
                contact_material_out[mcid] = vec2f(friction_in, restitution_in)

                # Increment active contact index
                contact_idx += 1

    # Return the generated function
    return add_multiple_contacts


###
# Primitive Colliders
###


@wp.func
def sphere_sphere(
    # Inputs:
    model_max_contacts: int32,
    world_max_contacts: int32,
    sphere1: Sphere,
    sphere2: Sphere,
    wid: int32,
    margin: float32,
    friction_in: float32,
    restitution_in: float32,
    # Contacts out:
    contacts_model_num_out: wp.array(dtype=int32),
    contacts_world_num_out: wp.array(dtype=int32),
    contact_wid_out: wp.array(dtype=int32),
    contact_cid_out: wp.array(dtype=int32),
    contact_body_A_out: wp.array(dtype=vec4f),
    contact_body_B_out: wp.array(dtype=vec4f),
    contact_gapfunc_out: wp.array(dtype=vec4f),
    contact_frame_out: wp.array(dtype=mat33f),
    contact_material_out: wp.array(dtype=vec2f),
):
    # Run the respective collider function to detect sphere-sphere contacts
    distance, position, normal = collide_sphere_sphere(sphere1.pos, sphere1.radius, sphere2.pos, sphere2.radius)

    # Add the active contact to the global contacts arrays
    add_single_contact(
        model_max_contacts,
        world_max_contacts,
        wid,
        sphere1.bid,
        sphere2.bid,
        margin,
        distance,
        position,
        normal,
        friction_in,
        restitution_in,
        contacts_model_num_out,
        contacts_world_num_out,
        contact_wid_out,
        contact_cid_out,
        contact_body_A_out,
        contact_body_B_out,
        contact_gapfunc_out,
        contact_frame_out,
        contact_material_out,
    )


@wp.func
def sphere_cylinder(
    # Inputs:
    model_max_contacts_in: int32,
    world_max_contacts_in: int32,
    sphere_in: Sphere,
    cylinder_in: Cylinder,
    wid_in: int32,
    margin: float32,
    friction_in: float32,
    restitution_in: float32,
    # Contacts out:
    contacts_model_num_out: wp.array(dtype=int32),
    contacts_world_num_out: wp.array(dtype=int32),
    contact_wid_out: wp.array(dtype=int32),
    contact_cid_out: wp.array(dtype=int32),
    contact_body_A_out: wp.array(dtype=vec4f),
    contact_body_B_out: wp.array(dtype=vec4f),
    contact_gapfunc_out: wp.array(dtype=vec4f),
    contact_frame_out: wp.array(dtype=mat33f),
    contact_material_out: wp.array(dtype=vec2f),
):
    # Use the tested collision calculation from collision_primitive.py
    dist, pos, normal = collide_sphere_cylinder(
        sphere_in.pos,
        sphere_in.radius,
        cylinder_in.pos,
        cylinder_in.axis,
        cylinder_in.radius,
        cylinder_in.half_height,
    )

    # Add the active contact to the global contacts arrays
    add_single_contact(
        model_max_contacts_in,
        world_max_contacts_in,
        wid_in,
        sphere_in.bid,
        cylinder_in.bid,
        margin,
        dist,
        pos,
        normal,
        friction_in,
        restitution_in,
        contacts_model_num_out,
        contacts_world_num_out,
        contact_wid_out,
        contact_cid_out,
        contact_body_A_out,
        contact_body_B_out,
        contact_gapfunc_out,
        contact_frame_out,
        contact_material_out,
    )


@wp.func
def sphere_cone():
    pass


@wp.func
def sphere_capsule(
    # Inputs:
    model_max_contacts_in: int32,
    world_max_contacts_in: int32,
    sphere_in: Sphere,
    capsule_in: Capsule,
    wid_in: int32,
    margin: float32,
    friction_in: float32,
    restitution_in: float32,
    # Contacts out:
    contacts_model_num_out: wp.array(dtype=int32),
    contacts_world_num_out: wp.array(dtype=int32),
    contact_wid_out: wp.array(dtype=int32),
    contact_cid_out: wp.array(dtype=int32),
    contact_body_A_out: wp.array(dtype=vec4f),
    contact_body_B_out: wp.array(dtype=vec4f),
    contact_gapfunc_out: wp.array(dtype=vec4f),
    contact_frame_out: wp.array(dtype=mat33f),
    contact_material_out: wp.array(dtype=vec2f),
):
    # Use the tested collision calculation from collision_primitive.py
    dist, pos, normal = collide_sphere_capsule(
        sphere_in.pos,
        sphere_in.radius,
        capsule_in.pos,
        capsule_in.axis,
        capsule_in.radius,
        capsule_in.half_length,
    )

    # Add the active contact to the global contacts arrays
    add_single_contact(
        model_max_contacts_in,
        world_max_contacts_in,
        wid_in,
        sphere_in.bid,
        capsule_in.bid,
        margin,
        dist,
        pos,
        normal,
        friction_in,
        restitution_in,
        contacts_model_num_out,
        contacts_world_num_out,
        contact_wid_out,
        contact_cid_out,
        contact_body_A_out,
        contact_body_B_out,
        contact_gapfunc_out,
        contact_frame_out,
        contact_material_out,
    )


@wp.func
def sphere_box(
    # Inputs:
    model_max_contacts_in: int32,
    world_max_contacts_in: int32,
    sphere_in: Sphere,
    box_in: Box,
    wid_in: int32,
    margin: float32,
    friction_in: float32,
    restitution_in: float32,
    # Contacts out:
    contacts_model_num_out: wp.array(dtype=int32),
    contacts_world_num_out: wp.array(dtype=int32),
    contact_wid_out: wp.array(dtype=int32),
    contact_cid_out: wp.array(dtype=int32),
    contact_body_A_out: wp.array(dtype=vec4f),
    contact_body_B_out: wp.array(dtype=vec4f),
    contact_gapfunc_out: wp.array(dtype=vec4f),
    contact_frame_out: wp.array(dtype=mat33f),
    contact_material_out: wp.array(dtype=vec2f),
):
    # Use the tested collision calculation from collision_primitive.py
    dist, pos, normal = collide_sphere_box(sphere_in.pos, sphere_in.radius, box_in.pos, box_in.rot, box_in.size)

    # Add the active contact to the global contacts arrays
    add_single_contact(
        model_max_contacts_in,
        world_max_contacts_in,
        wid_in,
        sphere_in.bid,
        box_in.bid,
        margin,
        dist,
        pos,
        normal,
        friction_in,
        restitution_in,
        contacts_model_num_out,
        contacts_world_num_out,
        contact_wid_out,
        contact_cid_out,
        contact_body_A_out,
        contact_body_B_out,
        contact_gapfunc_out,
        contact_frame_out,
        contact_material_out,
    )


@wp.func
def sphere_ellipsoid():
    pass


@wp.func
def cylinder_cylinder():
    pass


@wp.func
def cylinder_cone():
    pass


@wp.func
def cylinder_capsule():
    pass


@wp.func
def cylinder_box():
    pass


@wp.func
def cylinder_ellipsoid():
    pass


@wp.func
def cone_cone():
    pass


@wp.func
def cone_capsule():
    pass


@wp.func
def cone_box():
    pass


@wp.func
def cone_ellipsoid():
    pass


@wp.func
def capsule_capsule(
    # Inputs:
    model_max_contacts_in: int32,
    world_max_contacts_in: int32,
    capsule1_in: Capsule,
    capsule2_in: Capsule,
    wid_in: int32,
    margin: float32,
    friction_in: float32,
    restitution_in: float32,
    # Contacts out:
    contacts_model_num_out: wp.array(dtype=int32),
    contacts_world_num_out: wp.array(dtype=int32),
    contact_wid_out: wp.array(dtype=int32),
    contact_cid_out: wp.array(dtype=int32),
    contact_body_A_out: wp.array(dtype=vec4f),
    contact_body_B_out: wp.array(dtype=vec4f),
    contact_gapfunc_out: wp.array(dtype=vec4f),
    contact_frame_out: wp.array(dtype=mat33f),
    contact_material_out: wp.array(dtype=vec2f),
):
    # Use the tested collision calculation from collision_primitive.py
    dist, pos, normal = collide_capsule_capsule(
        capsule1_in.pos,
        capsule1_in.axis,
        capsule1_in.radius,
        capsule1_in.half_length,
        capsule2_in.pos,
        capsule2_in.axis,
        capsule2_in.radius,
        capsule2_in.half_length,
    )

    # Add the active contact to the global contacts arrays
    add_single_contact(
        model_max_contacts_in,
        world_max_contacts_in,
        wid_in,
        capsule1_in.bid,
        capsule2_in.bid,
        margin,
        dist,
        pos,
        normal,
        friction_in,
        restitution_in,
        contacts_model_num_out,
        contacts_world_num_out,
        contact_wid_out,
        contact_cid_out,
        contact_body_A_out,
        contact_body_B_out,
        contact_gapfunc_out,
        contact_frame_out,
        contact_material_out,
    )


@wp.func
def capsule_box(
    # Inputs:
    model_max_contacts_in: int32,
    world_max_contacts_in: int32,
    capsule_in: Capsule,
    box_in: Box,
    wid_in: int32,
    margin: float32,
    friction_in: float32,
    restitution_in: float32,
    # Contacts out:
    contacts_model_num_out: wp.array(dtype=int32),
    contacts_world_num_out: wp.array(dtype=int32),
    contact_wid_out: wp.array(dtype=int32),
    contact_cid_out: wp.array(dtype=int32),
    contact_body_A_out: wp.array(dtype=vec4f),
    contact_body_B_out: wp.array(dtype=vec4f),
    contact_gapfunc_out: wp.array(dtype=vec4f),
    contact_frame_out: wp.array(dtype=mat33f),
    contact_material_out: wp.array(dtype=vec2f),
):
    # Use the tested collision calculation from collision_primitive.py
    distances, positions, normals = collide_capsule_box(
        capsule_in.pos,
        capsule_in.axis,
        capsule_in.radius,
        capsule_in.half_length,
        box_in.pos,
        box_in.rot,
        box_in.size,
    )

    # Add the active contacts to the global contacts arrays (up to 2 contacts with per-contact normals)
    wp.static(make_add_multiple_contacts(2, False))(
        model_max_contacts_in,
        world_max_contacts_in,
        wid_in,
        capsule_in.bid,
        box_in.bid,
        margin,
        distances,
        positions,
        normals,
        friction_in,
        restitution_in,
        contacts_model_num_out,
        contacts_world_num_out,
        contact_wid_out,
        contact_cid_out,
        contact_body_A_out,
        contact_body_B_out,
        contact_gapfunc_out,
        contact_frame_out,
        contact_material_out,
    )


@wp.func
def capsule_ellipsoid():
    pass


@wp.func
def box_box(
    # Inputs:
    model_max_contacts_in: int32,
    world_max_contacts_in: int32,
    box1_in: Box,
    box2_in: Box,
    wid_in: int32,
    margin_in: float32,
    friction_in: float32,
    restitution_in: float32,
    # Outputs:
    contacts_model_num_out: wp.array(dtype=int32),
    contacts_world_num_out: wp.array(dtype=int32),
    contact_wid_out: wp.array(dtype=int32),
    contact_cid_out: wp.array(dtype=int32),
    contact_body_A_out: wp.array(dtype=vec4f),
    contact_body_B_out: wp.array(dtype=vec4f),
    contact_gapfunc_out: wp.array(dtype=vec4f),
    contact_frame_out: wp.array(dtype=mat33f),
    contact_material_out: wp.array(dtype=vec2f),
):
    # Use the tested collision calculation from collision_primitive.py
    distances, positions, normals = collide_box_box(
        box1_in.pos, box1_in.rot, box1_in.size, box2_in.pos, box2_in.rot, box2_in.size, margin_in
    )

    # Add the active contacts to the global contacts arrays (up to 8 contacts with per-contact normals)
    wp.static(make_add_multiple_contacts(8, False))(
        # Inputs:
        model_max_contacts_in,
        world_max_contacts_in,
        wid_in,
        box1_in.bid,
        box2_in.bid,
        margin_in,
        distances,
        positions,
        normals,
        friction_in,
        restitution_in,
        # Outputs:
        contacts_model_num_out,
        contacts_world_num_out,
        contact_wid_out,
        contact_cid_out,
        contact_body_A_out,
        contact_body_B_out,
        contact_gapfunc_out,
        contact_frame_out,
        contact_material_out,
    )


@wp.func
def box_ellipsoid():
    pass


@wp.func
def ellipsoid_ellipsoid():
    pass


@wp.func
def plane_sphere(
    # Inputs:
    model_max_contacts_in: int32,
    world_max_contacts_in: int32,
    plane_in: Plane,
    sphere_in: Sphere,
    wid_in: int32,
    margin: float32,
    friction_in: float32,
    restitution_in: float32,
    # Contacts out:
    contacts_model_num_out: wp.array(dtype=int32),
    contacts_world_num_out: wp.array(dtype=int32),
    contact_wid_out: wp.array(dtype=int32),
    contact_cid_out: wp.array(dtype=int32),
    contact_body_A_out: wp.array(dtype=vec4f),
    contact_body_B_out: wp.array(dtype=vec4f),
    contact_gapfunc_out: wp.array(dtype=vec4f),
    contact_frame_out: wp.array(dtype=mat33f),
    contact_material_out: wp.array(dtype=vec2f),
):
    # Use the tested collision calculation from collision_primitive.py
    # Note: collide_plane_sphere returns (distance, position) without normal
    dist, pos = collide_plane_sphere(plane_in.normal, plane_in.pos, sphere_in.pos, sphere_in.radius)

    # Use plane normal as contact normal
    normal = plane_in.normal

    # Add the active contact to the global contacts arrays
    add_single_contact(
        model_max_contacts_in,
        world_max_contacts_in,
        wid_in,
        plane_in.bid,
        sphere_in.bid,
        margin,
        dist,
        pos,
        normal,
        friction_in,
        restitution_in,
        contacts_model_num_out,
        contacts_world_num_out,
        contact_wid_out,
        contact_cid_out,
        contact_body_A_out,
        contact_body_B_out,
        contact_gapfunc_out,
        contact_frame_out,
        contact_material_out,
    )


@wp.func
def plane_box(
    # Inputs:
    model_max_contacts_in: int32,
    world_max_contacts_in: int32,
    plane_in: Plane,
    box_in: Box,
    wid_in: int32,
    margin: float32,
    friction_in: float32,
    restitution_in: float32,
    # Contacts out:
    contacts_model_num_out: wp.array(dtype=int32),
    contacts_world_num_out: wp.array(dtype=int32),
    contact_wid_out: wp.array(dtype=int32),
    contact_cid_out: wp.array(dtype=int32),
    contact_body_A_out: wp.array(dtype=vec4f),
    contact_body_B_out: wp.array(dtype=vec4f),
    contact_gapfunc_out: wp.array(dtype=vec4f),
    contact_frame_out: wp.array(dtype=mat33f),
    contact_material_out: wp.array(dtype=vec2f),
):
    # Use the tested collision calculation from collision_primitive.py
    distances, positions, normal = collide_plane_box(plane_in.normal, plane_in.pos, box_in.pos, box_in.rot, box_in.size)

    # Add the active contacts to the global contacts arrays (up to 4 contacts with shared normal)
    wp.static(make_add_multiple_contacts(4, True))(
        model_max_contacts_in,
        world_max_contacts_in,
        wid_in,
        plane_in.bid,
        box_in.bid,
        margin,
        distances,
        positions,
        normal,
        friction_in,
        restitution_in,
        contacts_model_num_out,
        contacts_world_num_out,
        contact_wid_out,
        contact_cid_out,
        contact_body_A_out,
        contact_body_B_out,
        contact_gapfunc_out,
        contact_frame_out,
        contact_material_out,
    )


@wp.func
def plane_ellipsoid(
    # Inputs:
    model_max_contacts_in: int32,
    world_max_contacts_in: int32,
    plane_in: Plane,
    ellipsoid_in: Ellipsoid,
    wid_in: int32,
    margin: float32,
    friction_in: float32,
    restitution_in: float32,
    # Contacts out:
    contacts_model_num_out: wp.array(dtype=int32),
    contacts_world_num_out: wp.array(dtype=int32),
    contact_wid_out: wp.array(dtype=int32),
    contact_cid_out: wp.array(dtype=int32),
    contact_body_A_out: wp.array(dtype=vec4f),
    contact_body_B_out: wp.array(dtype=vec4f),
    contact_gapfunc_out: wp.array(dtype=vec4f),
    contact_frame_out: wp.array(dtype=mat33f),
    contact_material_out: wp.array(dtype=vec2f),
):
    # Use the tested collision calculation from collision_primitive.py
    dist, pos, normal = collide_plane_ellipsoid(
        plane_in.normal, plane_in.pos, ellipsoid_in.pos, ellipsoid_in.rot, ellipsoid_in.size
    )

    # Add the active contact to the global contacts arrays
    add_single_contact(
        model_max_contacts_in,
        world_max_contacts_in,
        wid_in,
        plane_in.bid,
        ellipsoid_in.bid,
        margin,
        dist,
        pos,
        normal,
        friction_in,
        restitution_in,
        contacts_model_num_out,
        contacts_world_num_out,
        contact_wid_out,
        contact_cid_out,
        contact_body_A_out,
        contact_body_B_out,
        contact_gapfunc_out,
        contact_frame_out,
        contact_material_out,
    )


@wp.func
def plane_capsule(
    # Inputs:
    model_max_contacts_in: int32,
    world_max_contacts_in: int32,
    plane_in: Plane,
    capsule_in: Capsule,
    wid_in: int32,
    margin: float32,
    friction_in: float32,
    restitution_in: float32,
    # Contacts out:
    contacts_model_num_out: wp.array(dtype=int32),
    contacts_world_num_out: wp.array(dtype=int32),
    contact_wid_out: wp.array(dtype=int32),
    contact_cid_out: wp.array(dtype=int32),
    contact_body_A_out: wp.array(dtype=vec4f),
    contact_body_B_out: wp.array(dtype=vec4f),
    contact_gapfunc_out: wp.array(dtype=vec4f),
    contact_frame_out: wp.array(dtype=mat33f),
    contact_material_out: wp.array(dtype=vec2f),
):
    # Use the tested collision calculation from collision_primitive.py
    # Note: collide_plane_capsule returns a contact frame, not individual normals
    distances, positions, frame = collide_plane_capsule(
        plane_in.normal, plane_in.pos, capsule_in.pos, capsule_in.axis, capsule_in.radius, capsule_in.half_length
    )

    # Manually add contacts since plane_capsule returns a contact frame instead of normals
    # Count valid contacts
    num_contacts = int32(0)
    for i in range(2):
        if distances[i] != wp.inf and distances[i] <= margin:
            num_contacts += 1

    if num_contacts == 0:
        return

    # Assign body indices
    if capsule_in.bid < 0:
        bid_A = capsule_in.bid
        bid_B = plane_in.bid
    else:
        bid_A = plane_in.bid
        bid_B = capsule_in.bid

    # Increment the active contact counter
    mcio = wp.atomic_add(contacts_model_num_out, 0, num_contacts)
    wcio = wp.atomic_add(contacts_world_num_out, wid_in, num_contacts)

    # Retrieve the maximum number of contacts that can be stored
    max_num_contacts = wp.min(wp.min(model_max_contacts_in - mcio, world_max_contacts_in - wcio), num_contacts)

    # Extract normal from the contact frame (first column)
    normal = vec3f(frame[0, 0], frame[1, 0], frame[2, 0])

    # Add generated contacts data to the output arrays
    contact_idx = int32(0)
    for i in range(2):
        # Break if we've reached the maximum number of contacts
        if contact_idx >= max_num_contacts:
            break

        # If contact is valid, store it
        if distances[i] != wp.inf and distances[i] <= margin:
            # Compute the global contact index
            mcid = mcio + contact_idx

            # Get contact data
            dist = distances[i]
            pos = vec3f(positions[i, 0], positions[i, 1], positions[i, 2])
            dist_abs = wp.abs(dist)

            # Adjust normal direction based on body assignment
            contact_normal = normal
            if capsule_in.bid < 0:
                contact_normal = -contact_normal

            # Offset contact point by penetration depth
            pos_A = pos + 0.5 * contact_normal * dist_abs
            pos_B = pos - 0.5 * contact_normal * dist_abs

            # Store contact data
            contact_wid_out[mcid] = wid_in
            contact_cid_out[mcid] = wcio + contact_idx
            contact_body_A_out[mcid] = vec4f(pos_A[0], pos_A[1], pos_A[2], float32(bid_A))
            contact_body_B_out[mcid] = vec4f(pos_B[0], pos_B[1], pos_B[2], float32(bid_B))
            contact_gapfunc_out[mcid] = vec4f(contact_normal[0], contact_normal[1], contact_normal[2], dist)
            contact_frame_out[mcid] = frame
            contact_material_out[mcid] = vec2f(friction_in, restitution_in)

            # Increment active contact index
            contact_idx += 1


@wp.func
def plane_cylinder(
    # Inputs:
    model_max_contacts_in: int32,
    world_max_contacts_in: int32,
    plane_in: Plane,
    cylinder_in: Cylinder,
    wid_in: int32,
    margin: float32,
    friction_in: float32,
    restitution_in: float32,
    # Contacts out:
    contacts_model_num_out: wp.array(dtype=int32),
    contacts_world_num_out: wp.array(dtype=int32),
    contact_wid_out: wp.array(dtype=int32),
    contact_cid_out: wp.array(dtype=int32),
    contact_body_A_out: wp.array(dtype=vec4f),
    contact_body_B_out: wp.array(dtype=vec4f),
    contact_gapfunc_out: wp.array(dtype=vec4f),
    contact_frame_out: wp.array(dtype=mat33f),
    contact_material_out: wp.array(dtype=vec2f),
):
    # Use the tested collision calculation from collision_primitive.py
    distances, positions, normal = collide_plane_cylinder(
        plane_in.normal, plane_in.pos, cylinder_in.pos, cylinder_in.axis, cylinder_in.radius, cylinder_in.half_height
    )

    # Add the active contacts to the global contacts arrays (up to 4 contacts with shared normal)
    wp.static(make_add_multiple_contacts(4, True))(
        model_max_contacts_in,
        world_max_contacts_in,
        wid_in,
        plane_in.bid,
        cylinder_in.bid,
        margin,
        distances,
        positions,
        normal,
        friction_in,
        restitution_in,
        contacts_model_num_out,
        contacts_world_num_out,
        contact_wid_out,
        contact_cid_out,
        contact_body_A_out,
        contact_body_B_out,
        contact_gapfunc_out,
        contact_frame_out,
        contact_material_out,
    )


###
# Kernels
###


@wp.kernel
def _primitive_narrowphase(
    # Inputs
    geom_bid_in: wp.array(dtype=int32),
    geom_sid_in: wp.array(dtype=int32),
    geom_params_in: wp.array(dtype=vec4f),
    geom_offset_in: wp.array(dtype=transformf),
    geom_mid_in: wp.array(dtype=int32),
    geom_pose_in: wp.array(dtype=transformf),
    col_model_num_pairs_in: wp.array(dtype=int32),
    col_world_num_pairs_in: wp.array(dtype=int32),
    col_wid_in: wp.array(dtype=int32),
    col_geom_pair_in: wp.array(dtype=vec2i),
    contacts_model_max_num_in: wp.array(dtype=int32),
    contacts_world_max_num_in: wp.array(dtype=int32),
    # Outputs:
    contacts_model_num_out: wp.array(dtype=int32),
    contacts_world_num_out: wp.array(dtype=int32),
    contacts_wid_out: wp.array(dtype=int32),
    contacts_cid_out: wp.array(dtype=int32),
    contacts_body_A_out: wp.array(dtype=vec4f),
    contacts_body_B_out: wp.array(dtype=vec4f),
    contacts_gapfunc_out: wp.array(dtype=vec4f),
    contacts_frame_out: wp.array(dtype=mat33f),
    contacts_material_out: wp.array(dtype=vec2f),
):
    # Retrieve the thread id
    tid = wp.tid()

    # Skip if the thread id is greater than the number of pairs
    if tid >= col_model_num_pairs_in[0]:
        return

    # Retrieve the world index
    wid = col_wid_in[tid]

    # Retrieve the maximum number of contacts for this world
    model_max_contacts = contacts_model_max_num_in[0]
    world_max_contacts = contacts_world_max_num_in[wid]

    # Retrieve the geometry ids
    geom_pair = col_geom_pair_in[tid]
    gid1 = geom_pair[0]
    gid2 = geom_pair[1]

    # Retrieve the body ids
    bid1 = geom_bid_in[gid1]
    bid2 = geom_bid_in[gid2]

    # Retrieve the shape ids
    sid1 = geom_sid_in[gid1]
    sid2 = geom_sid_in[gid2]

    # TODO(team): static loop unrolling to remove unnecessary branching
    if sid1 == ShapeType.SPHERE and sid2 == ShapeType.SPHERE:
        sphere_sphere(
            model_max_contacts,
            world_max_contacts,
            make_sphere(geom_pose_in[gid1], geom_params_in[gid1], gid1, bid1),
            make_sphere(geom_pose_in[gid2], geom_params_in[gid2], gid2, bid2),
            wid,
            DEFAULT_MARGIN,
            float32(0.7),
            float32(0.0),
            contacts_model_num_out,
            contacts_world_num_out,
            contacts_wid_out,
            contacts_cid_out,
            contacts_body_A_out,
            contacts_body_B_out,
            contacts_gapfunc_out,
            contacts_frame_out,
            contacts_material_out,
        )

    elif sid1 == int32(ShapeType.SPHERE.value) and sid2 == int32(ShapeType.CYLINDER.value):
        sphere_cylinder(
            model_max_contacts,
            world_max_contacts,
            make_sphere(geom_pose_in[gid1], geom_params_in[gid1], gid1, bid1),
            make_cylinder(geom_pose_in[gid2], geom_params_in[gid2], gid2, bid2),
            wid,
            DEFAULT_MARGIN,
            float32(0.7),
            float32(0.0),
            contacts_model_num_out,
            contacts_world_num_out,
            contacts_wid_out,
            contacts_cid_out,
            contacts_body_A_out,
            contacts_body_B_out,
            contacts_gapfunc_out,
            contacts_frame_out,
            contacts_material_out,
        )

    elif sid1 == int32(ShapeType.SPHERE.value) and sid2 == int32(ShapeType.CONE.value):
        sphere_cone()

    elif sid1 == int32(ShapeType.SPHERE.value) and sid2 == int32(ShapeType.CAPSULE.value):
        sphere_capsule(
            model_max_contacts,
            world_max_contacts,
            make_sphere(geom_pose_in[gid1], geom_params_in[gid1], gid1, bid1),
            make_capsule(geom_pose_in[gid2], geom_params_in[gid2], gid2, bid2),
            wid,
            DEFAULT_MARGIN,
            float32(0.7),
            float32(0.0),
            contacts_model_num_out,
            contacts_world_num_out,
            contacts_wid_out,
            contacts_cid_out,
            contacts_body_A_out,
            contacts_body_B_out,
            contacts_gapfunc_out,
            contacts_frame_out,
            contacts_material_out,
        )

    elif sid1 == int32(ShapeType.SPHERE.value) and sid2 == int32(ShapeType.BOX.value):
        sphere_box(
            model_max_contacts,
            world_max_contacts,
            make_sphere(geom_pose_in[gid1], geom_params_in[gid1], gid1, bid1),
            make_box(geom_pose_in[gid2], geom_params_in[gid2], gid2, bid2),
            wid,
            DEFAULT_MARGIN,
            float32(0.7),
            float32(0.0),
            contacts_model_num_out,
            contacts_world_num_out,
            contacts_wid_out,
            contacts_cid_out,
            contacts_body_A_out,
            contacts_body_B_out,
            contacts_gapfunc_out,
            contacts_frame_out,
            contacts_material_out,
        )

    elif sid1 == int32(ShapeType.SPHERE.value) and sid2 == int32(ShapeType.ELLIPSOID.value):
        sphere_ellipsoid()

    elif sid1 == int32(ShapeType.CYLINDER.value) and sid2 == int32(ShapeType.CYLINDER.value):
        cylinder_cylinder()

    elif sid1 == int32(ShapeType.CYLINDER.value) and sid2 == int32(ShapeType.CONE.value):
        cylinder_cone()

    elif sid1 == int32(ShapeType.CYLINDER.value) and sid2 == int32(ShapeType.CAPSULE.value):
        cylinder_capsule()

    elif sid1 == int32(ShapeType.CYLINDER.value) and sid2 == int32(ShapeType.BOX.value):
        cylinder_box()

    elif sid1 == int32(ShapeType.CYLINDER.value) and sid2 == int32(ShapeType.ELLIPSOID.value):
        cylinder_ellipsoid()

    elif sid1 == int32(ShapeType.CONE.value) and sid2 == int32(ShapeType.CONE.value):
        cone_cone()

    elif sid1 == int32(ShapeType.CONE.value) and sid2 == int32(ShapeType.CAPSULE.value):
        cone_capsule()

    elif sid1 == int32(ShapeType.CONE.value) and sid2 == int32(ShapeType.BOX.value):
        cone_box()

    elif sid1 == int32(ShapeType.CONE.value) and sid2 == int32(ShapeType.ELLIPSOID.value):
        cone_ellipsoid()

    elif sid1 == int32(ShapeType.CAPSULE.value) and sid2 == int32(ShapeType.CAPSULE.value):
        capsule_capsule(
            model_max_contacts,
            world_max_contacts,
            make_capsule(geom_pose_in[gid1], geom_params_in[gid1], gid1, bid1),
            make_capsule(geom_pose_in[gid2], geom_params_in[gid2], gid2, bid2),
            wid,
            DEFAULT_MARGIN,
            float32(0.7),
            float32(0.0),
            contacts_model_num_out,
            contacts_world_num_out,
            contacts_wid_out,
            contacts_cid_out,
            contacts_body_A_out,
            contacts_body_B_out,
            contacts_gapfunc_out,
            contacts_frame_out,
            contacts_material_out,
        )

    elif sid1 == int32(ShapeType.CAPSULE.value) and sid2 == int32(ShapeType.BOX.value):
        capsule_box(
            model_max_contacts,
            world_max_contacts,
            make_capsule(geom_pose_in[gid1], geom_params_in[gid1], gid1, bid1),
            make_box(geom_pose_in[gid2], geom_params_in[gid2], gid2, bid2),
            wid,
            DEFAULT_MARGIN,
            float32(0.7),
            float32(0.0),
            contacts_model_num_out,
            contacts_world_num_out,
            contacts_wid_out,
            contacts_cid_out,
            contacts_body_A_out,
            contacts_body_B_out,
            contacts_gapfunc_out,
            contacts_frame_out,
            contacts_material_out,
        )

    elif sid1 == int32(ShapeType.CAPSULE.value) and sid2 == int32(ShapeType.ELLIPSOID.value):
        capsule_ellipsoid()

    elif sid1 == int32(ShapeType.BOX.value) and sid2 == int32(ShapeType.BOX.value):
        box_box(
            model_max_contacts,
            world_max_contacts,
            make_box(geom_pose_in[gid1], geom_params_in[gid1], gid1, bid1),
            make_box(geom_pose_in[gid2], geom_params_in[gid2], gid2, bid2),
            wid,
            DEFAULT_MARGIN,
            float32(0.7),
            float32(0.0),
            contacts_model_num_out,
            contacts_world_num_out,
            contacts_wid_out,
            contacts_cid_out,
            contacts_body_A_out,
            contacts_body_B_out,
            contacts_gapfunc_out,
            contacts_frame_out,
            contacts_material_out,
        )
    elif sid1 == int32(ShapeType.BOX.value) and sid2 == int32(ShapeType.ELLIPSOID.value):
        box_ellipsoid()
    elif sid1 == int32(ShapeType.ELLIPSOID.value) and sid2 == int32(ShapeType.ELLIPSOID.value):
        ellipsoid_ellipsoid()

    # Plane collisions (plane is always geometry 1, other shapes are geometry 2)
    elif sid1 == int32(ShapeType.PLANE.value) and sid2 == int32(ShapeType.SPHERE.value):
        plane_sphere(
            model_max_contacts,
            world_max_contacts,
            make_plane(geom_pose_in[gid1], geom_params_in[gid1], gid1, bid1),
            make_sphere(geom_pose_in[gid2], geom_params_in[gid2], gid2, bid2),
            wid,
            DEFAULT_MARGIN,
            float32(0.7),
            float32(0.0),
            contacts_model_num_out,
            contacts_world_num_out,
            contacts_wid_out,
            contacts_cid_out,
            contacts_body_A_out,
            contacts_body_B_out,
            contacts_gapfunc_out,
            contacts_frame_out,
            contacts_material_out,
        )

    elif sid1 == int32(ShapeType.PLANE.value) and sid2 == int32(ShapeType.BOX.value):
        plane_box(
            model_max_contacts,
            world_max_contacts,
            make_plane(geom_pose_in[gid1], geom_params_in[gid1], gid1, bid1),
            make_box(geom_pose_in[gid2], geom_params_in[gid2], gid2, bid2),
            wid,
            DEFAULT_MARGIN,
            float32(0.7),
            float32(0.0),
            contacts_model_num_out,
            contacts_world_num_out,
            contacts_wid_out,
            contacts_cid_out,
            contacts_body_A_out,
            contacts_body_B_out,
            contacts_gapfunc_out,
            contacts_frame_out,
            contacts_material_out,
        )

    elif sid1 == int32(ShapeType.PLANE.value) and sid2 == int32(ShapeType.ELLIPSOID.value):
        plane_ellipsoid(
            model_max_contacts,
            world_max_contacts,
            make_plane(geom_pose_in[gid1], geom_params_in[gid1], gid1, bid1),
            make_ellipsoid(geom_pose_in[gid2], geom_params_in[gid2], gid2, bid2),
            wid,
            DEFAULT_MARGIN,
            float32(0.7),
            float32(0.0),
            contacts_model_num_out,
            contacts_world_num_out,
            contacts_wid_out,
            contacts_cid_out,
            contacts_body_A_out,
            contacts_body_B_out,
            contacts_gapfunc_out,
            contacts_frame_out,
            contacts_material_out,
        )

    elif sid1 == int32(ShapeType.PLANE.value) and sid2 == int32(ShapeType.CAPSULE.value):
        plane_capsule(
            model_max_contacts,
            world_max_contacts,
            make_plane(geom_pose_in[gid1], geom_params_in[gid1], gid1, bid1),
            make_capsule(geom_pose_in[gid2], geom_params_in[gid2], gid2, bid2),
            wid,
            DEFAULT_MARGIN,
            float32(0.7),
            float32(0.0),
            contacts_model_num_out,
            contacts_world_num_out,
            contacts_wid_out,
            contacts_cid_out,
            contacts_body_A_out,
            contacts_body_B_out,
            contacts_gapfunc_out,
            contacts_frame_out,
            contacts_material_out,
        )

    elif sid1 == int32(ShapeType.PLANE.value) and sid2 == int32(ShapeType.CYLINDER.value):
        plane_cylinder(
            model_max_contacts,
            world_max_contacts,
            make_plane(geom_pose_in[gid1], geom_params_in[gid1], gid1, bid1),
            make_cylinder(geom_pose_in[gid2], geom_params_in[gid2], gid2, bid2),
            wid,
            DEFAULT_MARGIN,
            float32(0.7),
            float32(0.0),
            contacts_model_num_out,
            contacts_world_num_out,
            contacts_wid_out,
            contacts_cid_out,
            contacts_body_A_out,
            contacts_body_B_out,
            contacts_gapfunc_out,
            contacts_frame_out,
            contacts_material_out,
        )


###
# Kernel Launcher
###


@wp.func
def write_contact_kamino(
    contact_point_center: vec3f,
    contact_normal_a_to_b: vec3f,
    contact_distance: float32,
    radius_eff_a: float32,
    radius_eff_b: float32,
    thickness_a: float32,
    thickness_b: float32,
    bid_a: int32,
    bid_b: int32,
    wid: int32,
    margin: float32,
    model_max_contacts: int32,
    world_max_contacts: int32,
    friction: float32,
    restitution: float32,
    # outputs
    contacts_model_num_out: wp.array(dtype=int32),
    contacts_world_num_out: wp.array(dtype=int32),
    contact_wid_out: wp.array(dtype=int32),
    contact_cid_out: wp.array(dtype=int32),
    contact_body_A_out: wp.array(dtype=vec4f),
    contact_body_B_out: wp.array(dtype=vec4f),
    contact_gapfunc_out: wp.array(dtype=vec4f),
    contact_frame_out: wp.array(dtype=mat33f),
    contact_material_out: wp.array(dtype=vec2f),
):
    """
    Write a contact to Kamino-compatible output arrays.

    Converts NarrowPhase output format to Kamino Contacts format.
    """
    total_separation_needed = radius_eff_a + radius_eff_b + thickness_a + thickness_b

    # Normalize contact normal
    contact_normal_a_to_b = wp.normalize(contact_normal_a_to_b)

    # Compute contact points on each shape
    a_contact_world = contact_point_center - contact_normal_a_to_b * (0.5 * contact_distance + radius_eff_a)
    b_contact_world = contact_point_center + contact_normal_a_to_b * (0.5 * contact_distance + radius_eff_b)

    # Calculate penetration distance
    diff = b_contact_world - a_contact_world
    distance = wp.dot(diff, contact_normal_a_to_b)
    d = distance - total_separation_needed

    # Only write contact if within margin
    if d < margin:
        mcid = wp.atomic_add(contacts_model_num_out, 0, 1)
        wcid = wp.atomic_add(contacts_world_num_out, wid, 1)

        if mcid < model_max_contacts and wcid < world_max_contacts:
            # Perform body assignment (static body is always body A)
            if bid_b < 0:
                bid_A = bid_b
                bid_B = bid_a
                normal = -contact_normal_a_to_b
                pos_A = b_contact_world
                pos_B = a_contact_world
            else:
                bid_A = bid_a
                bid_B = bid_b
                normal = contact_normal_a_to_b
                pos_A = a_contact_world
                pos_B = b_contact_world

            # Store contact data in Kamino format
            contact_wid_out[mcid] = wid
            contact_cid_out[mcid] = wcid
            contact_body_A_out[mcid] = vec4f(pos_A[0], pos_A[1], pos_A[2], float32(bid_A))
            contact_body_B_out[mcid] = vec4f(pos_B[0], pos_B[1], pos_B[2], float32(bid_B))
            contact_gapfunc_out[mcid] = vec4f(normal[0], normal[1], normal[2], d)
            contact_frame_out[mcid] = make_contact_frame_znorm(normal)
            contact_material_out[mcid] = vec2f(friction, restitution)


@wp.kernel
def convert_narrow_phase_to_kamino_contacts_kernel(
    contact_pair: wp.array(dtype=vec2i),
    contact_position: wp.array(dtype=vec3f),
    contact_normal: wp.array(dtype=vec3f),
    contact_penetration: wp.array(dtype=float32),
    narrow_contact_count: wp.array(dtype=int32),
    geom_data: wp.array(dtype=vec4f),
    geom_type: wp.array(dtype=int32),
    geom_bid: wp.array(dtype=int32),
    geom_wid: wp.array(dtype=int32),
    margin: float32,
    friction: float32,
    restitution: float32,
    model_max_contacts: int32,
    world_max_contacts: int32,
    # Outputs (Kamino Contacts format)
    contacts_model_num_out: wp.array(dtype=int32),
    contacts_world_num_out: wp.array(dtype=int32),
    contact_wid_out: wp.array(dtype=int32),
    contact_cid_out: wp.array(dtype=int32),
    contact_body_A_out: wp.array(dtype=vec4f),
    contact_body_B_out: wp.array(dtype=vec4f),
    contact_gapfunc_out: wp.array(dtype=vec4f),
    contact_frame_out: wp.array(dtype=mat33f),
    contact_material_out: wp.array(dtype=vec2f),
):
    """
    Convert NarrowPhase output format to Kamino Contacts format.
    """
    idx = wp.tid()
    num_contacts = narrow_contact_count[0]

    if idx >= num_contacts:
        return

    # Get contact pair
    pair = contact_pair[idx]
    gid_a = pair[0]
    gid_b = pair[1]

    # Extract thickness values
    thickness_a = geom_data[gid_a][3]
    thickness_b = geom_data[gid_b][3]

    # Extract effective radius for sphere and capsule shapes
    type_a = geom_type[gid_a]
    type_b = geom_type[gid_b]

    radius_eff_a = float32(0.0)
    radius_eff_b = float32(0.0)

    # For spheres and capsules, extract the radius from scale[0]
    if type_a == int32(ShapeType.SPHERE.value) or type_a == int32(ShapeType.CAPSULE.value):
        radius_eff_a = geom_data[gid_a][0]

    if type_b == int32(ShapeType.SPHERE.value) or type_b == int32(ShapeType.CAPSULE.value):
        radius_eff_b = geom_data[gid_b][0]

    # Get contact data from narrow phase
    contact_point_center = contact_position[idx]
    contact_normal_a_to_b = contact_normal[idx]
    contact_distance = contact_penetration[idx] + thickness_a + thickness_b

    # Get body and world IDs
    bid_a = geom_bid[gid_a]
    bid_b = geom_bid[gid_b]
    wid = geom_wid[gid_a]  # Assume both geoms are in same world

    # Use write_contact_kamino to format the contact
    write_contact_kamino(
        contact_point_center,
        contact_normal_a_to_b,
        contact_distance,
        radius_eff_a,
        radius_eff_b,
        thickness_a,
        thickness_b,
        bid_a,
        bid_b,
        wid,
        margin,
        model_max_contacts,
        world_max_contacts,
        friction,
        restitution,
        contacts_model_num_out,
        contacts_world_num_out,
        contact_wid_out,
        contact_cid_out,
        contact_body_A_out,
        contact_body_B_out,
        contact_gapfunc_out,
        contact_frame_out,
        contact_material_out,
    )


class KaminoCollisionPipeline:
    """
    Collision pipeline for Kamino using NarrowPhase class.

    This wraps the Newton NarrowPhase API for use with Kamino's data structures.
    """

    def __init__(
        self,
        model: Model,
        max_contacts_per_pair: int = 10,
        margin: float = 1e-5,
        friction: float = 0.7,
        restitution: float = 0.0,
    ):
        """
        Initialize KaminoCollisionPipeline.

        Args:
            model: Kamino Model
            max_contacts_per_pair: Maximum contacts per collision pair
            margin: Contact margin threshold
            friction: Default friction coefficient
            restitution: Default restitution coefficient
        """
        from ....geometry.narrow_phase import NarrowPhase
        from ....geometry.types import GeoType

        self.model = model
        self.margin = margin
        self.friction = friction
        self.restitution = restitution

        # Compute maximum possible pairs (NxN)
        num_geoms = len(model.cgeoms.sid.numpy())
        max_pairs = (num_geoms * (num_geoms - 1)) // 2
        self.max_contacts = max_pairs * max_contacts_per_pair

        # Initialize NarrowPhase
        self.narrow_phase = NarrowPhase(
            max_candidate_pairs=max_pairs,
            max_triangle_pairs=1000000,
            device=None,
        )

        # Allocate buffers for narrow phase outputs
        self.narrow_contact_pair = wp.zeros(self.max_contacts, dtype=vec2i)
        self.narrow_contact_position = wp.zeros(self.max_contacts, dtype=vec3f)
        self.narrow_contact_normal = wp.zeros(self.max_contacts, dtype=vec3f)
        self.narrow_contact_penetration = wp.zeros(self.max_contacts, dtype=float32)
        self.narrow_contact_count = wp.zeros(1, dtype=int32)

        # Allocate buffers for geom data conversion
        self.geom_data = wp.zeros(num_geoms, dtype=vec4f)
        self.geom_transform = wp.zeros(num_geoms, dtype=transformf)
        self.geom_cutoff = wp.full(num_geoms, margin, dtype=float32)
        self.geom_type = wp.zeros(num_geoms, dtype=int32)
        self.geom_source_ptr = wp.zeros(num_geoms, dtype=wp.uint64)
        self.geom_collision_radius = wp.zeros(num_geoms, dtype=float32)

        # Store shape type mapping (Kamino ShapeType to Newton GeoType)
        self.shape_type_map = {
            ShapeType.SPHERE: int(GeoType.SPHERE),
            ShapeType.CAPSULE: int(GeoType.CAPSULE),
            ShapeType.BOX: int(GeoType.BOX),
            ShapeType.CYLINDER: int(GeoType.CYLINDER),
            ShapeType.PLANE: int(GeoType.PLANE),
            ShapeType.ELLIPSOID: int(GeoType.ELLIPSOID),
        }

    def launch(self, state: ModelData, collisions: Collisions, contacts: Contacts):
        """
        Run collision detection pipeline.

        Args:
            state: Current model state
            collisions: Collision pairs from broad phase
            contacts: Output contacts container
        """
        # Clear narrow phase contact count
        self.narrow_contact_count.zero_()

        # Prepare geometry data for NarrowPhase API
        self._prepare_geom_data(state)

        # Get collision pairs from broad phase
        num_pairs = collisions.cdata.model_num_collisions.numpy()[0]
        if num_pairs == 0:
            return

        # Run narrow phase
        self.narrow_phase.launch(
            candidate_pair=collisions.cdata.geom_pair,
            num_candidate_pair=collisions.cdata.model_num_collisions,
            geom_types=self.geom_type,
            geom_data=self.geom_data,
            geom_transform=self.geom_transform,
            geom_source=self.geom_source_ptr,
            geom_cutoff=self.geom_cutoff,
            geom_collision_radius=self.geom_collision_radius,
            contact_pair=self.narrow_contact_pair,
            contact_position=self.narrow_contact_position,
            contact_normal=self.narrow_contact_normal,
            contact_penetration=self.narrow_contact_penetration,
            contact_tangent=None,
            contact_pair_key=None,
            contact_key=None,
            contact_count=self.narrow_contact_count,
            device=None,
        )

        # Convert NarrowPhase output to Kamino Contacts format
        model_max = contacts.model_max_contacts.numpy()[0]
        world_max_list = contacts.world_max_contacts.numpy()

        wp.launch(
            kernel=convert_narrow_phase_to_kamino_contacts_kernel,
            dim=self.max_contacts,
            inputs=[
                self.narrow_contact_pair,
                self.narrow_contact_position,
                self.narrow_contact_normal,
                self.narrow_contact_penetration,
                self.narrow_contact_count,
                self.geom_data,
                self.geom_type,
                self.model.cgeoms.bid,
                self.model.cgeoms.wid,
                self.margin,
                self.friction,
                self.restitution,
                model_max,
                world_max_list[0] if len(world_max_list) > 0 else model_max,
            ],
            outputs=[
                contacts.model_num_contacts,
                contacts.world_num_contacts,
                contacts.wid,
                contacts.cid,
                contacts.body_A,
                contacts.body_B,
                contacts.gapfunc,
                contacts.frame,
                contacts.material,
            ],
        )

    def _prepare_geom_data(self, state: ModelData):
        """Prepare geometry data arrays for NarrowPhase API."""
        # Convert Kamino geometry data to Newton format
        num_geoms = len(self.model.cgeoms.sid.numpy())

        for i in range(num_geoms):
            # Convert shape type
            kamino_type = ShapeType(self.model.cgeoms.sid.numpy()[i])
            newton_type = self.shape_type_map.get(kamino_type, int(GeoType.BOX))
            self.geom_type.numpy()[i] = newton_type

            # Pack params into geom_data (scale xyz, thickness w)
            params = self.model.cgeoms.params.numpy()[i]
            self.geom_data.numpy()[i] = params

            # Set transform (world space)
            self.geom_transform.numpy()[i] = state.cgeoms.pose.numpy()[i]

            # Set collision radius (use largest param dimension)
            self.geom_collision_radius.numpy()[i] = max(params[0], params[1], params[2])


def primitive_narrowphase_unified(
    model: Model,
    state: ModelData,
    collisions: Collisions,
    contacts: Contacts,
    collision_pipeline: KaminoCollisionPipeline,
):
    """
    Unified narrow-phase collision detection using NarrowPhase class.

    This is a replacement for primitive_narrowphase that uses the collision pipeline.

    Args:
        model: Kamino Model
        state: Current model state
        collisions: Collision pairs from broad phase
        contacts: Output contacts container
        collision_pipeline: KaminoCollisionPipeline instance
    """
    collision_pipeline.launch(state, collisions, contacts)


def primitive_narrowphase(model: Model, state: ModelData, collisions: Collisions, contacts: Contacts):
    """
    Launches the narrow-phase collision detection kernel for primitive shapes.

    Arguments
    ------
        model (Model): The model containing the collision geometries.
        state (ModelData): The current state of the model.
        collisions (Collisions): The collision container holding collision pairs.
        contacts (Contacts): The contacts container to store detected contacts.
    """
    wp.launch(
        _primitive_narrowphase,
        dim=collisions.cmodel.num_model_geom_pairs,
        inputs=[
            # Inputs:
            model.cgeoms.bid,
            model.cgeoms.sid,
            model.cgeoms.params,
            model.cgeoms.offset,
            model.cgeoms.mid,
            state.cgeoms.pose,
            collisions.cdata.model_num_collisions,
            collisions.cdata.world_num_collisions,
            collisions.cdata.wid,
            collisions.cdata.geom_pair,
            contacts.model_max_contacts,
            contacts.world_max_contacts,
            # Outputs:
            contacts.model_num_contacts,
            contacts.world_num_contacts,
            contacts.wid,
            contacts.cid,
            contacts.body_A,
            contacts.body_B,
            contacts.gapfunc,
            contacts.frame,
            contacts.material,
        ],
    )
