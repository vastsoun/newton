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
KAMINO: Collision Detection: Narrow-phase operation for geometric primitives
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.kamino.core.math import FLOAT32_EPS, UNIT_Z
from newton._src.solvers.kamino.core.model import Model, ModelData
from newton._src.solvers.kamino.core.shapes import ShapeType
from newton._src.solvers.kamino.core.types import (
    float32,
    int32,
    mat33f,
    mat43f,
    mat83f,
    transformf,
    vec2f,
    vec2i,
    vec3f,
    vec4f,
    vec8f,
)
from newton._src.solvers.kamino.geometry.collisions import Collisions
from newton._src.solvers.kamino.geometry.contacts import Contacts
from newton._src.solvers.kamino.geometry.math import make_contact_frame_znorm

###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Constants
###

FLOAT32_MINVAL = wp.constant(float32(1e-15))
DEFAULT_MARGIN = FLOAT32_EPS
# DEFAULT_MARGIN = wp.constant(float32(1e-6))
# DEFAULT_MARGIN = wp.constant(float32(0.0))


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
def normalize_with_norm(x: vec3f):
    norm = wp.length(x)
    if norm == 0.0:
        return x, 0.0
    return x / norm, norm


###
# Functions
###


@wp.func
def add_active_contact(
    # Inputs:
    model_max_contacts_in: int32,
    world_max_contacts_in: int32,
    wid_in: int32,
    bid_1_in: int32,
    bid_2_in: int32,
    position_in: vec3f,
    normal_in: vec3f,
    distance_in: float32,
    margin_in: float32,
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
    active = (distance_in - margin_in) < 0.0
    if active:
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
            # The colliders compute the contact point in the middle, and thus to get the
            # per-geom contact points we need to offset by the penetration depth along the normal
            position_A_in = position_in + 0.5 * wp.abs(distance_in) * normal_in
            position_B_in = position_in - 0.5 * wp.abs(distance_in) * normal_in
            # Store the active contact output data
            contact_wid_out[mcid] = wid_in
            contact_cid_out[mcid] = wcid
            contact_body_A_out[mcid] = vec4f(position_A_in[0], position_A_in[1], position_A_in[2], float32(bid_A_in))
            contact_body_B_out[mcid] = vec4f(position_B_in[0], position_B_in[1], position_B_in[2], float32(bid_B_in))
            contact_gapfunc_out[mcid] = vec4f(normal_in[0], normal_in[1], normal_in[2], distance_in)
            contact_frame_out[mcid] = make_contact_frame_znorm(normal_in)
            contact_material_out[mcid][0] = friction_in
            contact_material_out[mcid][1] = restitution_in


@wp.func
def sphere_sphere(
    # Inputs:
    model_max_contacts: int32,
    world_max_contacts: int32,
    sphere1: Sphere,
    sphere2: Sphere,
    wid: int32,
    gap: float32,
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
    dir = sphere2.pos - sphere1.pos
    dist = wp.length(dir)
    if dist == 0.0:  # TODO: Shouldnt this use a numeric threshold, e.g. eps?
        normal = UNIT_Z
    else:
        normal = dir / dist
    dist = dist - (sphere1.radius + sphere2.radius)
    pos = sphere1.pos + normal * (sphere1.radius + 0.5 * dist)

    add_active_contact(
        model_max_contacts,
        world_max_contacts,
        wid,
        sphere1.bid,
        sphere2.bid,
        pos,
        normal,
        dist,
        margin,
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
def sphere_cylinder():
    pass


@wp.func
def sphere_cone():
    pass


@wp.func
def sphere_capsule():
    pass


@wp.func
def sphere_box(
    # Inputs:
    model_max_contacts_in: int32,
    world_max_contacts_in: int32,
    sphere_in: Sphere,
    box_in: Box,
    wid_in: int32,
    gap_in: float32,
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
    center = wp.transpose(box_in.rot) @ (sphere_in.pos - box_in.pos)
    clamped = wp.max(-box_in.size, wp.min(box_in.size, center))
    clamped_dir, dist = normalize_with_norm(clamped - center)

    if dist - sphere_in.radius > margin:
        return

    # sphere center inside box
    if dist <= FLOAT32_MINVAL:
        closest = 2.0 * (box_in.size[0] + box_in.size[1] + box_in.size[2])
        k = wp.int32(0)
        for i in range(6):
            face_dist = wp.abs(wp.where(i % 2, 1.0, -1.0) * box_in.size[i / 2] - center[i / 2])
            if closest > face_dist:
                closest = face_dist
                k = i

        nearest = vec3f(0.0)
        nearest[k / 2] = wp.where(k % 2, -1.0, 1.0)
        pos = center + nearest * (sphere_in.radius - closest) / 2.0
        contact_normal = box_in.rot @ nearest
        contact_dist = -closest - sphere_in.radius

    else:
        deepest = center + clamped_dir * sphere_in.radius
        pos = 0.5 * (clamped + deepest)
        contact_normal = box_in.rot @ clamped_dir
        contact_dist = dist - sphere_in.radius

    contact_pos = box_in.pos + box_in.rot @ pos

    add_active_contact(
        model_max_contacts_in,
        world_max_contacts_in,
        wid_in,
        sphere_in.bid,
        box_in.bid,
        contact_pos,
        contact_normal,
        contact_dist,
        margin,
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
def capsule_capsule():
    pass


@wp.func
def capsule_box():
    pass


@wp.func
def capsule_ellipsoid():
    pass


@wp.func
def _compute_rotmore(face_idx: int32) -> mat33f:
    rotmore = mat33f(0.0)
    if face_idx == 0:
        rotmore[0, 2] = -1.0
        rotmore[1, 1] = +1.0
        rotmore[2, 0] = +1.0
    elif face_idx == 1:
        rotmore[0, 0] = +1.0
        rotmore[1, 2] = -1.0
        rotmore[2, 1] = +1.0
    elif face_idx == 2:
        rotmore[0, 0] = +1.0
        rotmore[1, 1] = +1.0
        rotmore[2, 2] = +1.0
    elif face_idx == 3:
        rotmore[0, 2] = +1.0
        rotmore[1, 1] = +1.0
        rotmore[2, 0] = -1.0
    elif face_idx == 4:
        rotmore[0, 0] = +1.0
        rotmore[1, 2] = +1.0
        rotmore[2, 1] = -1.0
    elif face_idx == 5:
        rotmore[0, 0] = -1.0
        rotmore[1, 1] = +1.0
        rotmore[2, 2] = -1.0
    return rotmore


@wp.func
def box_box(
    # Inputs:
    model_max_contacts_in: int32,
    world_max_contacts_in: int32,
    box1_in: Box,
    box2_in: Box,
    wid_in: int32,
    gap_in: float32,
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
    # Compute transforms between box's frames
    # wp.printf("b1: d: %f, w: %f, h: %f\n", box1_in.size[0], box1_in.size[1], box1_in.size[2])
    # wp.printf("b2: d: %f, w: %f, h: %f\n", box2_in.size[0], box2_in.size[1], box2_in.size[2])

    pos21 = wp.transpose(box1_in.rot) @ (box2_in.pos - box1_in.pos)
    pos12 = wp.transpose(box2_in.rot) @ (box1_in.pos - box2_in.pos)

    rot21 = wp.transpose(box1_in.rot) @ box2_in.rot
    rot12 = wp.transpose(rot21)

    rot21abs = wp.matrix_from_rows(wp.abs(rot21[0]), wp.abs(rot21[1]), wp.abs(rot21[2]))
    rot12abs = wp.transpose(rot21abs)

    plen2 = rot21abs @ box2_in.size
    plen1 = rot12abs @ box1_in.size

    # Compute axis of maximum separation
    s_sum_3 = 3.0 * (box1_in.size + box2_in.size)
    separation = wp.float32(margin + s_sum_3[0] + s_sum_3[1] + s_sum_3[2])
    axis_code = wp.int32(-1)

    # First test: consider boxes' face normals
    for i in range(3):
        c1 = -wp.abs(pos21[i]) + box1_in.size[i] + plen2[i]
        c2 = -wp.abs(pos12[i]) + box2_in.size[i] + plen1[i]
        if c1 < -margin or c2 < -margin:
            return
        if c1 < separation:
            separation = c1
            axis_code = i + 3 * wp.int32(pos21[i] < 0) + 0  # Face of box1
        if c2 < separation:
            separation = c2
            axis_code = i + 3 * wp.int32(pos12[i] < 0) + 6  # Face of box2

    clnorm = wp.vec3(0.0)
    inv = wp.bool(False)
    cle1 = wp.int32(0)
    cle2 = wp.int32(0)

    # Second test: consider cross products of boxes' edges
    for i in range(3):
        for j in range(3):
            # Compute cross product of box edges (potential separating axis)
            if i == 0:
                cross_axis = wp.vec3(0.0, -rot12[j, 2], rot12[j, 1])
            elif i == 1:
                cross_axis = wp.vec3(rot12[j, 2], 0.0, -rot12[j, 0])
            else:
                cross_axis = wp.vec3(-rot12[j, 1], rot12[j, 0], 0.0)

            cross_length = wp.length(cross_axis)
            if cross_length < FLOAT32_MINVAL:
                continue

            cross_axis /= cross_length

            box_dist = wp.dot(pos21, cross_axis)
            c3 = wp.float32(0.0)

            # Project box half-sizes onto the potential separating axis
            for k in range(3):
                if k != i:
                    c3 += box1_in.size[k] * wp.abs(cross_axis[k])
                if k != j:
                    c3 += box2_in.size[k] * rot21abs[i, 3 - k - j] / cross_length

            c3 -= wp.abs(box_dist)

            # Early exit: no collision if separated along this axis
            if c3 < -margin:
                return

            # Track minimum separation and which edge-edge pair it occurs on
            if c3 < separation * (1.0 - 1e-12):
                separation = c3
                # Determine which corners/edges are closest
                cle1 = 0
                cle2 = 0

                for k in range(3):
                    if k != i and (int(cross_axis[k] > 0) ^ int(box_dist < 0)):
                        cle1 += 1 << k
                    if k != j and (int(rot21[i, 3 - k - j] > 0) ^ int(box_dist < 0) ^ int((k - j + 3) % 3 == 1)):
                        cle2 += 1 << k

                axis_code = 12 + i * 3 + j
                clnorm = cross_axis
                inv = box_dist < 0

    # No axis with separation < margin found
    if axis_code == -1:
        return

    points = mat83f()
    depth = vec8f()
    max_con_pair = 8
    # 8 contacts should suffice for most configurations

    if axis_code < 12:
        # Handle face-vertex collision
        face_idx = axis_code % 6
        box_idx = axis_code / 6
        rotmore = _compute_rotmore(face_idx)

        r = rotmore @ wp.where(box_idx, rot12, rot21)
        p = rotmore @ wp.where(box_idx, pos12, pos21)
        ss = wp.abs(rotmore @ wp.where(box_idx, box2_in.size, box1_in.size))
        s = wp.where(box_idx, box1_in.size, box2_in.size)
        rt = wp.transpose(r)

        lx, ly, hz = ss[0], ss[1], ss[2]
        p[2] -= hz

        clcorner = wp.int32(0)  # corner of non-face box with least axis separation

        for i in range(3):
            if r[2, i] < 0:
                clcorner += 1 << i

        lp = p
        for i in range(wp.static(3)):
            lp += rt[i] * s[i] * wp.where(clcorner & 1 << i, 1.0, -1.0)

        m = wp.int32(1)
        dirs = wp.int32(0)

        cn1 = wp.vec3(0.0)
        cn2 = wp.vec3(0.0)

        for i in range(3):
            if wp.abs(r[2, i]) < 0.5:
                if not dirs:
                    cn1 = rt[i] * s[i] * wp.where(clcorner & (1 << i), -2.0, 2.0)
                else:
                    cn2 = rt[i] * s[i] * wp.where(clcorner & (1 << i), -2.0, 2.0)
                dirs += 1

        k = dirs * dirs

        # Find potential contact points

        n = wp.int32(0)

        for i in range(k):
            for q in range(2):
                # lines_a and lines_b (lines between corners) computed on the fly
                lav = lp + wp.where(i < 2, wp.vec3(0.0), wp.where(i == 2, cn1, cn2))
                lbv = wp.where(i == 0 or i == 3, cn1, cn2)

                if wp.abs(lbv[q]) > FLOAT32_MINVAL:
                    br = 1.0 / lbv[q]
                    for j in range(-1, 2, 2):
                        ll = ss[q] * wp.float32(j)
                        c1 = (ll - lav[q]) * br
                        if c1 < 0 or c1 > 1:
                            continue
                        c2 = lav[1 - q] + lbv[1 - q] * c1
                        if wp.abs(c2) > ss[1 - q]:
                            continue

                        points[n] = lav + c1 * lbv
                        n += 1

        if dirs == 2:
            ax = cn1[0]
            bx = cn2[0]
            ay = cn1[1]
            by = cn2[1]
            C = 1.0 / (ax * by - bx * ay)

            for i in range(4):
                llx = wp.where(i / 2, lx, -lx)
                lly = wp.where(i % 2, ly, -ly)

                x = llx - lp[0]
                y = lly - lp[1]

                u = (x * by - y * bx) * C
                v = (y * ax - x * ay) * C

                if u > 0 and v > 0 and u < 1 and v < 1:
                    points[n] = wp.vec3(llx, lly, lp[2] + u * cn1[2] + v * cn2[2])
                    n += 1

        for i in range(1 << dirs):
            tmpv = lp + wp.float32(i & 1) * cn1 + wp.float32((i & 2) != 0) * cn2
            if tmpv[0] > -lx and tmpv[0] < lx and tmpv[1] > -ly and tmpv[1] < ly:
                points[n] = tmpv
                n += 1

        m = n
        n = wp.int32(0)

        for i in range(m):
            if points[i][2] > margin:
                continue
            if i != n:
                points[n] = points[i]
            points[n, 2] *= 0.5
            depth[n] = points[n, 2]
            n += 1

        # Set up contact frame
        rw = wp.where(box_idx, box2_in.rot, box1_in.rot) @ wp.transpose(rotmore)
        pw = wp.where(box_idx, box2_in.pos, box1_in.pos)
        normal = wp.where(box_idx, -1.0, 1.0) * wp.transpose(rw)[2]

    else:
        # Handle edge-edge collision
        edge1 = (axis_code - 12) / 3
        edge2 = (axis_code - 12) % 3

        # Set up non-contacting edges ax1, ax2 for box2 and pax1, pax2 for box 1
        ax1 = wp.int(1 - (edge2 & 1))
        ax2 = wp.int(2 - (edge2 & 2))

        pax1 = wp.int(1 - (edge1 & 1))
        pax2 = wp.int(2 - (edge1 & 2))

        if rot21abs[edge1, ax1] < rot21abs[edge1, ax2]:
            ax1, ax2 = ax2, ax1

        if rot12abs[edge2, pax1] < rot12abs[edge2, pax2]:
            pax1, pax2 = pax2, pax1

        rotmore = _compute_rotmore(wp.where(cle1 & (1 << pax2), pax2, pax2 + 3))

        # Transform coordinates for edge-edge contact calculation
        p = rotmore @ pos21
        rnorm = rotmore @ clnorm
        r = rotmore @ rot21
        rt = wp.transpose(r)
        s = wp.abs(wp.transpose(rotmore) @ box1_in.size)

        lx, ly, hz = s[0], s[1], s[2]
        p[2] -= hz

        # Calculate closest box2 face
        points[0] = (
            p
            + rt[ax1] * box2_in.size[ax1] * wp.where(cle2 & (1 << ax1), 1.0, -1.0)
            + rt[ax2] * box2_in.size[ax2] * wp.where(cle2 & (1 << ax2), 1.0, -1.0)
        )
        points[1] = points[0] - rt[edge2] * box2_in.size[edge2]
        points[0] += rt[edge2] * box2_in.size[edge2]
        points[2] = (
            p
            + rt[ax1] * box2_in.size[ax1] * wp.where(cle2 & (1 << ax1), -1.0, 1.0)
            + rt[ax2] * box2_in.size[ax2] * wp.where(cle2 & (1 << ax2), 1.0, -1.0)
        )
        points[3] = points[2] - rt[edge2] * box2_in.size[edge2]
        points[2] += rt[edge2] * box2_in.size[edge2]

        n = 4

        # Set up coordinate axes for contact face of box2
        axi_lp = points[0]
        axi_cn1 = points[1] - points[0]
        axi_cn2 = points[2] - points[0]

        # Check if contact normal is valid
        if wp.abs(rnorm[2]) < FLOAT32_MINVAL:
            return  # Shouldn't happen

        # Calculate inverse normal for projection
        innorm = wp.where(inv, -1.0, 1.0) / rnorm[2]

        pu = mat43f()

        # Project points onto contact plane
        for i in range(4):
            pu[i] = points[i]
            c_scl = points[i, 2] * wp.where(inv, -1.0, 1.0) * innorm
            points[i] -= rnorm * c_scl

        pts_lp = points[0]
        pts_cn1 = points[1] - points[0]
        pts_cn2 = points[2] - points[0]

        n = wp.int32(0)

        for i in range(4):
            for q in range(2):
                la = pts_lp[q] + wp.where(i < 2, 0.0, wp.where(i == 2, pts_cn1[q], pts_cn2[q]))
                lb = wp.where(i == 0 or i == 3, pts_cn1[q], pts_cn2[q])
                lc = pts_lp[1 - q] + wp.where(i < 2, 0.0, wp.where(i == 2, pts_cn1[1 - q], pts_cn2[1 - q]))
                ld = wp.where(i == 0 or i == 3, pts_cn1[1 - q], pts_cn2[1 - q])

                # linesu_a and linesu_b (lines between corners) computed on the fly
                lua = axi_lp + wp.where(i < 2, wp.vec3(0.0), wp.where(i == 2, axi_cn1, axi_cn2))
                lub = wp.where(i == 0 or i == 3, axi_cn1, axi_cn2)

                if wp.abs(lb) > FLOAT32_MINVAL:
                    br = 1.0 / lb
                    for j in range(-1, 2, 2):
                        if n == max_con_pair:
                            break
                        ll = s[q] * wp.float32(j)
                        c1 = (ll - la) * br
                        if c1 < 0 or c1 > 1:
                            continue
                        c2 = lc + ld * c1
                        if wp.abs(c2) > s[1 - q]:
                            continue
                        if (lua[2] + lub[2] * c1) * innorm > margin:
                            continue
                        points[n] = lua * 0.5 + c1 * lub * 0.5
                        points[n, q] += 0.5 * ll
                        points[n, 1 - q] += 0.5 * c2
                        depth[n] = points[n, 2] * innorm * 2.0
                        n += 1

        nl = n
        ax = pts_cn1[0]
        bx = pts_cn2[0]
        ay = pts_cn1[1]
        by = pts_cn2[1]
        C = 1.0 / (ax * by - bx * ay)

        for i in range(4):
            if n == max_con_pair:
                break
            llx = wp.where(i / 2, lx, -lx)
            lly = wp.where(i % 2, ly, -ly)

            x = llx - pts_lp[0]
            y = lly - pts_lp[1]

            u = (x * by - y * bx) * C
            v = (y * ax - x * ay) * C

            if nl == 0:
                if (u < 0 or u > 0) and (v < 0 or v > 1):
                    continue
            elif u < 0 or v < 0 or u > 1 or v > 1:
                continue

            u = wp.clamp(u, 0.0, 1.0)
            v = wp.clamp(v, 0.0, 1.0)
            w = 1.0 - u - v
            vtmp = pu[0] * w + pu[1] * u + pu[2] * v

            points[n] = wp.vec3(llx, lly, 0.0)

            vtmp2 = points[n] - vtmp
            tc1 = wp.length_sq(vtmp2)
            if vtmp[2] > 0 and tc1 > margin * margin:
                continue

            points[n] = 0.5 * (points[n] + vtmp)

            depth[n] = wp.sqrt(tc1) * wp.where(vtmp[2] < 0, -1.0, 1.0)
            n += 1

        nf = n

        for i in range(4):
            if n >= max_con_pair:
                break
            x = pu[i, 0]
            y = pu[i, 1]
            if nl == 0 and nf != 0:
                if (x < -lx or x > lx) and (y < -ly or y > ly):
                    continue
            elif x < -lx or x > lx or y < -ly or y > ly:
                continue

            c1 = wp.float32(0)

            for j in range(2):
                if pu[i, j] < -s[j]:
                    c1 += (pu[i, j] + s[j]) * (pu[i, j] + s[j])
                elif pu[i, j] > s[j]:
                    c1 += (pu[i, j] - s[j]) * (pu[i, j] - s[j])

            c1 += pu[i, 2] * innorm * pu[i, 2] * innorm

            if pu[i, 2] > 0 and c1 > margin * margin:
                continue

            tmp_p = wp.vec3(pu[i, 0], pu[i, 1], 0.0)

            for j in range(2):
                if pu[i, j] < -s[j]:
                    tmp_p[j] = -s[j] * 0.5
                elif pu[i, j] > s[j]:
                    tmp_p[j] = +s[j] * 0.5

            tmp_p += pu[i]
            points[n] = tmp_p * 0.5

            depth[n] = wp.sqrt(c1) * wp.where(pu[i, 2] < 0, -1.0, 1.0)
            n += 1

        # Set up contact data for all points
        rw = box1_in.rot @ wp.transpose(rotmore)
        pw = box1_in.pos
        normal = wp.where(inv, -1.0, 1.0) * rw @ rnorm

    # Assign body indices
    if box2_in.bid < 0:
        bid_A = box2_in.bid
        bid_B = box1_in.bid
        normal = -normal
    else:
        bid_A = box1_in.bid
        bid_B = box2_in.bid

    # Generate a contact frame
    frame = make_contact_frame_znorm(normal)

    # Increment the active contact counter
    mcio = wp.atomic_add(contacts_model_num_out, 0, n)
    wcio = wp.atomic_add(contacts_world_num_out, wid_in, n)
    nc = wp.min(wp.min(model_max_contacts_in - mcio, world_max_contacts_in - wcio), n)

    # Add generated contacts data to the output arrays
    for i in range(nc):
        points[i, 2] += hz
        pos = rw @ points[i] + pw
        mcid = mcio + i
        # This collider computes the contact point in the middle, and thus to get the
        # per-geom contact we need to offset the contact point by the penetration depth
        pos_A = pos + normal * wp.abs(depth[i])
        pos_B = pos - normal * wp.abs(depth[i])
        # Store contact data
        contact_wid_out[mcid] = wid_in
        contact_cid_out[mcid] = wcio + i
        contact_body_A_out[mcid] = vec4f(pos_A[0], pos_A[1], pos_A[2], float32(bid_A))
        contact_body_B_out[mcid] = vec4f(pos_B[0], pos_B[1], pos_B[2], float32(bid_B))
        contact_gapfunc_out[mcid] = vec4f(normal[0], normal[1], normal[2], 2.0 * depth[i])
        contact_frame_out[mcid] = frame
        # TODO: store margin contact_includemargin_out[mcid] = margin_in - gap_in
        contact_material_out[mcid] = vec2f(friction_in, restitution_in)


@wp.func
def box_ellipsoid():
    pass


@wp.func
def ellipsoid_ellipsoid():
    pass


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

    # Retrive the world index
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
    if sid1 == int32(ShapeType.SPHERE.value) and sid2 == int32(ShapeType.SPHERE.value):
        sphere_sphere(
            model_max_contacts,
            world_max_contacts,
            make_sphere(geom_pose_in[gid1], geom_params_in[gid1], gid1, bid1),
            make_sphere(geom_pose_in[gid2], geom_params_in[gid2], gid2, bid2),
            wid,
            float32(0.0),
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
        sphere_cylinder()

    elif sid1 == int32(ShapeType.SPHERE.value) and sid2 == int32(ShapeType.CONE.value):
        sphere_cone()

    elif sid1 == int32(ShapeType.SPHERE.value) and sid2 == int32(ShapeType.CAPSULE.value):
        sphere_capsule()

    elif sid1 == int32(ShapeType.SPHERE.value) and sid2 == int32(ShapeType.BOX.value):
        sphere_box(
            model_max_contacts,
            world_max_contacts,
            make_sphere(geom_pose_in[gid1], geom_params_in[gid1], gid1, bid1),
            make_box(geom_pose_in[gid2], geom_params_in[gid2], gid2, bid2),
            wid,
            float32(0.0),
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
        capsule_capsule()

    elif sid1 == int32(ShapeType.CAPSULE.value) and sid2 == int32(ShapeType.BOX.value):
        capsule_box()

    elif sid1 == int32(ShapeType.CAPSULE.value) and sid2 == int32(ShapeType.ELLIPSOID.value):
        capsule_ellipsoid()

    elif sid1 == int32(ShapeType.BOX.value) and sid2 == int32(ShapeType.BOX.value):
        box_box(
            model_max_contacts,
            world_max_contacts,
            make_box(geom_pose_in[gid1], geom_params_in[gid1], gid1, bid1),
            make_box(geom_pose_in[gid2], geom_params_in[gid2], gid2, bid2),
            wid,
            float32(0.0),
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


###
# Kernel Launcher
###


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
