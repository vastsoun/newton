########################################################################################################################
# KAMINO: Collision Detection: Narrow-phase operation for geometric primitives
########################################################################################################################

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
from newton._src.geometry.collision_primitive import (
    normalize_with_norm as collision_normalize_with_norm,
    collide_sphere_sphere,
    collide_sphere_box,
    collide_box_box,
)

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


# Use the tested version from collision_primitive.py
normalize_with_norm = collision_normalize_with_norm


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
    # Use the tested collision calculation from collision_primitive.py
    dist, pos, normal = collide_sphere_sphere(sphere1.pos, sphere1.radius, sphere2.pos, sphere2.radius)

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
    # Use the tested collision calculation from collision_primitive.py
    contact_dist, contact_pos, contact_normal = collide_sphere_box(
        sphere_in.pos, sphere_in.radius, box_in.pos, box_in.rot, box_in.size
    )

    if contact_dist > margin:
        return

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
    # Use the tested collision calculation from collision_primitive.py
    contact_dists, contact_positions, contact_normals = collide_box_box(
        box1_in.pos, box1_in.rot, box1_in.size, box2_in.pos, box2_in.rot, box2_in.size
    )

    # Count valid contacts (those with finite distance)
    n_contacts = wp.int32(0)
    for i in range(8):
        if contact_dists[i] != wp.inf and contact_dists[i] <= margin:
            n_contacts += 1

    if n_contacts == 0:
        return

    # Assign body indices
    if box2_in.bid < 0:
        bid_A = box2_in.bid
        bid_B = box1_in.bid
    else:
        bid_A = box1_in.bid
        bid_B = box2_in.bid

    # Increment the active contact counter
    mcio = wp.atomic_add(contacts_model_num_out, 0, n_contacts)
    wcio = wp.atomic_add(contacts_world_num_out, wid_in, n_contacts)
    nc = wp.min(wp.min(model_max_contacts_in - mcio, world_max_contacts_in - wcio), n_contacts)

    # Add generated contacts data to the output arrays
    contact_idx = wp.int32(0)
    for i in range(8):
        if contact_idx >= nc:
            break
        if contact_dists[i] != wp.inf and contact_dists[i] <= margin:
            mcid = mcio + contact_idx

            # Get contact data
            contact_pos = vec3f(contact_positions[i, 0], contact_positions[i, 1], contact_positions[i, 2])
            contact_normal = vec3f(contact_normals[i, 0], contact_normals[i, 1], contact_normals[i, 2])
            contact_dist = contact_dists[i]

            # Adjust normal direction based on body assignment
            if box2_in.bid < 0:
                contact_normal = -contact_normal

            # Generate a contact frame
            frame = make_contact_frame_znorm(contact_normal)

            # This collider computes the contact point in the middle, and thus to get the
            # per-geom contact we need to offset the contact point by the penetration depth
            pos_A = contact_pos + contact_normal * wp.abs(contact_dist) * 0.5
            pos_B = contact_pos - contact_normal * wp.abs(contact_dist) * 0.5

            # Store contact data
            contact_wid_out[mcid] = wid_in
            contact_cid_out[mcid] = wcio + contact_idx
            contact_body_A_out[mcid] = vec4f(pos_A[0], pos_A[1], pos_A[2], float32(bid_A))
            contact_body_B_out[mcid] = vec4f(pos_B[0], pos_B[1], pos_B[2], float32(bid_B))
            contact_gapfunc_out[mcid] = vec4f(contact_normal[0], contact_normal[1], contact_normal[2], contact_dist)
            contact_frame_out[mcid] = frame
            contact_material_out[mcid] = vec2f(friction_in, restitution_in)

            contact_idx += 1


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
