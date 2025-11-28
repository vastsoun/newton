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
Provides a specialization of Newton's unified collision-detection pipeline for Kamino.

This module provides interfaces and data-conversion specializations for Kamino that wraps
the broad-phase and narrow-phase of Newton's CollisionPipelineUnified, writing generated
contacts data directly into Kamino's respective format.
"""

import warp as wp
from warp.context import Devicelike

# Newton
from ....geometry.broad_phase_nxn import BroadPhaseAllPairs, BroadPhaseExplicit
from ....geometry.broad_phase_sap import BroadPhaseSAP
from ....geometry.collision_core import compute_tight_aabb_from_support
from ....geometry.contact_data import ContactData
from ....geometry.narrow_phase import NarrowPhase
from ....geometry.support_function import GenericShapeData, SupportMapDataProvider, pack_mesh_ptr
from ....geometry.types import GeoType
from ....sim.collide_unified import BroadPhaseMode

# Kamino
from ..core.builder import ModelBuilder
from ..core.geometry import update_collision_geometries_state
from ..core.model import Model, ModelData
from ..core.shapes import ShapeType
from ..core.types import float32, int32, mat33f, transformf, vec2f, vec2i, vec4f
from ..geometry.collisions import make_collision_pairs
from ..geometry.contacts import Contacts
from ..geometry.math import make_contact_frame_znorm

###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Types
###


@wp.struct
class ContactWriterDataKamino:
    """Contact writer data for writing contacts directly in Kamino format."""

    # Contact limits
    contact_max: int
    world_max_contacts: wp.array(dtype=int32)

    # Geometry information arrays
    geom_bid: wp.array(dtype=int32)  # Body ID for each geometry
    geom_wid: wp.array(dtype=int32)  # World ID for each geometry
    geom_mid: wp.array(dtype=int32)  # Material ID for each geometry

    # Material properties (indexed by material pair)
    material_friction: wp.array(dtype=float32)
    material_restitution: wp.array(dtype=float32)

    # Per-shape contact margin
    geom_contact_margin: wp.array(dtype=float32)

    # Output arrays (Kamino Contacts format)
    contacts_model_num: wp.array(dtype=int32)
    contacts_world_num: wp.array(dtype=int32)
    contact_wid: wp.array(dtype=int32)
    contact_cid: wp.array(dtype=int32)
    contact_body_A: wp.array(dtype=vec4f)
    contact_body_B: wp.array(dtype=vec4f)
    contact_gapfunc: wp.array(dtype=vec4f)
    contact_frame: wp.array(dtype=mat33f)
    contact_material: wp.array(dtype=vec2f)


###
# Functions
###


@wp.func
def kamino_write_contact_unified(
    contact_data: ContactData,
    writer_data: ContactWriterDataKamino,
):
    """
    Write a contact to Kamino-compatible output arrays.

    This function is used as a custom contact writer for NarrowPhase.launch_custom_write().
    It converts ContactData from the narrow phase directly to Kamino's contact format.

    Args:
        contact_data: ContactData struct from narrow phase containing contact information
        writer_data: ContactWriterDataKamino struct containing output arrays
    """
    total_separation_needed = (
        contact_data.radius_eff_a + contact_data.radius_eff_b + contact_data.thickness_a + contact_data.thickness_b
    )

    # Normalize contact normal
    contact_normal_a_to_b = wp.normalize(contact_data.contact_normal_a_to_b)

    # Compute contact points on each shape
    a_contact_world = contact_data.contact_point_center - contact_normal_a_to_b * (
        0.5 * contact_data.contact_distance + contact_data.radius_eff_a
    )
    b_contact_world = contact_data.contact_point_center + contact_normal_a_to_b * (
        0.5 * contact_data.contact_distance + contact_data.radius_eff_b
    )

    # Calculate penetration distance
    diff = b_contact_world - a_contact_world
    distance = wp.dot(diff, contact_normal_a_to_b)
    d = distance - total_separation_needed

    # Use per-shape contact margin (max of both shapes)
    margin_a = writer_data.geom_contact_margin[contact_data.shape_a]
    margin_b = writer_data.geom_contact_margin[contact_data.shape_b]
    margin = wp.max(margin_a, margin_b)

    # Only write contact if within margin
    if d < margin:
        # Get body and world IDs
        bid_a = writer_data.geom_bid[contact_data.shape_a]
        bid_b = writer_data.geom_bid[contact_data.shape_b]
        wid = writer_data.geom_wid[contact_data.shape_a]  # Assume both geoms are in same world

        # Get world max contacts for this world
        world_max = writer_data.world_max_contacts[wid]

        # Atomically increment contact counts
        mcid = wp.atomic_add(writer_data.contacts_model_num, 0, 1)
        wcid = wp.atomic_add(writer_data.contacts_world_num, wid, 1)

        if mcid < writer_data.contact_max and wcid < world_max:
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

            # Get material properties from material pair
            mid_a = writer_data.geom_mid[contact_data.shape_a]
            mid_b = writer_data.geom_mid[contact_data.shape_b]
            # Use average of material properties (simplified approach)
            # In a full implementation, you'd look up the material pair
            friction = float32(0.5) * (writer_data.material_friction[mid_a] + writer_data.material_friction[mid_b])
            restitution = float32(0.5) * (
                writer_data.material_restitution[mid_a] + writer_data.material_restitution[mid_b]
            )

            # Store contact data in Kamino format
            writer_data.contact_wid[mcid] = wid
            writer_data.contact_cid[mcid] = wcid
            writer_data.contact_body_A[mcid] = vec4f(pos_A[0], pos_A[1], pos_A[2], float32(bid_A))
            writer_data.contact_body_B[mcid] = vec4f(pos_B[0], pos_B[1], pos_B[2], float32(bid_B))
            writer_data.contact_gapfunc[mcid] = vec4f(normal[0], normal[1], normal[2], d)
            writer_data.contact_frame[mcid] = make_contact_frame_znorm(normal)
            writer_data.contact_material[mcid] = vec2f(friction, restitution)
        else:
            # Rollback the atomic add if we exceeded limits
            wp.atomic_sub(writer_data.contacts_model_num, 0, 1)
            wp.atomic_sub(writer_data.contacts_world_num, wid, 1)


###
# Kernels
###


@wp.kernel
def _kamino_compute_shape_aabbs(
    geom_pose: wp.array(dtype=transformf),
    geom_sid: wp.array(dtype=int32),
    geom_params: wp.array(dtype=vec4f),
    geom_ptr: wp.array(dtype=wp.uint64),
    geom_contact_margin: wp.array(dtype=float32),
    geom_collision_radius: wp.array(dtype=float32),
    # outputs
    aabb_lower: wp.array(dtype=wp.vec3),
    aabb_upper: wp.array(dtype=wp.vec3),
):
    """Compute axis-aligned bounding boxes for each Kamino geometry in world space.

    Uses support function for most shapes. Infinite planes and meshes use bounding sphere fallback.
    AABBs are enlarged by per-shape contact margin for contact detection.

    Converts Kamino shape parameters to Newton's scale format for AABB computation.
    """
    shape_id = wp.tid()

    # Get world transform (already computed in Kamino's cgeoms.pose)
    X_ws = geom_pose[shape_id]
    pos = wp.transform_get_translation(X_ws)
    orientation = wp.transform_get_rotation(X_ws)

    # Get shape type and params
    sid = geom_sid[shape_id]
    params = geom_params[shape_id]

    # Convert Kamino ShapeType to Newton GeoType and transform params to Newton scale
    geo_type = int(GeoType.BOX)
    scale = wp.vec3(params[0], params[1], params[2])

    if sid == int32(ShapeType.SPHERE.value):
        geo_type = int(GeoType.SPHERE)
        scale = wp.vec3(params[0], 0.0, 0.0)
    elif sid == int32(ShapeType.BOX.value):
        geo_type = int(GeoType.BOX)
        scale = wp.vec3(params[0] * 0.5, params[1] * 0.5, params[2] * 0.5)
    elif sid == int32(ShapeType.CAPSULE.value):
        geo_type = int(GeoType.CAPSULE)
        scale = wp.vec3(params[0], params[1] * 0.5, 0.0)
    elif sid == int32(ShapeType.CYLINDER.value):
        geo_type = int(GeoType.CYLINDER)
        scale = wp.vec3(params[0], params[1] * 0.5, 0.0)
    elif sid == int32(ShapeType.CONE.value):
        geo_type = int(GeoType.CONE)
        scale = wp.vec3(params[0], params[1] * 0.5, 0.0)
    elif sid == int32(ShapeType.ELLIPSOID.value):
        geo_type = int(GeoType.ELLIPSOID)
        scale = wp.vec3(params[0], params[1], params[2])
    elif sid == int32(ShapeType.PLANE.value):
        geo_type = int(GeoType.PLANE)
        scale = wp.vec3(0.0, 0.0, 0.0)  # Infinite plane

    # Enlarge AABB by per-shape contact margin for contact detection
    contact_margin = geom_contact_margin[shape_id]
    margin_vec = wp.vec3(contact_margin, contact_margin, contact_margin)

    # Check if this is an infinite plane or mesh - use bounding sphere fallback
    is_infinite_plane = (geo_type == int(GeoType.PLANE)) and (scale[0] == 0.0 and scale[1] == 0.0)
    is_mesh = geo_type == int(GeoType.MESH)
    is_sdf = geo_type == int(GeoType.SDF)

    if is_infinite_plane or is_mesh or is_sdf:
        # Use conservative bounding sphere approach
        radius = geom_collision_radius[shape_id]
        half_extents = wp.vec3(radius, radius, radius)
        aabb_lower[shape_id] = pos - half_extents - margin_vec
        aabb_upper[shape_id] = pos + half_extents + margin_vec
    else:
        # Use support function to compute tight AABB
        shape_data = GenericShapeData()
        shape_data.shape_type = geo_type
        shape_data.scale = scale
        shape_data.auxiliary = wp.vec3(0.0, 0.0, 0.0)

        # For CONVEX_MESH, pack the mesh pointer
        if geo_type == int(GeoType.CONVEX_MESH):
            shape_data.auxiliary = pack_mesh_ptr(geom_ptr[shape_id])

        data_provider = SupportMapDataProvider()

        # Compute tight AABB using helper function
        aabb_min_world, aabb_max_world = compute_tight_aabb_from_support(shape_data, orientation, pos, data_provider)

        aabb_lower[shape_id] = aabb_min_world - margin_vec
        aabb_upper[shape_id] = aabb_max_world + margin_vec


@wp.kernel
def _kamino_prepare_geom_data_kernel(
    geom_pose: wp.array(dtype=transformf),
    geom_sid: wp.array(dtype=int32),
    geom_params: wp.array(dtype=vec4f),
    # Outputs
    geom_data: wp.array(dtype=wp.vec4),  # scale xyz, thickness w
    geom_transform: wp.array(dtype=wp.transform),  # world space transform
    geom_type: wp.array(dtype=int32),  # Newton GeoType
):
    """Prepare geometry data arrays for NarrowPhase API from Kamino format.

    Converts Kamino shape parameters to Newton's scale format:
    - Kamino uses full dimensions, Newton uses half-extents for box
    - Kamino uses full height, Newton uses half-height for capsule/cylinder/cone
    - Plane has different semantics (Kamino: normal+distance, Newton: half-width/length)
    """
    idx = wp.tid()

    # Get Kamino shape type and params
    sid = geom_sid[idx]
    params = geom_params[idx]
    thickness = float32(0.0)

    # Convert Kamino ShapeType to Newton GeoType and transform params to Newton scale
    # Newton scale format: (see support_function.py GenericShapeData)
    #   BOX: half-extents (x, y, z)
    #   SPHERE: radius in x
    #   CAPSULE: radius in x, half-height in y
    #   CYLINDER: radius in x, half-height in y
    #   CONE: radius in x, half-height in y
    #   ELLIPSOID: semi-axes (x, y, z)
    #   PLANE: half-width in x, half-length in y

    geo_type = int32(GeoType.BOX)
    scale = wp.vec3(params[0], params[1], params[2])

    if sid == int32(ShapeType.SPHERE.value):
        # Kamino: (radius, 0, 0, 0) -> Newton: (radius, ?, ?)
        geo_type = int32(GeoType.SPHERE)
        scale = wp.vec3(params[0], 0.0, 0.0)

    elif sid == int32(ShapeType.BOX.value):
        # Kamino: (depth, width, height) full size -> Newton: half-extents
        geo_type = int32(GeoType.BOX)
        scale = wp.vec3(params[0] * 0.5, params[1] * 0.5, params[2] * 0.5)

    elif sid == int32(ShapeType.CAPSULE.value):
        # Kamino: (radius, height) full height -> Newton: (radius, half-height, ?)
        geo_type = int32(GeoType.CAPSULE)
        scale = wp.vec3(params[0], params[1] * 0.5, 0.0)

    elif sid == int32(ShapeType.CYLINDER.value):
        # Kamino: (radius, height) full height -> Newton: (radius, half-height, ?)
        geo_type = int32(GeoType.CYLINDER)
        scale = wp.vec3(params[0], params[1] * 0.5, 0.0)

    elif sid == int32(ShapeType.CONE.value):
        # Kamino: (radius, height) full height -> Newton: (radius, half-height, ?)
        geo_type = int32(GeoType.CONE)
        scale = wp.vec3(params[0], params[1] * 0.5, 0.0)

    elif sid == int32(ShapeType.ELLIPSOID.value):
        # Kamino: (a, b, c) semi-axes -> Newton: same
        geo_type = int32(GeoType.ELLIPSOID)
        scale = wp.vec3(params[0], params[1], params[2])

    elif sid == int32(ShapeType.PLANE.value):
        # Kamino: (normal_x, normal_y, normal_z, distance) infinite plane
        # Newton: (half_width, half_length, ?) finite plane
        # For infinite plane, use (0, 0, ?) to signal infinite
        geo_type = int32(GeoType.PLANE)
        scale = wp.vec3(0.0, 0.0, 0.0)  # Infinite plane

    geom_type[idx] = geo_type
    geom_data[idx] = wp.vec4(scale[0], scale[1], scale[2], thickness)

    # World space transform is already computed in Kamino
    geom_transform[idx] = geom_pose[idx]


###
# Interfaces
###


class CollisionPipelineUnifiedKamino:
    """
    Unified collision pipeline for Kamino using Newton's broad phase and narrow phase.

    This pipeline uses the same broad phase algorithms (NXN, SAP, EXPLICIT) and narrow phase
    (NarrowPhase with GJK/MPR) as Newton's CollisionPipelineUnified, but writes contacts
    directly in Kamino's format using a custom contact writer.

    This is an alternative to the existing KaminoCollisionPipeline and primitive_narrowphase
    which use Kamino's own collision detection kernels.
    """

    def __init__(
        self,
        builder: ModelBuilder,
        broadphase: BroadPhaseMode = BroadPhaseMode.EXPLICIT,
        max_contacts_per_pair: int = 10,
        default_contact_margin: float = 1e-3,
        default_friction: float = 0.7,
        default_restitution: float = 0.0,
        device: Devicelike = None,
    ):
        """
        Initialize CollisionPipelineUnifiedKamino.

        Args:
            builder: Kamino ModelBuilder (used to extract collision pair information)
            broad_phase_mode: Broad phase algorithm to use (NXN, SAP, or EXPLICIT)
            max_contacts_per_pair: Maximum contacts per collision pair
            default_contact_margin: Default contact margin for collision detection
            default_friction: Default friction coefficient
            default_restitution: Default restitution coefficient
            device: Device to allocate buffers on
        """
        self._device = device
        self._broadphase = broadphase
        self.default_contact_margin = default_contact_margin
        self.default_friction = default_friction
        self.default_restitution = default_restitution

        # Get geometry count from builder
        num_geoms = sum(builder._worlds[i].num_collision_geoms for i in range(builder.num_worlds))
        num_worlds = builder.num_worlds
        self.num_geoms = num_geoms
        self.num_worlds = num_worlds

        # Compute maximum possible pairs
        self.shape_pairs_max = (num_geoms * (num_geoms - 1)) // 2
        self.max_contacts = self.shape_pairs_max * max_contacts_per_pair

        # Build shape pairs for EXPLICIT mode
        if broadphase == BroadPhaseMode.EXPLICIT:
            _, model_nxn_geom_pair, _, _ = make_collision_pairs(builder)
            self.shape_pairs_filtered = wp.array(model_nxn_geom_pair, dtype=vec2i, device=device)
            self.shape_pairs_max = len(model_nxn_geom_pair)
            self.max_contacts = self.shape_pairs_max * max_contacts_per_pair
        else:
            self.shape_pairs_filtered = None

        # Build world index array for NXN/SAP modes
        geom_wid_list = []
        for wid in range(num_worlds):
            ncg = builder._worlds[wid].num_collision_geoms
            geom_wid_list.extend([wid] * ncg)

        # Build collision group array for NXN/SAP modes
        # Newton's broad phase uses a simpler collision group system than Kamino's bitmask approach
        # For NXN/SAP, we use a simple group ID (1 = collide with all, 0 = no collision)
        # The actual filtering based on Kamino's group/collides bitmasks is done in EXPLICIT mode
        # For NXN/SAP, we allow all pairs and let the narrow phase handle the actual collision
        geom_collision_group_list = [1] * num_geoms  # All geometries can collide

        with wp.ScopedDevice(device):
            self.geom_wid = wp.array(geom_wid_list, dtype=int32, device=device)
            self.geom_collision_group = wp.array(geom_collision_group_list, dtype=int32, device=device)

            # Allocate buffers for broad phase
            self.broad_phase_pair_count = wp.zeros(1, dtype=wp.int32, device=device)
            self.broad_phase_shape_pairs = wp.zeros(self.shape_pairs_max, dtype=wp.vec2i, device=device)
            self.shape_aabb_lower = wp.zeros(num_geoms, dtype=wp.vec3, device=device)
            self.shape_aabb_upper = wp.zeros(num_geoms, dtype=wp.vec3, device=device)

            # Allocate buffers for narrow phase input
            self.geom_data = wp.zeros(num_geoms, dtype=wp.vec4, device=device)
            self.geom_transform = wp.zeros(num_geoms, dtype=wp.transform, device=device)
            self.geom_type = wp.zeros(num_geoms, dtype=int32, device=device)
            self.geom_contact_margin = wp.full(num_geoms, default_contact_margin, dtype=float32, device=device)
            self.geom_collision_radius = wp.zeros(num_geoms, dtype=float32, device=device)

            # Default material arrays (will be updated from model)
            self.material_friction = wp.full(1, default_friction, dtype=float32, device=device)
            self.material_restitution = wp.full(1, default_restitution, dtype=float32, device=device)

        # Initialize broad phase
        if self._broadphase == BroadPhaseMode.NXN:
            self.nxn_broadphase = BroadPhaseAllPairs(self.geom_wid, shape_flags=None, device=device)
            self.sap_broadphase = None
            self.explicit_broadphase = None
        elif self._broadphase == BroadPhaseMode.SAP:
            self.sap_broadphase = BroadPhaseSAP(self.geom_wid, shape_flags=None, device=device)
            self.nxn_broadphase = None
            self.explicit_broadphase = None
        else:  # EXPLICIT
            self.explicit_broadphase = BroadPhaseExplicit()
            self.nxn_broadphase = None
            self.sap_broadphase = None

        # Initialize narrow phase with custom Kamino contact writer
        self.narrow_phase = NarrowPhase(
            max_candidate_pairs=self.shape_pairs_max,
            max_triangle_pairs=1000000,
            device=device,
            shape_aabb_lower=self.shape_aabb_lower,
            shape_aabb_upper=self.shape_aabb_upper,
            contact_writer_warp_func=kamino_write_contact_unified,
        )

    ###
    # Properties
    ###

    @property
    def device(self) -> Devicelike:
        """Returns the Warp device the pipeline operates on."""
        return self._device

    ###
    # Operations
    ###

    def collide(self, model: Model, state: ModelData, contacts: Contacts):
        """
        Run the unified collision pipeline.

        Args:
            model: Kamino Model
            state: Current model state (ModelData)
            contacts: Output contacts container (will be cleared and populated)
        """
        # Update geometry poses from body states
        update_collision_geometries_state(state.bodies.q_i, model.cgeoms, state.cgeoms)

        # Clear contacts
        contacts.clear()

        # Clear broad phase counter
        self.broad_phase_pair_count.zero_()

        # Compute AABBs for all geometries
        wp.launch(
            kernel=_kamino_compute_shape_aabbs,
            dim=self.num_geoms,
            inputs=[
                state.cgeoms.pose,
                model.cgeoms.sid,
                model.cgeoms.params,
                model.cgeoms.ptr,
                self.geom_contact_margin,
                self.geom_collision_radius,
            ],
            outputs=[
                self.shape_aabb_lower,
                self.shape_aabb_upper,
            ],
            device=self._device,
        )

        # Run broad phase
        if self._broadphase == BroadPhaseMode.NXN:
            self.nxn_broadphase.launch(
                self.shape_aabb_lower,
                self.shape_aabb_upper,
                None,  # AABBs are pre-expanded
                self.geom_collision_group,  # Simple collision groups (all = 1)
                self.geom_wid,
                self.num_geoms,
                self.broad_phase_shape_pairs,
                self.broad_phase_pair_count,
                device=self._device,
            )
        elif self._broadphase == BroadPhaseMode.SAP:
            self.sap_broadphase.launch(
                self.shape_aabb_lower,
                self.shape_aabb_upper,
                None,  # AABBs are pre-expanded
                self.geom_collision_group,  # Simple collision groups (all = 1)
                self.geom_wid,
                self.num_geoms,
                self.broad_phase_shape_pairs,
                self.broad_phase_pair_count,
                device=self._device,
            )
        else:  # EXPLICIT
            self.explicit_broadphase.launch(
                self.shape_aabb_lower,
                self.shape_aabb_upper,
                None,  # AABBs are pre-expanded
                self.shape_pairs_filtered,
                len(self.shape_pairs_filtered),
                self.broad_phase_shape_pairs,
                self.broad_phase_pair_count,
                device=self._device,
            )

        # Prepare geometry data for narrow phase
        wp.launch(
            kernel=_kamino_prepare_geom_data_kernel,
            dim=self.num_geoms,
            inputs=[
                state.cgeoms.pose,
                model.cgeoms.sid,
                model.cgeoms.params,
            ],
            outputs=[
                self.geom_data,
                self.geom_transform,
                self.geom_type,
            ],
            device=self._device,
        )

        # TODO: Why does this need to happen in every collide call?
        # Create writer data struct
        writer_data = ContactWriterDataKamino()
        writer_data.contact_max = contacts.num_model_max_contacts
        writer_data.world_max_contacts = contacts.world_max_contacts
        writer_data.geom_bid = model.cgeoms.bid
        writer_data.geom_wid = model.cgeoms.wid
        writer_data.geom_mid = model.cgeoms.mid
        writer_data.material_friction = self.material_friction
        writer_data.material_restitution = self.material_restitution
        writer_data.geom_contact_margin = self.geom_contact_margin
        writer_data.contacts_model_num = contacts.model_num_contacts
        writer_data.contacts_world_num = contacts.world_num_contacts
        writer_data.contact_wid = contacts.wid
        writer_data.contact_cid = contacts.cid
        writer_data.contact_body_A = contacts.body_A
        writer_data.contact_body_B = contacts.body_B
        writer_data.contact_gapfunc = contacts.gapfunc
        writer_data.contact_frame = contacts.frame
        writer_data.contact_material = contacts.material

        # Run narrow phase with custom Kamino contact writer
        self.narrow_phase.launch_custom_write(
            candidate_pair=self.broad_phase_shape_pairs,
            num_candidate_pair=self.broad_phase_pair_count,
            shape_types=self.geom_type,
            shape_data=self.geom_data,
            shape_transform=self.geom_transform,
            shape_source=model.cgeoms.ptr,
            shape_contact_margin=self.geom_contact_margin,
            shape_collision_radius=self.geom_collision_radius,
            writer_data=writer_data,
            device=self._device,
        )

    # TODO: Rework this and remove copy if possible
    def update_materials(self, model: Model):
        """
        Update material arrays from the model's material pairs.

        Args:
            model (Model): The Kamino model containing material-pair properties.
        """
        if model.mpairs is not None and model.mpairs.num_pairs > 0:
            # Reallocate material arrays if needed
            if self.material_friction.shape[0] != model.mpairs.num_pairs:
                self.material_friction = wp.zeros(model.mpairs.num_pairs, dtype=float32, device=self._device)
                self.material_restitution = wp.zeros(model.mpairs.num_pairs, dtype=float32, device=self._device)

            # Copy material properties
            wp.copy(self.material_friction, model.mpairs.dynamic_friction)
            wp.copy(self.material_restitution, model.mpairs.restitution)
