###########################################################################
# KAMINO: Collision Detection: GJK Operations
###########################################################################

from __future__ import annotations

import warp as wp

from newton._src.solvers.kamino.core.types import int32, float32, vec3f, mat43f

from newton._src.solvers.kamino.core.shapes import (
    SHAPE_EMPTY,
    SHAPE_SPHERE,
    SHAPE_CYLINDER,
    SHAPE_CONE,
    SHAPE_CAPSULE,
    SHAPE_BOX,
    SHAPE_ELLIPSOID,
    SHAPE_PLANE,
    SHAPE_CONVEX,
    SHAPE_MESH,
    SHAPE_SDF
)

from newton._src.solvers.kamino.geometry.types import FLOAT_MIN, FLOAT_MAX
from newton._src.solvers.kamino.geometry.math import gjk_normalize, orthonormal


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Functions
###

@wp.func
def gjk_support_geom(
   geom: Geom,
   shapeid: int32,
   dir: vec3f,
   verts: wp.array(dtype=vec3f)
):
    local_dir = wp.transpose(geom.rot) @ dir

    if shapeid == SHAPE_SPHERE:
        support_pt = geom.pos + geom.size[0] * dir

    elif shapeid == SHAPE_BOX:
        res = wp.cw_mul(wp.sign(local_dir), geom.size)
        support_pt = geom.rot @ res + geom.pos

    elif shapeid == SHAPE_CAPSULE:
        res = local_dir * geom.size[0]
        # add cylinder contribution
        res[2] += wp.sign(local_dir[2]) * geom.size[1]
        support_pt = geom.rot @ res + geom.pos

    elif shapeid == SHAPE_ELLIPSOID:
        res = wp.cw_mul(local_dir, geom.size)
        res = wp.normalize(res)
        # transform to ellipsoid
        res = wp.cw_mul(res, geom.size)
        support_pt = geom.rot @ res + geom.pos

    elif shapeid == SHAPE_CYLINDER:
        res = vec3f(0.0, 0.0, 0.0)
        # set result in XY plane: support on circle
        d = wp.sqrt(wp.dot(local_dir, local_dir))
        if d > float(1e-8):
            scl = geom.size[0] / d
            res[0] = local_dir[0] * scl
            res[1] = local_dir[1] * scl
        # set result in Z direction
        res[2] = wp.sign(local_dir[2]) * geom.size[1]
        support_pt = geom.rot @ res + geom.pos

    elif shapeid == SHAPE_CONVEX:
        max_dist = float(FLOAT_MIN)
        # exhaustive search over all vertices
        # TODO(team): consider hill-climb over graph data
        for i in range(geom.vertnum):
            vert = verts[geom.vertadr + i]
            dist = wp.dot(vert, local_dir)
            if dist > max_dist:
                max_dist = dist
                support_pt = vert
        support_pt = geom.rot @ support_pt + geom.pos

    # TODO: what are these values?
    return wp.dot(support_pt, dir), support_pt


@wp.func
def gjk_support(
    # In:
    geom1: Geom,
    geom2: Geom,
    geomtype1: int,
    geomtype2: int,
    dir: vec3f,
    verts: wp.array(dtype=vec3f),
):
    # Returns the distance between support points on two geoms, and the support point.
    # Negative distance means objects are not intersecting along direction `dir`.
    # Positive distance means objects are intersecting along the given direction `dir`.
    dist1, s1 = gjk_support_geom(geom1, geomtype1, dir, verts)
    dist2, s2 = gjk_support_geom(geom2, geomtype2, -dir, verts)
    support_pt = s1 - s2
    return dist1 + dist2, support_pt


def get_gjk(
  geomtype1: int,
  geomtype2: int,
  gjk_iterations: int,
):
    # determines if two objects intersect, returns simplex and normal
    @wp.func
    def _gjk(
        # Model:
        mesh_vert: wp.array(dtype=vec3f),
        # In:
        geom1: Geom,
        geom2: Geom,
    ):
        dir = vec3f(0.0, 0.0, 1.0)
        dir_n = -dir
        depth = float(FLOAT_MAX)

        dist_max, simplex0 = gjk_support(geom1, geom2, geomtype1, geomtype2, dir, mesh_vert)
        dist_min, simplex1 = gjk_support(geom1, geom2, geomtype1, geomtype2, dir_n, mesh_vert)

        if dist_max < dist_min:
            depth = dist_max
            normal = dir
        else:
            depth = dist_min
            normal = dir_n

        sd = simplex0 - simplex1
        dir = orthonormal(sd)

        dist_max, simplex3 = gjk_support(geom1, geom2, geomtype1, geomtype2, dir, mesh_vert)

        # Initialize a 2-simplex with simplex[2]==simplex[1]. This ensures the
        # correct winding order for face normals defined below. Face 0 and face 3
        # are degenerate, and face 1 and 2 have opposing normals.
        simplex = mat43f()
        simplex[0] = simplex0
        simplex[1] = simplex1
        simplex[2] = simplex[1]
        simplex[3] = simplex3

        if dist_max < depth:
            depth = dist_max
            normal = dir
        if dist_min < depth:
            depth = dist_min
            normal = dir_n

        plane = mat43f()
        for _ in range(gjk_iterations):
            # winding orders: plane[0] ccw, plane[1] cw, plane[2] ccw, plane[3] cw
            plane[0] = wp.cross(simplex[3] - simplex[2], simplex[1] - simplex[2])
            plane[1] = wp.cross(simplex[3] - simplex[0], simplex[2] - simplex[0])
            plane[2] = wp.cross(simplex[3] - simplex[1], simplex[0] - simplex[1])
            plane[3] = wp.cross(simplex[2] - simplex[0], simplex[1] - simplex[0])

            # Compute distance of each face halfspace to the origin. If dplane<0, then the
            # origin is outside the halfspace. If dplane>0 then the origin is inside
            # the halfspace defined by the face plane.

            dplane = wp.vec4(float(FLOAT_MAX))

            plane0, p0 = gjk_normalize(plane[0])
            plane1, p1 = gjk_normalize(plane[1])
            plane2, p2 = gjk_normalize(plane[2])
            plane3, p3 = gjk_normalize(plane[3])

            plane[0] = plane0
            plane[1] = plane1
            plane[2] = plane2
            plane[3] = plane3

            if p0:
                dplane[0] = wp.dot(plane[0], simplex[2])

            if p1:
                dplane[1] = wp.dot(plane[1], simplex[0])

            if p2:
                dplane[2] = wp.dot(plane[2], simplex[1])

            if p3:
                dplane[3] = wp.dot(plane[3], simplex[0])

            # pick plane normal with minimum distance to the origin
            i1 = wp.where(dplane[0] < dplane[1], 0, 1)
            i2 = wp.where(dplane[2] < dplane[3], 2, 3)
            index = wp.where(dplane[i1] < dplane[i2], i1, i2)

            if dplane[index] > 0.0:
                # origin is inside the simplex, objects are intersecting
                break

            # add new support point to the simplex
            dist, simplex_i = gjk_support(geom1, geom2, geomtype1, geomtype2, plane[index], mesh_vert)
            simplex[index] = simplex_i

            if dist < depth:
                depth = dist
                normal = plane[index]

            # preserve winding order of the simplex faces
            index1 = (index + 1) & 3
            index2 = (index + 2) & 3
            swap = simplex[index1]
            simplex[index1] = simplex[index2]
            simplex[index2] = swap

            if dist < 0.0:
                break  # objects are likely non-intersecting

        return simplex, normal

    return _gjk
