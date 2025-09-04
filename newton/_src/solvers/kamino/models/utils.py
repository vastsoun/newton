###########################################################################
# KAMINO: MODELS: MODEL BUILDER UTILITIES
###########################################################################

from typing import List
from newton._src.solvers.kamino.core import ModelBuilder
from newton._src.solvers.kamino.models.builders import (
    build_box_on_plane,
    build_box_pendulum,
    build_boxes_hinged,
    build_boxes_nunchaku,
    build_boxes_fourbar,
)


###
# Module interface
###

__all__ = [
    "make_single_builder",
    "make_homogeneous_builder",
    "make_heterogeneous_builder",
]


###
# Builder utilities
###

def make_single_builder(build_func=build_boxes_nunchaku) -> tuple[ModelBuilder, List[int], List[int]]:
    num_jcts = []
    num_bodies = []
    builder = ModelBuilder()
    build_func(builder)
    num_jcts.append(builder.world.num_joint_cts)
    num_bodies.append(builder.world.num_bodies)
    return builder, num_bodies, num_jcts


def make_homogeneous_builder(num_worlds: int, build_func=build_boxes_nunchaku) -> tuple[ModelBuilder, List[int], List[int]]:
    num_jcts = []
    num_bodies = []
    builder = ModelBuilder()
    build_func(builder)
    num_jcts.append(builder.world.num_joint_cts)
    num_bodies.append(builder.world.num_bodies)
    for _ in range(num_worlds - 1):
        other = ModelBuilder()
        build_func(other)
        num_jcts.append(other.world.num_joint_cts)
        num_bodies.append(other.world.num_bodies)
        builder.add_builder(other)
    return builder, num_bodies, num_jcts


def make_heterogeneous_builder() -> tuple[ModelBuilder, List[int], List[int]]:
    num_jcts = []
    num_bodies = []
    builder = ModelBuilder()
    build_boxes_fourbar(builder)
    num_jcts.append(builder.world.num_joint_cts)
    num_bodies.append(builder.world.num_bodies)

    other_boxes_nunchaku = ModelBuilder()
    build_boxes_nunchaku(other_boxes_nunchaku)
    num_jcts.append(other_boxes_nunchaku.world.num_joint_cts)
    num_bodies.append(other_boxes_nunchaku.world.num_bodies)
    builder.add_builder(other_boxes_nunchaku)

    other_boxes_hinged = ModelBuilder()
    build_boxes_hinged(other_boxes_hinged)
    num_jcts.append(other_boxes_hinged.world.num_joint_cts)
    num_bodies.append(other_boxes_hinged.world.num_bodies)
    builder.add_builder(other_boxes_hinged)

    other_box_pendulum = ModelBuilder()
    build_box_pendulum(other_box_pendulum)
    num_jcts.append(other_box_pendulum.world.num_joint_cts)
    num_bodies.append(other_box_pendulum.world.num_bodies)
    builder.add_builder(other_box_pendulum)

    builder_box_on_floor = ModelBuilder()
    build_box_on_plane(builder_box_on_floor)
    num_jcts.append(builder_box_on_floor.world.num_joint_cts)
    num_bodies.append(builder_box_on_floor.world.num_bodies)
    builder.add_builder(builder_box_on_floor)

    return builder, num_bodies, num_jcts
