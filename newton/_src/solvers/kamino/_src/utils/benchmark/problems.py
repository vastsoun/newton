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

import os
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import numpy as np
import warp as wp

import newton.utils

from ...core.builder import ModelBuilderKamino
from ...models import basics
from ...models.builders.utils import (
    add_ground_box,
    make_homogeneous_builder,
    set_uniform_body_pose_offset,
)
from ...utils import logger as msg
from ..io.usd import USDImporter

###
# Module interface
###

__all__ = [
    "BenchmarkProblemNameToConfigFn",
    "CameraConfig",
    "ControlConfig",
    "ProblemConfig",
    "ProblemDimensions",
    "ProblemSet",
    "make_benchmark_problems",
    "save_problem_dimensions_to_hdf5",
]

###
# Types
###


@dataclass
class ProblemDimensions:
    num_body_dofs: int = -1
    num_joint_dofs: int = -1
    min_delassus_dim: int = -1
    max_delassus_dim: int = -1
    num_worlds: int = -1


@dataclass
class ControlConfig:
    disable_controller: bool = False
    decimation: int | list[int] | None = None
    scale: float | list[float] | None = None


# TODO: Use set_camera_lookat params instead
@dataclass
class CameraConfig:
    position: tuple[float, float, float]
    pitch: float
    yaw: float


ProblemConfig = tuple[Callable[[int], ModelBuilderKamino], ControlConfig | None, CameraConfig | None]
"""
Defines the configurations for a single benchmark problem.

This contains:
- A callable taking the number of worlds, and returning a model builder that constructs the simulation
  worlds for the benchmark problem.
- Optional control configurations for perturbing the benchmark problem.
- Optional camera configurations for visualizing the benchmark problem.
"""


ProblemSet = dict[str, ProblemConfig]
"""
Defines a set of benchmark problems, indexed by a string name.

Each entry contains the configurations for a single
benchmark problem, including the model builder and
optional camera configurations for visualization.
"""


###
# Problem Definitions
###


def make_benchmark_problem_fourbar(
    gravity: bool = True,
    ground: bool = True,
) -> ProblemConfig:
    def builder_fn(num_worlds: int):
        builder = make_homogeneous_builder(
            num_worlds=num_worlds,
            build_fn=basics.build_boxes_fourbar,
            ground=ground,
        )
        for w in range(num_worlds):
            builder.gravity[w].enabled = gravity
        return builder

    control = ControlConfig(decimation=20, scale=10.0, disable_controller=False)
    camera = CameraConfig(
        position=(-0.2, -0.5, 0.1),
        pitch=-5.0,
        yaw=70.0,
    )
    return builder_fn, control, camera


def make_benchmark_problem_dr_legs(
    gravity: bool = True,
    ground: bool = True,
) -> ProblemConfig:
    # Set the path to the external USD assets
    asset_path = newton.utils.download_asset("disneyresearch")
    asset_file = str(asset_path / "dr_legs/usd" / "dr_legs_with_boxes.usda")

    def builder_fn(num_worlds: int):
        # Create a model builder from the imported USD
        msg.notif("Constructing builder from imported USD ...")
        importer = USDImporter()
        builder: ModelBuilderKamino = make_homogeneous_builder(
            num_worlds=num_worlds,
            build_fn=importer.import_from,
            load_static_geometry=True,
            source=asset_file,
            load_drive_dynamics=True,
            use_angular_drive_scaling=True,
        )
        # Offset the model to place it above the ground
        # NOTE: The USD model is centered at the origin
        offset = wp.transformf(0.0, 0.0, 0.265, 0.0, 0.0, 0.0, 1.0)
        set_uniform_body_pose_offset(builder=builder, offset=offset)
        # Add a static collision geometry for the plane
        if ground:
            for w in range(num_worlds):
                add_ground_box(builder, world_index=w)
        # Set gravity
        for w in range(builder.num_worlds):
            builder.gravity[w].enabled = gravity
        return builder

    # Set control configurations
    control = ControlConfig(decimation=20, scale=5.0, disable_controller=False)
    # Set the camera configuration for better visualization of the system
    camera = CameraConfig(
        position=(0.6, 0.6, 0.3),
        pitch=-10.0,
        yaw=225.0,
    )
    return builder_fn, control, camera


def make_benchmark_problem_olaf(
    gravity: bool = True,
    ground: bool = True,
) -> ProblemConfig:
    # Set the path to the external USD assets
    model_path = "D:/gmaloisel/Documents/Quick access shortcuts/Kamino - Data/kamino-assets-disney/usda/Olaf/olaf_articulated.usda"

    def builder_fn(num_worlds: int):
        # Create a model builder from the imported USD
        msg.notif("Constructing builder from imported USD ...")
        importer = USDImporter()
        builder: ModelBuilderKamino = make_homogeneous_builder(
            num_worlds=num_worlds,
            build_fn=importer.import_from,
            load_static_geometry=True,
            source=model_path,
            load_drive_dynamics=True,
            use_angular_drive_scaling=True,
        )
        # Offset the model to place it above the ground
        # NOTE: The USD model is centered at the origin
        offset = wp.transformf(0.0, 0.0, 0.265, 0.0, 0.0, 0.0, 1.0)
        set_uniform_body_pose_offset(builder=builder, offset=offset)
        # Add a static collision geometry for the plane
        if ground:
            for w in range(num_worlds):
                add_ground_box(builder, world_index=w)
        # Set gravity
        for w in range(builder.num_worlds):
            builder.gravity[w].enabled = gravity
        print(f"Num actuated coords: {builder.num_actuated_joint_coords}")
        return builder

    # Set control configurations
    control = ControlConfig(decimation=20, scale=150.0, disable_controller=False)
    # Set the camera configuration for better visualization of the system
    camera = CameraConfig(
        position=(1.0, 1.0, 0.7),
        pitch=-7.0,
        yaw=225.0,
    )
    return builder_fn, control, camera


def make_benchmark_problem_bdx_walking(
    gravity: bool = True,
    ground: bool = True,
) -> ProblemConfig:
    # Set the path to the external USD assets
    model_path = (
        "D:/gmaloisel/Documents/Quick access shortcuts/Kamino - Data/kamino-assets-disney/usda/bdx/bipedal.usda"
    )

    def builder_fn(num_worlds: int):
        # Create a model builder from the imported USD
        msg.notif("Constructing builder from imported USD ...")
        importer = USDImporter()
        builder: ModelBuilderKamino = make_homogeneous_builder(
            num_worlds=num_worlds,
            build_fn=importer.import_from,
            load_static_geometry=True,
            source=model_path,
            load_drive_dynamics=True,
            use_angular_drive_scaling=True,
        )
        # Offset the model to place it above the ground
        # NOTE: The USD model is centered at the origin
        offset = wp.transformf(0.0, 0.0, 0.335, 0.0, 0.0, 0.0, 1.0)
        set_uniform_body_pose_offset(builder=builder, offset=offset)
        # Add a static collision geometry for the plane
        if ground:
            for w in range(num_worlds):
                add_ground_box(builder, world_index=w)
        # Set gravity
        for w in range(builder.num_worlds):
            builder.gravity[w].enabled = gravity
        print(f"Num actuated coords: {builder.num_actuated_joint_coords}")
        return builder

    # Set control configurations
    control = ControlConfig(decimation=20, scale=150.0, disable_controller=False)
    # Set the camera configuration for better visualization of the system
    camera = CameraConfig(
        position=(1.0, 1.0, 0.7),
        pitch=-7.0,
        yaw=225.0,
    )
    return builder_fn, control, camera


def make_benchmark_problem_ragdoll(
    which: int,
    gravity: bool = True,
    ground: bool = True,
) -> ProblemConfig:
    # Set the path to the external USD assets
    EXAMPLE_ASSETS_PATH = "D:/gmaloisel/Code/kamino_dev/newton/newton/_src/solvers/kamino/_src/models/assets/examples"

    # Get path and dimensions based on selected model
    if which == 0:  # BDX (fixed pelvis)
        model_path = os.path.join(EXAMPLE_ASSETS_PATH, "BDX_fixed_pelvis/BDX_fixed_pelvis_robopt.usda")
        ground_height = -0.34118753
        top_height = 0.29422529
        hor_size = 0.630830843423171
    elif which == 1:  # Misha
        model_path = os.path.join(EXAMPLE_ASSETS_PATH, "Misha/Misha_robopt.usda")
        ground_height = -0.046935
        top_height = 0.32719702
        hor_size = 0.5533497180490314
    elif which == 2:  # Gazelle
        model_path = os.path.join(EXAMPLE_ASSETS_PATH, "Gazelle/Gazelle_robopt.usda")
        ground_height = 0.024
        top_height = 1.9647119
        hor_size = 1.311339009370567
    elif which == 3:  # Kansas
        model_path = os.path.join(EXAMPLE_ASSETS_PATH, "Kansas/Kansas_robopt.usda")
        ground_height = -0.58129031
        top_height = 1.78313203
        hor_size = 2.3935657183743704
    elif which == 4:  # Louis
        model_path = os.path.join(EXAMPLE_ASSETS_PATH, "Louis/Louis_robopt.usda")
        ground_height = -0.75961003
        top_height = 1.89627037
        hor_size = 3.3656623436200968
    elif which == 5:  # IronMan
        model_path = os.path.join(EXAMPLE_ASSETS_PATH, "IronMan/IronMan_robopt.usda")
        ground_height = -0.84514405
        top_height = 1.98352743
        hor_size = 1.688288111918465
    elif which == 6:  # A1000
        model_path = os.path.join(EXAMPLE_ASSETS_PATH, "A1000/A1000_robopt.usda")
        ground_height = -0.50427432
        top_height = 1.81120916
        hor_size = 1.9884896789039248
    else:
        raise ValueError("Unsupported model id for ragdoll problem")

    def builder_fn(num_worlds: int):
        # Create a model builder from the imported USD
        msg.notif("Constructing builder from imported USD ...")
        importer = USDImporter()
        builder: ModelBuilderKamino = make_homogeneous_builder(
            num_worlds=num_worlds, build_fn=importer.import_from, load_static_geometry=True, source=model_path
        )

        # Add a static collision geometry for the plane
        if ground:
            for w in range(num_worlds):
                add_ground_box(builder, world_index=w, z_offset=ground_height)
        # Set gravity
        for w in range(builder.num_worlds):
            builder.gravity[w].enabled = gravity
        return builder

    # Set control configurations
    control = ControlConfig(disable_controller=True)

    # Set the camera configuration for better visualization of the system
    camera_height = 0.5 * (ground_height + top_height)
    cam_fov_y = np.radians(45.0)
    model_half_height = 0.5 * (top_height - ground_height)
    target_half_height = model_half_height + 0.5 * hor_size * np.tan(0.5 * cam_fov_y)
    camera_dist = target_half_height / np.tan(0.5 * cam_fov_y)
    camera = CameraConfig(
        position=(0.5 * np.sqrt(2) * camera_dist, 0.5 * np.sqrt(2) * camera_dist, camera_height),
        pitch=0.0,
        yaw=-135.0,
    )

    return builder_fn, control, camera


###
# Problem Set Generator
###

BenchmarkProblemNameToConfigFn: dict[str, Callable[..., ProblemConfig]] = {
    "fourbar": make_benchmark_problem_fourbar,
    "bdx_ragdoll": partial(make_benchmark_problem_ragdoll, which=0),
    "misha": partial(make_benchmark_problem_ragdoll, which=1),
    "dr_legs": make_benchmark_problem_dr_legs,
    "gazelle": partial(make_benchmark_problem_ragdoll, which=2),
    "kansas": partial(make_benchmark_problem_ragdoll, which=3),
    "louis": partial(make_benchmark_problem_ragdoll, which=4),
    "iron_man": partial(make_benchmark_problem_ragdoll, which=5),
    "a1000": partial(make_benchmark_problem_ragdoll, which=6),
    "olaf": make_benchmark_problem_olaf,
    "bdx": make_benchmark_problem_bdx_walking,
}
"""
Defines a mapping from benchmark problem names to their
corresponding problem configuration generator functions.
"""


def make_benchmark_problems(
    names: list[str],
    gravity: bool = True,
    ground: bool = True,
) -> ProblemSet:
    # Ensure that problem names are provided and valid
    if names is None:
        raise ValueError("Problem names must be provided as a list of strings.")

    # Define common generator kwargs for all problems to avoid repetition
    generator_kwargs = {"gravity": gravity, "ground": ground}

    # Generate the problem configurations for each specified problem name
    problems: ProblemSet = {}
    for name in names:
        if name not in BenchmarkProblemNameToConfigFn.keys():
            raise ValueError(
                f"Unsupported problem name: {name}.\nSupported names are: {list(BenchmarkProblemNameToConfigFn.keys())}"
            )

        problems[name] = BenchmarkProblemNameToConfigFn[name](**generator_kwargs)
    return problems


def save_problem_dimensions_to_hdf5(problem_dims: dict[str, ProblemDimensions], datafile):
    for problem_name, dims in problem_dims.items():
        scope = f"Problems/{problem_name}"
        datafile[f"{scope}/num_body_dofs"] = dims.num_body_dofs
        datafile[f"{scope}/num_joint_dofs"] = dims.num_joint_dofs
        datafile[f"{scope}/min_delassus_dim"] = dims.min_delassus_dim
        datafile[f"{scope}/max_delassus_dim"] = dims.max_delassus_dim
        datafile[f"{scope}/num_worlds"] = dims.num_worlds
