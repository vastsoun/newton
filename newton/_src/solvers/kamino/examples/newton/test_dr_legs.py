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

###########################################################################
# Example Robot DR Legs
#
# Shows how to simulate DR Legs with multiple worlds using SolverKamino.
#
# Command: python -m newton.examples robot_dr_legs --num-worlds 16
#
###########################################################################

import time

import warp as wp

import newton
from newton._src.solvers.kamino._src.utils import logger as msg

num_worlds = 1024
device = wp.get_device()

# Create a single-robot model builder and register the Kamino-specific custom attributes
robot_builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
newton.solvers.SolverKamino.register_custom_attributes(robot_builder)
robot_builder.default_shape_cfg.margin = 1e-6
robot_builder.default_shape_cfg.gap = 0.01

tic = time.time()

# Load the DR Legs USD and add it to the builder
asset_path = newton.utils.download_asset("disneyresearch")
asset_file = str(asset_path / "dr_legs/usd" / "dr_legs_with_meshes_and_boxes.usda")
robot_builder.add_usd(
    asset_file,
    joint_ordering=None,
    force_show_colliders=True,
    force_position_velocity_actuation=True,
    collapse_fixed_joints=False,  # TODO @cavemor: Fails when True, investigate (doesn't have fixed joints)
    enable_self_collisions=False,
    hide_collision_shapes=True,
)

toc = time.time()
msg.warning(f"add_usd(): {(toc - tic) * 1000:.2f} ms")

tic = time.time()

# Create the multi-world model by duplicating the single-robot
# builder for the specified number of worlds
builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
for _ in range(num_worlds):
    builder.add_world(robot_builder)

toc = time.time()
msg.warning(f"add_world() x {num_worlds}: {(toc - tic) * 1000:.2f} ms")

# Add a global ground plane applied to all worlds
builder.add_ground_plane()

tic = time.time()

# Create the model from the builder
model = builder.finalize(skip_validation_joints=True)

toc = time.time()
msg.warning(f"finalize(): {(toc - tic) * 1000:.2f} ms")

model.shape_margin.fill_(1e-6)
model.shape_gap.fill_(0.01)

tic = time.time()

# Create the Kamino solver for the given model
solver = newton.solvers.SolverKamino(model)

toc = time.time()
msg.warning(f"SolverKamino(): {(toc - tic) * 1000:.2f} ms")
