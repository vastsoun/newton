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
# Example Renderer Settings
#
# Demonstrates configurable renderer lighting and shadow settings.
# No simulation — static shapes are placed on a ground plane so you
# can tweak diffuse/specular scales, shadow radius, shadow extents,
# and the spotlight toggle via command-line arguments.
#
# Command: python -m newton.examples renderer_settings
# Matte:   python -m newton.examples renderer_settings --diffuse-scale 0.8 --specular-scale 0.2
#
###########################################################################

import warp as wp

import newton
import newton.examples


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim_time = 0.0
        self.fps = 60
        self.frame_dt = 1.0 / self.fps

        builder = newton.ModelBuilder()
        builder.add_ground_plane()

        # Add some shapes at different positions
        for i, y in enumerate([-2, 0, 2]):
            b = builder.add_body(xform=wp.transform((0, y, 0.5), wp.quat_identity()))
            if i == 0:
                builder.add_shape_sphere(b, radius=0.5)
            elif i == 1:
                builder.add_shape_box(b, hx=0.5, hy=0.35, hz=0.5)
            else:
                builder.add_shape_capsule(b, radius=0.3, half_height=0.5)

        self.model = builder.finalize()
        self.state = self.model.state()

        self.viewer.set_model(self.model)

        # Apply lighting settings from command-line arguments
        renderer = getattr(self.viewer, "renderer", None)
        if renderer is not None:
            if hasattr(args, "diffuse_scale") and args.diffuse_scale is not None:
                renderer.diffuse_scale = args.diffuse_scale
            if hasattr(args, "specular_scale") and args.specular_scale is not None:
                renderer.specular_scale = args.specular_scale
            if hasattr(args, "shadow_radius") and args.shadow_radius is not None:
                renderer.shadow_radius = args.shadow_radius
            if hasattr(args, "shadow_extents") and args.shadow_extents is not None:
                renderer.shadow_extents = args.shadow_extents
            if hasattr(args, "spotlight") and args.spotlight is not None:
                renderer.spotlight_enabled = args.spotlight

            print(f"diffuse_scale:    {renderer.diffuse_scale}")
            print(f"specular_scale:   {renderer.specular_scale}")
            print(f"shadow_radius:    {renderer.shadow_radius}")
            print(f"shadow_extents:   {renderer.shadow_extents}")
            print(f"spotlight_enabled: {renderer.spotlight_enabled}")

        self.viewer.set_camera(pos=wp.vec3(5.0, 0.0, 2.0), pitch=-15.0, yaw=-180.0)

    def step(self):
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        self.viewer.end_frame()

    def test_final(self):
        pass


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--diffuse-scale", type=float, default=None, help="Diffuse light scale")
    parser.add_argument("--specular-scale", type=float, default=None, help="Specular light scale")
    parser.add_argument("--shadow-radius", type=float, default=None, help="PCF shadow softness radius")
    parser.add_argument("--shadow-extents", type=float, default=None, help="Shadow map half-size in world units")
    parser.add_argument("--spotlight", type=bool, default=None, help="Use cone spotlight")

    viewer, args = newton.examples.init(parser)

    example = Example(viewer, args)
    newton.examples.run(example, args)
