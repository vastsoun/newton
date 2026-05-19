# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
TODO
"""

import dataclasses
import os

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.kamino._src.metrics import SolutionMetricsLogger, SolutionMetricsNewton
from newton._src.solvers.kamino._src.utils import logger as msg

###
# Scaffolding
###


###
# Helpers
###


###
# Runtime
###



if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    example.viewer._paused = True  # Start paused to inspect the initial configuration

    # If only a single-world is created, set initial
    # camera position for better view of the system
    if hasattr(example.viewer, "set_camera"):
        camera_pos = wp.vec3(5.0, 0.0, 2.0)
        pitch = -15.0
        yaw = -180.0
        example.viewer.set_camera(camera_pos, pitch, yaw)

    newton.examples.run(example, args)
    example.test_final()
