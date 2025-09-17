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
KAMINO: MODELS: MODEL CONSTRUCTION UTILITIES & ASSETS
"""

from . import builders

__all__ = [
    "builders",
    "get_examples_usd_assets_path",
    "get_primitives_usd_assets_path",
    "get_tests_usd_assets_path",
]

###
# Asset path utilities
###


def get_examples_usd_assets_path() -> str:
    import os

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets/examples/usd")
    if not os.path.exists(path):
        raise FileNotFoundError(f"The USD assets path for example models does not exist: {path}")
    return path


def get_primitives_usd_assets_path() -> str:
    import os

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets/primitives")
    if not os.path.exists(path):
        raise FileNotFoundError(f"The USD assets path for primitive models does not exist: {path}")
    return path


def get_tests_usd_assets_path() -> str:
    import os

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets/tests")
    if not os.path.exists(path):
        raise FileNotFoundError(f"The USD assets path for testing models does not exist: {path}")
    return path
