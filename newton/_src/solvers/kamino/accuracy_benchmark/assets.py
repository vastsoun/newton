# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Central override point for the paper-benchmark USD asset root.

Assets used by the ``example_benchmark_robot_*`` scripts (Iron Man, BDX, Olaf,
DR Legs) are not fetchable through ``newton.utils.download_asset`` — they live
on a local shared drive. This module exposes a single ``paper_assets_root()``
helper that returns the root directory; individual scripts store only the
relative path to their USD asset under this root.

Override via the ``NEWTON_KAMINO_PAPER_ASSETS_ROOT`` environment variable when
running on a machine where the default path is not mounted, or set
``paper_assets_root.default`` from Python before the first call.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["paper_assets_root", "resolve_asset"]

_DEFAULT_ROOT = "D:/gmaloisel/Documents/Quick access shortcuts/Kamino - Data/kamino-assets-disney"
_ENV_VAR = "NEWTON_KAMINO_PAPER_ASSETS_ROOT"


def paper_assets_root() -> Path:
    """Return the root directory that holds the paper-benchmark USD assets.

    Precedence: environment variable ``NEWTON_KAMINO_PAPER_ASSETS_ROOT`` first,
    then the module-level default. The returned path is not required to exist
    at call time; existence is checked lazily by :func:`resolve_asset`.
    """
    return Path(os.environ.get(_ENV_VAR, _DEFAULT_ROOT))


def resolve_asset(relative_path: str) -> str:
    """Resolve ``relative_path`` under :func:`paper_assets_root` and check existence.

    Returns the absolute path as a string (USD loaders in Newton expect ``str``
    rather than ``Path``). Raises ``FileNotFoundError`` with a message pointing
    at both the relative path and the env-var override.
    """
    full = paper_assets_root() / relative_path
    if not full.exists():
        raise FileNotFoundError(
            f"Paper asset not found: {full}\n"
            f"  relative path: {relative_path}\n"
            f"  root: {paper_assets_root()} (override via ${_ENV_VAR})"
        )
    return str(full)
