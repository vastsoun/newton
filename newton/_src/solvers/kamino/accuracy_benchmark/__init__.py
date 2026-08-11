# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Cross-solver accuracy benchmark for the Kamino paper experiments.

This subpackage exposes:

* :class:`SolverSetup` and :class:`SetupRunner` — per-solver simulation
  containers and their multi-solver driver.
* :class:`PhysicsMetrics` and the ``compute_*`` helpers — closed-form
  contact / joint constraint residuals computed directly from the Newton
  :class:`State` and :class:`Contacts`.
* :class:`PhysicsMetricsLogger` — on-device rolling/bounded history logger
  for the per-world summary fields, with ``plot_comparison`` /
  ``table_comparison`` classmethods that emit the paper's PDFs and CSV.
"""

from .assets import paper_assets_root, resolve_asset
from .logging import PhysicsMetricsLogger
from .metrics import (
    ConstraintMetrics,
    PhysicsMetrics,
    compute_contact_constraint_metrics,
    compute_joint_constraint_metrics,
    compute_per_world_contact_constraint_summary,
    compute_per_world_joint_constraint_summary,
)
from .setup import MODE_INDEPENDENT, MODE_TIED, MODE_TIED_REFERENCE, SetupRunner, SolverSetup

###
# Module interface
###

__all__ = [
    "MODE_INDEPENDENT",
    "MODE_TIED",
    "MODE_TIED_REFERENCE",
    "ConstraintMetrics",
    "PhysicsMetrics",
    "PhysicsMetricsLogger",
    "SetupRunner",
    "SolverSetup",
    "compute_contact_constraint_metrics",
    "compute_joint_constraint_metrics",
    "compute_per_world_contact_constraint_summary",
    "compute_per_world_joint_constraint_summary",
    "paper_assets_root",
    "resolve_asset",
]
