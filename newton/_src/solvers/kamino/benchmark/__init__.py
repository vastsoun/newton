# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Benchmark utilities for the Kamino solver package.

This subpackage exposes:

* :class:`SolverSetup` — turn-key construction of a Newton model/state/solver
  trio for benchmark sweeps.
* :class:`PhysicsMetrics` and the ``compute_*`` helpers — closed-form contact
  constraint residuals and per-world summaries computed directly from the
  Newton :class:`Contacts` container.
* :class:`PhysicsMetricsLogger` — on-device rolling/bounded history logger
  for the per-world summary fields produced by
  :func:`compute_per_world_contact_constraint_summary` and
  :func:`compute_per_world_joint_constraint_summary`.
"""

from .logging import PhysicsMetricsLogger
from .metrics import (
    ConstraintMetrics,
    PhysicsMetrics,
    compute_contact_constraint_metrics,
    compute_contact_velocities,
    compute_joint_constraint_metrics,
    compute_per_world_contact_constraint_summary,
    compute_per_world_joint_constraint_summary,
)
from .setup import SolverSetup

###
# Module interface
###

__all__ = [
    "ConstraintMetrics",
    "PhysicsMetrics",
    "PhysicsMetricsLogger",
    "SolverSetup",
    "compute_contact_constraint_metrics",
    "compute_contact_velocities",
    "compute_joint_constraint_metrics",
    "compute_per_world_contact_constraint_summary",
    "compute_per_world_joint_constraint_summary",
]
