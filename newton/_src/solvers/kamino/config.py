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
Defines the :class:`SolverKamino` class, providing a physics backend for
simulating constrained multi-body systems for arbitrary mechanical assemblies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import warp as wp

from ...sim import Model, ModelBuilder

###
# Module interface
###

__all__ = [
    "CollisionDetectorConfig",
    "ConstrainedDynamicsConfig",
    "ForwardKinematicsSolverConfig",
    "PADMMSolverConfig",
]


###
# Types
###


@dataclass
class CollisionDetectorConfig:
    """
    A container to hold configurations for the internal collision detector used for contact generation.
    """

    pipeline: Literal["primitive", "unified"] = "unified"
    """
    The type of collision-detection pipeline to use, either `primitive` or `unified`.\n
    Defaults to `unified`.
    """

    broadphase: Literal["nxn", "sap", "explicit"] = "explicit"
    """
    The broad-phase collision-detection to use (`nxn`, `sap`, or `explicit`).\n
    Defaults to `explicit`.
    """

    bvtype: Literal["aabb", "bs"] = "aabb"
    """
    The type of bounding volume to use in the broad-phase.\n
    Defaults to `aabb`.
    """

    max_contacts: int | None = None
    """
    The maximum number of contacts to generate over the entire model.\n
    Used to compute the total maximum contacts allocated for the model,
    in conjunction with the total number of candidate geom-pairs.\n
    Defaults to `None`, allowing contact allocations to occur according to the model.
    """

    max_contacts_per_world: int | None = None
    """
    The per-world maximum contacts allocation override.\n
    If specified, it will override the per-world maximum number of contacts
    computed according to the candidate geom-pairs represented in the model.\n
    Defaults to `None`, allowing contact allocations to occur according to the model.
    """

    max_contacts_per_pair: int | None = None
    """
    The maximum number of contacts to generate per candidate geom-pair.\n
    Used to compute the total maximum contacts allocated for the model,
    in conjunction with the total number of candidate geom-pairs.\n
    Defaults to `None`, allowing contact allocations to occur according to the model.
    """

    max_triangle_pairs: int | None = None
    """
    The maximum number of triangle-primitive shape pairs to consider in the narrow-phase.\n
    Used only when the model contains triangle meshes or heightfields.\n
    Defaults to `None`, allowing contact allocations to occur according to the model.
    """

    default_gap: float = 0.0
    """
    The default detection gap [m] applied as a floor to per-geometry gaps.\n
    Defaults to `0.0`.
    """


@dataclass
class ConstrainedDynamicsConfig:
    """
    A container to hold configurations for the construction of the constrained forward dynamics problem.
    """

    alpha: float = 0.01
    """
    Global default Baumgarte stabilization parameter for bilateral joint constraints.\n
    Must be in range `[0, 1.0]`.\n
    Defaults to `0.01`.
    """

    beta: float = 0.01
    """
    Global default Baumgarte stabilization parameter for unilateral joint-limit constraints.\n
    Must be in range `[0, 1.0]`.\n
    Defaults to `0.01`.
    """

    gamma: float = 0.01
    """
    Global default Baumgarte stabilization parameter for unilateral contact constraints.\n
    Must be in range `[0, 1.0]`.\n
    Defaults to `0.01`.
    """

    delta: float = 1.0e-6
    """
    Contact penetration margin used for unilateral contact constraints.\n
    Must be non-negative.\n
    Defaults to `1.0e-6`.
    """

    preconditioning: bool = True
    """
    Set to `True` to enable preconditioning of the dual problem.\n
    Defaults to `True`.
    """


@dataclass
class PADMMSolverConfig:
    """
    A container to hold configurations for the PADMM forward dynamics solver.
    """

    primal_tolerance: float = 1e-6
    """
    The target tolerance on the total primal residual `r_primal`.\n
    Must be greater than zero. Defaults to `1e-6`.
    """

    dual_tolerance: float = 1e-6
    """
    The target tolerance on the total dual residual `r_dual`.\n
    Must be greater than zero. Defaults to `1e-6`.
    """

    compl_tolerance: float = 1e-6
    """
    The target tolerance on the total complementarity residual `r_compl`.\n
    Must be greater than zero. Defaults to `1e-6`.
    """

    restart_tolerance: float = 0.999
    """
    The tolerance on the total combined primal-dual residual `r_comb`,
    for determining when gradient acceleration should be restarted.\n
    Must be greater than zero. Defaults to `0.999`.
    """

    eta: float = 1e-5
    """
    The proximal regularization parameter.\n
    Must be greater than zero. Defaults to `1e-5`.
    """

    rho_0: float = 1.0
    """
    The initial value of the ALM penalty parameter.\n
    Must be greater than zero. Defaults to `1.0`.
    """

    rho_min: float = 1e-5
    """
    The lower-bound applied to the ALM penalty parameter.\n
    Used to ensure numerical stability when adaptive penalty updates are used.\n
    Must be greater than zero. Defaults to `1e-5`.
    """

    a_0: float = 1.0
    """
    The initial value of the acceleration parameter.\n
    Must be greater than zero. Defaults to `1.0`.
    """

    alpha: float = 10.0
    """
    The primal-dual residual threshold used to determine when penalty updates are needed.
    Must be greater than one. Defaults to `10.0`.
    """

    tau: float = 1.5
    """
    The factor by which the ALM penalty is increased/decreased when
    the primal-dual residual ratios exceed the threshold `alpha`.\n
    Must be greater than `1.0`. Defaults to `1.5`.
    """

    max_iterations: int = 200
    """
    The maximum number of solver iterations.\n
    Must be greater than zero. Defaults to `200`.
    """

    penalty_update_freq: int = 1
    """
    The permitted frequency of penalty updates.\n
    If zero, no updates are performed. Otherwise, updates are performed every
    `penalty_update_freq` iterations. Defaults to `1`.
    """

    penalty_update_method: Literal["fixed", "balanced"] = "fixed"
    """
    The penalty update method used to adapt the penalty parameter.\n
    Defaults to `fixed`. See :class:`PADMMPenaltyUpdate` for details.
    """

    linear_solver_tolerance: float = 0.0
    """
    The default absolute tolerance for the iterative linear solver.\n
    When positive, the iterative solver's atol is initialized to this value
    at the start of each ADMM solve.\n
    When zero, the iterative solver's own tolerance is left unchanged.\n
    Must be non-negative. Defaults to `0.0`.
    """

    linear_solver_tolerance_ratio: float = 0.0
    """
    The ratio used to adapt the iterative linear solver tolerance from the ADMM primal residual.\n
    When positive, the linear solver absolute tolerance is set to
    `ratio * ||r_primal||_2` at each ADMM iteration.\n
    When zero, the linear solver tolerance is not adapted (fixed tolerance).\n
    Must be non-negative. Defaults to `0.0`.
    """

    @classmethod
    def register_custom_attributes(cls, builder: ModelBuilder) -> None:
        """
        Register custom attributes for this config.

        Args:
            builder (ModelBuilder): The model builder to register the custom attributes to.
        """

        # Register KaminoSceneAPI attributes so the USD importer will store them on the model
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="padmm_primal_tolerance",
                frequency=Model.AttributeFrequency.ONCE,
                assignment=Model.AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=1e-6,
                namespace="kamino",
                usd_attribute_name="newton:kamino:padmm:primalTolerance",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="padmm_dual_tolerance",
                frequency=Model.AttributeFrequency.ONCE,
                assignment=Model.AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=1e-6,
                namespace="kamino",
                usd_attribute_name="newton:kamino:padmm:dualTolerance",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="padmm_complementarity_tolerance",
                frequency=Model.AttributeFrequency.ONCE,
                assignment=Model.AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=1e-6,
                namespace="kamino",
                usd_attribute_name="newton:kamino:padmm:complementarityTolerance",
            )
        )

        # Separately register `newton:maxSolverIterations` from `KaminoSceneAPI` so we have access
        # to it through the model.
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="max_solver_iterations",
                frequency=Model.AttributeFrequency.ONCE,
                assignment=Model.AttributeAssignment.MODEL,
                dtype=wp.int32,
                default=-1,
                namespace="kamino",
                usd_attribute_name="newton:maxSolverIterations",
            )
        )


@dataclass
class ForwardKinematicsSolverConfig:
    """
    A container to hold configurations for the Gauss-Newton forward kinematics solver used for state resets.
    """

    preconditioner: Literal["none", "jacobi_diagonal", "jacobi_block_diagonal"] = "jacobi_block_diagonal"
    """
    Preconditioner to use for the Conjugate Gradient solver if sparsity is enabled
    Changing this setting after the solver's initialization leads to undefined behavior.
    Defaults to `jacobi_block_diagonal`.
    """

    max_newton_iterations: int = 30
    """
    Maximal number of Gauss-Newton iterations.
    Changes to this setting after the solver's initialization will have no effect.
    Defaults to `30`.
    """

    max_line_search_iterations: int = 20
    """
    Maximal line search iterations in the inner loop.
    Changes to this setting after the solver's initialization will have no effect.
    Defaults to `20`.
    """

    tolerance: float = 1e-6
    """
    Maximal absolute kinematic constraint value that is acceptable at the solution.
    Changes to this setting after the solver's initialization will have no effect.
    Defaults to `1e-6`.
    """

    TILE_SIZE_CTS: int = 8
    """
    Tile size for kernels along the dimension of kinematic constraints.
    Changes to this setting after the solver's initialization will have no effect.
    Defaults to `8`.
    """

    TILE_SIZE_VRS: int = 8
    """
    Tile size for kernels along the dimension of rigid body pose variables.
    Changes to this setting after the solver's initialization will have no effect.
    Defaults to `8`.
    """

    use_sparsity: bool = False
    """
    Whether to use sparse Jacobian and solver; otherwise, dense versions are used.
    Changes to this setting after the solver's initialization lead to undefined behavior.
    Defaults to `False`.
    """

    use_adaptive_cg_tolerance: bool = True
    """
    Whether to use an adaptive tolerance strategy for the Conjugate Gradient solver if sparsity
    is enabled, which reduces the number of CG iterations in most cases.
    Changes to this setting after graph capture will have no effect.
    Defaults to `True`.
    """

    reset_state: bool = True
    """
    Whether to reset the state to initial states, to use as initial guess.
    Changes to this setting after graph capture will have no effect.
    Defaults to `True`.
    """
