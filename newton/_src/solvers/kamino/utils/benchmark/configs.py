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


from ...linalg.linear import LinearSolverNameToType
from ...solver_kamino import SolverKaminoSettings
from ...solvers.padmm import PADMMPenaltyUpdate, PADMMWarmStartMode
from ...solvers.warmstart import WarmstarterContacts

###
# Module interface
###

__all__ = [
    "make_benchmark_configs",
    "make_solver_config_default",
    "make_solver_config_dense_lltb_fast_dr_legs",
    "make_solver_config_sparse_cr_adaptiv_rho_balanced_dr_legs",
    "make_solver_config_sparse_cr_fast_dr_legs",
]


###
# Solver configurations
###


def make_solver_config_default() -> tuple[str, SolverKaminoSettings]:
    # ------------------------------------------------------------------------------
    name = "Default"
    # ------------------------------------------------------------------------------
    settings = SolverKaminoSettings()
    # ------------------------------------------------------------------------------
    # Constraint stabilization
    settings.problem.alpha = 0.1
    # ------------------------------------------------------------------------------
    # Jacobian representation
    settings.sparse = False
    settings.sparse_jacobian = False
    # ------------------------------------------------------------------------------
    # Linear system solver
    settings.linear_solver_type = LinearSolverNameToType["LLTB"]
    settings.linear_solver_kwargs = {}
    # ------------------------------------------------------------------------------
    # PADMM
    settings.padmm.max_iterations = 200
    settings.padmm.primal_tolerance = 1e-6
    settings.padmm.dual_tolerance = 1e-6
    settings.padmm.compl_tolerance = 1e-6
    settings.padmm.restart_tolerance = 0.999
    settings.padmm.eta = 1e-5
    settings.padmm.rho_0 = 1.0
    settings.padmm.rho_min = 1e-5
    settings.padmm.penalty_update_method = PADMMPenaltyUpdate.FIXED
    settings.padmm.penalty_update_freq = 1
    settings.use_solver_acceleration = True
    settings.avoid_graph_conditionals = False
    # ------------------------------------------------------------------------------
    # Warm-starting
    settings.warmstart_mode = PADMMWarmStartMode.CONTAINERS
    settings.contact_warmstart_method = WarmstarterContacts.Method.GEOM_PAIR_NET_FORCE
    # ------------------------------------------------------------------------------
    return name, settings


def make_solver_config_dense_lltb_fast_dr_legs() -> tuple[str, SolverKaminoSettings]:
    # ------------------------------------------------------------------------------
    name = "Dense LLTB Fast DR Legs"
    # ------------------------------------------------------------------------------
    settings = SolverKaminoSettings()
    # ------------------------------------------------------------------------------
    # Constraint stabilization
    settings.problem.alpha = 0.1
    # ------------------------------------------------------------------------------
    # Jacobian representation
    settings.sparse = False
    settings.sparse_jacobian = False
    # ------------------------------------------------------------------------------
    # Linear system solver
    settings.linear_solver_type = LinearSolverNameToType["LLTB"]
    settings.linear_solver_kwargs = {}
    # ------------------------------------------------------------------------------
    # PADMM
    settings.padmm.max_iterations = 200
    settings.padmm.primal_tolerance = 1e-4
    settings.padmm.dual_tolerance = 1e-4
    settings.padmm.compl_tolerance = 1e-4
    settings.padmm.restart_tolerance = 0.999
    settings.padmm.eta = 1e-5
    settings.padmm.rho_0 = 0.05
    settings.padmm.rho_min = 1e-5
    settings.padmm.penalty_update_method = PADMMPenaltyUpdate.FIXED
    settings.padmm.penalty_update_freq = 1
    settings.use_solver_acceleration = True
    settings.avoid_graph_conditionals = False
    # ------------------------------------------------------------------------------
    # Warm-starting
    settings.warmstart_mode = PADMMWarmStartMode.CONTAINERS
    settings.contact_warmstart_method = WarmstarterContacts.Method.GEOM_PAIR_NET_FORCE
    # ------------------------------------------------------------------------------
    return name, settings


# TODO @cavemor: PLEASE ADD YOUR BEST CONFIGS HERE
def make_solver_config_sparse_cr_fast_dr_legs() -> tuple[str, SolverKaminoSettings]:
    # ------------------------------------------------------------------------------
    name = "Sparse CR Fast DR Legs"
    # ------------------------------------------------------------------------------
    settings = SolverKaminoSettings()
    # ------------------------------------------------------------------------------
    # Constraint stabilization
    settings.problem.alpha = 0.1
    # ------------------------------------------------------------------------------
    # Jacobian representation
    settings.sparse = True
    settings.sparse_jacobian = True
    # ------------------------------------------------------------------------------
    # Linear system solver
    settings.linear_solver_type = LinearSolverNameToType["CR"]
    settings.linear_solver_kwargs = {"maxiter": 9}
    # ------------------------------------------------------------------------------
    # PADMM
    settings.padmm.max_iterations = 100
    settings.padmm.primal_tolerance = 1e-4
    settings.padmm.dual_tolerance = 1e-4
    settings.padmm.compl_tolerance = 1e-4
    settings.padmm.restart_tolerance = 0.999
    settings.padmm.eta = 1e-5
    settings.padmm.rho_0 = 0.05
    settings.padmm.rho_min = 0.01
    settings.padmm.penalty_update_method = PADMMPenaltyUpdate.FIXED
    settings.padmm.penalty_update_freq = 1
    settings.use_solver_acceleration = True
    settings.avoid_graph_conditionals = False
    # ------------------------------------------------------------------------------
    # Warm-starting
    settings.warmstart_mode = PADMMWarmStartMode.CONTAINERS
    settings.contact_warmstart_method = WarmstarterContacts.Method.GEOM_PAIR_NET_FORCE
    # ------------------------------------------------------------------------------
    return name, settings


def make_solver_config_sparse_cr_adaptiv_rho_balanced_dr_legs() -> tuple[str, SolverKaminoSettings]:
    # ------------------------------------------------------------------------------
    name = "Sparse CR Adaptive Rho Balanced DR Legs"
    # ------------------------------------------------------------------------------
    settings = SolverKaminoSettings()
    # ------------------------------------------------------------------------------
    # Constraint stabilization
    settings.problem.alpha = 0.1
    # ------------------------------------------------------------------------------
    # Jacobian representation
    settings.sparse = True
    settings.sparse_jacobian = True
    # ------------------------------------------------------------------------------
    # Linear system solver
    settings.linear_solver_type = LinearSolverNameToType["CR"]
    settings.linear_solver_kwargs = {"maxiter": 9}
    # ------------------------------------------------------------------------------
    # PADMM
    settings.padmm.max_iterations = 100
    settings.padmm.primal_tolerance = 1e-4
    settings.padmm.dual_tolerance = 1e-4
    settings.padmm.compl_tolerance = 1e-4
    settings.padmm.restart_tolerance = 0.999
    settings.padmm.eta = 1e-5
    settings.padmm.rho_0 = 0.05
    settings.padmm.rho_min = 0.01
    settings.padmm.penalty_update_method = PADMMPenaltyUpdate.BALANCED
    settings.padmm.penalty_update_freq = 10
    settings.use_solver_acceleration = True
    settings.avoid_graph_conditionals = False
    # ------------------------------------------------------------------------------
    # Warm-starting
    settings.warmstart_mode = PADMMWarmStartMode.CONTAINERS
    settings.contact_warmstart_method = WarmstarterContacts.Method.GEOM_PAIR_NET_FORCE
    # ------------------------------------------------------------------------------
    return name, settings


###
# Utilities
###


def make_benchmark_configs() -> dict[str, SolverKaminoSettings]:
    generators = [
        make_solver_config_default,
        make_solver_config_dense_lltb_fast_dr_legs,
        make_solver_config_sparse_cr_fast_dr_legs,
        make_solver_config_sparse_cr_adaptiv_rho_balanced_dr_legs,
    ]
    solver_configs: dict[str, SolverKaminoSettings] = {}
    for gen in generators:
        name, config = gen()
        solver_configs[name] = config
    return solver_configs
