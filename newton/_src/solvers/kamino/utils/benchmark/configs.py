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

import h5py

from ...linalg.linear import LinearSolverNameToType, LinearSolverTypeToName
from ...solver_kamino import SolverKaminoConfig
from ...solvers.padmm import PADMMPenaltyUpdate
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


def make_solver_config_default() -> tuple[str, SolverKaminoConfig]:
    # ------------------------------------------------------------------------------
    name = "Default"
    # ------------------------------------------------------------------------------
    config = SolverKaminoConfig()
    # ------------------------------------------------------------------------------
    # Constraint stabilization
    config.problem.alpha = 0.1
    # ------------------------------------------------------------------------------
    # Jacobian representation
    config.sparse = False
    config.sparse_jacobian = False
    # ------------------------------------------------------------------------------
    # Linear system solver
    config.linear_solver_type = LinearSolverNameToType["LLTB"]
    config.linear_solver_kwargs = {}
    # ------------------------------------------------------------------------------
    # PADMM
    config.padmm.max_iterations = 200
    config.padmm.primal_tolerance = 1e-6
    config.padmm.dual_tolerance = 1e-6
    config.padmm.compl_tolerance = 1e-6
    config.padmm.restart_tolerance = 0.999
    config.padmm.eta = 1e-5
    config.padmm.rho_0 = 1.0
    config.padmm.rho_min = 1e-5
    config.padmm.penalty_update_method = PADMMPenaltyUpdate.FIXED
    config.padmm.penalty_update_freq = 1
    config.use_solver_acceleration = True
    config.avoid_graph_conditionals = False
    # ------------------------------------------------------------------------------
    # Warm-starting
    config.warmstart_mode = "containers"
    config.contact_warmstart_method = WarmstarterContacts.Method.GEOM_PAIR_NET_FORCE
    # ------------------------------------------------------------------------------
    return name, config


def make_solver_config_dense_lltb_fast_dr_legs() -> tuple[str, SolverKaminoConfig]:
    # ------------------------------------------------------------------------------
    name = "Dense LLTB Fast DR Legs"
    # ------------------------------------------------------------------------------
    config = SolverKaminoConfig()
    # ------------------------------------------------------------------------------
    # Constraint stabilization
    config.problem.alpha = 0.1
    # ------------------------------------------------------------------------------
    # Jacobian representation
    config.sparse = False
    config.sparse_jacobian = False
    # ------------------------------------------------------------------------------
    # Linear system solver
    config.linear_solver_type = LinearSolverNameToType["LLTB"]
    config.linear_solver_kwargs = {}
    # ------------------------------------------------------------------------------
    # PADMM
    config.padmm.max_iterations = 200
    config.padmm.primal_tolerance = 1e-4
    config.padmm.dual_tolerance = 1e-4
    config.padmm.compl_tolerance = 1e-4
    config.padmm.restart_tolerance = 0.999
    config.padmm.eta = 1e-5
    config.padmm.rho_0 = 0.05
    config.padmm.rho_min = 1e-5
    config.padmm.penalty_update_method = PADMMPenaltyUpdate.FIXED
    config.padmm.penalty_update_freq = 1
    config.use_solver_acceleration = True
    config.avoid_graph_conditionals = False
    # ------------------------------------------------------------------------------
    # Warm-starting
    config.warmstart_mode = "containers"
    config.contact_warmstart_method = WarmstarterContacts.Method.GEOM_PAIR_NET_FORCE
    # ------------------------------------------------------------------------------
    return name, config


# TODO @cavemor: PLEASE ADD YOUR BEST CONFIGS HERE
def make_solver_config_sparse_cr_fast_dr_legs() -> tuple[str, SolverKaminoConfig]:
    # ------------------------------------------------------------------------------
    name = "Sparse CR Fast DR Legs"
    # ------------------------------------------------------------------------------
    config = SolverKaminoConfig()
    # ------------------------------------------------------------------------------
    # Constraint stabilization
    config.problem.alpha = 0.1
    # ------------------------------------------------------------------------------
    # Jacobian representation
    config.sparse = True
    config.sparse_jacobian = True
    # ------------------------------------------------------------------------------
    # Linear system solver
    config.linear_solver_type = LinearSolverNameToType["CR"]
    config.linear_solver_kwargs = {"maxiter": 9}
    # ------------------------------------------------------------------------------
    # PADMM
    config.padmm.max_iterations = 100
    config.padmm.primal_tolerance = 1e-4
    config.padmm.dual_tolerance = 1e-4
    config.padmm.compl_tolerance = 1e-4
    config.padmm.restart_tolerance = 0.999
    config.padmm.eta = 1e-5
    config.padmm.rho_0 = 0.05
    config.padmm.rho_min = 0.01
    config.padmm.penalty_update_method = PADMMPenaltyUpdate.FIXED
    config.padmm.penalty_update_freq = 1
    config.use_solver_acceleration = True
    config.avoid_graph_conditionals = False
    # ------------------------------------------------------------------------------
    # Warm-starting
    config.warmstart_mode = "containers"
    config.contact_warmstart_method = WarmstarterContacts.Method.GEOM_PAIR_NET_FORCE
    # ------------------------------------------------------------------------------
    return name, config


def make_solver_config_sparse_cr_adaptiv_rho_balanced_dr_legs() -> tuple[str, SolverKaminoConfig]:
    # ------------------------------------------------------------------------------
    name = "Sparse CR Adaptive Rho Balanced DR Legs"
    # ------------------------------------------------------------------------------
    config = SolverKaminoConfig()
    # ------------------------------------------------------------------------------
    # Constraint stabilization
    config.problem.alpha = 0.1
    # ------------------------------------------------------------------------------
    # Jacobian representation
    config.sparse = True
    config.sparse_jacobian = True
    # ------------------------------------------------------------------------------
    # Linear system solver
    config.linear_solver_type = LinearSolverNameToType["CR"]
    config.linear_solver_kwargs = {"maxiter": 9}
    # ------------------------------------------------------------------------------
    # PADMM
    config.padmm.max_iterations = 100
    config.padmm.primal_tolerance = 1e-4
    config.padmm.dual_tolerance = 1e-4
    config.padmm.compl_tolerance = 1e-4
    config.padmm.restart_tolerance = 0.999
    config.padmm.eta = 1e-5
    config.padmm.rho_0 = 0.05
    config.padmm.rho_min = 0.01
    config.padmm.penalty_update_method = PADMMPenaltyUpdate.BALANCED
    config.padmm.penalty_update_freq = 10
    config.use_solver_acceleration = True
    config.avoid_graph_conditionals = False
    # ------------------------------------------------------------------------------
    # Warm-starting
    config.warmstart_mode = "containers"
    config.contact_warmstart_method = WarmstarterContacts.Method.GEOM_PAIR_NET_FORCE
    # ------------------------------------------------------------------------------
    return name, config


###
# Utilities
###


def make_benchmark_configs() -> dict[str, SolverKaminoConfig]:
    generators = [
        make_solver_config_default,
        make_solver_config_dense_lltb_fast_dr_legs,
        make_solver_config_sparse_cr_fast_dr_legs,
        make_solver_config_sparse_cr_adaptiv_rho_balanced_dr_legs,
    ]
    solver_configs: dict[str, SolverKaminoConfig] = {}
    for gen in generators:
        name, config = gen()
        solver_configs[name] = config
    return solver_configs


###
# Functions
###


def save_solver_configs_to_hdf5(configs: dict[str, SolverKaminoConfig], datafile: h5py.File):
    for config_name, config in configs.items():
        scope = f"Solver/{config_name}"
        # ------------------------------------------------------------------------------
        datafile[f"{scope}/constraints/alpha"] = config.problem.alpha
        datafile[f"{scope}/constraints/beta"] = config.problem.beta
        datafile[f"{scope}/constraints/gamma"] = config.problem.gamma
        datafile[f"{scope}/constraints/delta"] = config.problem.delta
        datafile[f"{scope}/constraints/preconditioning"] = config.problem.preconditioning
        # ------------------------------------------------------------------------------
        datafile[f"{scope}/sparse_solver"] = config.sparse
        datafile[f"{scope}/sparse_jacobian"] = config.sparse_jacobian
        # ------------------------------------------------------------------------------
        datafile[f"{scope}/linear_solver/type"] = str(LinearSolverTypeToName[config.linear_solver_type])
        datafile[f"{scope}/linear_solver/args"] = f"{config.linear_solver_kwargs}"
        # ------------------------------------------------------------------------------
        datafile[f"{scope}/padmm/max_iterations"] = config.padmm.max_iterations
        datafile[f"{scope}/padmm/primal_tolerance"] = config.padmm.primal_tolerance
        datafile[f"{scope}/padmm/dual_tolerance"] = config.padmm.dual_tolerance
        datafile[f"{scope}/padmm/compl_tolerance"] = config.padmm.compl_tolerance
        datafile[f"{scope}/padmm/restart_tolerance"] = config.padmm.restart_tolerance
        datafile[f"{scope}/padmm/eta"] = config.padmm.eta
        datafile[f"{scope}/padmm/rho_0"] = config.padmm.rho_0
        datafile[f"{scope}/padmm/rho_min"] = config.padmm.rho_min
        datafile[f"{scope}/padmm/a_0"] = config.padmm.a_0
        datafile[f"{scope}/padmm/alpha"] = config.padmm.alpha
        datafile[f"{scope}/padmm/tau"] = config.padmm.tau
        datafile[f"{scope}/padmm/penalty_update_method"] = config.padmm.penalty_update_method
        datafile[f"{scope}/padmm/penalty_update_freq"] = config.padmm.penalty_update_freq
        datafile[f"{scope}/padmm/linear_solver_tolerance"] = config.padmm.linear_solver_tolerance
        datafile[f"{scope}/padmm/linear_solver_tolerance_ratio"] = config.padmm.linear_solver_tolerance_ratio
        datafile[f"{scope}/padmm/use_solver_acceleration"] = config.use_solver_acceleration
        datafile[f"{scope}/padmm/avoid_graph_conditionals"] = config.avoid_graph_conditionals
        # ------------------------------------------------------------------------------
        datafile[f"{scope}/warmstarting/warmstart_mode"] = config.warmstart_mode
        datafile[f"{scope}/warmstarting/contact_warmstart_method"] = config.contact_warmstart_method.value


def load_solver_configs_to_hdf5(datafile: h5py.File) -> dict[str, SolverKaminoConfig]:
    configs = {}
    for config_name in datafile["Solver"].keys():
        config = SolverKaminoConfig()
        # ------------------------------------------------------------------------------
        config.problem.alpha = float(datafile[f"Solver/{config_name}/constraints/alpha"][()])
        config.problem.beta = float(datafile[f"Solver/{config_name}/constraints/beta"][()])
        config.problem.gamma = float(datafile[f"Solver/{config_name}/constraints/gamma"][()])
        config.problem.delta = float(datafile[f"Solver/{config_name}/constraints/delta"][()])
        config.problem.preconditioning = bool(datafile[f"Solver/{config_name}/constraints/preconditioning"][()])
        # ------------------------------------------------------------------------------
        config.sparse = bool(datafile[f"Solver/{config_name}/sparse_solver"][()])
        config.sparse_jacobian = bool(datafile[f"Solver/{config_name}/sparse_jacobian"][()])
        # ------------------------------------------------------------------------------
        config.linear_solver_type = LinearSolverNameToType[
            datafile[f"Solver/{config_name}/linear_solver/type"][()].decode("utf-8")
        ]
        config.linear_solver_kwargs = eval(datafile[f"Solver/{config_name}/linear_solver/args"][()].decode("utf-8"))
        # ------------------------------------------------------------------------------
        config.padmm.max_iterations = float(datafile[f"Solver/{config_name}/padmm/max_iterations"][()])
        config.padmm.primal_tolerance = float(datafile[f"Solver/{config_name}/padmm/primal_tolerance"][()])
        config.padmm.dual_tolerance = float(datafile[f"Solver/{config_name}/padmm/dual_tolerance"][()])
        config.padmm.compl_tolerance = float(datafile[f"Solver/{config_name}/padmm/compl_tolerance"][()])
        config.padmm.restart_tolerance = float(datafile[f"Solver/{config_name}/padmm/restart_tolerance"][()])
        config.padmm.eta = float(datafile[f"Solver/{config_name}/padmm/eta"][()])
        config.padmm.rho_0 = float(datafile[f"Solver/{config_name}/padmm/rho_0"][()])
        config.padmm.rho_min = float(datafile[f"Solver/{config_name}/padmm/rho_min"][()])
        config.padmm.a_0 = float(datafile[f"Solver/{config_name}/padmm/a_0"][()])
        config.padmm.alpha = float(datafile[f"Solver/{config_name}/padmm/alpha"][()])
        config.padmm.tau = float(datafile[f"Solver/{config_name}/padmm/tau"][()])
        config.padmm.penalty_update_method = PADMMPenaltyUpdate(
            int(datafile[f"Solver/{config_name}/padmm/penalty_update_method"][()])
        )
        config.padmm.penalty_update_freq = int(datafile[f"Solver/{config_name}/padmm/penalty_update_freq"][()])
        config.padmm.linear_solver_tolerance = float(
            datafile[f"Solver/{config_name}/padmm/linear_solver_tolerance"][()]
        )
        config.padmm.linear_solver_tolerance_ratio = float(
            datafile[f"Solver/{config_name}/padmm/linear_solver_tolerance_ratio"][()]
        )
        config.use_solver_acceleration = bool(datafile[f"Solver/{config_name}/padmm/use_solver_acceleration"][()])
        config.avoid_graph_conditionals = bool(datafile[f"Solver/{config_name}/padmm/avoid_graph_conditionals"][()])
        # ------------------------------------------------------------------------------
        config.warmstart_mode = str(datafile[f"Solver/{config_name}/warmstarting/warmstart_mode"][()])
        config.contact_warmstart_method = WarmstarterContacts.Method(
            int(datafile[f"Solver/{config_name}/warmstarting/contact_warmstart_method"][()])
        )
        # ------------------------------------------------------------------------------
        configs[config_name] = config
    return configs
