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

from ...linalg.linear import LinearSolverNameToType, LinearSolverTypeToName
from ...solver_kamino import SolverKamino

###
# Module interface
###

__all__ = [
    "make_benchmark_configs",
    "make_solver_config_default",
    "make_solver_config_dense_jacobian_llt_accurate",
    "make_solver_config_dense_jacobian_llt_fast",
    "make_solver_config_sparse_delassus_cr_accurate",
    "make_solver_config_sparse_delassus_cr_fast",
    "make_solver_config_sparse_jacobian_llt_accurate",
    "make_solver_config_sparse_jacobian_llt_fast",
]


###
# Solver configurations
###


def make_solver_config_default() -> tuple[str, SolverKamino.Config]:
    # ------------------------------------------------------------------------------
    name = "Default"
    # ------------------------------------------------------------------------------
    config = SolverKamino.Config()
    # ------------------------------------------------------------------------------
    # Constraint stabilization
    config.problem.alpha = 0.1
    # ------------------------------------------------------------------------------
    # Jacobian representation
    config.sparse_jacobian = False
    config.sparse_dynamics = False
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
    config.padmm.penalty_update_method = "fixed"
    config.padmm.penalty_update_freq = 1
    config.use_solver_acceleration = True
    config.avoid_graph_conditionals = False
    # ------------------------------------------------------------------------------
    # Warm-starting
    config.warmstart_mode = "containers"
    config.contact_warmstart_method = "geom_pair_net_force"
    # ------------------------------------------------------------------------------
    return name, config


def make_solver_config_dense_jacobian_llt_accurate() -> tuple[str, SolverKamino.Config]:
    # ------------------------------------------------------------------------------
    name = "Dense Jacobian LLT accurate"
    # ------------------------------------------------------------------------------
    config = SolverKamino.Config()
    # ------------------------------------------------------------------------------
    # Constraint stabilization
    config.problem.alpha = 0.1
    # ------------------------------------------------------------------------------
    # Jacobian representation
    config.sparse_dynamics = False
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
    config.padmm.rho_0 = 0.1
    config.padmm.rho_min = 1e-5
    config.padmm.penalty_update_method = "fixed"
    config.padmm.penalty_update_freq = 1
    config.use_solver_acceleration = True
    config.avoid_graph_conditionals = False
    # ------------------------------------------------------------------------------
    # Warm-starting
    config.warmstart_mode = "containers"
    config.contact_warmstart_method = "geom_pair_net_force"
    # ------------------------------------------------------------------------------
    return name, config


def make_solver_config_dense_jacobian_llt_fast() -> tuple[str, SolverKamino.Config]:
    # ------------------------------------------------------------------------------
    name = "Dense Jacobian LLT fast"
    # ------------------------------------------------------------------------------
    config = SolverKamino.Config()
    # ------------------------------------------------------------------------------
    # Constraint stabilization
    config.problem.alpha = 0.1
    # ------------------------------------------------------------------------------
    # Jacobian representation
    config.sparse_dynamics = False
    config.sparse_jacobian = False
    # ------------------------------------------------------------------------------
    # Linear system solver
    config.linear_solver_type = LinearSolverNameToType["LLTB"]
    config.linear_solver_kwargs = {}
    # ------------------------------------------------------------------------------
    # PADMM
    config.padmm.max_iterations = 100
    config.padmm.primal_tolerance = 1e-4
    config.padmm.dual_tolerance = 1e-4
    config.padmm.compl_tolerance = 1e-4
    config.padmm.restart_tolerance = 0.999
    config.padmm.eta = 1e-5
    config.padmm.rho_0 = 0.02
    config.padmm.rho_min = 1e-5
    config.padmm.penalty_update_method = "fixed"
    config.padmm.penalty_update_freq = 1
    config.use_solver_acceleration = True
    config.avoid_graph_conditionals = False
    # ------------------------------------------------------------------------------
    # Warm-starting
    config.warmstart_mode = "containers"
    config.contact_warmstart_method = "geom_pair_net_force"
    # ------------------------------------------------------------------------------
    return name, config


def make_solver_config_sparse_jacobian_llt_accurate() -> tuple[str, SolverKamino.Config]:
    # ------------------------------------------------------------------------------
    name = "Sparse Jacobian LLT accurate"
    # ------------------------------------------------------------------------------
    config = SolverKamino.Config()
    # ------------------------------------------------------------------------------
    # Constraint stabilization
    config.problem.alpha = 0.1
    # ------------------------------------------------------------------------------
    # Jacobian representation
    config.sparse_dynamics = False
    config.sparse_jacobian = True
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
    config.padmm.rho_0 = 0.1
    config.padmm.rho_min = 1e-5
    config.padmm.penalty_update_method = "fixed"
    config.padmm.penalty_update_freq = 1
    config.use_solver_acceleration = True
    config.avoid_graph_conditionals = False
    # ------------------------------------------------------------------------------
    # Warm-starting
    config.warmstart_mode = "containers"
    config.contact_warmstart_method = "geom_pair_net_force"
    # ------------------------------------------------------------------------------
    return name, config


def make_solver_config_sparse_jacobian_llt_fast() -> tuple[str, SolverKamino.Config]:
    # ------------------------------------------------------------------------------
    name = "Sparse Jacobian LLT fast"
    # ------------------------------------------------------------------------------
    config = SolverKamino.Config()
    # ------------------------------------------------------------------------------
    # Constraint stabilization
    config.problem.alpha = 0.1
    # ------------------------------------------------------------------------------
    # Jacobian representation
    config.sparse_dynamics = False
    config.sparse_jacobian = True
    # ------------------------------------------------------------------------------
    # Linear system solver
    config.linear_solver_type = LinearSolverNameToType["LLTB"]
    config.linear_solver_kwargs = {}
    # ------------------------------------------------------------------------------
    # PADMM
    config.padmm.max_iterations = 100
    config.padmm.primal_tolerance = 1e-4
    config.padmm.dual_tolerance = 1e-4
    config.padmm.compl_tolerance = 1e-4
    config.padmm.restart_tolerance = 0.999
    config.padmm.eta = 1e-5
    config.padmm.rho_0 = 0.02
    config.padmm.rho_min = 1e-5
    config.padmm.penalty_update_method = "fixed"
    config.padmm.penalty_update_freq = 1
    config.use_solver_acceleration = True
    config.avoid_graph_conditionals = False
    # ------------------------------------------------------------------------------
    # Warm-starting
    config.warmstart_mode = "containers"
    config.contact_warmstart_method = "geom_pair_net_force"
    # ------------------------------------------------------------------------------
    return name, config


def make_solver_config_sparse_delassus_cr_accurate() -> tuple[str, SolverKamino.Config]:
    # ------------------------------------------------------------------------------
    name = "Sparse Delassus CR accurate"
    # ------------------------------------------------------------------------------
    config = SolverKamino.Config()
    # ------------------------------------------------------------------------------
    # Constraint stabilization
    config.problem.alpha = 0.1
    # ------------------------------------------------------------------------------
    # Jacobian representation
    config.sparse_dynamics = True
    config.sparse_jacobian = True
    # ------------------------------------------------------------------------------
    # Linear system solver
    config.linear_solver_type = LinearSolverNameToType["CR"]
    config.linear_solver_kwargs = {"maxiter": 30}
    # ------------------------------------------------------------------------------
    # PADMM
    config.padmm.max_iterations = 200
    config.padmm.primal_tolerance = 1e-6
    config.padmm.dual_tolerance = 1e-6
    config.padmm.compl_tolerance = 1e-6
    config.padmm.restart_tolerance = 0.999
    config.padmm.eta = 1e-5
    config.padmm.rho_0 = 0.1
    config.padmm.rho_min = 1e-5
    config.padmm.penalty_update_method = "fixed"
    config.padmm.penalty_update_freq = 1
    config.use_solver_acceleration = True
    config.avoid_graph_conditionals = False
    # ------------------------------------------------------------------------------
    # Warm-starting
    config.warmstart_mode = "containers"
    config.contact_warmstart_method = "geom_pair_net_force"
    # ------------------------------------------------------------------------------
    return name, config


def make_solver_config_sparse_delassus_cr_fast() -> tuple[str, SolverKamino.Config]:
    # ------------------------------------------------------------------------------
    name = "Sparse Delassus CR fast"
    # ------------------------------------------------------------------------------
    config = SolverKamino.Config()
    # ------------------------------------------------------------------------------
    # Constraint stabilization
    config.problem.alpha = 0.1
    # ------------------------------------------------------------------------------
    # Jacobian representation
    config.sparse_dynamics = True
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
    config.padmm.rho_0 = 0.02
    config.padmm.rho_min = 1e-5
    config.padmm.penalty_update_method = "fixed"
    config.padmm.penalty_update_freq = 1
    config.use_solver_acceleration = True
    config.avoid_graph_conditionals = False
    # ------------------------------------------------------------------------------
    # Warm-starting
    config.warmstart_mode = "containers"
    config.contact_warmstart_method = "geom_pair_net_force"
    # ------------------------------------------------------------------------------
    return name, config


###
# Utilities
###


def make_benchmark_configs(include_default: bool = True) -> dict[str, SolverKamino.Config]:
    if include_default:
        generators = [make_solver_config_default]
    else:
        generators = []
    generators.extend(
        [
            make_solver_config_dense_jacobian_llt_accurate,
            make_solver_config_dense_jacobian_llt_fast,
            make_solver_config_sparse_jacobian_llt_accurate,
            make_solver_config_sparse_jacobian_llt_fast,
            make_solver_config_sparse_delassus_cr_accurate,
            make_solver_config_sparse_delassus_cr_fast,
        ]
    )
    solver_configs: dict[str, SolverKamino.Config] = {}
    for gen in generators:
        name, config = gen()
        solver_configs[name] = config
    return solver_configs


###
# Functions
###


def save_solver_configs_to_hdf5(configs: dict[str, SolverKamino.Config], datafile):
    for config_name, config in configs.items():
        scope = f"Solver/{config_name}"
        # ------------------------------------------------------------------------------
        datafile[f"{scope}/constraints/alpha"] = config.problem.alpha
        datafile[f"{scope}/constraints/beta"] = config.problem.beta
        datafile[f"{scope}/constraints/gamma"] = config.problem.gamma
        datafile[f"{scope}/constraints/delta"] = config.problem.delta
        datafile[f"{scope}/constraints/preconditioning"] = config.problem.preconditioning
        # ------------------------------------------------------------------------------
        datafile[f"{scope}/sparse_jacobian"] = config.sparse_jacobian
        datafile[f"{scope}/sparse_dynamics"] = config.sparse_dynamics
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
        datafile[f"{scope}/warmstarting/contact_warmstart_method"] = config.contact_warmstart_method


def load_solver_configs_to_hdf5(datafile) -> dict[str, SolverKamino.Config]:
    configs = {}
    for config_name in datafile["Solver"].keys():
        config = SolverKamino.Config()
        # ------------------------------------------------------------------------------
        config.problem.alpha = float(datafile[f"Solver/{config_name}/constraints/alpha"][()])
        config.problem.beta = float(datafile[f"Solver/{config_name}/constraints/beta"][()])
        config.problem.gamma = float(datafile[f"Solver/{config_name}/constraints/gamma"][()])
        config.problem.delta = float(datafile[f"Solver/{config_name}/constraints/delta"][()])
        config.problem.preconditioning = bool(datafile[f"Solver/{config_name}/constraints/preconditioning"][()])
        # ------------------------------------------------------------------------------
        config.sparse_jacobian = bool(datafile[f"Solver/{config_name}/sparse_jacobian"][()])
        config.sparse_dynamics = bool(datafile[f"Solver/{config_name}/sparse_dynamics"][()])
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
        config.padmm.penalty_update_method = str(datafile[f"Solver/{config_name}/padmm/penalty_update_method"][()])
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
        config.contact_warmstart_method = str(
            datafile[f"Solver/{config_name}/warmstarting/contact_warmstart_method"][()]
        )
        # ------------------------------------------------------------------------------
        configs[config_name] = config
    return configs
