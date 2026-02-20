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
from ...solvers.padmm import PADMMWarmStartMode
from ...solvers.warmstart import WarmstarterContacts
from ...utils.sim import SimulatorSettings

###
# Functions
###


def make_default_solver_config() -> SolverKaminoSettings:
    settings = SolverKaminoSettings()

    # Constraint stabilization
    settings.problem.alpha = 0.1

    # Jacobian representation
    settings.sparse = False
    settings.sparse_jacobian = False

    # Linear system solver
    settings.linear_solver_type = LinearSolverNameToType["LLTB"]
    settings.linear_solver_kwargs = {}

    # PADMM
    settings.padmm.primal_tolerance = 1e-6
    settings.padmm.dual_tolerance = 1e-6
    settings.padmm.compl_tolerance = 1e-6
    settings.padmm.max_iterations = 200
    settings.padmm.eta = 1e-5
    settings.padmm.rho_0 = 0.05
    settings.use_solver_acceleration = True
    settings.avoid_graph_conditionals = False

    # Warm-starting
    settings.warmstart_mode = PADMMWarmStartMode.CONTAINERS
    settings.contact_warmstart_method = WarmstarterContacts.Method.GEOM_PAIR_NET_FORCE

    # Logging
    settings.compute_metrics = False
    settings.collect_solver_info = False

    return settings


def make_default_simulator_config() -> SimulatorSettings:
    settings = SimulatorSettings()
    settings.solver = make_default_solver_config()
    settings.dt = 0.001
    return settings
