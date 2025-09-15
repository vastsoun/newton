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

"""KAMINO: Utilities: Linear Algebra"""

from .admm import (
    ADMMInfo,
    ADMMSolver,
    ADMMStatus,
    compute_lambdas,
    compute_u_plus,
)
from .eigval import (
    GramIteration,
    PowerIteration,
)
from .linear import (
    ComputationInfo,
    ConjugateGradientSolver,
    DirectSolver,
    GaussSeidelSolver,
    IndirectSolver,
    JacobiSolver,
    LDLTBlockedSolver,
    LDLTBunchKaufmanSolver,
    LDLTEigen3Solver,
    LDLTNoPivotSolver,
    LDLTSciPySolver,
    LinearSolution,
    LinearSolver,
    LinearSolverType,
    LLTNumPySolver,
    LLTSciPySolver,
    LLTStdSolver,
    LUNoPivotSolver,
    LUSciPySolver,
    MinimumResidualSolver,
    NumPySolver,
    SciPySolver,
    SORSolver,
    conjugate_gradient,
    gauss_seidel,
    jacobi,
    minimum_residual,
    successive_over_relaxation,
)
from .matrix import (
    MatrixComparison,
    MatrixSign,
    RectangularMatrixProperties,
    SquareSymmetricMatrixProperties,
    is_square_matrix,
    is_symmetric_matrix,
)

###
# Module API
###


__all__ = [
    "ADMMInfo",
    "ADMMSolver",
    "ADMMStatus",
    "ComputationInfo",
    "ConjugateGradientSolver",
    "DirectSolver",
    "GaussSeidelSolver",
    "GramIteration",
    "IndirectSolver",
    "JacobiSolver",
    "LDLTBlockedSolver",
    "LDLTBunchKaufmanSolver",
    "LDLTEigen3Solver",
    "LDLTNoPivotSolver",
    "LDLTSciPySolver",
    "LLTNumPySolver",
    "LLTSciPySolver",
    "LLTStdSolver",
    "LUNoPivotSolver",
    "LUSciPySolver",
    "LinearSolution",
    "LinearSolver",
    "LinearSolverType",
    "MatrixComparison",
    "MatrixSign",
    "MinimumResidualSolver",
    "NumPySolver",
    "PowerIteration",
    "RectangularMatrixProperties",
    "SORSolver",
    "SciPySolver",
    "SquareSymmetricMatrixProperties",
    "compute_lambdas",
    "compute_u_plus",
    "conjugate_gradient",
    "gauss_seidel",
    "is_square_matrix",
    "is_symmetric_matrix",
    "jacobi",
    "minimum_residual",
    "successive_over_relaxation",
]
