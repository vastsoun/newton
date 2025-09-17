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
KAMINO: Utilities: Linear Algebra
"""

from .admm import ADMMInfo, ADMMSolver, ADMMStatus
from .cholesky import Cholesky
from .ldlt_bk import LDLTBunchKaufman
from .ldlt_blocked import LDLTBlocked
from .ldlt_eigen3 import LDLTEigen3
from .ldlt_nopivot import LDLTNoPivot
from .matrix import MatrixComparison, SquareSymmetricMatrixProperties, is_square_matrix, is_symmetric_matrix

FactorizerType = Cholesky | LDLTNoPivot | LDLTBunchKaufman | LDLTBlocked | LDLTEigen3


###
# Module API
###

__all__ = [
    "ADMMInfo",
    "ADMMSolver",
    "ADMMStatus",
    "Cholesky",
    "FactorizerType",
    "LDLTBlocked",
    "LDLTBunchKaufman",
    "LDLTEigen3",
    "LDLTNoPivot",
    "MatrixComparison",
    "SquareSymmetricMatrixProperties",
    "is_square_matrix",
    "is_symmetric_matrix",
]
