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

"""KAMINO: Utilities: Linear Algebra: Matrix Factorizations"""

from .ldlt_bk import (
    ldlt_bk_lower,
    ldlt_bk_lower_reconstruct,
    ldlt_bk_lower_solve,
    ldlt_bk_lower_unpack,
)
from .ldlt_blocked import (
    ldlt_blocked_lower,
    ldlt_blocked_upper,
)
from .ldlt_eigen3 import (
    ldlt_eigen3_lower,
    ldlt_eigen3_lower_inplace,
    ldlt_eigen3_lower_solve,
    ldlt_eigen3_lower_solve_inplace,
    ldlt_eigen3_lower_unpack,
)
from .ldlt_nopivot import (
    ldlt_nopivot_lower,
    ldlt_nopivot_lower_reconstruct,
    ldlt_nopivot_lower_solve_inplace,
    ldlt_nopivot_upper,
    ldlt_nopivot_upper_reconstruct,
    ldlt_nopivot_upper_solve_inplace,
)
from .llt_std import (
    llt_std_lower,
    llt_std_lower_reconstruct,
    llt_std_lower_solve,
    llt_std_lower_with_tolerance,
    llt_std_lower_without_conditionals,
    llt_std_upper,
    llt_std_upper_reconstruct,
    llt_std_upper_solve,
    llt_std_upper_with_tolerance,
    llt_std_upper_without_conditionals,
)
from .lu_nopivot import (
    lu_nopiv,
    lu_nopiv_solve,
    lu_nopiv_solve_backward_upper,
    lu_nopiv_solve_forward_lower,
)

###
# Module API
###

__all__ = [
    "ldlt_bk_lower",
    "ldlt_bk_lower_reconstruct",
    "ldlt_bk_lower_solve",
    "ldlt_bk_lower_unpack",
    "ldlt_blocked_lower",
    "ldlt_blocked_upper",
    "ldlt_eigen3_lower",
    "ldlt_eigen3_lower_inplace",
    "ldlt_eigen3_lower_solve",
    "ldlt_eigen3_lower_solve_inplace",
    "ldlt_eigen3_lower_unpack",
    "ldlt_nopivot_lower",
    "ldlt_nopivot_lower_reconstruct",
    "ldlt_nopivot_lower_solve_inplace",
    "ldlt_nopivot_upper",
    "ldlt_nopivot_upper_reconstruct",
    "ldlt_nopivot_upper_solve_inplace",
    "llt_std_lower",
    "llt_std_lower_reconstruct",
    "llt_std_lower_solve",
    "llt_std_lower_with_tolerance",
    "llt_std_lower_without_conditionals",
    "llt_std_upper",
    "llt_std_upper_reconstruct",
    "llt_std_upper_solve",
    "llt_std_upper_with_tolerance",
    "llt_std_upper_without_conditionals",
    "lu_nopiv",
    "lu_nopiv_solve",
    "lu_nopiv_solve_backward_upper",
    "lu_nopiv_solve_forward_lower",
]
