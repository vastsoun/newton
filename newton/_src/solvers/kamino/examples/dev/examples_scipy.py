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

"""TODO"""

import numpy as np
from scipy import linalg

# TODO:
#   1. Experiment w/ SciPy/NumPy LU
#   2. Create Factorizer class to wrap:
#       2.1 NumPy Cholesky
#       2.3 SciPy LU
#       2.2 SciPy Cholesky
#       2.4 SciPy LDL
#   3. Test the FPI solvers
#   4. Modularize each FPI solver as a class
#   5. Unify FPI and factorizers under common base class
#   6. Determine a mechanism to create randomized RBD problems
#       6.1 Randomized GMM given: number of bodies, mass ratio, min-max mass
#       6.2 Randomized Jacobian given: number of constraints, lever-arm ration, min-max lever-arm
#
# NOTE:
#   1. ...
#
# L @ L.T @ x = b
#
# L @ y = b
# L.T @ x = y
# -----
# P @ L @ U @ x = b
# L @ U @ x = P.T @ b
# L @ z = P.T @ b
# U @ x = z
# -----


if __name__ == "__main__":
    dtype = np.float64
    # dtype = np.float32

    # ----------------------------
    A = np.array([[3, 2, 0], [1, -1, 0], [0, 5, 1]], dtype=dtype)
    A = A @ A.T  # Make SPD
    b = A @ np.array([2, 4, -1], dtype=dtype)

    # ---------------------------- LinearSolverNumpy
    x_np = np.linalg.solve(A, b)
    r_np = A @ x_np - b

    # ---------------------------- LinearSolverSciPy
    x_sp = linalg.solve(A, b)
    r_sp = A @ x_sp - b

    # ---------------------------- FactorizerLUSciPy
    lu, piv = linalg.lu_factor(A)
    x_lu = linalg.lu_solve((lu, piv), b)
    r_lu = A @ x_lu - b

    # ---------------------------- FactorizerLUSciPy
    P, L, U = linalg.lu(A, permute_l=False)
    y_lu2 = linalg.solve_triangular(L, P.T @ b, lower=True)
    x_lu2 = linalg.solve_triangular(U, y_lu2, lower=False)
    r_lu2 = A @ x_lu2 - b

    # ---------------------------- FactorizerCholeskyNumPy
    L_ch_np = np.linalg.cholesky(A)
    y_ch_np = linalg.solve_triangular(L_ch_np, b, lower=True)
    x_ch_np = linalg.solve_triangular(L_ch_np.T, y_ch_np, lower=False)
    r_ch_np = A @ x_ch_np - b

    # ---------------------------- FactorizerCholeskySciPy
    L_ch_sp = linalg.cholesky(A, lower=True)
    y_chl_sp = linalg.solve_triangular(L_ch_sp, b, lower=True)
    x_chl_sp = linalg.solve_triangular(L_ch_sp.T, y_chl_sp, lower=False)
    r_chl_sp = A @ x_chl_sp - b

    # ---------------------------- FactorizerCholeskySciPy
    U_ch_sp = linalg.cholesky(A, lower=False)
    y_chu_sp = linalg.solve_triangular(U_ch_sp.T, b, lower=True)
    x_chu_sp = linalg.solve_triangular(U_ch_sp, y_chu_sp, lower=False)
    r_chu_sp = A @ x_chu_sp - b

    # ---------------------------- FactorizerLDLSciPy
    PL_ldl_sp, D_ldl_sp, perm_ldl_sp = linalg.ldl(A)
    L_ldl_sp = PL_ldl_sp[perm_ldl_sp, :]
    b_ldl_sp = b[perm_ldl_sp]
    z_ldl_sp = linalg.solve_triangular(L_ldl_sp, b_ldl_sp, lower=True)
    y_ldl_sp = z_ldl_sp / np.diag(D_ldl_sp)
    x_ldl_sp = linalg.solve_triangular(L_ldl_sp.T, y_ldl_sp, lower=False)
    x_ldl_sp = x_ldl_sp[np.argsort(perm_ldl_sp)]
    r_ldl_sp = A @ x_ldl_sp - b

    # ----------------------------
    print("----------------------------")
    print(f"dtype: {dtype}")

    # ----------------------------
    r_np_l2 = np.linalg.norm(r_np, ord=2)
    r_sp_l2 = np.linalg.norm(r_sp, ord=2)
    r_lu_l2 = np.linalg.norm(r_lu, ord=2)
    r_lu2_l2 = np.linalg.norm(r_lu2, ord=2)
    r_ch_np_l2 = np.linalg.norm(r_ch_np, ord=2)
    r_chl_sp_l2 = np.linalg.norm(r_chl_sp, ord=2)
    r_chu_sp_l2 = np.linalg.norm(r_chu_sp, ord=2)
    r_ldl_sp_l2 = np.linalg.norm(r_ldl_sp, ord=2)
    print("----------------------------")
    print(f"r_np_l2: {r_np_l2}")
    print(f"r_sp_l2: {r_sp_l2}")
    print(f"r_lu_l2: {r_lu_l2}")
    print(f"r_lu2_l2: {r_lu2_l2}")
    print(f"r_ch_np_l2: {r_ch_np_l2}")
    print(f"r_chl_sp_l2: {r_chl_sp_l2}")
    print(f"r_chu_sp_l2: {r_chu_sp_l2}")
    print(f"r_ldl_sp_l2: {r_ldl_sp_l2}")

    print("----------------------------")
    r_np_infnorm = np.linalg.norm(r_np, ord=np.inf)
    r_sp_infnorm = np.linalg.norm(r_sp, ord=np.inf)
    r_lu_infnorm = np.linalg.norm(r_lu, ord=np.inf)
    r_lu2_infnorm = np.linalg.norm(r_lu2, ord=np.inf)
    r_ch_np_infnorm = np.linalg.norm(r_ch_np, ord=np.inf)
    r_chl_sp_infnorm = np.linalg.norm(r_chl_sp, ord=np.inf)
    r_chu_sp_infnorm = np.linalg.norm(r_chu_sp, ord=np.inf)
    r_ldl_sp_infnorm = np.linalg.norm(r_ldl_sp, ord=np.inf)
    print(f"r_np_infnorm: {r_np_infnorm}")
    print(f"r_sp_infnorm: {r_sp_infnorm}")
    print(f"r_lu_infnorm: {r_lu_infnorm}")
    print(f"r_lu2_infnorm: {r_lu2_infnorm}")
    print(f"r_ch_np_infnorm: {r_ch_np_infnorm}")
    print(f"r_chl_sp_infnorm: {r_chl_sp_infnorm}")
    print(f"r_chu_sp_infnorm: {r_chu_sp_infnorm}")
    print(f"r_ldl_sp_infnorm: {r_ldl_sp_infnorm}")
