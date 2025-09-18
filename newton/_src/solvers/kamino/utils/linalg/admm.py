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

"""KAMINO: Utilities: Linear Algebra: ADMM constrained dynamics solver implemented in NumPy"""

from enum import IntEnum

import numpy as np

from .linear import LinearSolverType, NumPySolver

###
# Utilities
###


def compute_u_plus(
    u_minus: np.ndarray,
    invM: np.ndarray,
    J: np.ndarray,
    h: np.ndarray,
    lambdas: np.ndarray,
) -> np.ndarray:
    """
    Compute the next-step generalized velocity vector given the problem and constraint reactions.
    """
    return u_minus + invM @ ((J.T @ lambdas) + h)


def compute_lambdas(v: np.ndarray, J: np.ndarray, u_plus: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Compute the constraint reactions vector given the problem and next-step generalized velocity.
    """
    r_epsilon = v.dtype.type(1) / v.dtype.type(epsilon)
    return r_epsilon * (v - J @ u_plus)


###
# ADMM Solver
###


class ADMMResult(IntEnum):
    SUCCESS = 0
    MAXITER = 1
    DIVERGE = 2
    ERROR = 3


class ADMMStatus:
    def __init__(self, dtype: np.dtype = np.float64):
        self.result: ADMMResult = ADMMResult.ERROR
        self.converged: bool = False
        self.iterations: int = 0
        self.r_p: float = dtype.type(np.inf)
        self.r_d: float = dtype.type(np.inf)
        self.r_c: float = dtype.type(np.inf)
        self.r_i: float = dtype.type(np.inf)


class ADMMInfo:
    def __init__(self, maxiter: int = 0, dtype: np.dtype = np.float64):
        self.r_p: np.ndarray = np.zeros(maxiter, dtype=dtype)
        self.r_d: np.ndarray = np.zeros(maxiter, dtype=dtype)
        self.r_c: np.ndarray = np.zeros(maxiter, dtype=dtype)
        self.r_i: np.ndarray = np.zeros(maxiter, dtype=dtype)

    def reset(self):
        self.r_p.fill(0)
        self.r_d.fill(0)
        self.r_c.fill(0)
        self.r_i.fill(0)


class ADMMSolver:
    def __init__(
        self,
        dtype: np.dtype = np.float64,
        primal_tolerance: float = 1e-6,
        dual_tolerance: float = 1e-6,
        compl_tolerance: float = 1e-6,
        iter_tolerance: float = 0.0,
        diverge_tolerance: float = 1e-1,
        maxvalue: float = 1e20,
        eta: float = 1e-5,
        rho: float = 1.0,
        omega: float = 1.0,
        maxiter: int = 200,
        kkt_solver: LinearSolverType | None = None,
        schur_solver: LinearSolverType | None = None,
    ):
        # Meta-data
        self.dtype: np.dtype = dtype
        self.status: ADMMStatus | None = None
        self.info: ADMMInfo | None = None
        self.nbd: int = 0
        self.ncts: int = 0

        # Settings
        eps = float(np.finfo(self.dtype).eps)
        dtmax = float(np.finfo(self.dtype).max)
        self.primal_tolerance: float = self.dtype(max(primal_tolerance, eps))
        self.dual_tolerance: float = self.dtype(max(dual_tolerance, eps))
        self.compl_tolerance: float = self.dtype(max(compl_tolerance, eps))
        self.iter_tolerance: float = self.dtype(iter_tolerance)
        self.diverge_tolerance: float = self.dtype(diverge_tolerance)
        self.maxvalue: float = self.dtype(min(maxvalue, dtmax))
        self.eta: float = self.dtype(eta)
        self.rho: float = self.dtype(rho)
        self.omega: float = self.dtype(omega)
        self.maxiter: int = maxiter

        # Residuals
        self.r_p: np.ndarray | None = None
        self.r_d: np.ndarray | None = None
        self.r_c: np.ndarray | None = None
        self.r_i: np.ndarray | None = None

        # Linear system solver
        self.kkt_solver: LinearSolverType = kkt_solver or NumPySolver(dtype=self.dtype)
        self.schur_solver: LinearSolverType = schur_solver or NumPySolver(dtype=self.dtype)

        # KKT linear problem
        self.K: np.ndarray | None = None
        self.k: np.ndarray | None = None

        # Primal Schur complement linear problem
        self.P: np.ndarray | None = None
        self.p: np.ndarray | None = None

        # Dual Schur complement linear problem
        self.D: np.ndarray | None = None
        self.d: np.ndarray | None = None

        # Intermediates
        self.ux: np.ndarray | None = None
        self.u: np.ndarray | None = None
        self.w: np.ndarray | None = None
        self.v: np.ndarray | None = None

        # State
        self.x: np.ndarray | None = None
        self.y: np.ndarray | None = None
        self.z: np.ndarray | None = None
        self.x_p: np.ndarray | None = None
        self.y_p: np.ndarray | None = None
        self.z_p: np.ndarray | None = None

        # Solution
        self.u_plus: np.ndarray | None = None
        self.v_plus: np.ndarray | None = None
        self.lambdas: np.ndarray | None = None

    ###
    # Internal operations
    ###

    def _set_tolerance_dtype(self):
        """Ensure that all tolerances are of the correct dtype and at least equal to the corresponding machine epsilon."""
        eps = np.finfo(self.dtype).eps
        self.primal_tolerance = self.dtype.type(max(self.primal_tolerance, eps))
        self.dual_tolerance = self.dtype.type(max(self.dual_tolerance, eps))
        self.compl_tolerance = self.dtype.type(max(self.compl_tolerance, eps))
        self.iter_tolerance = self.dtype.type(max(self.iter_tolerance, eps))

    def _check_state(self) -> bool:
        """Check and correct the ADMM state vectors: primal, slack and dual variables."""
        x_valid = np.all(np.isfinite(self.x)) and np.all(np.abs(self.x) < self.maxvalue)
        y_valid = np.all(np.isfinite(self.y)) and np.all(np.abs(self.y) < self.maxvalue)
        z_valid = np.all(np.isfinite(self.z)) and np.all(np.abs(self.z) < self.maxvalue)
        if not (x_valid and y_valid and z_valid):
            self.status.result = ADMMResult.ERROR
            return False
        return True

    def _update_state(self):
        """Update the ADMM state vectors: primal, slack and dual variables."""
        self.x[:] = self.omega * self.x + (self.dtype.type(1) - self.omega) * self.y_p
        self.y[:] = self.x - (self.dtype.type(1) / self.rho) * self.z_p
        self.z[:] = self.z_p + self.rho * (self.y - self.x)

    def _compute_residuals(self, iter: int):
        """Compute the residuals for the current state."""
        # eps = np.finfo(self.dtype).eps
        self.r_p = self.x - self.y
        self.r_d = self.eta * (self.x - self.x_p) + self.rho * (self.y - self.y_p)
        self.r_c = np.dot(self.x, self.z)
        self.r_i = self.y - self.y_p
        self.status.r_p = np.max(np.abs(self.r_p))
        self.status.r_d = np.max(np.abs(self.r_d))
        self.status.r_c = np.max(np.abs(self.r_c))
        # self.status.r_i = np.max(np.abs(self.r_i))
        # self.status.r_i = np.max(np.abs(self.r_i)) / np.max(np.abs(self.y_p))
        # self.status.r_i = np.max(np.abs(self.r_i)) / (eps + np.max(np.abs(self.y_p)))
        self.status.r_i = np.max(np.abs(self.r_i)) / (self.dtype.type(1) + np.max(np.abs(self.y)))
        self.info.r_p[iter] = self.status.r_p
        self.info.r_d[iter] = self.status.r_d
        self.info.r_c[iter] = self.status.r_c
        self.info.r_i[iter] = self.status.r_i

    def _check_converged(self, iter: int) -> bool:
        """Check if the solver has converged based on the current iteration."""
        meets_tolerances: bool = (
            iter > 0
            and self.status.r_p < self.primal_tolerance
            and self.status.r_d < self.dual_tolerance
            and self.status.r_c < self.compl_tolerance
        )
        has_stagnated: bool = iter > 0 and self.status.r_i < self.iter_tolerance
        self.status.converged = meets_tolerances or has_stagnated
        return self.status.converged

    def _update_previous(self):
        self.x_p[:] = self.x
        self.y_p[:] = self.y
        self.z_p[:] = self.z

    def _set_result(self):
        if self.status.converged:
            self.status.result = ADMMResult.SUCCESS
        elif self.status.iterations >= self.maxiter:
            if (
                self.status.r_p > self.diverge_tolerance
                or self.status.r_d > self.diverge_tolerance
                or self.status.r_c > self.diverge_tolerance
                or self.status.r_i > self.diverge_tolerance
            ):
                self.status.result = ADMMResult.DIVERGE
            else:
                self.status.result = ADMMResult.MAXITER

    def _truncate_info(self):
        self.info.r_p = self.info.r_p[: self.status.iterations]
        self.info.r_d = self.info.r_d[: self.status.iterations]
        self.info.r_c = self.info.r_c[: self.status.iterations]
        self.info.r_i = self.info.r_i[: self.status.iterations]

    ###
    # Public API
    ###

    def solve_kkt(
        self,
        M: np.ndarray,
        J: np.ndarray,
        h: np.ndarray,
        u_minus: np.ndarray,
        v_star: np.ndarray,
    ) -> ADMMStatus:
        self.dtype = M.dtype
        self._set_tolerance_dtype()

        self.status = ADMMStatus(dtype=self.dtype)
        self.info = ADMMInfo(maxiter=self.maxiter, dtype=self.dtype)

        self.w = np.zeros_like(u_minus)
        self.v = np.zeros_like(v_star)

        self.x = np.zeros_like(v_star)
        self.y = np.zeros_like(v_star)
        self.z = np.zeros_like(v_star)
        self.x_p = np.zeros_like(v_star)
        self.y_p = np.zeros_like(v_star)
        self.z_p = np.zeros_like(v_star)

        self.nbd = M.shape[0]
        self.ncts = J.shape[0]

        kdim = self.nbd + self.ncts
        self.ux = np.zeros((kdim,), dtype=self.dtype)
        self.k = np.zeros((kdim,), dtype=self.dtype)
        self.K = np.zeros((kdim, kdim), dtype=self.dtype)

        self.w[:] = M @ u_minus + h
        self.k[self.ncts :] = self.w
        self.K[self.ncts :, self.ncts :] = M
        self.K[self.ncts :, : self.ncts] = J.T
        self.K[: self.ncts, self.ncts :] = J
        self.K[: self.ncts, : self.ncts] = -(self.eta + self.rho) * np.eye(self.ncts, dtype=self.dtype)

        self.kkt_solver.compute(A=self.K)

        for i in range(self.maxiter):
            self.status.iterations += 1

            self.v[:] = -v_star + self.eta * self.x_p + self.rho * self.y_p + self.z_p
            self.k[: self.ncts] = self.v
            self.ux[:] = self.kkt_solver.solve(b=self.k)
            self.x[:] = -self.ux[: self.ncts]
            if not self._check_state():
                break

            self._update_state()
            self._compute_residuals(i)
            if self._check_converged(i):
                break
            self._update_previous()

        self._set_result()
        self._truncate_info()
        self.lambdas = self.y.copy()
        self.v_plus = self.z.copy()
        self.u_plus = self.ux[self.ncts :]

        return self.status

    def solve_schur_primal(
        self,
        M: np.ndarray,
        J: np.ndarray,
        h: np.ndarray,
        u_minus: np.ndarray,
        v_star: np.ndarray,
    ) -> ADMMStatus:
        self.dtype = M.dtype
        self._set_tolerance_dtype()

        self.status = ADMMStatus(dtype=self.dtype)
        self.info = ADMMInfo(maxiter=self.maxiter, dtype=self.dtype)

        self.u = np.zeros_like(u_minus)
        self.w = np.zeros_like(u_minus)
        self.v = np.zeros_like(v_star)

        self.x = np.zeros_like(v_star)
        self.y = np.zeros_like(v_star)
        self.z = np.zeros_like(v_star)
        self.x_p = np.zeros_like(v_star)
        self.y_p = np.zeros_like(v_star)
        self.z_p = np.zeros_like(v_star)

        self.nbd = M.shape[0]
        self.ncts = J.shape[0]

        epsilon = self.eta + self.rho
        r_epsilon = self.dtype.type(1) / epsilon
        self.w = M @ u_minus + h
        self.p = np.zeros_like(self.w)
        self.P = M + r_epsilon * (J.T @ J)

        self.schur_solver.compute(A=self.P)

        for i in range(self.maxiter):
            self.status.iterations += 1

            self.v[:] = -v_star + self.eta * self.x_p + self.rho * self.y_p + self.z_p
            self.p[:] = self.w + r_epsilon * (J.T @ self.v)
            self.u[:] = self.schur_solver.solve(b=self.p)
            self.x[:] = r_epsilon * (self.v - J @ self.u)
            if not self._check_state():
                break

            self._update_state()
            self._compute_residuals(i)
            if self._check_converged(i):
                break
            self._update_previous()

        self._set_result()
        self._truncate_info()
        self.lambdas = self.y.copy()
        self.v_plus = self.z.copy()
        self.u_plus = self.u.copy()

        return self.status

    def solve_schur_dual(
        self,
        D: np.ndarray,
        v_f: np.ndarray,
        v_star: np.ndarray,
        u_minus: np.ndarray,
        invM: np.ndarray,
        J: np.ndarray,
        h: np.ndarray,
        use_preconditioning: bool = False,
    ) -> ADMMStatus:
        self.dtype = D.dtype
        self._set_tolerance_dtype()

        self.status = ADMMStatus(dtype=self.dtype)
        self.info = ADMMInfo(maxiter=self.maxiter, dtype=self.dtype)

        self.x = np.zeros_like(v_f)
        self.y = np.zeros_like(v_f)
        self.z = np.zeros_like(v_f)
        self.x_p = np.zeros_like(v_f)
        self.y_p = np.zeros_like(v_f)
        self.z_p = np.zeros_like(v_f)

        self.nbd = J.shape[1]
        self.ncts = J.shape[0]

        self.v = -(v_star + v_f)
        self.d = np.zeros_like(v_f)
        self.D = np.copy(D)
        if use_preconditioning:
            S = np.sqrt(np.reciprocal(np.abs(self.D.diagonal())))
            S = np.diag(S)
            self.v = S @ self.v
            self.D = S @ self.D @ S
        self.D += (self.eta + self.rho) * np.eye(D.shape[0])

        self.schur_solver.compute(A=self.D)

        for i in range(self.maxiter):
            self.status.iterations += 1

            self.d[:] = self.v + self.eta * self.x_p + self.rho * self.y_p + self.z_p
            self.x[:] = self.schur_solver.solve(b=self.d)
            if not self._check_state():
                break

            self._update_state()
            self._compute_residuals(i)
            if self._check_converged(i):
                break
            self._update_previous()

        self._set_result()
        self._truncate_info()
        if use_preconditioning:
            self.lambdas = S @ self.y
            self.v_plus = S @ self.z
        else:
            self.lambdas = self.y.copy()
            self.v_plus = self.z.copy()
        self.u_plus = compute_u_plus(u_minus=u_minus, invM=invM, J=J, h=h, lambdas=self.lambdas)

        return self.status

    def save_info(self, path: str, suffix: str = ""):
        import os  # noqa: PLC0415

        import matplotlib.pyplot as plt  # noqa: PLC0415

        plt.figure(figsize=(8, 4))
        plt.plot(self.info.r_p)
        plt.title("ADMM Primal Residual")
        plt.xlabel("Iteration")
        plt.ylabel("r_p")
        plt.grid(True)
        plt.savefig(os.path.join(path, f"admm_info_res_prim{suffix}.png"))
        plt.close()

        plt.figure(figsize=(8, 4))
        plt.plot(self.info.r_d)
        plt.title("ADMM Dual Residual")
        plt.xlabel("Iteration")
        plt.ylabel("r_d")
        plt.grid(True)
        plt.savefig(os.path.join(path, f"admm_info_res_dual{suffix}.png"))
        plt.close()

        plt.figure(figsize=(8, 4))
        plt.plot(self.info.r_c)
        plt.title("ADMM Compl. Residual")
        plt.xlabel("Iteration")
        plt.ylabel("r_c")
        plt.grid(True)
        plt.savefig(os.path.join(path, f"admm_info_res_compl{suffix}.png"))
        plt.close()

        plt.figure(figsize=(8, 4))
        plt.plot(self.info.r_i)
        plt.title("ADMM Iter. Residual")
        plt.xlabel("Iteration")
        plt.ylabel("r_i")
        plt.grid(True)
        plt.savefig(os.path.join(path, f"admm_info_res_iter{suffix}.png"))
        plt.close()
