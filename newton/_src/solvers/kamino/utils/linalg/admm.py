###########################################################################
# KAMINO: Utilities: Linear Algebra: ADMM solver in NumPy
###########################################################################

import numpy as np

from newton._src.solvers.kamino.utils.linalg.ldlt_eigen3 import LDLTEigen3
from newton._src.solvers.kamino.utils.linalg.llt_std import LLT

###
# Solvers
###


class ADMMStatus:
    def __init__(self, converged: bool = False, iterations: int = 0, message: str = ""):
        self.message = message
        self.converged: bool = converged
        self.iterations: int = iterations
        self.r_p: float = 0.0
        self.r_d: float = 0.0
        self.r_c: float = 0.0

    def __str__(self) -> str:
        return f"Converged: {self.converged}, Iterations: {self.iterations}, Info: {self.message}"


class ADMMInfo:
    def __init__(self, maxiter: int = 0, dtype: np.dtype = np.float32):
        self.r_p: np.ndarray = np.zeros(maxiter, dtype=dtype)
        self.r_d: np.ndarray = np.zeros(maxiter, dtype=dtype)
        self.r_c: np.ndarray = np.zeros(maxiter, dtype=dtype)

    def reset(self):
        self.r_p.fill(0.0)
        self.r_d.fill(0.0)
        self.r_c.fill(0.0)


class ADMMSolver:
    def __init__(
        self,
        primal_tolerance: float = 1e-6,
        dual_tolerance: float = 1e-6,
        compl_tolerance: float = 1e-6,
        eta: float = 1e-5,
        rho: float = 1.0,
        omega: float = 1.0,
        maxiter: int = 200,
    ):
        # Settings
        self.primal_tolerance = primal_tolerance
        self.dual_tolerance = dual_tolerance
        self.compl_tolerance = compl_tolerance
        self.eta = eta
        self.rho = rho
        self.omega = omega
        self.maxiter = maxiter

        # Meta-data
        self.status = ADMMStatus()
        self.info: ADMMInfo | None = None

        # Residuals
        self.r_p: float = 0.0
        self.r_d: float = 0.0
        self.r_c: float = 0.0

        # Factorizers
        self.llt = None
        self.ldlt = None

        # Schur Linear Problem
        self.v: np.ndarray | None = None
        self.D: np.ndarray | None = None
        self.vp: np.ndarray | None = None
        self.Dp: np.ndarray | None = None

        # KKT Linear Problem
        self.k: np.ndarray | None = None
        self.K: np.ndarray | None = None

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

    def _factorize(self, mat: np.ndarray, use_cholesky: bool, use_ldlt: bool):
        if use_cholesky:
            try:
                self.llt = LLT(mat)
            except np.linalg.LinAlgError:
                raise ValueError("Matrix is not positive definite!")
        elif use_ldlt:
            try:
                self.ldlt = LDLTEigen3(mat)
            except np.linalg.LinAlgError:
                raise ValueError("Matrix is not positive definite!")

    def _solve_system(self, mat: np.ndarray, vec: np.ndarray, use_cholesky: bool, use_ldlt: bool) -> np.ndarray:
        if use_cholesky:
            return self.llt.solve(vec)
        elif use_ldlt:
            return self.ldlt.solve(vec)
        else:
            return np.linalg.solve(mat, vec)

    def _update_variables(self):
        self.x = self.omega * self.x + (1.0 - self.omega) * self.y_p
        self.y = self.x - (1.0 / self.rho) * self.z_p
        self.z = self.z_p + self.rho * (self.y - self.x)

    def _compute_residuals(self, iter: int):
        """Compute the residuals for the current state."""
        self.r_p = self.x - self.y
        self.r_d = self.eta * (self.x - self.x_p) + self.rho * (self.y - self.y_p)
        self.r_c = np.dot(self.x, self.z)
        self.status.r_p = np.max(np.abs(self.r_p))
        self.status.r_d = np.max(np.abs(self.r_d))
        self.status.r_c = np.max(np.abs(self.r_c))
        self.info.r_p[iter] = self.status.r_p
        self.info.r_d[iter] = self.status.r_d
        self.info.r_c[iter] = self.status.r_c

    def _has_converged(self, iter: int) -> bool:
        """Check if the solver has converged based on the current iteration."""
        return (
            iter > 0
            and self.status.r_p < self.primal_tolerance
            and self.status.r_d < self.dual_tolerance
            and self.status.r_c < self.compl_tolerance
        )

    def _update_previous(self):
        self.x_p = self.x.copy()
        self.y_p = self.y.copy()
        self.z_p = self.z.copy()

    def _truncate_residuals(self):
        self.info.r_p = self.info.r_p[: self.status.iterations]
        self.info.r_d = self.info.r_d[: self.status.iterations]
        self.info.r_c = self.info.r_c[: self.status.iterations]

    def solve_schur(
        self, D: np.ndarray, v_f: np.ndarray, use_cholesky: bool = False, use_ldlt: bool = False
    ) -> ADMMStatus:
        self.info = ADMMInfo(maxiter=self.maxiter, dtype=D.dtype)

        self.v = np.zeros_like(v_f)
        self.D = np.copy(D)
        self.D += (self.eta + self.rho) * np.eye(D.shape[0])
        self._factorize(self.D, use_cholesky, use_ldlt)

        self.x = np.zeros_like(v_f)
        self.y = np.zeros_like(v_f)
        self.z = np.zeros_like(v_f)
        self.x_p = np.zeros_like(v_f)
        self.y_p = np.zeros_like(v_f)
        self.z_p = np.zeros_like(v_f)

        self.u_plus = None
        self.v_plus = np.zeros_like(v_f)
        self.lambdas = np.zeros_like(v_f)

        self.status = ADMMStatus()
        for i in range(self.maxiter):
            self.status.iterations += 1

            self.v = -v_f + self.eta * self.x_p + self.rho * self.y_p + self.z_p

            self.x = self._solve_system(self.D, self.v, use_cholesky, use_ldlt)

            self._update_variables()
            self._compute_residuals(i)
            if self._has_converged(i):
                self.status.converged = True
                break
            self._update_previous()

        self._truncate_residuals()
        self.v_plus = self.z.copy()
        self.lambdas = self.y.copy()

        return self.status

    def solve_schur_preconditioned(
        self, D: np.ndarray, v_f: np.ndarray, use_cholesky: bool = False, use_ldlt: bool = False
    ) -> ADMMStatus:
        self.info = ADMMInfo(maxiter=self.maxiter, dtype=D.dtype)

        self.v = np.zeros_like(v_f)
        self.vp = np.zeros_like(v_f)
        self.Dp = np.copy(D)

        S = np.sqrt(np.reciprocal(np.abs(self.Dp.diagonal())))
        S = np.diag(S)

        self.vp = S @ v_f
        self.Dp = S @ self.Dp @ S
        self.Dp += (self.eta + self.rho) * np.eye(self.Dp.shape[0])
        self._factorize(self.Dp, use_cholesky, use_ldlt)

        self.x = np.zeros_like(v_f)
        self.y = np.zeros_like(v_f)
        self.z = np.zeros_like(v_f)
        self.x_p = np.zeros_like(v_f)
        self.y_p = np.zeros_like(v_f)
        self.z_p = np.zeros_like(v_f)

        self.u_plus = None
        self.v_plus = np.zeros_like(v_f)
        self.lambdas = np.zeros_like(v_f)

        self.status = ADMMStatus()
        for i in range(self.maxiter):
            self.status.iterations += 1

            self.v = -self.vp + self.eta * self.x_p + self.rho * self.y_p + self.z_p

            self.x = self._solve_system(self.Dp, self.v, use_cholesky, use_ldlt)

            self._update_variables()
            self._compute_residuals(i)
            if self._has_converged(i):
                self.status.converged = True
                break
            self._update_previous()

        self._truncate_residuals()
        self.v_plus = S @ self.z
        self.lambdas = S @ self.y

        return self.status

    def solve_kkt(
        self, M: np.ndarray, J: np.ndarray, h: np.ndarray, u_p: np.ndarray, v_star: np.ndarray, use_ldlt: bool = False
    ) -> ADMMStatus:
        self.info = ADMMInfo(maxiter=self.maxiter, dtype=M.dtype)

        nbd = M.shape[0]
        ncts = J.shape[0]

        kdim = nbd + ncts
        self.K = np.zeros((kdim, kdim), dtype=M.dtype)
        self.k = np.zeros((kdim,), dtype=M.dtype)

        # self.K[:nbd, :nbd] = M
        # self.K[:nbd, nbd:] = J.T
        # self.K[nbd:, :nbd] = J
        # self.K[nbd:, nbd:] = - (self.eta + self.rho) * np.eye(ncts, dtype=M.dtype)
        # self.k[:nbd] = M @ u_p + h
        self.K[:ncts, :ncts] = -(self.eta + self.rho) * np.eye(ncts, dtype=M.dtype)
        self.K[:ncts, ncts:] = J
        self.K[ncts:, :ncts] = J.T
        self.K[ncts:, ncts:] = M
        self.k[ncts:] = M @ u_p + h
        self._factorize(self.K, False, use_ldlt)

        self.x = np.zeros_like(v_star)
        self.y = np.zeros_like(v_star)
        self.z = np.zeros_like(v_star)
        self.x_p = np.zeros_like(v_star)
        self.y_p = np.zeros_like(v_star)
        self.z_p = np.zeros_like(v_star)

        self.u_plus = np.zeros_like(u_p)
        self.v_plus = np.zeros_like(v_star)
        self.lambdas = np.zeros_like(v_star)

        self.status = ADMMStatus()
        for i in range(self.maxiter):
            self.status.iterations += 1

            # self.k[nbd:] = - v_star + self.eta * self.x_p + self.rho * self.y_p + self.z_p
            self.k[:ncts] = -v_star + self.eta * self.x_p + self.rho * self.y_p + self.z_p

            ux = self._solve_system(self.K, self.k, False, use_ldlt)

            # self.u_plus = ux[:nbd]
            # self.x = - ux[nbd:]
            self.x = -ux[:ncts]
            self.u_plus = ux[ncts:]

            self._update_variables()
            self._compute_residuals(i)
            if self._has_converged(i):
                self.status.converged = True
                break
            self._update_previous()

        self._truncate_residuals()
        self.lambdas = self.y.copy()

        return self.status

    def save_info(self, path: str, suffix: str = ""):
        import os

        import matplotlib.pyplot as plt

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
