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
KAMINO: Proximal ADMM DualProblem Solver

Provides a forward dynamics solver for constrained rigid multi-body systems using
the Alternating Direction Method of Multipliers (ADMM). This implementation realizes
the Proximal ADMM algorithm described in [1] and is based on the work of J. Carpentier
et al in [2]. It solves the Lagrange dual of the constrained forward dynamics problem
in constraint reactions (i.e. impulses) and post-event constraint-space velocities.

Notes
----
- ADMM is based on the Augmented Lagrangian Method (ALM).
- Proximal ADMM introduces an additional proximal regularization term to the optimization objective.
- Uses (optional) over-relaxation factor to improve convergence.
- Uses (optional) adaptive penalty updates based on the primal-dual residual balancing.

References
----
- [1] https://arxiv.org/abs/2504.19771
- [2] https://arxiv.org/pdf/2405.17020
- [3] https://onlinelibrary.wiley.com/doi/full/10.1002/nme.6693
"""

from enum import IntEnum

import warp as wp
from warp.context import Devicelike

from newton._src.solvers.kamino.core.math import FLOAT32_EPS
from newton._src.solvers.kamino.core.model import Model, ModelSize
from newton._src.solvers.kamino.core.types import float32, int32, vec3f
from newton._src.solvers.kamino.dynamics.dual import DualProblem
from newton._src.solvers.kamino.geometry.contacts import Contacts
from newton._src.solvers.kamino.kinematics.limits import Limits

###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Types
###


class PADMMPenaltyUpdate(IntEnum):
    """
    An enumeration of the penalty update methods used in PADMM.
    """

    FIXED = 0
    """Fixed penalty, no updates are performed."""
    LINEAR = 1
    """Linear penalty updates, rho is increased by a fixed factor."""
    SPECTRAL = 2
    """Spectral penalty updates, rho is increased by the spectral radius of the Delassus matrix."""


@wp.struct
class PADMMConfig:
    """
    A data type to hold the PADMM solver configurations. Intended for in-device storage.
    """

    primal_tolerance: float32
    """The tolerance applied to the primal residuals."""
    dual_tolerance: float32
    """The tolerance applied to the dual residuals."""
    compl_tolerance: float32
    """The tolerance applied to the complementarity residuals."""
    eta: float32
    """The proximal regularization parameter. Must be greater than zero."""
    rho_0: float32
    """The initial value of the penalty parameter. Must be greater than zero."""
    omega: float32
    """The over-relaxation factor. Must be in the range [0.0, 2.0]."""
    alpha: float32
    """The primal-dual residual threshold used to determine when penalty updates are needed."""
    tau_inc: float32
    """The factor by which the penalty is increased when the primal-dual residual exceeds the threshold."""
    max_iterations: int32
    """The maximum number of solver iterations."""
    penalty_update_freq: int32
    """The frequency of penalty updates. If zero, no updates are performed."""
    penalty_update_method: int32
    """The method used to update the penalty parameter. See :class:`PADMMPenaltyUpdate` for details."""


@wp.struct
class PADMMStatus:
    """
    A data type to hold the PADMM solver status. Intended for in-device storage.
    """

    converged: int32
    """A flag indicating whether the solver has converged (1) or not (0)."""
    iterations: int32
    """The number of iterations performed by the solver."""

    r_p: float32
    """The total primal residual according to the appropriate metric norm (currently the infinity norm)."""
    r_d: float32
    """The total dual residual according to the appropriate metric norm (currently the infinity norm)."""
    r_c: float32
    """The total complementarity residual according to the appropriate metric norm (currently the infinity norm)."""


@wp.struct
class PADMMPenalty:
    """
    A data type to hold the PADMM solver penalty state. Intended for in-device storage.
    """

    rho_updates: int32
    """The number of penalty updates performed. Also equals the number of Delassus factorizations."""
    rho_p: float32
    """The previous value of the ALM penalty parameter."""
    rho: float32
    """The current value of the ALM penalty parameter."""


###
# Containers
###


class PADMMSettings:
    """
    A class to hold the PADMM solver settings.
    """

    def __init__(self):
        self.primal_tolerance: float = 1e-6
        """The tolerance applied to the primal residuals."""
        self.dual_tolerance: float = 1e-6
        """The tolerance applied to the dual residuals."""
        self.compl_tolerance: float = 1e-6
        """The tolerance applied to the complementarity residuals."""
        self.eta: float = 1e-5
        """The proximal regularization parameter. Must be greater than zero."""
        self.rho_0: float = 1.0
        """The initial value of the penalty parameter. Must be greater than zero."""
        self.omega: float = 1.0
        """The over-relaxation factor. Must be in the range [0.0, 2.0]."""
        self.alpha: float = 10.0
        """The primal-dual residual threshold used to determine when penalty updates are needed."""
        self.tau_inc: float = 1.5
        """The factor by which the penalty is increased when the primal-dual residual exceeds the threshold."""
        self.max_iterations: int = 200
        """The maximum number of solver iterations."""
        self.penalty_update_freq: int = 1
        """The frequency of penalty updates. If zero, no updates are performed."""
        self.penalty_update_method: PADMMPenaltyUpdate = PADMMPenaltyUpdate.FIXED
        """The method used to update the penalty parameter. Defaults to fixed penalty (i.e. not adaptive)."""

    def to_config(self) -> PADMMConfig:
        """
        Convert the settings to a PADMMConfig object.
        """
        config = PADMMConfig()
        config.primal_tolerance = self.primal_tolerance
        config.dual_tolerance = self.dual_tolerance
        config.compl_tolerance = self.compl_tolerance
        config.eta = self.eta
        config.rho_0 = self.rho_0
        config.omega = self.omega
        config.alpha = self.alpha
        config.tau_inc = self.tau_inc
        config.max_iterations = self.max_iterations
        config.penalty_update_freq = self.penalty_update_freq
        config.penalty_update_method = self.penalty_update_method
        return config


class PADMMState:
    """
    An interface container to the PADMM solver internal state arrays.
    """

    def __init__(self, size: ModelSize | None = None):
        self.done: wp.array(dtype=int32) | None = None
        """A global flag indicating if the solver should terminate the solve operation."""

        self.s: wp.array(dtype=float32) | None = None
        """The De Saxce correction vector."""

        self.v: wp.array(dtype=float32) | None = None
        """The total velocity bias vector computed from the PADMM state and proximal parameters rho and eta."""

        self.x: wp.array(dtype=float32) | None = None
        """The current PADMM primal variables."""

        self.x_p: wp.array(dtype=float32) | None = None
        """The previous PADMM primal variables."""

        self.y: wp.array(dtype=float32) | None = None
        """The current PADMM slack variables."""

        self.y_p: wp.array(dtype=float32) | None = None
        """The previous PADMM slack variables."""

        self.z: wp.array(dtype=float32) | None = None
        """The current PADMM dual variables."""

        self.z_p: wp.array(dtype=float32) | None = None
        """The previous PADMM dual variables."""

        # Perform memory allocations if size is specified
        if size is not None:
            self.allocate(size)

    def allocate(self, size: ModelSize):
        self.done = wp.zeros(1, dtype=int32)
        self.s = wp.zeros(size.sum_of_max_total_cts, dtype=float32)
        self.v = wp.zeros(size.sum_of_max_total_cts, dtype=float32)
        self.x = wp.zeros(size.sum_of_max_total_cts, dtype=float32)
        self.x_p = wp.zeros(size.sum_of_max_total_cts, dtype=float32)
        self.y = wp.zeros(size.sum_of_max_total_cts, dtype=float32)
        self.y_p = wp.zeros(size.sum_of_max_total_cts, dtype=float32)
        self.z = wp.zeros(size.sum_of_max_total_cts, dtype=float32)
        self.z_p = wp.zeros(size.sum_of_max_total_cts, dtype=float32)

    def zero(self):
        self.s.zero_()
        self.v.zero_()
        self.x.zero_()
        self.x_p.zero_()
        self.y.zero_()
        self.y_p.zero_()
        self.z.zero_()
        self.z_p.zero_()


class PADMMResiduals:
    """
    An interface container to the PADMM solver residuals arrays.
    """

    def __init__(self, size: ModelSize | None = None):
        self.r_primal: wp.array(dtype=float32) | None = None
        """The PADMM primal residual vector."""
        self.r_dual: wp.array(dtype=float32) | None = None
        """The PADMM dual residual vector."""
        self.r_compl: wp.array(dtype=float32) | None = None
        """The PADMM complementarity residual vector."""

        # Perform memory allocations if size is specified
        if size is not None:
            self.allocate(size)

    def allocate(self, size: ModelSize):
        self.r_primal = wp.zeros(size.sum_of_max_total_cts, dtype=float32)
        self.r_dual = wp.zeros(size.sum_of_max_total_cts, dtype=float32)
        self.r_compl = wp.zeros(size.sum_of_max_unilaterals, dtype=float32)


class PADMMSolution:
    """
    An interface container to the PADMM solver solution arrays.
    """

    def __init__(self, size: ModelSize | None = None):
        self.lambdas: wp.array(dtype=float32) | None = None
        """The constraint reactions (i.e. impulses) solution array."""
        self.v_plus: wp.array(dtype=float32) | None = None
        """The post-event constraint-space velocities solution array."""

        # Perform memory allocations if size is specified
        if size is not None:
            self.allocate(size)

    def allocate(self, size: ModelSize):
        self.lambdas = wp.zeros(size.sum_of_max_total_cts, dtype=float32)
        self.v_plus = wp.zeros(size.sum_of_max_total_cts, dtype=float32)


class PADMMInfo:
    """
    A container to hold the PADMM solver convergence info arrays.

    Notes:
    - The length of the arrays is determined by the maximum number of iterations
    and is filled up to the number of iterations performed by the solver on each
    solve. This allows for post-solve analysis of the convergence behavior.
    - This has a significant impact on solver performance and memory usage, so it
    is recommended to only enable this for testing and debugging purposes.
    """

    def __init__(
        self,
        size: ModelSize | None = None,
        max_iters: int | None = None,
    ):
        self.lambdas: wp.array(dtype=float32) | None = None
        """
        The constraint reactions (i.e. impulses) of each world.\n
        """

        self.v_plus: wp.array(dtype=float32) | None = None
        """
        The post-event constraint-space velocities of each world.\n
        This is computed using the current solution as: `v_plus := v_f + D @ lambdas`.
        """

        self.v_aug: wp.array(dtype=float32) | None = None
        """
        The post-event augmented constraint-space velocities of each world.\n
        This is computed using the current solution as: `v_aug := v_plus + s`.
        """

        self.s: wp.array(dtype=float32) | None = None
        """
        The De Saxce correction velocities of each world.\n
        This is computed using the current solution as: `s := G(v_plus)`.
        """

        self.offsets: wp.array(dtype=int32) | None = None
        """The residuals index offset of each world."""

        self.norm_s: wp.array(dtype=float32) | None = None
        """The per-solve history of the L2 norm of the De Saxce correction variables."""

        self.norm_x: wp.array(dtype=float32) | None = None
        """The per-solve history of the L2 norm of the primal variables."""

        self.norm_y: wp.array(dtype=float32) | None = None
        """The per-solve history of the L2 norm of the slack variables."""

        self.norm_z: wp.array(dtype=float32) | None = None
        """The per-solve history of the L2 norm of the dual variables."""

        self.f_ccp: wp.array(dtype=float32) | None = None
        """The per-solve history of the CCP optimization objective."""

        self.f_ncp: wp.array(dtype=float32) | None = None
        """The per-solve history of the NCP optimization objective."""

        self.r_dx: wp.array(dtype=float32) | None = None
        """The per-solve history of the primal iterate residual."""

        self.r_dy: wp.array(dtype=float32) | None = None
        """The per-solve history of the slack iterate residual."""

        self.r_dz: wp.array(dtype=float32) | None = None
        """The per-solve history of the dual iterate residual."""

        self.r_primal: wp.array(dtype=float32) | None = None
        """The per-solve history of PADMM primal residuals."""

        self.r_dual: wp.array(dtype=float32) | None = None
        """The per-solve history of PADMM dual residuals."""

        self.r_compl: wp.array(dtype=float32) | None = None
        """The per-solve history of PADMM complementarity residuals."""

        self.r_pd: wp.array(dtype=float32) | None = None
        """The per-solve history of PADMM primal-dual residuals ratio."""

        self.r_dp: wp.array(dtype=float32) | None = None
        """The per-solve history of PADMM dual-primal residuals ratio."""

        self.r_ncp_primal: wp.array(dtype=float32) | None = None
        """The per-solve history of NCP primal residuals."""

        self.r_ncp_dual: wp.array(dtype=float32) | None = None
        """The per-solve history of NCP dual residuals."""

        self.r_ncp_compl: wp.array(dtype=float32) | None = None
        """The per-solve history of NCP complementarity residuals."""

        self.r_ncp_natmap: wp.array(dtype=float32) | None = None
        """The per-solve history of NCP natural-map residuals."""

        # Perform memory allocations if size is specified
        if max_iters is not None:
            self.allocate(size=size, max_iters=max_iters)

    def allocate(self, size: ModelSize, max_iters: int):
        # Ensure num_worlds is valid
        if not isinstance(size.num_worlds, int) or size.num_worlds <= 0:
            raise ValueError("num_worlds must be a positive integer specifying the number of worlds.")

        # Ensure max_iters is valid
        if not isinstance(max_iters, int) or max_iters <= 0:
            raise TypeError("max_iters must be a positive integer specifying the maximum number of iterations.")

        # Allocate intermediate arrays
        self.lambdas = wp.zeros(size.sum_of_max_total_cts, dtype=float32)
        self.v_plus = wp.zeros(size.sum_of_max_total_cts, dtype=float32)
        self.v_aug = wp.zeros(size.sum_of_max_total_cts, dtype=float32)
        self.s = wp.zeros(size.sum_of_max_total_cts, dtype=float32)

        # Compute the index offsets for the info of each world
        maxsize = max_iters * size.num_worlds
        offsets = [max_iters * i for i in range(size.num_worlds)]

        # Allocate the in-device solver info data arrays
        self.offsets = wp.array(offsets, dtype=int32)
        self.norm_s = wp.zeros(maxsize, dtype=float32)
        self.norm_x = wp.zeros(maxsize, dtype=float32)
        self.norm_y = wp.zeros(maxsize, dtype=float32)
        self.norm_z = wp.zeros(maxsize, dtype=float32)
        self.r_dx = wp.zeros(maxsize, dtype=float32)
        self.r_dy = wp.zeros(maxsize, dtype=float32)
        self.r_dz = wp.zeros(maxsize, dtype=float32)
        self.f_ccp = wp.zeros(maxsize, dtype=float32)
        self.f_ncp = wp.zeros(maxsize, dtype=float32)
        self.r_primal = wp.zeros(maxsize, dtype=float32)
        self.r_dual = wp.zeros(maxsize, dtype=float32)
        self.r_compl = wp.zeros(maxsize, dtype=float32)
        self.r_pd = wp.zeros(maxsize, dtype=float32)
        self.r_dp = wp.zeros(maxsize, dtype=float32)
        self.r_ncp_primal = wp.zeros(maxsize, dtype=float32)
        self.r_ncp_dual = wp.zeros(maxsize, dtype=float32)
        self.r_ncp_compl = wp.zeros(maxsize, dtype=float32)
        self.r_ncp_natmap = wp.zeros(maxsize, dtype=float32)

    def zero(self):
        self.lambdas.zero_()
        self.v_plus.zero_()
        self.v_aug.zero_()
        self.s.zero_()
        self.norm_s.zero_()
        self.norm_x.zero_()
        self.norm_y.zero_()
        self.norm_z.zero_()
        self.f_ccp.zero_()
        self.f_ncp.zero_()
        self.r_dx.zero_()
        self.r_dy.zero_()
        self.r_dz.zero_()
        self.r_primal.zero_()
        self.r_dual.zero_()
        self.r_compl.zero_()
        self.r_pd.zero_()
        self.r_dp.zero_()
        self.r_ncp_primal.zero_()
        self.r_ncp_dual.zero_()
        self.r_ncp_compl.zero_()
        self.r_ncp_natmap.zero_()


# An interface container to the solver data
class PADMMData:
    """
    An interface container to the PADMM solver data arrays.
    This high-level container bundles all solver data into a single object.
    """

    def __init__(
        self, size: ModelSize | None = None, max_iters: int = 0, collect_info: bool = False, device: Devicelike = None
    ):
        self.config: wp.array(dtype=PADMMConfig) | None = None
        """The array of PADMM solver configurations. Shape is (nw,) and type :class:`PADMMConfig`."""
        self.status: wp.array(dtype=PADMMStatus) | None = None
        """The array of PADMM solver status. Shape is (nw,) and type :class:`PADMMStatus`."""
        self.penalty: wp.array(dtype=PADMMPenalty) | None = None
        """The array of PADMM solver penalty parameters. Shape is (nw,) and type :class:`PADMMPenalty`."""
        self.state: PADMMState | None = None
        """The PADMM internal solver state."""
        self.residuals: PADMMResiduals | None = None
        """The PADMM residuals."""
        self.solution: PADMMSolution | None = None
        """The PADMM solution."""
        self.info: PADMMInfo | None = None
        """The (optional) PADMM convergence info."""

        # Perform memory allocations if size is specified
        if size is not None:
            self.allocate(size=size, max_iters=max_iters, collect_info=collect_info, device=device)

    def allocate(self, size: ModelSize, max_iters: int = 0, collect_info: bool = False, device: Devicelike = None):
        with wp.ScopedDevice(device):
            self.config = wp.zeros(shape=(size.num_worlds,), dtype=PADMMConfig)
            self.status = wp.zeros(shape=(size.num_worlds,), dtype=PADMMStatus)
            self.penalty = wp.zeros(shape=(size.num_worlds,), dtype=PADMMPenalty)
            self.state = PADMMState(size)
            self.residuals = PADMMResiduals(size)
            self.solution = PADMMSolution(size)
            if collect_info and max_iters > 0:
                self.info = PADMMInfo(size=size, max_iters=max_iters)


###
# Functions
###


@wp.func
def project_to_coulomb_cone(x: vec3f, mu: float32, epsilon: float32 = 0.0) -> vec3f:
    """
    Projects a 3D vector `x` onto an isotropic Coulomb friction cone defined by the friction coefficient `mu`.

    Args:
        x (vec3f): The input vector to be projected.
        mu (float32): The friction coefficient defining the aperture of the cone.
        epsilon (float32, optional): A numerical tolerance applied to the cone boundary. Defaults to 0.0.

    Returns:
        vec3f: The vector projected onto the Coulomb cone.
    """
    xn = x[2]
    xt_norm = wp.sqrt(x[0] * x[0] + x[1] * x[1])
    y = wp.vec3f(0.0)
    if mu * xt_norm > -xn + epsilon:
        if xt_norm <= mu * xn + epsilon:
            y = x
        else:
            ys = (mu * xt_norm + xn) / (mu * mu + 1.0)
            yts = mu * ys / xt_norm
            y[0] = yts * x[0]
            y[1] = yts * x[1]
            y[2] = ys
    return y


@wp.func
def project_to_coulomb_dual_cone(x: vec3f, mu: float32, epsilon: float32 = 0.0) -> vec3f:
    """
    Projects a 3D vector `x` onto the dual of an isotropic Coulomb friction cone defined by the friction coefficient `mu`.

    Args:
        x (vec3f): The input vector to be projected.
        mu (float32): The friction coefficient defining the aperture of the cone.
        epsilon (float32, optional): A numerical tolerance applied to the cone boundary. Defaults to 0.0.

    Returns:
        vec3f: The vector projected onto the dual Coulomb cone.
    """
    xn = x[2]
    xt_norm = wp.sqrt(x[0] * x[0] + x[1] * x[1])
    y = wp.vec3f(0.0)
    if xt_norm > -mu * xn + epsilon:
        if mu * xt_norm <= xn + epsilon:
            y = x
        else:
            ys = (xt_norm + mu * xn) / (mu * mu + 1.0)
            yts = ys / xt_norm
            y[0] = yts * x[0]
            y[1] = yts * x[1]
            y[2] = mu * ys
    return y


@wp.func
def compute_inf_norm(
    dim: int32,
    vio: int32,
    x: wp.array(dtype=float32),
) -> float32:
    norm = float(0.0)
    for i in range(dim):
        norm = wp.max(norm, wp.abs(x[vio + i]))
    return norm


@wp.func
def compute_l1_norm(
    dim: int32,
    vio: int32,
    x: wp.array(dtype=float32),
) -> float32:
    sum = float(0.0)
    for i in range(dim):
        sum += wp.abs(x[vio + i])
    return sum


@wp.func
def compute_l2_norm(
    dim: int32,
    vio: int32,
    x: wp.array(dtype=float32),
) -> float32:
    sum = float(0.0)
    for i in range(dim):
        x_i = x[vio + i]
        sum += x_i * x_i
    return wp.sqrt(sum)


@wp.func
def compute_dot_product(
    dim: int32,
    vio: int32,
    x: wp.array(dtype=float32),
    y: wp.array(dtype=float32),
) -> float32:
    """
    Computes the dot (i.e. inner) product between two vectors `x` and `y` stored in flat arrays.\n
    Both vectors are of dimension `dim`, starting from the vector index offset `vio`.

    Args:
        dim (int32): The dimension (i.e. size) of the vectors.
        vio (int32): The vector index offset (i.e. start index).
        x (wp.array(dtype=float32)): The first vector.
        y (wp.array(dtype=float32)): The second vector.

    Returns:
        float32: The dot product of the two vectors.
    """
    product = float(0.0)
    for i in range(dim):
        v_i = vio + i
        product += x[v_i] * y[v_i]
    return product


@wp.func
def compute_double_dot_product(
    dim: int32,
    vio: int32,
    x: wp.array(dtype=float32),
    y: wp.array(dtype=float32),
    z: wp.array(dtype=float32),
) -> float32:
    """
    Computes the the inner product `x.T @ (y + z)` between a vector `x` and the sum of two vectors `y` and `z`.\n
    All vectors are stored in flat arrays, with dimension `dim` and starting from the vector index offset `vio`.

    Args:
        dim (int32): The dimension (i.e. size) of the vectors.
        vio (int32): The vector index offset (i.e. start index).
        x (wp.array(dtype=float32)): The first vector.
        y (wp.array(dtype=float32)): The second vector.
        z (wp.array(dtype=float32)): The third vector.

    Returns:
        float32: The inner product of `x` with the sum of `y` and `z`.
    """
    product = float(0.0)
    for i in range(dim):
        v_i = vio + i
        product += x[v_i] * (y[v_i] + z[v_i])
    return product


@wp.func
def compute_vector_sum(
    dim: int32, vio: int32, x: wp.array(dtype=float32), y: wp.array(dtype=float32), z: wp.array(dtype=float32)
):
    """
    Computes the sum of two vectors `x` and `y` and stores the result in vector `z`.\n
    All vectors are stored in flat arrays, with dimension `dim` and starting from the vector index offset `vio`.

    Args:
        dim (int32): The dimension (i.e. size) of the vectors.
        vio (int32): The vector index offset (i.e. start index).
        x (wp.array(dtype=float32)): The first vector.
        y (wp.array(dtype=float32)): The second vector.
        z (wp.array(dtype=float32)): The output vector where the sum is stored.

    Returns:
        None: The result is stored in the output vector `z`.
    """
    for i in range(dim):
        v_i = vio + i
        z[v_i] = x[v_i] + y[v_i]


@wp.func
def compute_cwise_vec_mul(
    dim: int32,
    vio: int32,
    a: wp.array(dtype=float32),
    x: wp.array(dtype=float32),
    y: wp.array(dtype=float32),
):
    """
    Computes the coefficient-wise vector-vector product `y =  a * x`.\n
    """
    for i in range(dim):
        v_i = vio + i
        y[v_i] = a[v_i] * x[v_i]


@wp.func
def compute_gemv(
    maxdim: int32,
    dim: int32,
    vio: int32,
    mio: int32,
    sigma: float32,
    P: wp.array(dtype=float32),
    A: wp.array(dtype=float32),
    x: wp.array(dtype=float32),
    b: wp.array(dtype=float32),
    c: wp.array(dtype=float32),
):
    """
    Computes the generalized matrix-vector product `c =  b + (A - sigma * I_n)@ x`.\n

    The matrix `A` is stored using row-major order in flat array with allocation size `maxdim x maxdim`,
    starting from the matrix index offset `mio`. The active dimensions of the matrix are `dim x dim`,
    where `dim` is the number of rows and columns. The vectors `x, b, c` are stored in flat arrays with
    dimensions `dim`, starting from the vector index offset `vio`.

    Args:
        maxdim (int32): The maximum dimension of the matrix `A`.
        dim (int32): The active dimension of the matrix `A` and the vectors `x, b, c`.
        vio (int32): The vector index offset (i.e. start index) for the vectors `x, b, c`.
        mio (int32): The matrix index offset (i.e. start index) for the matrix `A`.
        A (wp.array(dtype=float32)):
            The matrix `A` stored in row-major order.
        x (wp.array(dtype=float32)):
            The input vector `x` to be multiplied with the matrix `A`.
        b (wp.array(dtype=float32)):
            The input vector `b` to be added to the product `A @ x`.
        c (wp.array(dtype=float32)):
            The output vector `c` where the result of the operation is stored.

    Returns:
        None: The result is stored in the output vector `c`.
    """
    b_i = float(0.0)
    x_j = float(0.0)
    for i in range(dim):
        v_i = vio + i
        m_i = mio + maxdim * i
        b_i = b[v_i]
        for j in range(dim):
            x_j = x[vio + j]
            b_i += A[m_i + j] * x_j
        b_i -= sigma * x[v_i]
        c[v_i] = (1.0 / P[v_i]) * b_i


@wp.func
def compute_desaxce_corrections(
    nc: int32,
    cio: int32,
    vio: int32,
    ccgo: int32,
    mu: wp.array(dtype=float32),
    v_plus: wp.array(dtype=float32),
    s: wp.array(dtype=float32),
):
    """
    Computes the De Saxce correction for each active contact.

    `s = G(v) := [ 0, 0 , mu * || vt ||_2,]^T, where v := [vtx, vty, vn]^T, vt := [vtx, vty]^T`

    Args:
        nc (int32): The number of active contact constraints.
        cio (int32): The contact index offset (i.e. start index) for the contacts.
        vio (int32): The vector index offset (i.e. start index)
        ccgo (int32): The contact constraint group offset (i.e. start index)
        mu (wp.array(dtype=float32)):
            The array of friction coefficients for each contact constraint.
        v_plus (wp.array(dtype=float32)):
            The post-event constraint-space velocities array, which contains the tangential velocities `vtx`
            and `vty` for each contact constraint.
        s (wp.array(dtype=float32)):
            The output array where the De Saxce corrections are stored.\n
            The size of this array should be at least `vio + ccgo + 3 * nc`, where `vio` is the vector index offset,
            `ccgo` is the contact constraint group offset, and `nc` is the number of active contact constraints.

    Returns:
        None: The De Saxce corrections are stored in the output array `s`.
    """
    # Iterate over each active contact
    for k in range(nc):
        # Compute the contact index
        c_k = cio + k

        # Compute the constraint vector index
        v_k = vio + ccgo + 3 * k

        # Retrieve the friction coefficient for this contact
        mu_k = mu[c_k]

        # Compute the 2D norm of the tangential velocity
        vtx_k = v_plus[v_k]
        vty_k = v_plus[v_k + 1]
        vt_norm_k = wp.sqrt(vtx_k * vtx_k + vty_k * vty_k)

        # Store De Saxce correction for this block
        s[v_k] = 0.0
        s[v_k + 1] = 0.0
        s[v_k + 2] = mu_k * vt_norm_k


@wp.func
def compute_ncp_primal_residual(
    nl: int32,
    nc: int32,
    vio: int32,
    lcgo: int32,
    ccgo: int32,
    cio: int32,
    mu: wp.array(dtype=float32),
    lambdas: wp.array(dtype=float32),
) -> float32:
    """
    Computes the NCP primal residual as: `r_p := || lambda - proj_K(lambda) ||_inf`, where:
    - `lambda` is the vector of constraint reactions (i.e. impulses)
    - `proj_K()` is the projection operator onto the cone `K`
    - `K` is the total cone defined by the unilateral constraints such as limits and contacts
    - `|| . ||_inf` is the infinity norm (i.e. maximum absolute value of the vector components)

    Notes:
    - The cone for joint constraints is all of `R^njc`, so projection is a no-op.
    - For limit constraints, the cone is defined as `K_l := { lambda | lambda >= 0 }`
    - For contact constraints, the cone is defined as `K_c := { lambda | || lambda ||_2 <= mu * || vn ||_2 }`

    Args:
        nl (int32): The number of active limit constraints.
        nc (int32): The number of active contact constraints.
        vio (int32): The vector index offset (i.e. start index) for the constraints.
        lcgo (int32): The limit constraint group offset (i.e. start index).
        ccgo (int32): The contact constraint group offset (i.e. start index).
        cio (int32): The contact index offset (i.e. start index) for the contacts.
        mu (wp.array(dtype=float32)):
            The array of friction coefficients for each contact.
        lambdas (wp.array(dtype=float32)):
            The array of constraint reactions (i.e. impulses).

    Returns:
        float32: The maximum primal residual across all constraints, computed as the infinity norm.
    """
    # Initialize the primal residual
    r_ncp_p = float(0.0)

    # NOTE: We skip the joint constraint reactions are not bounded, the cone is all of R^njc

    for lid in range(nl):
        # Compute the limit constraint index offset
        lcio = vio + lcgo + lid
        # Compute the primal residual for the limit constraints
        lambda_l = lambdas[lcio]
        lambda_l -= wp.max(0.0, lambda_l)
        r_ncp_p = wp.max(r_ncp_p, wp.abs(lambda_l))

    for cid in range(nc):
        # Compute the contact constraint index offset
        ccio = vio + ccgo + 3 * cid
        # Retrieve the friction coefficient for this contact
        mu_c = mu[cio + cid]
        # Compute the primal residual for the contact constraints
        lambda_c = vec3f(lambdas[ccio], lambdas[ccio + 1], lambdas[ccio + 2])
        lambda_c -= project_to_coulomb_cone(lambda_c, mu_c)
        r_ncp_p = wp.max(r_ncp_p, wp.max(wp.abs(lambda_c)))

    # Return the maximum primal residual
    return r_ncp_p


@wp.func
def compute_ncp_dual_residual(
    njc: int32,
    nl: int32,
    nc: int32,
    vio: int32,
    lcgo: int32,
    ccgo: int32,
    cio: int32,
    mu: wp.array(dtype=float32),
    v_aug: wp.array(dtype=float32),
) -> float32:
    """
    Computes the NCP dual residual as: `r_d := || v_aug - proj_K^*(v_aug) ||_inf`, where:
    - `v_aug` is the vector of augmented constraint velocities: v_aug := v_plus + s
    - `v_plus` is the post-event constraint-space velocities
    - `s` is the De Saxce correction vector
    - `proj_K^*()` is the projection operator onto the dual cone `K^*`
    - `K^*` is the dual of the total cone defined by the unilateral constraints such as limits and contacts
    - `|| . ||_inf` is the infinity norm (i.e. maximum absolute value of the vector components)

    Notes:
    - The dual cone for joint constraints is the origin point x=0.
    - For limit constraints, the cone is defined as `K_l := { lambda | lambda >= 0 }`
    - For contact constraints, the cone is defined as `K_c := { lambda | || lambda ||_2 <= mu * || vn ||_2 }`

    Args:
        njc (int32): The number of joint constraints.
        nl (int32): The number of active limit constraints.
        nc (int32): The number of active contact constraints.
        vio (int32): The vector index offset (i.e. start index) for the constraints.
        lcgo (int32): The limit constraint group offset (i.e. start index).
        ccgo (int32): The contact constraint group offset (i.e. start index).
        cio (int32): The contact index offset (i.e. start index) for the contacts.
        mu (wp.array(dtype=float32)):
            The array of friction coefficients for each contact constraint.
        v_aug (wp.array(dtype=float32)):
            The array of augmented constraint velocities.

    Returns:
        float32: The maximum dual residual across all constraints, computed as the infinity norm.
    """
    # Initialize the dual residual
    r_ncp_d = float(0.0)

    for jid in range(njc):
        # Compute the joint constraint index offset
        jcio_j = vio + jid
        # Compute the dual residual for the joint constraints
        # NOTE #1: Each constraint-space velocity for joint should be zero
        # NOTE #2: the dual of R^njc is the origin zero vector
        v_j = v_aug[jcio_j]
        r_ncp_d = wp.max(r_ncp_d, wp.abs(v_j))

    for lid in range(nl):
        # Compute the limit constraint index offset
        lcio_l = vio + lcgo + lid
        # Compute the dual residual for the limit constraints
        # NOTE: Each constraint-space velocity should be non-negative
        v_l = v_aug[lcio_l]
        v_l -= wp.max(0.0, v_l)
        r_ncp_d = wp.max(r_ncp_d, wp.abs(v_l))

    for cid in range(nc):
        # Compute the contact constraint index offset
        ccio_c = vio + ccgo + 3 * cid
        # Retrieve the friction coefficient for this contact
        mu_c = mu[cio + cid]
        # Compute the dual residual for the contact constraints
        # NOTE: Each constraint-space velocity should be lie in the dual of the Coulomb friction cone
        v_c = vec3f(v_aug[ccio_c], v_aug[ccio_c + 1], v_aug[ccio_c + 2])
        v_c -= project_to_coulomb_dual_cone(v_c, mu_c)
        r_ncp_d = wp.max(r_ncp_d, wp.max(wp.abs(v_c)))

    # Return the maximum dual residual
    return r_ncp_d


@wp.func
def compute_ncp_complementarity_residual(
    nl: int32,
    nc: int32,
    vio: int32,
    lcgo: int32,
    ccgo: int32,
    v_aug: wp.array(dtype=float32),
    lambdas: wp.array(dtype=float32),
) -> float32:
    """
    Computes the NCP complementarity residual as `r_c := || lambda.dot(v_plus + s) ||_inf`

    Satisfaction of the complementarity condition `lambda _|_ (v_plus + s))` is measured
    using the per-constraint entity inner product, i.e. per limit and per contact. Thus,
    for each limit constraint `k`, we compute `v_k * lambda_k` and for each contact
    constraint `k`, we compute `v_k.dot(lambda_k)`.

    Args:
        nl (int32): The number of active limit constraints.
        nc (int32): The number of active contact constraints.
        vio (int32): The vector index offset (i.e. start index) for the constraints.
        lcgo (int32): The limit constraint group offset (i.e. start index).
        ccgo (int32): The contact constraint group offset (i.e. start index).
        v_aug (wp.array(dtype=float32)):
            The array of augmented constraint velocities.
        lambdas (wp.array(dtype=float32)):
            The array of constraint reactions (i.e. impulses).

    Returns:
        float32: The maximum complementarity residual across all constraints, computed as the infinity norm.
    """
    # Initialize the complementarity residual
    r_ncp_c = float(0.0)

    for lid in range(nl):
        # Compute the limit constraint index offset
        lcio = vio + lcgo + lid
        # Compute the complementarity residual for the limit constraints
        v_l = v_aug[lcio]
        lambda_l = lambdas[lcio]
        r_ncp_c = wp.max(r_ncp_c, wp.abs(v_l * lambda_l))

    for cid in range(nc):
        # Compute the contact constraint index offset
        ccio = vio + ccgo + 3 * cid
        # Compute the complementarity residual for the contact constraints
        v_c = vec3f(v_aug[ccio], v_aug[ccio + 1], v_aug[ccio + 2])
        lambda_c = vec3f(lambdas[ccio], lambdas[ccio + 1], lambdas[ccio + 2])
        r_ncp_c = wp.max(r_ncp_c, wp.abs(wp.dot(v_c, lambda_c)))

    # Return the maximum complementarity residual
    return r_ncp_c


@wp.func
def compute_ncp_natural_map_residual(
    nl: int32,
    nc: int32,
    vio: int32,
    lcgo: int32,
    ccgo: int32,
    cio: int32,
    mu: wp.array(dtype=float32),
    v_aug: wp.array(dtype=float32),
    lambdas: wp.array(dtype=float32),
) -> float32:
    """
    Computes the natural-map residuals as: `r_natmap = || lambda - proj_K(lambda - (v + s)) ||_inf`

    Args:
        nl (int32): The number of active limit constraints.
        nc (int32): The number of active contact constraints.
        vio (int32): The vector index offset (i.e. start index) for the constraints.
        lcgo (int32): The limit constraint group offset (i.e. start index).
        ccgo (int32): The contact constraint group offset (i.e. start index).
        cio (int32): The contact index offset (i.e. start index) for the contacts.
        mu (wp.array(dtype=float32)):
            The array of friction coefficients for each contact.
        v_aug (wp.array(dtype=float32)):
            The array of augmented constraint velocities.
        lambdas (wp.array(dtype=float32)):
            The array of constraint reactions (i.e. impulses).
    """

    # Initialize the natural-map residual
    r_ncp_natmap = float(0.0)

    for lid in range(nl):
        # Compute the limit constraint index offset
        lcio = vio + lcgo + lid
        # Compute the natural-map residual for the limit constraints
        v_l = v_aug[lcio]
        lambda_l = lambdas[lcio]
        lambda_l -= wp.max(0.0, lambda_l - v_l)
        r_ncp_natmap = wp.max(r_ncp_natmap, wp.abs(lambda_l))

    for cid in range(nc):
        # Compute the contact constraint index offset
        ccio = vio + ccgo + 3 * cid
        # Retrieve the friction coefficient for this contact
        mu_c = mu[cio + cid]
        # Compute the natural-map residual for the contact constraints
        v_c = vec3f(v_aug[ccio], v_aug[ccio + 1], v_aug[ccio + 2])
        lambda_c = vec3f(lambdas[ccio], lambdas[ccio + 1], lambdas[ccio + 2])
        lambda_c -= project_to_coulomb_cone(lambda_c - v_c, mu_c)
        r_ncp_natmap = wp.max(r_ncp_natmap, wp.max(wp.abs(lambda_c)))

    # Return the maximum natural-map residual
    return r_ncp_natmap


@wp.func
def compute_preconditioned_iterate_residual(
    ncts: int32, vio: int32, P: wp.array(dtype=float32), x: wp.array(dtype=float32), x_p: wp.array(dtype=float32)
) -> float32:
    """
    Computes the iterate residual as: `r_dx := || P @ (x - x_p) ||_inf`

    Args:
        ncts (int32): The number of active constraints in the world.
        vio (int32): The vector index offset (i.e. start index) for the constraints.
        x (wp.array(dtype=float32)):
            The current solution vector.
        x_p (wp.array(dtype=float32)):
            The previous solution vector.

    Returns:
        float32: The maximum iterate residual across all active constraints, computed as the infinity norm.
    """
    # Initialize the iterate residual
    r_dx = float(0.0)
    for i in range(ncts):
        # Compute the index offset of the vector block of the world
        v_i = vio + i
        # Update the iterate and proximal-point residuals
        r_dx = wp.max(r_dx, P[v_i] * wp.abs(x[v_i] - x_p[v_i]))
    # Return the maximum iterate residual
    return r_dx


@wp.func
def compute_inverse_preconditioned_iterate_residual(
    ncts: int32, vio: int32, P: wp.array(dtype=float32), x: wp.array(dtype=float32), x_p: wp.array(dtype=float32)
) -> float32:
    """
    Computes the iterate residual as: `r_dx := || P^{-1} @ (x - x_p) ||_inf`

    Args:
        ncts (int32): The number of active constraints in the world.
        vio (int32): The vector index offset (i.e. start index) for the constraints.
        x (wp.array(dtype=float32)):
            The current solution vector.
        x_p (wp.array(dtype=float32)):
            The previous solution vector.

    Returns:
        float32: The maximum iterate residual across all active constraints, computed as the infinity norm.
    """
    # Initialize the iterate residual
    r_dx = float(0.0)
    for i in range(ncts):
        # Compute the index offset of the vector block of the world
        v_i = vio + i
        # Update the iterate and proximal-point residuals
        r_dx = wp.max(r_dx, (1.0 / P[v_i]) * wp.abs(x[v_i] - x_p[v_i]))
    # Return the maximum iterate residual
    return r_dx


###
# Kernels
###


# TODO: Make multiple initialization kernels for the different penalty update methods
@wp.kernel
def _initialize_solver(
    # Inputs:
    solver_config: wp.array(dtype=PADMMConfig),
    # Outputs:
    solver_status: wp.array(dtype=PADMMStatus),
    solver_penalty: wp.array(dtype=PADMMPenalty),
):
    # Retrive the world index as thread index
    wid = wp.tid()

    # Retrieve the solver configuration
    config = solver_config[wid]

    # Initialize status
    s = solver_status[wid]
    s.iterations = int(0)
    s.converged = int(0)
    s.r_p = float(0.0)
    s.r_d = float(0.0)
    s.r_c = float(0.0)
    solver_status[wid] = s

    # Initialize penalty
    # NOTE: Currently only fixed penalty is used
    p = solver_penalty[wid]
    p.rho = config.rho_0
    p.rho_p = float(0.0)
    p.rho_updates = int(0)
    solver_penalty[wid] = p


@wp.kernel
def _update_delassus_proximal_regularization(
    # Inputs:
    problem_maxdim: wp.array(dtype=int32),
    problem_dim: wp.array(dtype=int32),
    problem_mio: wp.array(dtype=int32),
    solver_config: wp.array(dtype=PADMMConfig),
    solver_penalty: wp.array(dtype=PADMMPenalty),
    solver_status: wp.array(dtype=PADMMStatus),
    # Outputs:
    D: wp.array(dtype=float32),
):
    # Retrieve the thread index
    wid, tid = wp.tid()

    # Retrieve the number of active constraints in the world
    ncts = problem_dim[wid]

    # Retrieve the solver status
    status = solver_status[wid]

    # Skip if row index exceed the problem size or if the solver has already converged
    if tid >= ncts or status.converged > 0:
        return

    # Retrieve the maximum number of dimensions of the world
    maxdim = problem_maxdim[wid]

    # Retrieve the matrix index offset of the world
    mio = problem_mio[wid]

    # Retrieve the solver parameters
    cfg = solver_config[wid]
    pen = solver_penalty[wid]

    # Extract the regularization parameters
    rho_p = pen.rho_p
    rho = pen.rho
    eta = cfg.eta

    # Add the proximal regularization to the diagonal of the Delassus matrix
    D[mio + maxdim * tid + tid] += eta + (rho - rho_p)


@wp.kernel
def _compute_desaxce_correction(
    # Inputs:
    problem_nc: wp.array(dtype=int32),
    problem_cio: wp.array(dtype=int32),
    problem_ccgo: wp.array(dtype=int32),
    problem_vio: wp.array(dtype=int32),
    problem_mu: wp.array(dtype=float32),
    solver_status: wp.array(dtype=PADMMStatus),
    solver_z_p: wp.array(dtype=float32),
    # Outputs:
    solver_s: wp.array(dtype=float32),
):
    # Retrieve the thread index as the contact index
    wid, cid = wp.tid()

    # Retrieve the number of contact active in the world
    nc = problem_nc[wid]

    # Retrieve the solver status
    status = solver_status[wid]

    # Skip if row index exceed the problem size or if the solver has already converged
    if cid >= nc or status.converged > 0:
        return

    # Retrieve the contacts index offset of the world
    cio = problem_cio[wid]

    # Retrieve the index offset of the vector block of the world
    vio = problem_vio[wid]

    # Retrieve the contact constraints group offset of the world
    ccgo = problem_ccgo[wid]

    # Compute the index offset of the corresponding contact constraint
    ccio_k = vio + ccgo + 3 * cid

    # Retrieve the contact index w.r.t the model
    cio_k = cio + cid

    # Compute the norm of the tangential components
    vtx = solver_z_p[ccio_k]
    vty = solver_z_p[ccio_k + 1]
    vt_norm = wp.sqrt(vtx * vtx + vty * vty)

    # Store De Saxce correction for this block
    solver_s[ccio_k] = 0.0
    solver_s[ccio_k + 1] = 0.0
    solver_s[ccio_k + 2] = problem_mu[cio_k] * vt_norm


@wp.kernel
def _compute_velocity_bias(
    # Inputs:
    problem_dim: wp.array(dtype=int32),
    problem_vio: wp.array(dtype=int32),
    problem_v_f: wp.array(dtype=float32),
    solver_config: wp.array(dtype=PADMMConfig),
    solver_penalty: wp.array(dtype=PADMMPenalty),
    solver_status: wp.array(dtype=PADMMStatus),
    solver_s: wp.array(dtype=float32),
    solver_x_p: wp.array(dtype=float32),
    solver_y_p: wp.array(dtype=float32),
    solver_z_p: wp.array(dtype=float32),
    # Outputs:
    solver_v: wp.array(dtype=float32),
):
    # Retrieve the thread indices as the world and constraint index
    wid, tid = wp.tid()

    # Retrieve the total number of active constraints in the world
    ncts = problem_dim[wid]

    # Retrieve the solver status
    status = solver_status[wid]

    # Skip if row index exceed the problem size or if the solver has already converged
    if tid >= ncts or status.converged > 0:
        return

    # Retrieve the index offset of the vector block of the world
    vio = problem_vio[wid]

    # Retrive solver parameters
    eta = solver_config[wid].eta
    rho = solver_penalty[wid].rho

    # Compute the index offset of the vector block of the world
    tio = vio + tid

    # Retrieve the solver state
    v_f = problem_v_f[tio]
    s = solver_s[tio]
    x_p = solver_x_p[tio]
    y_p = solver_y_p[tio]
    z_p = solver_z_p[tio]

    # v = - v_f - s + eta * x_p + rho * y_p + z_p
    solver_v[tio] = -v_f - s + eta * x_p + rho * y_p + z_p


@wp.kernel
def _apply_overrelaxation_and_compute_projection_argument(
    # Inputs
    problem_dim: wp.array(dtype=int32),
    problem_vio: wp.array(dtype=int32),
    solver_config: wp.array(dtype=PADMMConfig),
    solver_penalty: wp.array(dtype=PADMMPenalty),
    solver_status: wp.array(dtype=PADMMStatus),
    solver_y_p: wp.array(dtype=float32),
    solver_z_p: wp.array(dtype=float32),
    # Outputs
    solver_x: wp.array(dtype=float32),
    solver_y: wp.array(dtype=float32),
):
    # Retrieve the thread indices as the world and constraint index
    wid, tid = wp.tid()

    # Retrieve the total number of active constraints in the world
    ncts = problem_dim[wid]

    # Retrieve the solver status
    status = solver_status[wid]

    # Skip if row index exceed the problem size or if the solver has already converged
    if tid >= ncts or status.converged > 0:
        return

    # Retrieve the index offset of the vector block of the world
    vio = problem_vio[wid]

    # Retrive the relaxation factor
    omega = solver_config[wid].omega

    # Capture the ALM penalty
    rho = solver_penalty[wid].rho

    # Compute the index offset of the vector block of the world
    tio = vio + tid

    # Retrive the solver state variables
    y_p = solver_y_p[tio]
    z_p = solver_z_p[tio]
    x = solver_x[tio]
    y = solver_y[tio]

    # Compute the over-relaxation update
    x = omega * x + (1.0 - omega) * y_p

    # Compute argument for the projection operator
    y = x - (1.0 / rho) * z_p

    # Store the updated values back to the solver state
    solver_x[tio] = x
    solver_y[tio] = y


# TODO: Break this up to two kernels launched simultaneously (1x for limits, 1x for contacts)
@wp.kernel
def _project_to_feasible_cone(
    # Inputs:
    problem_nl: wp.array(dtype=int32),
    problem_nc: wp.array(dtype=int32),
    problem_cio: wp.array(dtype=int32),
    problem_lcgo: wp.array(dtype=int32),
    problem_ccgo: wp.array(dtype=int32),
    problem_vio: wp.array(dtype=int32),
    problem_mu: wp.array(dtype=float32),
    solver_status: wp.array(dtype=PADMMStatus),
    # Outputs:
    solver_y: wp.array(dtype=float32),
):
    # Retrieve the thread index as the unilateral entity index
    wid, uid = wp.tid()

    # Retrieve the solver status
    status = solver_status[wid]

    # Retrieve the number of active limits and contacts in the world
    nl = problem_nl[wid]
    nc = problem_nc[wid]

    # Skip if row index exceed the problem size or if the solver has already converged
    if uid >= (nl + nc) or status.converged > 0:
        return

    # Retrieve the index offset of the vector block of the world
    vio = problem_vio[wid]

    # Check if the thread should handle a limit
    if nl > 0 and uid < nl:
        # Retrieve the limit constraint group offset of the world
        lcgo = problem_lcgo[wid]
        # Compute the constraint index offset of the limit element
        lcio_j = vio + lcgo + uid
        # Project to the non-negative orthant
        solver_y[lcio_j] = wp.max(solver_y[lcio_j], 0.0)

    # Check if the thread should handle a contact
    elif nc > 0 and uid >= nl:
        # Retrieve the contact index offset of the world
        cio = problem_cio[wid]
        # Retrieve the limit constraint group offset of the world
        ccgo = problem_ccgo[wid]
        # Compute the index of the contact element in the unilaterals array
        # NOTE: We need to substract the number of active limits
        cid = uid - nl
        # Compute the index offset of the contact constraint
        ccio_j = vio + ccgo + 3 * cid
        # Capture a 3D vector
        x = vec3f(solver_y[ccio_j], solver_y[ccio_j + 1], solver_y[ccio_j + 2])
        # Project to the coulomb friction cone
        y_proj = project_to_coulomb_cone(x, problem_mu[cio + cid])
        # Copy vec3 projection into the slack variable array
        solver_y[ccio_j] = y_proj[0]
        solver_y[ccio_j + 1] = y_proj[1]
        solver_y[ccio_j + 2] = y_proj[2]


@wp.kernel
def _update_dual_variables_and_compute_primal_dual_residuals(
    # Inputs:
    problem_dim: wp.array(dtype=int32),
    problem_vio: wp.array(dtype=int32),
    problem_P: wp.array(dtype=float32),
    solver_config: wp.array(dtype=PADMMConfig),
    solver_penalty: wp.array(dtype=PADMMPenalty),
    solver_status: wp.array(dtype=PADMMStatus),
    solver_x: wp.array(dtype=float32),
    solver_x_p: wp.array(dtype=float32),
    solver_y: wp.array(dtype=float32),
    solver_y_p: wp.array(dtype=float32),
    solver_z_p: wp.array(dtype=float32),
    # Outputs:
    solver_z: wp.array(dtype=float32),
    solver_r_p: wp.array(dtype=float32),
    solver_r_d: wp.array(dtype=float32),
):
    # Retrieve the thread indices as the world and constraint index
    wid, tid = wp.tid()

    # Retrieve the total number of active constraints in the world
    ncts = problem_dim[wid]

    # Retrieve the solver status
    status = solver_status[wid]

    # Skip if row index exceed the problem size or if the solver has already converged
    if tid >= ncts or status.converged > 0:
        return

    # Retrieve the index offset of the vector block of the world
    vio = problem_vio[wid]

    # Capture proximal parameter and the ALM penalty
    eta = solver_config[wid].eta
    rho = solver_penalty[wid].rho

    # Compute the index offset of the vector block of the world
    tio = vio + tid

    # Retrieve
    P_i = problem_P[tio]

    # Retrieve the solver state inputs
    x = solver_x[tio]
    y = solver_y[tio]
    x_p = solver_x_p[tio]
    y_p = solver_y_p[tio]
    z_p = solver_z_p[tio]

    # Compute the dual variable update
    solver_z[tio] = z_p + rho * (y - x)

    # Compute the primal residual as the concensus of the primal and slack variable
    solver_r_p[tio] = P_i * (x - y)

    # Compute the dual residual using the ADMM-specific shortcut
    solver_r_d[tio] = (1.0 / P_i) * (rho * (y - y_p) + eta * (x - x_p))


# TODO: Break this up to two kernels launched simultaneously (1x for limits, 1x for contacts)?
@wp.kernel
def _compute_complementarity_residuals(
    # Inputs:
    problem_nl: wp.array(dtype=int32),
    problem_nc: wp.array(dtype=int32),
    problem_vio: wp.array(dtype=int32),
    problem_uio: wp.array(dtype=int32),
    problem_lcgo: wp.array(dtype=int32),
    problem_ccgo: wp.array(dtype=int32),
    solver_status: wp.array(dtype=PADMMStatus),
    solver_x: wp.array(dtype=float32),
    solver_z: wp.array(dtype=float32),
    # Outputs:
    solver_r_c: wp.array(dtype=float32),
):
    # Retrieve the thread index as the unilateral entity index
    wid, uid = wp.tid()

    # Retrieve the solver status
    status = solver_status[wid]

    # Retrieve the number of active limits and contacts in the world
    nl = problem_nl[wid]
    nc = problem_nc[wid]

    # Skip if row index exceed the problem size or if the solver has already converged
    if uid >= (nl + nc) or status.converged > 0:
        return

    # Retrieve the index offsets of the unilateral elements
    uio = problem_uio[wid]

    # Retrieve the index offset of the vector block of the world
    vio = problem_vio[wid]

    # Compute the index offset of the vector block of the world
    uio_u = uio + uid

    # Check if the thread should handle a limit
    if nl > 0 and uid < nl:
        # Retrieve the limit constraint group offset of the world
        lcgo = problem_lcgo[wid]
        # Compute the constraint index offset of the limit element
        lcio_j = vio + lcgo + uid
        # Compute the scalar product of the primal and dual variables
        solver_r_c[uio_u] = solver_x[lcio_j] * solver_z[lcio_j]

    # Check if the thread should handle a contact
    elif nc > 0 and uid >= nl:
        # Retrieve the limit constraint group offset of the world
        ccgo = problem_ccgo[wid]
        # Compute the index of the contact element in the unilaterals array
        # NOTE: We need to substract the number of active limits
        cid = uid - nl
        # Compute the index offset of the contact constraint
        ccio_j = vio + ccgo + 3 * cid
        # Capture 3D vectors
        x_c = vec3f(solver_x[ccio_j], solver_x[ccio_j + 1], solver_x[ccio_j + 2])
        z_c = vec3f(solver_z[ccio_j], solver_z[ccio_j + 1], solver_z[ccio_j + 2])
        # Compute the inner product of the primal and dual variables
        solver_r_c[uio_u] = wp.dot(x_c, z_c)


@wp.kernel
def _compute_infnorm_residuals_serially(
    # Inputs:
    problem_nl: wp.array(dtype=int32),
    problem_nc: wp.array(dtype=int32),
    problem_uio: wp.array(dtype=int32),
    problem_dim: wp.array(dtype=int32),
    problem_vio: wp.array(dtype=int32),
    solver_config: wp.array(dtype=PADMMConfig),
    solver_r_p: wp.array(dtype=float32),
    solver_r_d: wp.array(dtype=float32),
    solver_r_c: wp.array(dtype=float32),
    # Outputs:
    solver_state_done: wp.array(dtype=int32),
    solver_status: wp.array(dtype=PADMMStatus),
):
    # Retrieve the thread index as the world index
    wid = wp.tid()

    # Retrieve the solver status
    status = solver_status[wid]

    # Skip this step if already converged
    if status.converged:
        return

    # Update iteration counter
    status.iterations += 1

    # Capture the size of the residuals arrays
    nl = problem_nl[wid]
    nc = problem_nc[wid]
    ncts = problem_dim[wid]

    # Retrieve the solver configurations
    config = solver_config[wid]

    # Retrieve the index offsets of the vector block and unilateral elements
    vio = problem_vio[wid]
    uio = problem_uio[wid]

    # Extract the solver tolerances
    eps_p = config.primal_tolerance
    eps_d = config.dual_tolerance
    eps_c = config.compl_tolerance

    # Extract the maximum number of iterations
    maxiters = config.max_iterations

    # Compute element-wise max over each residual vector to compute the infinity-norm
    r_p_max = float(0.0)
    r_d_max = float(0.0)
    for j in range(ncts):
        rio_j = vio + j
        r_p_max = wp.max(r_p_max, wp.abs(solver_r_p[rio_j]))
        r_d_max = wp.max(r_d_max, wp.abs(solver_r_d[rio_j]))

    # Compute the infinity-norm of the complementarity residuals
    nu = nl + nc
    r_c_max = float(0.0)
    for j in range(nu):
        r_c_max = wp.max(r_c_max, wp.abs(solver_r_c[uio + j]))

    # Store the scalar metric residuals in the solver status
    status.r_p = r_p_max
    status.r_d = r_d_max
    status.r_c = r_c_max

    # Check and store convergence state
    if status.iterations > 1 and r_p_max <= eps_p and r_d_max <= eps_d and r_c_max <= eps_c:
        status.converged = 1

    # If converged or reached max iterations, decrement the number of active worlds
    if status.converged or status.iterations >= maxiters:
        solver_state_done[0] -= 1

    # Store the updated status
    solver_status[wid] = status


@wp.kernel
def _collect_solver_convergence_info(
    # Inputs:
    problem_nl: wp.array(dtype=int32),
    problem_nc: wp.array(dtype=int32),
    problem_cio: wp.array(dtype=int32),
    problem_lcgo: wp.array(dtype=int32),
    problem_ccgo: wp.array(dtype=int32),
    problem_maxdim: wp.array(dtype=int32),
    problem_dim: wp.array(dtype=int32),
    problem_vio: wp.array(dtype=int32),
    problem_mio: wp.array(dtype=int32),
    problem_mu: wp.array(dtype=float32),
    problem_v_f: wp.array(dtype=float32),
    problem_D: wp.array(dtype=float32),
    problem_P: wp.array(dtype=float32),
    solver_state_s: wp.array(dtype=float32),
    solver_state_x: wp.array(dtype=float32),
    solver_state_x_p: wp.array(dtype=float32),
    solver_state_y: wp.array(dtype=float32),
    solver_state_y_p: wp.array(dtype=float32),
    solver_state_z: wp.array(dtype=float32),
    solver_state_z_p: wp.array(dtype=float32),
    solver_config: wp.array(dtype=PADMMConfig),
    solver_penalty: wp.array(dtype=PADMMPenalty),
    solver_status: wp.array(dtype=PADMMStatus),
    # Outputs:
    solver_info_lambdas: wp.array(dtype=float32),
    solver_info_v_plus: wp.array(dtype=float32),
    solver_info_v_aug: wp.array(dtype=float32),
    solver_info_s: wp.array(dtype=float32),
    solver_info_offset: wp.array(dtype=int32),
    solver_info_norm_s: wp.array(dtype=float32),
    solver_info_norm_x: wp.array(dtype=float32),
    solver_info_norm_y: wp.array(dtype=float32),
    solver_info_norm_z: wp.array(dtype=float32),
    solver_info_f_ccp: wp.array(dtype=float32),
    solver_info_f_ncp: wp.array(dtype=float32),
    solver_info_r_dx: wp.array(dtype=float32),
    solver_info_r_dy: wp.array(dtype=float32),
    solver_info_r_dz: wp.array(dtype=float32),
    solver_info_r_primal: wp.array(dtype=float32),
    solver_info_r_dual: wp.array(dtype=float32),
    solver_info_r_compl: wp.array(dtype=float32),
    solver_info_r_pd: wp.array(dtype=float32),
    solver_info_r_dp: wp.array(dtype=float32),
    solver_info_r_ncp_primal: wp.array(dtype=float32),
    solver_info_r_ncp_dual: wp.array(dtype=float32),
    solver_info_r_ncp_compl: wp.array(dtype=float32),
    solver_info_r_ncp_natmap: wp.array(dtype=float32),
):
    # Retrieve the thread index as the world index
    wid = wp.tid()

    # Retrieve the world-specific data
    nl = problem_nl[wid]
    nc = problem_nc[wid]
    maxncts = problem_maxdim[wid]
    ncts = problem_dim[wid]
    cio = problem_cio[wid]
    lcgo = problem_lcgo[wid]
    ccgo = problem_ccgo[wid]
    vio = problem_vio[wid]
    mio = problem_mio[wid]
    rio = solver_info_offset[wid]
    config = solver_config[wid]
    penalty = solver_penalty[wid]
    status = solver_status[wid]

    # Retrieve parameters
    iter = status.iterations - 1

    # Compute additional info
    njc = ncts - (nl + 3 * nc)

    # Compute total diagonal regularization applied to the Delassus matrix
    sigma = config.eta + penalty.rho

    # Compute and store the norms of the current solution state
    norm_s = compute_l2_norm(ncts, vio, solver_state_s)
    norm_x = compute_l2_norm(ncts, vio, solver_state_x)
    norm_y = compute_l2_norm(ncts, vio, solver_state_y)
    norm_z = compute_l2_norm(ncts, vio, solver_state_z)

    # Compute (division safe) residual ratios
    r_pd = status.r_p / (status.r_d + FLOAT32_EPS)
    r_dp = status.r_d / (status.r_p + FLOAT32_EPS)

    # Compute the post-event constraint-space velocity from the current solution: v_plus = v_f + D @ lambda
    compute_cwise_vec_mul(ncts, vio, problem_P, solver_state_y, solver_info_lambdas)

    # Compute the post-event constraint-space velocity from the current solution: v_plus = v_f + D @ lambda
    compute_gemv(maxncts, ncts, vio, mio, sigma, problem_P, problem_D, solver_state_y, problem_v_f, solver_info_v_plus)

    # Compute the De Saxce correction for each contact as: s = G(v_plus)
    compute_desaxce_corrections(nc, cio, vio, ccgo, problem_mu, solver_info_v_plus, solver_info_s)

    # Compute the CCP optimization objective as: f_ccp = 0.5 * lambda.dot(v_plus + v_f)
    f_ccp = 0.5 * compute_double_dot_product(ncts, vio, solver_info_lambdas, solver_info_v_plus, problem_v_f)

    # Compute the NCP optimization objective as:  f_ncp = f_ccp + lambda.dot(s)
    f_ncp = compute_dot_product(ncts, vio, solver_info_lambdas, solver_info_s)
    f_ncp += f_ccp

    # Compute the augmented post-event constraint-space velocity as: v_aug = v_plus + s
    compute_vector_sum(ncts, vio, solver_info_v_plus, solver_info_s, solver_info_v_aug)

    # Compute the NCP primal residual as: r_p := || lambda - proj_K(lambda) ||_inf
    r_ncp_p = compute_ncp_primal_residual(nl, nc, vio, lcgo, ccgo, cio, problem_mu, solver_info_lambdas)

    # Compute the NCP dual residual as: r_d := || v_plus + s - proj_dual_K(v_plus + s)  ||_inf
    r_ncp_d = compute_ncp_dual_residual(njc, nl, nc, vio, lcgo, ccgo, cio, problem_mu, solver_info_v_aug)

    # Compute the NCP complementarity (lambda _|_ (v_plus + s)) residual as r_c := || lambda.dot(v_plus + s) ||_inf
    r_ncp_c = compute_ncp_complementarity_residual(nl, nc, vio, lcgo, ccgo, solver_info_v_aug, solver_info_lambdas)

    # Compute the natural-map residuals as: r_natmap = || lambda - proj_K(lambda - (v + s)) ||_inf
    r_ncp_natmap = compute_ncp_natural_map_residual(
        nl, nc, vio, lcgo, ccgo, cio, problem_mu, solver_info_v_aug, solver_info_lambdas
    )

    # Compute the iterate residual as: r_iter := || y - y_p ||_inf
    r_dx = compute_preconditioned_iterate_residual(ncts, vio, problem_P, solver_state_x, solver_state_x_p)
    r_dy = compute_preconditioned_iterate_residual(ncts, vio, problem_P, solver_state_y, solver_state_y_p)
    r_dz = compute_inverse_preconditioned_iterate_residual(ncts, vio, problem_P, solver_state_z, solver_state_z_p)

    # Compute index offset for the info of the current iteration
    iio = rio + iter

    # Store the convergence information in the solver info arrays
    solver_info_norm_s[iio] = norm_s
    solver_info_norm_x[iio] = norm_x
    solver_info_norm_y[iio] = norm_y
    solver_info_norm_z[iio] = norm_z
    solver_info_r_dx[iio] = r_dx
    solver_info_r_dy[iio] = r_dy
    solver_info_r_dz[iio] = r_dz
    solver_info_r_primal[iio] = status.r_p
    solver_info_r_dual[iio] = status.r_d
    solver_info_r_compl[iio] = status.r_c
    solver_info_r_pd[iio] = r_pd
    solver_info_r_dp[iio] = r_dp
    solver_info_r_ncp_primal[iio] = r_ncp_p
    solver_info_r_ncp_dual[iio] = r_ncp_d
    solver_info_r_ncp_compl[iio] = r_ncp_c
    solver_info_r_ncp_natmap[iio] = r_ncp_natmap
    solver_info_f_ccp[iio] = f_ccp
    solver_info_f_ncp[iio] = f_ncp


@wp.kernel
def _apply_dual_preconditioner_to_state(
    # Inputs:
    problem_dim: wp.array(dtype=int32),
    problem_vio: wp.array(dtype=int32),
    problem_P: wp.array(dtype=float32),
    # Outputs:
    solver_x: wp.array(dtype=float32),
    solver_y: wp.array(dtype=float32),
    solver_z: wp.array(dtype=float32),
):
    # Retrieve the thread index
    wid, tid = wp.tid()

    # Retrieve the number of active constraints in the world
    ncts = problem_dim[wid]

    # Skip if row index exceed the problem size
    if tid >= ncts:
        return

    # Retrieve the vector index offset of the world
    vio = problem_vio[wid]

    # Compute the global index of the vector entry
    v_i = vio + tid

    # Retrieve the i-th entries of the target vectors
    x_i = solver_x[v_i]
    y_i = solver_y[v_i]
    z_i = solver_z[v_i]

    # Retrieve the i-th entry of the diagonal preconditioner
    P_i = problem_P[v_i]

    # Store the preconditioned i-th entry of the vector
    solver_x[v_i] = P_i * x_i
    solver_y[v_i] = P_i * y_i
    solver_z[v_i] = (1.0 / P_i) * z_i


@wp.kernel
def _compute_final_desaxce_correction(
    problem_nc: wp.array(dtype=int32),
    problem_cio: wp.array(dtype=int32),
    problem_ccgo: wp.array(dtype=int32),
    problem_vio: wp.array(dtype=int32),
    problem_mu: wp.array(dtype=float32),
    solver_z: wp.array(dtype=float32),
    # Outputs:
    solver_s: wp.array(dtype=float32),
):
    # Retrieve the thread index
    wid, cid = wp.tid()

    # Retrieve the number of contact active in the world
    nc = problem_nc[wid]

    # Retrieve the limit constraint group offset of the world
    ccgo = problem_ccgo[wid]

    # Skip if row index exceed the problem size or if the solver has already converged
    if cid >= nc:
        return

    # Retrieve the index offset of the vector block of the world
    cio = problem_cio[wid]

    # Retrieve the index offset of the vector block of the world
    vio = problem_vio[wid]

    # Compute the vector index offset of the corresponding contact constraint
    ccio_k = vio + ccgo + 3 * cid

    # Compute the norm of the tangential components
    vtx = solver_z[ccio_k]
    vty = solver_z[ccio_k + 1]
    vt_norm = wp.sqrt(vtx * vtx + vty * vty)

    # Store De Saxce correction for this block
    solver_s[ccio_k] = 0.0
    solver_s[ccio_k + 1] = 0.0
    solver_s[ccio_k + 2] = problem_mu[cio + cid] * vt_norm


# Update solution vectors
@wp.kernel
def _compute_solution_vectors(
    # Inputs:
    problem_dim: wp.array(dtype=int32),
    problem_vio: wp.array(dtype=int32),
    solver_s: wp.array(dtype=float32),
    solver_y: wp.array(dtype=float32),
    solver_z: wp.array(dtype=float32),
    # Outputs:
    solver_v_plus: wp.array(dtype=float32),
    solver_lambdas: wp.array(dtype=float32),
):
    # Retrieve the thread index
    wid, tid = wp.tid()

    # Retrieve the total number of active constraints in the world
    ncts = problem_dim[wid]

    # Skip if row index exceed the problem size or if the solver has already converged
    if tid >= ncts:
        return

    # Retrieve the index offset of the vector block of the world
    vio = problem_vio[wid]

    # Compute the index offset of the vector block of the world
    tio = vio + tid

    # Retrieve the solver state
    z = solver_z[tio]
    s = solver_s[tio]
    y = solver_y[tio]

    # Update constraint velocity: v_plus = z - s;
    solver_v_plus[tio] = z - s

    # Update constraint reactions: lambda = y
    solver_lambdas[tio] = y


###
# Solver
###


class PADMMDualSolver:
    """
    A Proximal ADMM solver for constrained rigid multi-body systems.

    This solver implements the Proximal ADMM algorithm to solve the Lagrange dual of the
    constrained forward dynamics problem in constraint reactions (i.e. impulses) and
    post-event constraint-space velocities.

    Notes:
    - The solver is designed to work with the DualProblem formulation.
    - The solver operates on the Lagrange dual of the constrained forward dynamics problem.
    - The solver is based on the Proximal ADMM algorithm, which introduces a proximal regularization term
    - The solver supports multiple penalty update methods, including fixed, linear, and spectral updates.
    - The solver can be configured with various tolerances, penalty parameters, and other settings.

    Attributes:
    - TODO
    """

    @staticmethod
    def _check_settings(
        model: Model | None = None, settings: list[PADMMSettings] | PADMMSettings | None = None
    ) -> list[PADMMSettings]:
        # If no settings are provided, use defaults
        if settings is None:
            # If no model is provided, use a single default settings object
            if model is None:
                settings = [PADMMSettings()]

            # If a model is provided, create a list of default settings
            # objects based on the number of worlds in the model
            else:
                num_worlds = model.info.num_worlds
                settings = [PADMMSettings()] * num_worlds

        # If a single settings object is provided, convert it to a list
        elif isinstance(settings, PADMMSettings):
            settings = [settings] * (model.info.num_worlds if model else 1)

        # If a list of settings is provided, ensure it matches the number
        # of worlds and that all settings are instances of PADMMSettings
        elif isinstance(settings, list):
            if model is not None and len(settings) != model.info.num_worlds:
                raise ValueError(f"Expected {model.info.num_worlds} settings, got {len(settings)}")
            if not all(isinstance(s, PADMMSettings) for s in settings):
                raise TypeError("All settings must be instances of PADMMSettings")

        # Return the validated settings
        return settings

    def __init__(
        self,
        model: Model | None = None,
        limits: Limits | None = None,
        contacts: Contacts | None = None,
        settings: list[PADMMSettings] | PADMMSettings | None = None,
        collect_info: bool = False,
        device: Devicelike = None,
    ):
        # Declare the internal solver settings cache
        self._settings: list[PADMMSettings] = []
        self._max_iters: int = 0
        self._collect_info = False

        # Declare the model size cache
        self._size: ModelSize | None = None

        # Declare the solver data container
        self._data: PADMMData | None = None

        # Cache the requested device
        self._device = device

        # Perform memory allocations if a model is provided
        if model is not None:
            self.allocate(
                model=model,
                limits=limits,
                contacts=contacts,
                settings=settings,
                collect_info=collect_info,
                device=device,
            )

    @property
    def settings(self) -> list[PADMMSettings]:
        """
        The host-side cache of the solver settings.\n
        They are used to construct the warp array of type `PADMMConfig` on the target device.
        """
        return self._settings

    @property
    def size(self) -> ModelSize:
        """
        Returns the host-side cache of the solver allocation sizes.
        """
        return self._size

    @property
    def data(self) -> PADMMData:
        """
        The solver data container. This is a high-level container that bundles all solver data into a single object.
        """
        if self._data is None:
            raise RuntimeError("Solver data has not been allocated yet. Call `allocate()` first.")
        return self._data

    @property
    def device(self) -> Devicelike:
        """
        Returns the device on which the solver data is allocated.
        """
        return self._device

    def allocate(
        self,
        model: Model | None = None,
        limits: Limits | None = None,
        contacts: Contacts | None = None,
        settings: list[PADMMSettings] | PADMMSettings | None = None,
        collect_info: bool = False,
        device: Devicelike = None,
    ):
        # Ensure the model is valid
        if model is None:
            raise ValueError("A model of type `Model` must be provided to allocate the Delassus operator.")
        elif not isinstance(model, Model):
            raise ValueError("Invalid model provided. Must be an instance of `Model`.")

        # Ensure the limits container is valid if provided
        if limits is not None:
            if not isinstance(limits, Limits):
                raise ValueError("Invalid limits container provided. Must be an instance of `Limits`.")

        # Ensure the contacts container is valid if provided
        if contacts is not None:
            if not isinstance(contacts, Contacts):
                raise ValueError("Invalid contacts container provided. Must be an instance of `Contacts`.")

        # Override the current info collection flag if specified at allocation time
        if collect_info != self._collect_info:
            self._collect_info = collect_info

        # Override the current device if specified at allocation time
        if device is not None:
            self._device = device

        # Cache the solver settings
        if settings is not None:
            self._settings = self._check_settings(model, settings)
        elif len(self._settings) == 0:
            self._settings = self._check_settings(model, None)

        # Capture reference to the model size
        self._size = model.size

        # Store the max number of iterations across all worlds
        # NOTE: This is necessary as the main solver loop must iterate over
        # the maximum number of iterations configured for any world
        self._max_iters = max([s.max_iterations for s in self._settings])

        # Allocate memory in device global memory
        self._data = PADMMData(
            size=self._size, max_iters=self._max_iters, collect_info=self._collect_info, device=self._device
        )

        # Write algorithm configs into device memory
        configs = [s.to_config() for s in self._settings]
        with wp.ScopedDevice(self._device):
            self._data.config = wp.array(configs, dtype=PADMMConfig)

    # TODO(team): We should think about how to introduce an optional systematic warm-starting mechanism here

    def coldstart(self):
        # Initialize state arrays to zero
        self._data.state.zero()

    def initialize(self):
        # Initialize solver status and penalty parameters from the set configurations
        wp.launch(
            kernel=_initialize_solver,
            dim=self._size.num_worlds,
            inputs=[self._data.config, self._data.status, self._data.penalty],
        )

        # Initialize the global while condition flag
        # NOTE: We use a single-element array that is initialized
        # to number of worlds and decremented by each world that
        # converges or reaches the maximum number of iterations
        self._data.state.done.fill_(self._size.num_worlds)

    def update_regularization(self, problem: DualProblem):
        # Update the proximal regularization term in the Delassus matrix
        wp.launch(
            kernel=_update_delassus_proximal_regularization,
            dim=(self._size.num_worlds, self._size.max_of_max_total_cts),
            inputs=[
                # Inputs:
                problem.data.maxdim,
                problem.data.dim,
                problem.data.mio,
                self._data.config,
                self._data.penalty,
                self._data.status,
                # Outputs:
                problem.data.D,
            ],
        )

        # Compute Choleky/LDLT factorization of the Delassus matrix
        problem._delassus.factorize(reset_to_zero=True)

    def update_desaxce_correction(self, problem: DualProblem):
        wp.launch(
            kernel=_compute_desaxce_correction,
            dim=(self._size.num_worlds, self._size.max_of_max_contacts),
            inputs=[
                # Inputs:
                problem.data.nc,
                problem.data.cio,
                problem.data.ccgo,
                problem.data.vio,
                problem.data.mu,
                self._data.status,
                self._data.state.z_p,
                # Outputs:
                self._data.state.s,
            ],
        )

    def update_velocity_bias(self, problem: DualProblem):
        wp.launch(
            kernel=_compute_velocity_bias,
            dim=(self._size.num_worlds, self._size.max_of_max_total_cts),
            inputs=[
                # Inputs:
                problem.data.dim,
                problem.data.vio,
                problem.data.v_f,
                self._data.config,
                self._data.penalty,
                self._data.status,
                self._data.state.s,
                self._data.state.x_p,
                self._data.state.y_p,
                self._data.state.z_p,
                # Outputs:
                self._data.state.v,
            ],
        )

    def update_unconstrained_solution(self, problem: DualProblem):
        # TODO: We should do this in-place
        # wp.copy(self._data.state.x, self._data.state.v)
        # problem._delassus.solve_inplace(x=self._data.state.x)
        problem._delassus.solve(v=self._data.state.v, x=self._data.state.x)

    def update_projection_to_feasible_set(self, problem: DualProblem):
        # Apply over-relaxation and compute the argument to the projection operator
        wp.launch(
            kernel=_apply_overrelaxation_and_compute_projection_argument,
            dim=(self._size.num_worlds, self._size.max_of_max_total_cts),
            inputs=[
                # Inputs:
                problem.data.dim,
                problem.data.vio,
                self._data.config,
                self._data.penalty,
                self._data.status,
                self._data.state.y_p,
                self._data.state.z_p,
                # Outputs:
                self._data.state.x,
                self._data.state.y,
            ],
        )

        # Project to the feasible set defined by the cone K := R^{njd} x R_+^{nld} x K_{mu}^{nc}
        wp.launch(
            kernel=_project_to_feasible_cone,
            dim=(self._size.num_worlds, self._size.max_of_max_unilaterals),
            inputs=[
                # Inputs:
                problem.data.nl,
                problem.data.nc,
                problem.data.cio,
                problem.data.lcgo,
                problem.data.ccgo,
                problem.data.vio,
                problem.data.mu,
                self._data.status,
                # Outputs:
                self._data.state.y,
            ],
        )

    def update_dual_variables_and_residuals(self, problem: DualProblem):
        # Update the dual variables and compute primal-dual residuals from the current state
        # NOTE: These are combined into a single kernel to reduce kernel launch overhead
        wp.launch(
            kernel=_update_dual_variables_and_compute_primal_dual_residuals,
            dim=(self._size.num_worlds, self._size.max_of_max_total_cts),
            inputs=[
                # Inputs:
                problem.data.dim,
                problem.data.vio,
                problem.data.P,
                self._data.config,
                self._data.penalty,
                self._data.status,
                self._data.state.x,
                self._data.state.x_p,
                self._data.state.y,
                self._data.state.y_p,
                self._data.state.z_p,
                # Outputs:
                self._data.state.z,
                self._data.residuals.r_primal,
                self._data.residuals.r_dual,
            ],
        )

        # Compute complementarity residual from the current state
        wp.launch(
            kernel=_compute_complementarity_residuals,
            dim=(self._size.num_worlds, self._size.max_of_max_unilaterals),
            inputs=[
                # Inputs:
                problem.data.nl,
                problem.data.nc,
                problem.data.cio,
                problem.data.uio,
                problem.data.lcgo,
                problem.data.ccgo,
                self._data.status,
                self._data.state.x,
                self._data.state.z,
                # Outputs:
                self._data.residuals.r_compl,
            ],
        )

    def update_convergence_check(self, problem: DualProblem):
        # Compute infinity-norm of all residuals and check for convergence
        wp.launch(
            kernel=_compute_infnorm_residuals_serially,
            dim=self._size.num_worlds,
            inputs=[
                # Inputs:
                problem.data.nl,
                problem.data.nc,
                problem.data.uio,
                problem.data.dim,
                problem.data.vio,
                self._data.config,
                self._data.residuals.r_primal,
                self._data.residuals.r_dual,
                self._data.residuals.r_compl,
                # Outputs:
                self._data.state.done,
                self._data.status,
            ],
        )

    def update_solver_info(self, problem: DualProblem):
        # First reset the internal buffer arrays to zero
        # to ensure we do not accumulate values across iterations
        self.data.info.v_plus.zero_()
        self.data.info.v_aug.zero_()
        self._data.info.s.zero_()

        # Collect convergence information from the current state
        wp.launch(
            kernel=_collect_solver_convergence_info,
            dim=self._size.num_worlds,
            inputs=[
                # Inputs:
                problem.data.nl,
                problem.data.nc,
                problem.data.cio,
                problem.data.lcgo,
                problem.data.ccgo,
                problem.data.maxdim,
                problem.data.dim,
                problem.data.vio,
                problem.data.mio,
                problem.data.mu,
                problem.data.v_f,
                problem.data.D,
                problem.data.P,
                self._data.state.s,
                self._data.state.x,
                self._data.state.x_p,
                self._data.state.y,
                self._data.state.y_p,
                self._data.state.z,
                self._data.state.z_p,
                self._data.config,
                self._data.penalty,
                self._data.status,
                # Outputs:
                self._data.info.lambdas,
                self._data.info.v_plus,
                self._data.info.v_aug,
                self._data.info.s,
                self._data.info.offsets,
                self._data.info.norm_s,
                self._data.info.norm_x,
                self._data.info.norm_y,
                self._data.info.norm_z,
                self._data.info.f_ccp,
                self._data.info.f_ncp,
                self._data.info.r_dx,
                self._data.info.r_dy,
                self._data.info.r_dz,
                self._data.info.r_primal,
                self._data.info.r_dual,
                self._data.info.r_compl,
                self._data.info.r_pd,
                self._data.info.r_dp,
                self._data.info.r_ncp_primal,
                self._data.info.r_ncp_dual,
                self._data.info.r_ncp_compl,
                self._data.info.r_ncp_natmap,
            ],
        )

    def update_previous_state(self):
        wp.copy(self._data.state.x_p, self._data.state.x)
        wp.copy(self._data.state.y_p, self._data.state.y)
        wp.copy(self._data.state.z_p, self._data.state.z)

    def update_solution(self, problem: DualProblem):
        # Apply the dual preconditioner to recover the final PADMM state
        wp.launch(
            kernel=_apply_dual_preconditioner_to_state,
            dim=(self._size.num_worlds, self._size.max_of_max_total_cts),
            inputs=[
                # Inputs:
                problem.data.dim,
                problem.data.vio,
                problem.data.P,
                # Outputs:
                self._data.state.x,
                self._data.state.y,
                self._data.state.z,
            ],
        )

        # Update the De Saxce correction from terminal PADMM dual variables
        wp.launch(
            kernel=_compute_final_desaxce_correction,
            dim=(self._size.num_worlds, self._size.max_of_max_contacts),
            inputs=[
                # Inputs:
                problem.data.nc,
                problem.data.cio,
                problem.data.ccgo,
                problem.data.vio,
                problem.data.mu,
                self._data.state.z,
                # Outputs:
                self._data.state.s,
            ],
        )

        # Update solution vectors from the terminal PADMM state
        wp.launch(
            kernel=_compute_solution_vectors,
            dim=(self._size.num_worlds, self._size.max_of_max_total_cts),
            inputs=[
                # Inputs:
                problem.data.dim,
                problem.data.vio,
                self._data.state.s,
                self._data.state.y,
                self._data.state.z,
                # Outputs:
                self._data.solution.v_plus,
                self._data.solution.lambdas,
            ],
        )

    def step(self, problem: DualProblem):
        # Compute De Saxce correction from the previous dual variables
        self.update_desaxce_correction(problem)

        # Compute the total velocity bias, i.e. rhs vector of the unconstrained linear system
        self.update_velocity_bias(problem)

        # Compute the unconstrained solution and store in the primal variables
        self.update_unconstrained_solution(problem)

        # Project the over-relaxed primal variables to the feasible set
        self.update_projection_to_feasible_set(problem)

        # Update the dual variables and compute residuals from the current state
        self.update_dual_variables_and_residuals(problem)

        # Compute infinity-norm of all residuals and check for convergence
        self.update_convergence_check(problem)

        # Optionally record internal solver info
        if self._collect_info:
            self.update_solver_info(problem)

        # Update caches of previous state variables
        self.update_previous_state()

    def solve(self, problem: DualProblem):
        # Cold-start the solver state
        self.coldstart()

        # Initialize the solver status and ALM penalty parameter
        self.initialize()

        # Add the diagonal proximal regularization to the Delassus matrix
        # D_{eta,rho} := D + (eta + rho) * I_{nd}
        self.update_regularization(problem)

        # Reset the solver info to zero if collection is enabled
        if self._collect_info:
            self._data.info.zero()

        # Iterate until convergence or maximum number of iterations is reached
        wp.capture_while(self._data.state.done, while_body=self.step, problem=problem)

        # Update the final solution from the terminal PADMM state
        self.update_solution(problem)
