###########################################################################
# KAMINO: Utilities: Linear Algebra
###########################################################################

from typing import Union

from .matrix import (
    is_square_matrix,
    is_symmetric_matrix,
    SquareSymmetricMatrixProperties,
    MatrixComparison
)

from .cholesky import Cholesky
from .ldlt_nopivot import LDLTNoPivot
from .ldlt_bk import LDLTBunchKaufman
from .ldlt_blocked import LDLTBlocked
from .ldlt_eigen3 import LDLTEigen3

from .admm import (
    ADMMStatus,
    ADMMInfo,
    ADMMSolver
)

FactorizerType = Union[Cholesky, LDLTNoPivot, LDLTBunchKaufman, LDLTBlocked, LDLTEigen3]


###
# Module API
###

__all__ = [
    "is_square_matrix",
    "is_symmetric_matrix",
    "SquareSymmetricMatrixProperties",
    "MatrixComparison",
    "Cholesky",
    "LDLTNoPivot",
    "LDLTBunchKaufman",
    "LDLTBlocked",
    "LDLTEigen3",
    "FactorizerType",
    "ADMMStatus",
    "ADMMInfo",
    "ADMMSolver"
]
