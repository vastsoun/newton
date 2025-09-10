###########################################################################
# KAMINO: Utilities: Linear Algebra
###########################################################################

from typing import Union

from .admm import ADMMInfo, ADMMSolver, ADMMStatus
from .cholesky import Cholesky
from .ldlt_bk import LDLTBunchKaufman
from .ldlt_blocked import LDLTBlocked
from .ldlt_eigen3 import LDLTEigen3
from .ldlt_nopivot import LDLTNoPivot
from .matrix import MatrixComparison, SquareSymmetricMatrixProperties, is_square_matrix, is_symmetric_matrix

FactorizerType = Union[Cholesky, LDLTNoPivot, LDLTBunchKaufman, LDLTBlocked, LDLTEigen3]


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
