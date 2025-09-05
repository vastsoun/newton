###########################################################################
# KAMINO: Utilities: Linear Algebra
###########################################################################

# from typing import Union

from .admm import ADMMInfo, ADMMSolver, ADMMStatus
from .ldlt_bk import LDLTBunchKaufman
from .ldlt_blocked import LDLTBlocked
from .ldlt_eigen3 import LDLTEigen3
from .ldlt_nopivot import LDLTNoPivot
from .llt_std import LLT
from .lu_nopiv import LUNoPivot
from .matrix import MatrixComparison, SquareSymmetricMatrixProperties, is_square_matrix, is_symmetric_matrix

# FactorizerType = Union[LLT, LDLTNoPivot, LDLTBunchKaufman, LDLTBlocked, LDLTEigen3]


###
# Module API
###

__all__ = [
    "LLT",
    "ADMMInfo",
    "ADMMSolver",
    "ADMMStatus",
    # "FactorizerType",
    "LDLTBlocked",
    "LDLTBunchKaufman",
    "LDLTEigen3",
    "LDLTNoPivot",
    "LUNoPivot",
    "MatrixComparison",
    "SquareSymmetricMatrixProperties",
    "is_square_matrix",
    "is_symmetric_matrix",
]
