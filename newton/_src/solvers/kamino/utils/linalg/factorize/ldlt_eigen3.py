###########################################################################
# KAMINO: Utilities: Linear Algebra: LDLT based on that of Eigen3
###########################################################################

import numpy as np

from ..matrix import (
    MatrixSign,
    _make_tolerance,
    assert_is_square_matrix,
    assert_is_symmetric_matrix,
)

###
# Module interface
###

__all__ = [
    "ldlt_eigen3_lower",
    "ldlt_eigen3_lower_inplace",
    "ldlt_eigen3_lower_solve",
    "ldlt_eigen3_lower_solve_inplace",
    "ldlt_eigen3_lower_unpack",
]


###
# Factorize
###


def ldlt_eigen3_lower_inplace(mat, transpositions, temp, sign_ref):
    # mat: (n,n) numpy array (real or complex), modified in-place
    # transpositions: (n,) int array, modified in-place
    # temp: (n,) workspace array (same dtype as mat), modified in-place
    # sign_ref: 1-element list storing MatrixSign value, modified in-place
    # returns: bool ret
    n = mat.shape[0]
    ret = True
    found_zero_pivot = False

    if n <= 1:
        # transpositions.setIdentity()
        if transpositions.shape[0] != n:
            raise ValueError("transpositions size mismatch")
        for i in range(n):
            transpositions[i] = i
        if n == 0:
            sign_ref[0] = MatrixSign.ZeroSign
        else:
            akk = mat[0, 0].real
            if akk > 0:
                sign_ref[0] = MatrixSign.PositiveSemiDef
            elif akk < 0:
                sign_ref[0] = MatrixSign.NegativeSemiDef
            else:
                sign_ref[0] = MatrixSign.ZeroSign
        return True

    is_complex = np.iscomplexobj(mat)

    for k in range(n):
        # Find largest diagonal element in absolute value on the trailing diagonal
        max_abs = -1.0
        index_of_biggest_in_corner = k
        for i in range(k, n):
            a = abs(mat[i, i])
            if a > max_abs:
                max_abs = a
                index_of_biggest_in_corner = i

        transpositions[k] = int(index_of_biggest_in_corner)

        if k != index_of_biggest_in_corner:
            p = index_of_biggest_in_corner

            # swap row heads: mat.row(k).head(k) <-> mat.row(p).head(k)
            for j in range(k):
                tmp = mat[k, j]
                mat[k, j] = mat[p, j]
                mat[p, j] = tmp

            # swap column tails: mat.col(k).tail(s) <-> mat.col(p).tail(s) where s = n - p - 1
            # s = n - p - 1
            for i in range(p + 1, n):
                tmp = mat[i, k]
                mat[i, k] = mat[i, p]
                mat[i, p] = tmp

            # swap diagonal entries mat[k,k] <-> mat[p,p]
            tmpkk = mat[k, k]
            mat[k, k] = mat[p, p]
            mat[p, p] = tmpkk

            # for i in k+1..p-1:
            for i in range(k + 1, p):
                tmp = mat[i, k]
                mat[i, k] = np.conj(mat[p, i])
                mat[p, i] = np.conj(tmp)

            if is_complex:
                mat[p, k] = np.conj(mat[p, k])

        # Partition sizes
        rs = n - k - 1  # trailing size

        # A10 = mat[k, 0:k]
        # A21 = mat[k+1:n, k]
        # A20 = mat[k+1:n, 0:k]
        if k > 0:
            # temp[0:k] = real(diag[0:k]) * conj(A10)
            for i in range(k):
                temp[i] = mat[i, i].real * np.conj(mat[k, i])

            # mat[k,k] -= A10 * temp[0:k]
            accum = mat.dtype.type(0)
            for i in range(k):
                accum += mat[k, i] * temp[i]
            mat[k, k] = mat[k, k] - accum

            # A21 -= A20 * temp[0:k]
            if rs > 0:
                for r in range(rs):
                    acc = mat.dtype.type(0)
                    for i in range(k):
                        acc += mat[k + 1 + r, i] * temp[i]
                    mat[k + 1 + r, k] = mat[k + 1 + r, k] - acc

        realAkk = float(mat[k, k].real)
        pivot_is_valid = abs(realAkk) > 0.0

        if k == 0 and (not pivot_is_valid):
            # Entire diagonal is zero; fill transpositions with identity and check strictly lower part is zero
            sign_ref[0] = MatrixSign.ZeroSign
            for j in range(n):
                transpositions[j] = j
                all_zero = True
                for r in range(j + 1, n):
                    if mat[r, j] != 0:
                        all_zero = False
                        break
                ret = ret and all_zero
            return ret

        if rs > 0 and pivot_is_valid:
            # A21 /= realAkk
            inv = 1.0 / realAkk
            for r in range(rs):
                mat[k + 1 + r, k] = mat[k + 1 + r, k] * inv
        elif rs > 0:
            # Must be zero to be valid
            all_zero = True
            for r in range(rs):
                if mat[k + 1 + r, k] != 0:
                    all_zero = False
                    break
            ret = ret and all_zero

        if found_zero_pivot and pivot_is_valid:
            ret = False
        elif not pivot_is_valid:
            found_zero_pivot = True

        # Update sign
        if sign_ref[0] == MatrixSign.PositiveSemiDef:
            if realAkk < 0.0:
                sign_ref[0] = MatrixSign.Indefinite
        elif sign_ref[0] == MatrixSign.NegativeSemiDef:
            if realAkk > 0.0:
                sign_ref[0] = MatrixSign.Indefinite
        elif sign_ref[0] == MatrixSign.ZeroSign:
            if realAkk > 0.0:
                sign_ref[0] = MatrixSign.PositiveSemiDef
            elif realAkk < 0.0:
                sign_ref[0] = MatrixSign.NegativeSemiDef

    return ret


def ldlt_eigen3_lower(
    A: np.ndarray, itype: np.dtype = np.int32, check_symmetry: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, MatrixSign, bool]:
    assert_is_square_matrix(A)
    if check_symmetry:
        assert_is_symmetric_matrix(A)

    # Retrieve the problem size
    dim = A.shape[0]
    dtype = A.dtype

    # Initialize the factorization
    LD = np.copy(A)
    LD[np.triu_indices_from(LD, k=1)] = 0
    transpositions = np.arange(dim, dtype=itype)
    scratch = np.zeros(dim, dtype=dtype)
    sign = [MatrixSign.ZeroSign]
    success = True

    # Handle the case were the matrix actually represents a scalar
    if dim <= 1:
        A00 = A[0, 0]
        zero_ = dtype(0)
        LD[0, 0] = A00
        transpositions[0] = 0
        if dim == 0:
            sign[0] = MatrixSign.ZeroSign
        elif A00 > zero_:
            sign[0] = MatrixSign.PositiveSemiDef
        elif A00 < zero_:
            sign[0] = MatrixSign.NegativeSemiDef
        else:
            sign[0] = MatrixSign.ZeroSign

    # Otherwise, proceed with the factorization
    else:
        success = ldlt_eigen3_lower_inplace(LD, transpositions, scratch, sign)

    # Return the tuple of factorization data
    return LD, transpositions, scratch, sign[0], success


###
# Solve
###


def ldlt_eigen3_lower_solve_inplace(
    LD: np.ndarray,
    transpositions: np.ndarray,
    x: np.ndarray,
    tol: float | None = None,
) -> None:
    # Solve A x = b where A = P (L D L^*) P^T and mat stores L (strict lower) and D on the diagonal.
    tol = _make_tolerance(tol, dtype=LD.dtype)

    n = LD.shape[0]

    # Apply P^T: reverse the recorded swaps
    for k in range(n):
        p = int(transpositions[k])
        if p != k:
            x[k], x[p] = x[p], x[k]

    # Forward solve: L z = P^T b  (L has unit diagonal; stored in strict lower of mat)
    for i in range(n):
        s = LD.dtype.type(0)
        # sum_{j < i} L[i,j] * z[j]
        for j in range(i):
            s += LD[i, j] * x[j]
        x[i] = x[i] - s

    # Diagonal solve: D w = z  (D is real on the diagonal of mat)
    for i in range(n):
        di = LD[i, i]
        if np.abs(di) > tol:
            x[i] = x[i] / di
        else:
            x[i] = 0.0

    # Backward solve: L^T y = w
    for i in range(n - 1, -1, -1):
        s = LD.dtype.type(0)
        for j in range(i + 1, n):
            s += LD[j, i] * x[j]
        x[i] = x[i] - s

    # Apply P: forward recorded swaps
    for k in range(n - 1, -1, -1):
        p = int(transpositions[k])
        if p != k:
            x[k], x[p] = x[p], x[k]


def ldlt_eigen3_lower_solve(mat: np.ndarray, transpositions: np.ndarray, b: np.ndarray, tol: float = 1e-8) -> bool:
    x = b.astype(mat.dtype, copy=True)
    ldlt_eigen3_lower_solve_inplace(mat, transpositions, x, tol)
    return x


###
# Unpack
###


def ldlt_eigen3_lower_unpack(
    LD: np.ndarray,
    transpositions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dim = LD.shape[0]
    dtype = LD.dtype

    # Extract L and D from LD (L has unit diagonal, D is real diagonal)
    L = np.tril(LD, k=-1) + np.eye(dim, dtype=dtype)
    D = np.diag(LD.diagonal().real.astype(dtype))

    # Build an explicit permutation matrix
    perm = np.arange(dim, dtype=np.int32)
    for k, p in enumerate(transpositions):
        if p != k:
            perm[k], perm[p] = perm[p], perm[k]
    P = np.eye(dim, dtype=dtype)[:, perm]

    # Return the unpacked factorization
    return L, D, P
