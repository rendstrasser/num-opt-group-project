import numpy as np
from mpmath import mp
from mpmath import matrix as mpmatrix
from typing import Tuple


def invert_matrix(A: np.ndarray) -> np.ndarray:
    """Compute the inverse of a square matrix using gaussian elimination with partial row pivoting.

    See `Algorithm A.1` and `Wikipedia <https://en.wikipedia.org/wiki/LU_decomposition>`.

    :param A: A non-singular n-by-n matrix
    :return: The inverse of ``A``
    """

    if not isinstance(A, np.ndarray):
        raise TypeError("'A' must be of type numpy.ndarray")

    if len(A.shape) > 2:
        raise ValueError("'A' must be two-dimensional")

    n, m = A.shape

    if n != m:
        raise ValueError("'A' must be square")

    A = A.astype(float)
    LU, P = _lup_decompose(A)
    B = np.dot(P, np.eye(n))
    X = _lup_solve(LU, B)

    return X


def _lup_decompose(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the LU decomposition with partial pivoting (LUP) of a square matrix

    See `Algorithm A.1` and `Wikipedia <https://en.wikipedia.org/wiki/LU_decomposition>`.

    :param A: ``A`` as described in :func:`invert_matrix`
    :return: The combined lower and upper triangular n-by-n matrix, the n-by-n permutation matrix
    """

    n = len(A)
    LU, P = np.copy(A), np.eye(n)

    for i in range(n - 1):
        # Find maximum pivot in column
        j = i + np.argmax(np.abs(LU[i:, i]))

        if np.abs(LU[i, j]) < 10e-6:
            raise ValueError("'A' must be non-singular")

        # Permute rows
        LU[[i, j]], P[[i, j]] = LU[[j, i]], P[[j, i]]

        # Perform Gaussian elimination
        LU[i+1 :, i] /= LU[i, i]
        LU[i+1 :, i+1 :] -= np.outer(LU[i+1 :, i], LU[i, i+1 :])

    return LU, P

def _lup_solve(LU: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Solve a system of linear equations by forward and backward substitution

    :param LU: ``LU`` as returned by :func:`_lup_decompose`
    :param B: A n-by-m matrix
    :return: The n-by-m solution matrix
    """

    n = len(LU)
    X = np.copy(B)

    # Forward substitution
    for i in range(n):
        X[i] -= np.dot(LU[i, :i], X[:i])

    # Backward substitution
    for i in reversed(range(n)):
        X[i] -= np.dot(LU[i, i+1 :], X[i+1 :])
        X[i] /= LU[i,i]

    return X