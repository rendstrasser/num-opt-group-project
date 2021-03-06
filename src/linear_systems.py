import numpy as np

from src.matrix_inversion import invert_matrix

def solve(A: np.ndarray, b: np.ndarray,
          use_cholesky=True,
          use_gaussian_elimination_inverse=False) -> np.ndarray:
    """
    Solves the linear system of Ax = b using either cholesky factorization, gaussian elimination with pivoting
    or the default numpy implementation.

    :param A: non-singular matrix
    :param b: solution to Ax
    :param use_cholesky: setting to use cholesky factorization for solving
    :param use_gaussian_elimination_inverse: setting to use gaussian elimination with pivoting for solving
    :return: solution x of Ax = b
    """

    if use_cholesky:
        return solve_cholesky(A, b)

    if use_gaussian_elimination_inverse:
        return solve_gaussian_elimination_inverse(A, b)

    return np.linalg.solve(A, b)

def solve_gaussian_elimination_inverse(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solves
    :param A:
    :param b:
    :return:
    """
    A_inv = invert_matrix(A)

    return A_inv @ b


def solve_cholesky(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solves Ax = b without computing the inverse of A but uses forward/backward substitution and the specified substitution

    :param A: symmetric, positive-definite matrix
    :param b: solution to linear system
    :param cholesky: if set to false, PLU factorization is used. use cholesky only for positive definite matrices, else LU
    :return: x (np.array with shape n x 1)
    """

    L = cholesky_factorization(A)
    y = forward(L, b)
    solution = backward(L.T, y)

    return solution


def cholesky_factorization(A):
    """
    factorizes input matrix in lower triangular matrix, requires the matrix to be symmetric and positive definite
    :param A: has to me a square matrix
    :return: lower triangular matrix
    """
    assert A.shape[0] == A.shape[1]

    N = A.shape[1]

    L = np.zeros_like(A, dtype=np.float64)

    for i in range(N):
        for j in range(i+1):
            s = np.dot(L[i, :j], L[j, :j])
            if i == j:
                L[i, j] = np.sqrt(np.max(A[i, i] - s, 0))
            else:
                L[i, j] = (1 / L[j, j]) * (A[i, j] - s)

    return L

def forward(L, b):
    y = []
    for i in range(len(b)):
        y.append(b[i])
        for j in range(i):
            y[i] = y[i] - (L[i,j]*y[j])
        y[i] = y[i]/L[i,i]

    return y


def backward(Lt, y):
    x = np.zeros_like(y)
    for i in range(x.shape[0] , 0, -1):
        x[i - 1] = (y[i -1] - np.dot(Lt[i-1, i:],x[i:]))/Lt[i-1, i-1]

    return x
