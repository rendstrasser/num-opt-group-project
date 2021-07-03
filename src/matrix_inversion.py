import numpy as np
from mpmath import mp
from mpmath import matrix as mpmatrix


def invert_matrix(matrix: np.ndarray, custom_matrix_inversion_enabled: bool) -> np.ndarray:
    # does matrix inversion using the Gauss-Jordan Algorithm
    # returns the inverse matrix
    # requires the input matrix to have the same number of rows as columns

    # we use mpmath here as we require the floating point precision for our matrix inversion
    matrix = mpmatrix(matrix)

    if matrix.cols != matrix.rows:
        raise ValueError("Matrix must have same number of rows as columns to be invertible")

    if not custom_matrix_inversion_enabled:
        return np.array(mp.inverse(matrix).tolist(), dtype=np.float64)

    inverse = mp.eye(len(matrix))

    # do inversion for lower left part
    for j in range(matrix.cols):
        for i in range(matrix.rows):
            if i > j:
                v = matrix[i, j]
                matrix[i, :] -= v * matrix[j, :]
                inverse[i, :] -= v * inverse[j, :]
                continue

            if i == j:
                v = matrix[i, j]

                # diagonal elements will be adjusted all entries on the left are 0.
                # what is left is to get the diagonal to be 1
                if v != 0:
                    matrix[i, :] /= v
                    inverse[i, :] /= v

                # as diagonal is 0, we need to adjust the rows of the matrix in order
                # to get have a non-zero diagonal entry (which is required for invertible matrices)
                else:
                    for x in range(j + 1, matrix.cols):
                        v = matrix[x, j]
                        if v != 0:
                            matrix[i, :] += (matrix[x, :] / v)
                            inverse[i, :] += (inverse[x, :] / v)
                            break
                    # exit with value 0 implies non-invertible matrix
                    if v == 0:
                        raise ValueError("Matrix is not invertible")
                continue

    # do inversion for upper right part:
    for j in reversed(range(matrix.cols)):
        for i in reversed(range(matrix.rows)):
            if i < j:
                v = matrix[i, j]
                matrix[i, :] -= v * matrix[j, :]
                inverse[i, :] -= v * inverse[j, :]

    # convert back to numpy
    inverse = np.array(inverse.tolist(), dtype=np.float64)
    return inverse


if __name__ == "__main__":
    mp.mps = 40
    A = np.array([[1, 2, 3], [5, 2, 4], [3, 5, 1]])
    correct_inverse = invert_matrix(A, custom_matrix_inversion_enabled=False)
    custom_inverse = invert_matrix(A, custom_matrix_inversion_enabled=True)
    print("Correct")
    print(correct_inverse)
    print("Custom")
    print(custom_inverse)

    print("==============\nDifference:")
    print(correct_inverse - custom_inverse)

