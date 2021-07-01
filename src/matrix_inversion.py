import numpy as np


# TODO improve
def invert_matrix(matrix: np.ndarray) -> np.ndarray:
    # does matrix inversion using the Gauss-Jordan Algorithm
    # returns the inverse matrix
    # requires the input matrix to have the same number of rows as columns

    if len(matrix)!=len(matrix[0]):
        raise ValueError("Matrix must have same number of rows as columns to be invertible")
    inverse=np.identity(len(matrix))
    #do inversion for lower left part
    for j, column in enumerate(np.transpose(matrix)):
        for i, row in enumerate(matrix):
            if i==j:
                if matrix[i][j]!=0:

                    selected=matrix[i][j]

                    matrix[i]=matrix[i]/selected
                    inverse[i] = inverse[i] / selected

                else:
                    for x in range(len(matrix)):
                        if matrix[x][j]!=0 and x>j:
                            selected=matrix[x][j]
                            matrix[i] = np.add(matrix[i], matrix[x]/selected)
                            inverse[i] = inverse[i] + inverse[x]/selected
                            break
                        elif x==len(matrix)-1:
                            raise ValueError("Matrix is not invertible")
            if i>j:
                selected=matrix[i][j]
                matrix[i]=matrix[i]-selected* matrix[j]
                inverse[i] = inverse[i] - selected * inverse[j]

    #do inversion for upper right part:
    for j, column in reversed(list(enumerate(np.transpose(matrix)))):
        for i, row in reversed(list(enumerate(matrix))):
            if i<j:
                selected=matrix[i][j]
                matrix[i]=matrix[i]-selected* matrix[j]
                inverse[i] = inverse[i] - selected * inverse[j]

    return inverse


