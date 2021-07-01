import numpy as np


def scaling_ruiz(A):
    """
    This function computes the matrices D and E that scale the problem to attain rows and columns that are
    normalized to 1

    :param A: matrix A, that is being rescaled
    :return: D, E
    """

    threshold = 1 + 10**-5  # upper bound on the ratio between the largest and smallest row and column

    m, n = A.shape  # dimensions of the matrix A

    # stores the ratios, used for scaling the problem
    d1 = np.ones(shape=m)
    d2 = np.ones(shape=n)
    
    B = A.copy()  # initializes B as a copy of A
    
    r1 = np.inf  # ratio of rows, used for stopping the loop
    r2 = np.inf  # ratio of columns, used for stopping the loop

    while r1 > threshold or r2 > threshold:  # performs the algorithm until the ratio is small enough
        row_norm = np.sum(np.abs(B)**2, axis=1)**(1/2)  # l2 norm of rows
        col_norm = np.sum(np.abs(B)**2, axis=0)**(1/2)  # l2 norm of columns
        
        d1 = np.multiply(d1, row_norm**(-1/2))  # normalizes the d1 (corresponds to norms of the rows of matrix)
        d2 = np.multiply(d2, (m/n)**1/4 * col_norm**(-1/2))  # normalizes d2 (corr. to norms of the cols of matrix)
        B = np.diag(d1) @ A @ np.diag(d2)  # computes new B (rescaled version of A)
        
        row_norm = np.sum(np.abs(B)**2, axis=1)**(1/2)  # l2 norm of rows
        col_norm = np.sum(np.abs(B)**2, axis=0)**(1/2)  # l2 norm of columns
        
        r1 = np.max(row_norm) / np.min(row_norm)  # ratio of rows, used for stopping the loop
        r2 = np.max(col_norm) / np.min(col_norm)  # ratio of columns, used for stopping the loop

    # matrices used for scaling the problem to a normalized space
    D = np.diag(d1)
    E = np.diag(d2)

    return D, E
