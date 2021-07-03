import numpy as np

from src.classes import MinimizationProblem


def scale_problem(problem):
    """
    This function creates a scaled problem in the quadratic case.
    Uses the Ruiz algorithm presented in https://arxiv.org/abs/1610.03871 at Algorithm 2.
    E transforms the computed minimum back into the original space.

    :param problem: the problem (objective) that we want to minimize
    :return: a descaling matrix E and the problem formulated as a rescaled variant
    """

    if not problem.settings.variable_scaling_enabled:
        return None, problem

    if problem.A is None:  # we only scale quadratic problems
        return None, problem

    D, E = scaling_ruiz(problem.A)  # returns the matrices that transform the problem into the new space
    A = D @ problem.A @ E  # defines the new A
    b = D @ problem.b  # defines the new b

    # the function and the derivatives need to be defined again since we changed our A and b

    def f(x):
        """
        Function that we want to minimize (antiderivative of Ax-b)
        Calculates the function value at x.

        :param x: input x (1D list with n elements)
        :return: scalar value at f(x)
        """
        return 1/2 * x @ A @ x - b @ x

    def d_f(x):
        """
        First derivative of function that we want to minimize.
        Calculates the gradient at x.

        :param x: input x (1D list with n elements)
        :return: 1D array at f'(x)
        """
        return A @ x - b

    def d2_f(x):
        """
        Second derivative of function that we want to minimize.
        Calculates the Hessian at x.

        :param x: input x (1D list with n elements)
        :return: 2D array at f''(x)
        """
        return A

    # we define the starting point in the new space as the starting point in the original space
    x0 = problem.x0

    return E, MinimizationProblem(A=A, b=b, f=f, solution=problem.solution, x0=x0,
                                  settings=problem.settings,
                                  gradient_f=d_f if problem.gradient_f else None,
                                  hessian_f=d2_f if problem.hessian_f else None)


def scaling_ruiz(A):
    """
    This function computes the matrices D and E that scale the problem to attain rows and columns that are
    normalized to 1.
    Uses the Ruiz algorithm presented in https://arxiv.org/abs/1610.03871 at Algorithm 2.

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
