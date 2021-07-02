import numpy as np

from src.classes import MinimizationProblem, MinimizationProblemSettings


def create_quadratic_problem(n, random_state=None, settings=MinimizationProblemSettings()):
    """
    Creates a quadratic problem of the form Ax=b with x element of the real numbers to the power n.

    A is created by sampling random integers between 1 and 10 into a matrix A_gen
    and then calculating A_gen * A_gen^T which gives us a positive-semi-definite matrix and with
    a high probability positive-definite. For the actual 5 problems that we create, we
    choose a seed that ensures positive-definite A.

    The solution x is randomly sampled with integers between 1 and 10.

    We choose the starting point x0 to be all 0.

    :param settings: Settings for the minimization problem
    :param n: Dimensionality of x
    :param random_state: (optional) Random state to enforce a specific seed and therefore determinism of the outcome.
    :param supply_gradient: (bool) Whether to supply the gradient or not. Default to `True`.
    :param supply_hessian: (bool) Whether to supply the gradient or not. Default to `True`.
    :return: a random quadratic MinimizationProblem of the form Ax=b
    """

    # Set the seed if given to ensure a deterministic outcome of this function.
    if random_state is not None:
        np.random.seed(random_state)

    # Sample a generator matrix for A randomly
    A_generator = np.random.randint(low=1, high=10, size=(n, n))

    # Calculate A_gen*A_gen^T to ensure positive-(semi-)definite matrix
    A = A_generator @ A_generator.T

    # Sample a solution x randomly
    solution = np.random.randint(low=1, high=10, size=n)

    if settings.degenerate_problem:  # if we want to degenerate the problem
        scaling_matrix = np.identity(A.shape[0])  # initialize a scaling matrix
        scaling_matrix[0, 0] = 1000  # set one element much larger than the others
        A = scaling_matrix @ A  # multiply with old A

    # Calculate b
    b = A @ solution

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

    # Starting point x0 set to all 0
    x0 = np.zeros(n)

    # Construct the final minimization problem and return it
    return MinimizationProblem(A=A, b=b, f=f, solution=solution, x0=x0,
                               settings=settings,
                               gradient_f=d_f if not settings.gradient_approximation_enabled else None,
                               hessian_f=d2_f if not settings.hessian_approximation_enabled else None)


def create_non_quadratic_problem(random_state=None, settings=MinimizationProblemSettings()):
    """
    Creates a 1-dimensional non-quadratic problem of the form (x-a)*(x-b)*(x-c).

    We sample a as a random integer between -10 and -1.
    We sample b as a random integer between -1 and 7.
    We sample c as a random integer between 7 and 15.

    The solution x is randomly sampled with numbers between 1 and 10.

    We choose the starting point x0 to be -20 s.t.
    matrices B, which represent the Hessian or an approximation of it,
    are positive-definite and we therefore are guaranteed to get a
    local minimizer.

    :param random_state: (optional) Random state to enforce a specific seed and therefore determinism of the outcome.
    :param supply_gradient: (bool) Whether to supply the gradient or not. Default to `True`.
    :param supply_hessian: (bool) Whether to supply the gradient or not. Default to `True`.
    :return: a random non-quadratic MinimizationProblem of the form (x-a)*(x-b)*(x-c)
    """

    # Set the seed if given to ensure a deterministic outcome of this function.
    if random_state is not None:
        np.random.seed(random_state)

    # Sample a,b,c randomly
    a = np.random.randint(low=-10, high=-1)
    b = np.random.randint(low=-1, high=7)
    c = np.random.randint(low=7, high=15)

    # Define the solution to be one of the 2 possible local minimizers
    solution = [np.array([a]), np.array([c])]

    def f(x):
        """
        Function that we want to minimize (antiderivative of (x-a)(x-b)(x-c))
        Calculates the function value at x.

        :param x: input x (1D list with only 1 element)
        :return: scalar value at f(x)
        """
        return 1/12 * x[0]*(-4*x[0]**2* (a+b+c) + 6*x[0]*(a*(b+c) + b*c) - 12*a*b*c + 3*x[0]**3)

    def d_f(x):
        """
        First derivative of function that we want to minimize.
        Calculates the gradient at x.

        :param x: input x (1D list with only 1 element)
        :return: 1D array at f'(x) (with a single element)
        """
        return (x-a)*(x-b)*(x-c)

    def d2_f(x):
        """
        Second derivative of function that we want to minimize.
        Calculates the Hessian at x.

        :param x: input x (1D list with only 1 element)
        :return: 2D array at f''(x) (with a single element)
        """
        return np.array([a*(b+c-2*x) + b*(c-2*x) + x*(3*x-2*c)])

    # Choose x0=-20 s.t. we can assume global convergence for Newton due to positive Hessian.
    x0 = np.array([-20], dtype=np.float64)

    # Construct the final minimization problem and return it
    return MinimizationProblem(A=None, b=None, f=f, solution=solution, x0=x0,
                               settings=settings,
                               gradient_f=d_f if not settings.gradient_approximation_enabled else None,
                               hessian_f=d2_f if not settings.hessian_approximation_enabled else None)