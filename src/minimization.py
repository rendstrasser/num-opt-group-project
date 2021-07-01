import numpy as np
from typing import Callable, Tuple

from src.classes import MinimizationProblem, IterationState
from src.stopping_criterion import check_stopping_criterion
from src.scaling import scaling_ruiz


def scale_problem(problem):
    """
    This function creates a scaled problem in the quadratic case
    E transforms the computed minimum back into the original space

    :param problem: the problem (objective) that we want to minimize
    :return: E, MinimizationProblem
    """

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
    
    # Starting point x0 set to all 0
    x0 = problem.x0  # we define the point in the new space as the point in the original space
    solution = problem.solution
    
    return E, MinimizationProblem(A=A, b=b, f=f, solution=solution, x0=x0,
                               gradient_f = d_f if problem.gradient_f else None, 
                               hessian_f = d2_f if problem.hessian_f else None)

def find_minimizer(
        problem_orignal: MinimizationProblem,
        direction_method: Callable[[np.ndarray, MinimizationProblem, IterationState], IterationState],
        a0=1,
        tolerance=1e-5,
        max_iter=10_000):
    """
    Executes a minimization procedure on a given problem with a given direction method
    until a local minimizer is found or we reach the maximum number of iterations.

    :param problem: the problem (objective) that we want to minimize
    :param direction_method: method which takes an input x, the minimization problem
                             and an iteration state to calculate another iteration state;
                             i.e., calculates the direction p for an input x and a minimization problem
    :param a0: (optional, default=1) Starting point for the step size length a for each backtracking line search execution
    :param tolerance: (optional, default=1e-5) Tolerance of approximated minimizer - controls how close we want to get to the real solution
    :param max_iter: (optional, default=10000) Maximum number of iterations that we want to execute
    :return: the minimizer x and a list of the L2 gradient norms for all iterations
    """

    E, problem = scale_problem(problem_orignal)
    
    # We choose the starting point as defined in the minimization problem
    x = problem.x0

    # Holds a direction state for the last iteration
    # Initialized as None, because we don't have a previous iteration initially.
    direction_state = None

    # L2 gradient norms for the iterations, will be returned later to allow plotting.
    gradients = []

    # Actual minimization procedure for max_iter steps
    for i in range(max_iter):
        # Perform the backtracking line search at the current point x
        x, new_direction_state = backtracking_line_search(problem, x, direction_state, direction_method, a0)
        
        if i == 0:
            first_direction_state = new_direction_state
            
        # Evaluate if a stopping criterion is fulfilled, which allows us to early exit the minimization procedure
        stopping_criteria_met, grad_norm = check_stopping_criterion(problem, direction_state, first_direction_state, new_direction_state, tolerance)

        # Append the L2 gradient norm to the list of gradient norms, which is returned at the end.
        gradients.append(grad_norm)

        # Early exit if stopping criterion is fulfilled
        if stopping_criteria_met:
            break

        # Remember the current iteration state as the previous one for the next iteration.
        direction_state = new_direction_state

    if E is not None:
        x = E @ x
    # Return the approximated minimizer x and the L2 gradient norms for the iterations
    return x, gradients


# smallest stable floating point number (TODO probably obsolete with mpmath)
_epsilon = np.sqrt(np.finfo(float).eps)


# performs backtracking line search
def backtracking_line_search(
        problem: MinimizationProblem,
        x: np.ndarray,
        prev_p_state: IterationState,
        direction_method: Callable[[np.ndarray, MinimizationProblem, IterationState], IterationState],
        a0=1, c=0.4, p=0.8) -> Tuple[np.ndarray, IterationState]:
    """
    Backtracking line search as described in the book at page 37, algorithm 3.1.

    :param problem: the problem (objective) that we want to minimize
    :param x: current approximated minimizer x
    :param prev_p_state: IterationState from previous minimization iteration. May be None.
    :param direction_method: method which takes an input x, the minimization problem
                             and an iteration state to calculate another iteration state;
                             i.e., calculates the direction p for an input x and a minimization problem
    :param a0: (optional, default=1) Starting point for the step size length a (alpha in the book)
               for each backtracking line search execution
    :param c: (optional, default=0.4) c as in the book at page 37, algorithm 3.1.
    :param p: (optional, default=0.8) p as in the book at page 37, algorithm 3.1.
    :return: new approximated minimizer x and the IterationState of this iteration
    """

    # We start with alpha=a0
    a = a0

    # Calculate the current IterationState which also contains the direction p.
    p_state = direction_method(x, problem, prev_p_state)

    # Check the first Wolfe condition
    while problem.f(x + a * p_state.direction) > (problem.f(x) + c * a * (p_state.gradient @ p_state.direction)):
        a *= p

        if a * np.linalg.norm(p_state.direction) < _epsilon:
            # step must not become smaller than precision, early exit to ensure valid a
            # TODO probably obsolete with mpmath
            break

    # Return the new approximated minimizer x and the IterationState of this iteration
    return x + a * p_state.direction, p_state
