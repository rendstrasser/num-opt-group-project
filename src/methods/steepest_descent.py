import numpy as np
from src.classes import MinimizationProblem, IterationState


def steepest_descent_direction(
        x,
        problem: MinimizationProblem,
        prev_state: IterationState):
    """
    Calculates the direction p with the steepest descent method (see book page 20 at the end of the page).
    Outputs an iteration state, which contains the current x, direction p and the gradient at x.

    :param x: Current approximated minimizer x
    :param problem: The problem we are trying to minimize.
    :param prev_state: IterationState from previous minimization iteration. May be None.
    :return: Returns an IterationState which represents the current executed iteration
    """

    grad = problem.calc_gradient_at(x)
    p = -grad

    return IterationState(x, p, grad)
