import numpy as np
from src.classes import MinimizationProblem, IterationState
from src.matrix_inversion import invert_matrix


def newton_direction(
        x,
        problem: MinimizationProblem,
        prev_state: IterationState):
    """
    Calculates the direction p with the newton method (see book page 22, 2.15).
    Outputs an iteration state, which contains the current x, direction p and the gradient at x.

    :param x: Current approximated minimizer x
    :param problem: The problem we are trying to minimize.
    :param prev_state: IterationState from previous minimization iteration. May be None.
    :return: Returns an IterationState which represents the current executed iteration
    """

    hessian = problem.calc_hessian_at(x)
    hessian_inv = invert_matrix(hessian, problem.settings.custom_matrix_inversion_enabled)
    grad = problem.calc_gradient_at(x)
    p = -hessian_inv @ grad

    return IterationState(x, p, grad)
