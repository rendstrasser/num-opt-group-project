import numpy as np
from src.classes import MinimizationProblem, IterationState
from src.matrix_inversion import invert_matrix
from src.linear_systems import solve


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
    grad = problem.calc_gradient_at(x)

    p = -solve(hessian, grad,
               problem.settings.cholesky_linear_systems_solver_enabled,
               problem.settings.gaussian_elimination_linear_systems_solver_enabled)

    return IterationState(x, p, grad)
