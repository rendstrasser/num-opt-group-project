import numpy as np
from src.classes import MinimizationProblem, IterationState


def fr_conjugate_direction(
        x,
        problem: MinimizationProblem,
        prev_state: IterationState):
    """
    Calculates the direction p with the FR conjugate method (see book page 121, algorithm 5.4).
    Outputs an iteration state, which contains the current x, direction p and the gradient at x.

    :param x: Current approximated minimizer x
    :param problem: The problem we are trying to minimize.
    :param prev_state: IterationState from previous minimization iteration. May be None.
    :return: Returns an IterationState which represents the current executed iteration
    """

    grad = problem.calc_gradient_at(x)
    p = -grad

    if prev_state is not None:
        prev_grad = prev_state.gradient
        beta = (grad @ grad) / (prev_grad @ prev_grad)
        p += beta * prev_state.direction

    return IterationState(x, p, grad)
