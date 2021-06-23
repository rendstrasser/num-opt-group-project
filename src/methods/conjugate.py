import numpy as np
from src.classes import MinimizationProblem, IterationState

# calculates conjugate direction with Fletcher-Reeves method
def fr_conjugate_direction(
        x,
        problem: MinimizationProblem,
        prev_state: IterationState):

    grad = problem.calc_gradient_at(problem.f, x)
    p = -grad

    if prev_state is not None:
        prev_grad = prev_state.gradient
        beta = (grad @ grad) / (prev_grad @ prev_grad)
        p += beta * prev_state.direction

    return IterationState(x, p, grad)
