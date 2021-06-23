import numpy as np
from src.classes import MinimizationProblem, IterationState

# calculates steepest descent direction
def steepest_descent_direction(
        x,
        problem: MinimizationProblem,
        prev_state: IterationState):

    grad = problem.calc_gradient_at(problem.f, x)
    p = -grad

    return IterationState(x, p, grad)
