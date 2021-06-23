import numpy as np
from typing import Callable, Tuple

from src.classes import MinimizationProblem, IterationState
from src.stopping_criterion import check_stopping_criterion


def find_minimizer(
        problem: MinimizationProblem,
        x0: np.ndarray,
        direction_method: Callable[[np.ndarray, MinimizationProblem, IterationState], IterationState],
        a0=1,
        tolerance=1e-5,
        max_iter=10_000):
    x = x0

    direction_state = None
    gradients = []
    for i in range(max_iter):
        x, new_direction_state = backtracking_line_search(problem, x, direction_state, direction_method, a0)

        stopping_criteria_met, grad_norm = check_stopping_criterion(direction_state, new_direction_state, tolerance)
        gradients.append(grad_norm)

        if stopping_criteria_met:
            break

        direction_state = new_direction_state

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

    a = a0

    p_state = direction_method(x, problem, prev_p_state)

    # first wolfe condition
    while problem.f(x + a * p_state.direction) > (problem.f(x) + c * a * (p_state.gradient @ p_state.direction)):
        a *= p

        if a * np.linalg.norm(p_state.direction) < _epsilon:
            # step must not become smaller than precision
            break

    return x + a * p_state.direction, p_state
