import numpy as np
from dataclasses import dataclass, make_dataclass
from typing import Callable, Tuple
from decimal import Decimal

from numpy.lib import gradient


@dataclass
class MinimizationProblem:
    """ 
    Data class containing all necessary information of a minimization problem to support 
    steepest descent, newton, quasi-newton and conjugate minimization.
    """
    f: Callable[[np.ndarray], np.ndarray]
    gradient_f: Callable[[np.ndarray], np.ndarray]
    hessian_f: Callable[[np.ndarray], np.ndarray]
    solution: np.ndarray
    x0: np.ndarray

    # TODO improve
    def calc_gradient_at(self, f: Callable[[np.ndarray], np.ndarray], x: np.ndarray) -> np.ndarray:
        if self.gradient_f is not None:
            return self.gradient_f(x)

        # TODO
        raise NotImplementedError()

    # TODO improve (maybe)
    def calc_hessian_at(self, f: Callable[[np.ndarray], np.ndarray], x: np.ndarray) -> np.ndarray:
        if self.hessian_f is not None:
            return self.hessian_f(x)

        # TODO
        raise NotImplementedError()


@dataclass
class IterationState:
    """
    Data class containing the state of a direction calculation iteration, 
    holding interesting data for consumers and for the next iteration of the direction calculation.
    """
    x: np.ndarray
    direction: np.ndarray
    gradient: np.ndarray


@dataclass
class BfgsQuasiNewtonState(IterationState):
    H: np.ndarray

_epsilon = np.sqrt(np.finfo(float).eps)

def find_minimizer(
    problem: MinimizationProblem, 
    x0: np.ndarray, 
    direction_method: Callable[[np.ndarray, MinimizationProblem, IterationState], IterationState],
    a0 = 1, 
    tolerance = 1e-5, 
    max_iter = 10_000):

    x = x0

    direction_state = None
    gradients = []
    for i in range(max_iter):
        x, new_direction_state = _backtracking_line_search(problem, x, direction_state, direction_method, a0)

        stopping_criteria_met, grad_norm = check_stopping_criterion(direction_state, new_direction_state, tolerance)
        gradients.append(grad_norm)

        if stopping_criteria_met:
            break

        direction_state = new_direction_state

    return x, gradients

# TODO improve
def check_stopping_criterion(prev_direction_state, new_direction_state, tolerance):
    grad_norm = np.linalg.norm(new_direction_state.gradient)

    if grad_norm < tolerance:
        return True, grad_norm

    return False, grad_norm

# TODO improve
def invert_matrix(A: np.ndarray) -> np.ndarray:
    return np.linalg.inv(A)


# calculates conjugate direction with Fletcher-Reeves method
def fr_conjugate_direction(x, problem: MinimizationProblem, prev_state: IterationState):
    grad = problem.calc_gradient_at(problem.f, x)
    p = -grad
    
    if prev_state is not None:
        prev_grad = prev_state.gradient
        beta = (grad @ grad) / (prev_grad @ prev_grad)
        p += beta * prev_state.direction

    return IterationState(x, p, grad)

# calculates quasi-newton direction with BFGS method
def bfgs_quasi_newton_direction(x, problem: MinimizationProblem, prev_state: BfgsQuasiNewtonState):
    I = np.identity(len(x))
    H = I
    grad = problem.calc_gradient_at(problem.f, x)

    if prev_state is not None:
        s = x - prev_state.x
        y = grad - prev_state.gradient
        rho_denominator = y @ s

        if rho_denominator != 0: # safety condition
            rho = 1/rho_denominator
            H = (I - rho * np.outer(s, y)) @ prev_state.H @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)

    p = -H @ grad
    return BfgsQuasiNewtonState(x, p, grad, H)

# calculates steepest descent direction
def steepest_descent_direction(x, problem: MinimizationProblem, prev_state: IterationState):
    grad = problem.calc_gradient_at(problem.f, x)
    p = -grad

    return IterationState(x, p, grad)

# calculates newton direction
def newton_direction(x, problem: MinimizationProblem, prev_state: IterationState):
    hessian = problem.calc_hessian_at(problem.f, x)
    hessian_inv = invert_matrix(hessian)
    grad = problem.calc_gradient_at(problem.f, x)
    p = -hessian_inv @ grad

    return IterationState(x, p, grad)

# performs backtracking line search
def _backtracking_line_search(
    problem: MinimizationProblem, 
    x: np.ndarray, 
    prev_p_state: IterationState,
    direction_method: Callable[[np.ndarray, MinimizationProblem, IterationState], IterationState],
    a0 = 1, c = 0.4, p = 0.8) -> Tuple[np.ndarray, IterationState]:

    a = a0

    p_state = direction_method(x, problem, prev_p_state)

    # first wolfe condition
    while problem.f(x + a * p_state.direction) > (problem.f(x) + c * a * (p_state.gradient @ p_state.direction)):
        a *= p 

        if a * np.linalg.norm(p_state.direction) < _epsilon:
            # step must not become smaller than precision
            break

    return x + a * p_state.direction, p_state