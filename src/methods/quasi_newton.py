import numpy as np
from src.classes import MinimizationProblem, BfgsQuasiNewtonState



# calculates quasi-newton direction with BFGS method
def bfgs_quasi_newton_direction(
        x,
        problem: MinimizationProblem,
        prev_state: BfgsQuasiNewtonState):

    I = np.identity(len(x))
    H = I
    grad = problem.calc_gradient_at(problem.f, x)

    if prev_state is not None:
        s = x - prev_state.x
        y = grad - prev_state.gradient
        rho_denominator = y @ s

        if rho_denominator != 0:  # safety condition
            rho = 1 / rho_denominator
            H = (I - rho * np.outer(s, y)) @ prev_state.H @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)

    p = -H @ grad
    return BfgsQuasiNewtonState(x, p, grad, H)
