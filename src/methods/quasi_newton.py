import numpy as np
from src.classes import MinimizationProblem, BfgsQuasiNewtonState


def bfgs_quasi_newton_direction(
        x,
        problem: MinimizationProblem,
        prev_state: BfgsQuasiNewtonState):
    """
    Calculates the direction p with the BFGS quasi-newton method (see book page 140, 6.17).
    Outputs an iteration state, which contains the current x, direction p and the gradient at x.

    :param x: Current approximated minimizer x
    :param problem: The problem we are trying to minimize.
    :param prev_state: IterationState from previous minimization iteration. May be None.
    :return: Returns an IterationState which represents the current executed iteration
    """

    I = np.identity(len(x))
    H = I
    grad = problem.calc_gradient_at(problem.f, x)

    if prev_state is not None:
        s = x - prev_state.x
        y = grad - prev_state.gradient
        rho_denominator = y @ s

        # safety condition (use H from previous iteration if rho_denominator=0)
        if rho_denominator != 0:
            rho = 1 / rho_denominator
            H = (I - rho * np.outer(s, y)) @ prev_state.H @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)
        else:
            H = prev_state.H

    p = -H @ grad
    return BfgsQuasiNewtonState(x, p, grad, H)
