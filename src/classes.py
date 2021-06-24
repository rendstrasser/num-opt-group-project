import numpy as np
from typing import Callable
from dataclasses import dataclass


@dataclass
class MinimizationProblem:
    """
    Data class containing all necessary information of a minimization problem to support
    steepest descent, newton, quasi-newton and conjugate minimization.

    Args:
        f (Callable): The function (objective) we are trying to minimize.
        gradient_f (Callable): The gradient function of f.
        hessian_f (Callable): The hessian function of f.
        solution (list): The solution(s) to the minimization problem.
                             Might contain multiple if there are multiple local minimizers.
        x0 (list): The starting point for the minimization procedure.
    """
    f: Callable[[np.ndarray], np.ndarray]
    gradient_f: Callable[[np.ndarray], np.ndarray]
    hessian_f: Callable[[np.ndarray], np.ndarray]
    solution: np.ndarray
    x0: np.ndarray

    # TODO improve
    def calc_gradient_at(self,
            f: Callable[[np.ndarray], np.ndarray],
            x: np.ndarray) -> np.ndarray:

        if self.gradient_f is not None:
            return self.gradient_f(x)

        # TODO
        raise NotImplementedError()

    # TODO improve (maybe)
    def calc_hessian_at(self,
            f: Callable[[np.ndarray], np.ndarray],
            x: np.ndarray) -> np.ndarray:

        if self.hessian_f is not None:
            return self.hessian_f(x)

        # TODO
        raise NotImplementedError()


@dataclass
class IterationState:
    """
    Data class containing the state of a direction calculation iteration
    within a minimization procedure,holding interesting data for consumers
    and for the next iteration of the direction calculation.

    Args:
        x (list): The approximated minimizer input for a specific iteration.
        direction (list): The calculated direction (p) of a specific iteration.
        gradient (list): The calculated gradient at x at a specific iteration.
    """
    x: np.ndarray
    direction: np.ndarray
    gradient: np.ndarray


@dataclass
class BfgsQuasiNewtonState(IterationState):
    """
    Data class containing the state of a direction calculation iteration
    within a BFGS quasi-newton minimization procedure, holding interesting data for consumers
    and for the next iteration of the direction calculation.

    Args:
        x (list): The approximated minimizer input for a specific iteration.
        direction (list): The calculated direction (p) of a specific iteration.
        gradient (list): The calculated gradient at x at a specific iteration.
        H: (2D list): The H that was used to calculate the direction (p).
                      Stored in the state to avoid recomputation at the next iteration.
    """
    H: np.ndarray