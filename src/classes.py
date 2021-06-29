import numpy as np
from typing import Callable
from dataclasses import dataclass

from numpy.lib.function_base import gradient


@dataclass
class MinimizationProblem:
    """
    Data class containing all necessary information of a minimization problem to support
    steepest descent, newton, quasi-newton and conjugate minimization.

    Args:
        f (Callable): The function (objective) we are trying to minimize.
        solution (list): The solution(s) to the minimization problem.
                             Might contain multiple if there are multiple local minimizers.
        x0 (list): The starting point for the minimization procedure.
        gradient_f (Callable): The gradient function of f. Optional.
        hessian_f (Callable): The hessian function of f. Optional.
    """
    f: Callable[[np.ndarray], np.ndarray]
    solution: np.ndarray
    x0: np.ndarray
    gradient_f: Callable[[np.ndarray], np.ndarray] = None
    hessian_f: Callable[[np.ndarray], np.ndarray] = None

    # Why do we pass f, when it is saved in the state?
    def calc_gradient_at(self, f: Callable, x: np.ndarray) -> np.ndarray:
        """Calculate gradient at point `x`. Uses an approximation, if the gradient is not explicitly known.

        Args:
            f (Callable): Function. 
            x (np.ndarray): Array representing some point in the domain of the function.

        Returns:
            np.ndarray: (Approxmiated or true) gradient at point `x`.
        """

        return (self.gradient_f or self.central_difference_gradient)(x)

    def central_difference_gradient(self, x: np.ndarray) -> np.ndarray:
        """Approximate gradient as described in equation (8.7), called the 'central difference formula'.

        Args:
            x (np.ndarray): Function input.

        Returns:
            np.ndarray: Approximated gradient.
        """

        # Given the datatype of x, the below is the least number such that `1.0 + eps != 1.0`.
        eps = np.finfo(x.dtype).eps

        return np.array([
            (self.f(x + eps*unitvector) + self.f(x - eps*unitvector))/2*self.epsilon
            for unitvector in np.eye(N=len(x))
        ])
        

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