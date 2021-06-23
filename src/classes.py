import numpy as np
from typing import Callable
from dataclasses import dataclass


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
    Data class containing the state of a direction calculation iteration,
    holding interesting data for consumers and for the next iteration of the direction calculation.
    """
    x: np.ndarray
    direction: np.ndarray
    gradient: np.ndarray


@dataclass
class BfgsQuasiNewtonState(IterationState):
    H: np.ndarray