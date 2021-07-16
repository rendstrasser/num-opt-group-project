import numpy as np
from typing import Callable
from dataclasses import dataclass

from src.gradient_approximation import gradient_approximation, hessian_approximation

@dataclass
class MinimizationProblemSettings:
    """
    Data class containing settings to activate/deactivate newly
    implemented features for project phase 2.

    Args:
        gradient_approximation_enabled (bool): Enables approximation of gradients as described in equation (8.7) in the book.
        hessian_approximation_enabled (bool): Enables approximation of hessians as described in equation (8.7) in the book.
        cholesky_linear_systems_solver_enabled (bool): Enables cholesky-based linear systems solving method as described in algorithm (A.2) in the book
        gaussian_elimination_linear_systems_solver_enabled (bool): Enables gaussian elimination with partial row pivoting linear system solving method as described in algorithm (A.1) in the book
        variable_scaling_enabled (bool): Enables variable scaling inversion for quadratic problems using the Ruiz algorithm presented in https://arxiv.org/pdf/1610.03871.pdf
        advanced_stopping_criteria_enabled (bool): Advanced stopping criteria enabled as described in the PDF of phase 2
        degenerate_problem (bool): (only for quadratic problems) Transforms problem into a degenerated problem by scaling first row/column of A by 100
        scale_problem (bool): (only for quadratic problems) Transforms problem into a scaled problem by scaling A and b by 10^5
    """
    gradient_approximation_enabled: bool = False
    hessian_approximation_enabled: bool = False
    cholesky_linear_systems_solver_enabled: bool = False
    gaussian_elimination_linear_systems_solver_enabled: bool = False
    variable_scaling_enabled: bool = False
    advanced_stopping_criteria_enabled: bool = False
    degenerate_problem: bool = False
    scale_problem: bool = False


@dataclass
class MinimizationProblem:
    """
    Data class containing all necessary information of a minimization problem to support
    steepest descent, newton, quasi-newton and conjugate minimization.

    Args:
        A (np.ndarray): Matrix A, used for solving the problem
        b (np.ndarray): Vector b, used for solving the problem
        f (Callable): The function (objective) we are trying to minimize.
        solution (list): The solution(s) to the minimization problem.
                             Might contain multiple if there are multiple local minimizers.
        x0 (list): The starting point for the minimization procedure.
        gradient_f (Callable): The gradient function of f. Optional.
        hessian_f (Callable): The hessian function of f. Optional.
    """
    A: np.ndarray
    b: np.ndarray
    f: Callable[[np.ndarray], np.ndarray]
    solution: np.ndarray
    x0: np.ndarray
    settings: MinimizationProblemSettings = MinimizationProblemSettings()
    gradient_f: Callable[[np.ndarray], np.ndarray] = None
    hessian_f: Callable[[np.ndarray], np.ndarray] = None

    # Why do we pass f, when it is saved in the state?
    def calc_gradient_at(self, x: np.ndarray) -> np.ndarray:
        """Calculate gradient at point `x`. Uses an approximation, if the gradient is not explicitly known.

        Args:
            x (np.ndarray): Array representing some point in the domain of the function.

        Returns:
            np.ndarray: (Approximated or true) gradient at point `x`.
        """

        if self.gradient_f:
            # if the problem has knowledge about the gradient, use it directly without approximation
            return self.gradient_f(x)

        return gradient_approximation(self.f, x)

    def calc_hessian_at(self, x: np.ndarray) -> np.ndarray:
        """ Calculate hessian at point `x`. Uses an approximation, if the hessian is not explicitly known.

        Args:
            x (np.ndarray): Array representing some point in the domain of the function.

        Returns:
            np.ndarray: (Approximated or true) gradient at point `x`.
        """

        if self.hessian_f:
            # if the problem has knowledge about the hessian, use it directly without approximation
            return self.hessian_f(x)

        return hessian_approximation(self.f, x)


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
                      Stored in the state to avoid re-computation at the next iteration.
    """
    H: np.ndarray
