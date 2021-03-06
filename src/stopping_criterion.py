import numpy as np

from src.classes import MinimizationProblem, IterationState


def check_stopping_criterion(
        problem: MinimizationProblem,
        prev_direction_state: IterationState,
        first_direction_state: IterationState,
        new_direction_state: IterationState,
        tolerance: float):
    """
    This function determines if the minimization process should be stopped, based on multiple criteria

    :param problem: the problem (objective) that we want to minimize
    :param prev_direction_state: State of the previous iteration
    :param first_direction_state: State after the first iteration
    :param new_direction_state: Current state
    :param tolerance: Tolerance value, at which we stop the minimization

    :return: bool (True if we should stop, else False), grad_norm (norm of gradient)
    """

    grad_norm = np.linalg.norm(new_direction_state.gradient)  # norm of gradient

    # if advanced stopping criteria are disabled by the settings, just look at the gradient norm vs tolerance
    if not problem.settings.advanced_stopping_criteria_enabled:
        if grad_norm < tolerance:
            return True, grad_norm

        return False, grad_norm


    x0 = problem.x0  # starting position
    f0 = problem.f(x0)  # function value at starting position
    grad_x0 = problem.gradient_f(x0)  # gradient at x0

    # norm of gradient at x0 times factor (tolerance)
    # for our problems, the tolerance needs to be multiplied with 1e-2 to guarantee close enough results
    tolerance_norm = tolerance * 1e-2 * np.linalg.norm(grad_x0)
    
    # checks if the ratio between the current gradient and the gradient at x0 surpasses the threshold
    if grad_norm < tolerance_norm:
        print("ratio of gradients too small:", grad_norm)
        return True, grad_norm

    # additional checks which require a previous iteration 
    # (we could consider x0 as an iteration but not doing this check for the very first step doesn't hurt)
    if prev_direction_state:  # first iterations has no previous state
        
        x_prev = prev_direction_state.x  # previous position
        x_cur = new_direction_state.x  # current position
        f_prev = problem.f(x_prev)  # previous function value
        f_cur = problem.f(x_cur)  # current function value
        
        f_norm_step = np.abs(f_cur - f_prev)  # distance between function values of latest step
        x_norm_step = np.linalg.norm(x_cur - x_prev)  # distance between current position and last position

        x1 = first_direction_state.x  # position after first step
        f1 = problem.f(x1)  # function value after first step
        f_norm_start = np.abs(f1 - f0)  # length of the first step
        x_norm_start = np.linalg.norm(x1 - x0)  # distance between positions after first step

        # checks if the ratio between the function values of the current step and the first step surpass the threshold
        if f_norm_step < tolerance * f_norm_start:
            print("ratio f(x) of cur step and f(x) of first step too close:", f_norm_step)
            return True, grad_norm
       
        # checks if the ratio of the current step (position) and the first step surpasses the threshold
        if x_norm_step < tolerance * x_norm_start:
            print("ratio of cur step (x) and first step too close:", x_norm_step)
            return True, grad_norm

    return False, grad_norm
