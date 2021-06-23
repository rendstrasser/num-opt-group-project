import numpy as np


# TODO improve
def check_stopping_criterion(prev_direction_state, new_direction_state, tolerance):
    grad_norm = np.linalg.norm(new_direction_state.gradient)

    if grad_norm < tolerance:
        return True, grad_norm

    return False, grad_norm