import numpy as np


# TODO improve
def check_stopping_criterion(problem, prev_direction_state, first_direction_state, new_direction_state, tolerance):
    
    tolerance = 10**-8 # debugging
    
    # part 1
    grad_norm = np.linalg.norm(new_direction_state.gradient)
    
    # part 2
    x0 = problem.x0
    grad_x0 = problem.gradient_f(x0)
    tolerance_norm = tolerance * np.linalg.norm(grad_x0)
    
    if prev_direction_state:  # first iterations has no previous state
        
        x_prev = prev_direction_state.x
        x_cur = new_direction_state.x
        f_prev = problem.f(x_prev)
        f_cur = problem.f(x_cur)
        
        f_norm = np.abs(f_cur - f_prev)
        x_norm = np.linalg.norm(x_cur - x_prev)
        
        # part 1 f(x)
#         if f_norm < tolerance:
#             print("f(x) too close:", f_norm)
#             return True, f_norm
        
        # part 2 f(x)
#         if f_norm < tolerance_norm:
#             print("normalized f(x) too close:", f_norm)
#             return True, f_norm
        
        
        # part 1 x
#         if x_norm < tolerance:
#             print("x too close", x_norm)
#             return True, x_norm
        
        # part 2 x
#         if x_norm < tolerance_norm:
#             print("normalized x too close", x_norm)
#             return True, x_norm
        
        
    # Part 1 g(x)
#     if grad_norm < tolerance:
#         print("gradient too small", grad_norm)
#         return True, grad_norm
#     coin = np.random.randint(0,1000)
#     if coin == 238:
#         print(grad_norm)
        
    # part 2 g(x)  
    if grad_norm < tolerance_norm:
        print("normalized gradient too small", grad_norm)
        return True, grad_norm

    
    return False, grad_norm