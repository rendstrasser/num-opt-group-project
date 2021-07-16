import numpy as np
import matplotlib.pyplot as plt

def final_printout(x_0,x_optimal,x_appr,f,grad,tolerance, **kwargs):
    """
    Parameters
    --------------------------------------------------------------------------------------------------------------
    x_0: numpy 1D array, corresponds to initial point
    x_optimal: numpy 1D array, corresponds to optimal point, which you know, or have solved analytically
    x_appr: numpy 1D array, corresponds to approximated point, which your algorithm returned
    --------------------------------------------------------------------------------------------------------------
    f: function which takes 2 inputs: x (initial, optimal, or approximated)
                                      **args
       Function f returns a scalar output.
    --------------------------------------------------------------------------------------------------------------
    grad: function which takes 3 inputs: x (initial, optimal, or approximated),
                                         function f,
                                         args (which are submitted, because you might need
                                              to call f(x,**args) inside your gradient function implementation).
          Function grad approximates gradient at given point and returns a 1d np array.
    --------------------------------------------------------------------------------------------------------------
    args: dictionary, additional (except of x) arguments to function f
    tolerance: float number, absolute tolerance, precision to which, you compare optimal and approximated solution.
    """

    print(f'Initial x is :\t\t{x_0}')
    print(f'Optimal x is :\t\t{x_optimal}')
    print(f'Approximated x is :\t{x_appr}')
    print(f'Is close verification: \t{np.isclose(x_appr,x_optimal,atol=tolerance)}\n')
    f_opt = f(x_optimal,**kwargs)
    f_appr = f(x_appr,**kwargs)
    print(f'Function value in optimal point:\t{f_opt}')
    print(f'Function value in approximated point:   {f_appr}')
    print(f'Is close verification:\t{np.isclose(f_opt,f_appr,atol=tolerance)}\n')
    print(f'Gradient approximation in optimal point is:\n{grad(x_optimal,**kwargs)}\n')
    grad_appr = grad(x_appr,**kwargs)
    print(f'Gradient approximation in approximated point is:\n{grad_appr}\n')
    print(f'Is close verification:\n{np.isclose(grad_appr,np.zeros(grad_appr.shape),atol=tolerance)}')


def plot_grad_norms(problem_grad_norms):
    """
    Plots the evolution of the gradient norms over the iterations for the minimized problems.

    :param problem_grad_norms: List of gradient-L2-norm-per-iteration lists over problems
    """
    for i, grad_norms in enumerate(problem_grad_norms):
        plt.plot(range(len(grad_norms)), grad_norms, label=f"Problem {i+1}")

    plt.title("Gradient evolution")
    plt.xlabel("Iterations")
    plt.ylabel("Gradient L2 norm")
    plt.legend()
    plt.show()