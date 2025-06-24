import numpy as np
from scipy.optimize import root_scalar

def solve_EC(omega_q, Delta, g, chi_target):

    def chi_expr(EC):
        return (g**2 * EC) / (Delta * (Delta + EC)) - chi_target

    result = root_scalar(chi_expr, bracket=[0.01e9*2*np.pi, 0.5e9*2*np.pi], method='brentq')
    return result.root if result.converged else None

def get_EJ(omega_q, EC):

    return ((omega_q + EC)**2) / (8 * EC)
