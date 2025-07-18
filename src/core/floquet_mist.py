import numpy as np
import matplotlib.pyplot as plt
from qutip import *

def estimate_photon_number(Omega_q_mag, g, chi):
    epsilon_t = Omega_q_mag * g / chi
    return (epsilon_t / (2 * g))**2

def critical_photon_number(delta_r, g):
    return delta_r / (2 * g)**2

def check_mist_warning(params, chi, g, delta_r):
    Omega_q_mag, _ = params
    n_r = estimate_photon_number(Omega_q_mag, g, chi)
    n_crit = critical_photon_number(delta_r, g)
    return n_r >= n_crit, n_r, n_crit

def estimate_photon_number_from_params(params, chi, g):
    Omega_q_mag, _ = params
    epsilon_t = Omega_q_mag * g / chi
    return (epsilon_t / (2 * g))**2
