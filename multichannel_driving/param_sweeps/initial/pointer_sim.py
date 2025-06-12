import numpy as np
import pandas as pd
from scipy.optimize import minimize

energy_levels = 3
t = np.linspace(0, 10, 1000)

# Fixed drive parameters
Omega_q1_mag = 3.0
phi_q1 = np.pi
Omega_q2_mag = 2.0
phi_q2 = np.pi / 2
params = [Omega_q1_mag, phi_q1, Omega_q2_mag, phi_q2]


# Optimization parameters
optimize_type = 'min'  # 'min', 'avg', 'spacing'
initial_params = [0.1, 0, 0.1, np.pi/2]  
bounds = [
    (0.001, 2.0),    # Omega_q1_mag
    (0, 2*np.pi),   # phi_q1
    (0.001, 2.0),    # Omega_q2_mag
    (0, 2*np.pi)    # phi_q2
]

states = []
for s1 in range(energy_levels):
    for s2 in range(energy_levels):
        sigma_z1 = 2 * s1 - 1
        sigma_z2 = 2 * s2 - 1
        states.append((sigma_z1, sigma_z2))

def compute_chis(g_1, g_2, delta_r1, delta_r2):
    chi_1 = g_1**2 / delta_r1
    chi_2 = g_2**2 / delta_r2
    return chi_1, chi_2

def calculate_steady_state(sigma_z1, sigma_z2, params, chi_1, chi_2, delta_r1, delta_r2, kappa, g_1, g_2, delta_resonator):
    Omega_q1_mag, phi_q1, Omega_q2_mag, phi_q2 = params
    Omega_q1 = Omega_q1_mag * np.exp(1j * phi_q1)
    Omega_q2 = Omega_q2_mag * np.exp(1j * phi_q2)
    Omega_r = 0.0

    epsilon = 1j * Omega_r - (Omega_q1 * chi_1 * sigma_z1 / g_1 + Omega_q2 * chi_2 * sigma_z2 / g_2)
    delta_eff = (chi_1 * sigma_z1) + (chi_2 * sigma_z2)
    return epsilon / (1j * kappa / 2 + delta_eff + delta_resonator)

def calculate_snr(state1, state2, params, chi_1, chi_2, delta_r1, delta_r2, kappa, g_1, g_2, delta_resonator):
    alpha1 = calculate_steady_state(*state1, params, chi_1, chi_2, delta_r1, delta_r2, kappa, g_1, g_2, delta_resonator)
    alpha2 = calculate_steady_state(*state2, params, chi_1, chi_2, delta_r1, delta_r2, kappa, g_1, g_2, delta_resonator)
    return np.abs(alpha1 - alpha2)**2

def calculate_all_snrs(params, chi_1, chi_2, delta_r1, delta_r2, kappa, g_1, g_2, delta_resonator):
    snrs = []
    for i, state1 in enumerate(states):
        for j, state2 in enumerate(states):
            if i != j:
                snrs.append(calculate_snr(state1, state2, params, chi_1, chi_2, delta_r1, delta_r2, kappa, g_1, g_2, delta_resonator))
    return np.array(snrs)

def calculate_spacing_metric(params, chi_1, chi_2, delta_r1, delta_r2, kappa, g_1, g_2, delta_resonator):
    steady_states = []
    for state in states:
        alpha = calculate_steady_state(*state, params, chi_1, chi_2, delta_r1, delta_r2, kappa, g_1, g_2, delta_resonator)
        steady_states.append(alpha)
    steady_states = np.array(steady_states)
    
    distances = []
    for i in range(len(steady_states)):
        for j in range(i+1, len(steady_states)):
            dist = np.abs(steady_states[i] - steady_states[j])
            distances.append(dist)
    distances = np.array(distances)
    
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    
    if min_dist < 0.1:
        return float('inf')
    
    uniformity = std_dist / mean_dist
    return -(min_dist * (1 - uniformity))

def objective_function(params, chi_1, chi_2, delta_r1, delta_r2, kappa, g_1, g_2, delta_resonator):
    if optimize_type == 'spacing':
        return calculate_spacing_metric(params, chi_1, chi_2, delta_r1, delta_r2, kappa, g_1, g_2, delta_resonator)
    else:
        snrs = calculate_all_snrs(params, chi_1, chi_2, delta_r1, delta_r2, kappa, g_1, g_2, delta_resonator)
        if optimize_type == 'min':
            return -np.min(snrs)
        else:  # 'avg'
            return -np.mean(snrs)

def optimize_parameters(chi_1, chi_2, delta_r1, delta_r2, kappa, g_1, g_2, delta_resonator):
    result = minimize(
        lambda params: objective_function(params, chi_1, chi_2, delta_r1, delta_r2, kappa, g_1, g_2, delta_resonator),
        initial_params,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 50000}
    )
    
    optimized_params = result.x
    snrs = calculate_all_snrs(optimized_params, chi_1, chi_2, delta_r1, delta_r2, kappa, g_1, g_2, delta_resonator)
    
    return {
        'delta_r1': delta_r1,
        'delta_r2': delta_r2,
        'g_1': g_1,
        'g_2': g_2, 
        'kappa': kappa,
        'delta_resonator': delta_resonator,
        'Omega_q1_mag': optimized_params[0],
        'phi_q1': optimized_params[1],
        'Omega_q2_mag': optimized_params[2],
        'phi_q2': optimized_params[3],
        'min_snr': np.min(snrs),
        'avg_snr': np.mean(snrs),
        'max_snr': np.max(snrs),
        'std_snr': np.std(snrs)
    }
