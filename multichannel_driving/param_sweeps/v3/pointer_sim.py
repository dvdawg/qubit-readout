import numpy as np
from scipy.optimize import minimize
from scipy import integrate

# dispersive shift calculation
def compute_chis(g_1, g_2, delta_r1, delta_r2):
    chi_1 = g_1**2 / delta_r1
    chi_2 = g_2**2 / delta_r2
    return chi_1, chi_2

# steady-state pointer amplitude calculation
def calculate_steady_state(sigma_z1, sigma_z2, params, chi_1, chi_2, delta_r1, delta_r2, kappa, g_1, g_2, delta_resonator):
    Omega_q1_mag, phi_q1, Omega_q2_mag, phi_q2 = params
    Omega_q1 = Omega_q1_mag * np.exp(1j * phi_q1)
    Omega_q2 = Omega_q2_mag * np.exp(1j * phi_q2)
    Omega_r = 0.0

    epsilon = (1j * Omega_r - (Omega_q1 * chi_1 * sigma_z1 / g_1 + Omega_q2 * chi_2 * sigma_z2 / g_2))
    delta_eff = chi_1 * sigma_z1 + chi_2 * sigma_z2

    return epsilon / (1j * kappa/2 + delta_eff + delta_resonator)

def alpha_traj(t, sigma_z1, sigma_z2, params, chi_1, chi_2, kappa, g_1, g_2, delta_resonator):
    Omega_q1_mag, phi_q1, Omega_q2_mag, phi_q2 = params
    Omega_q1 = Omega_q1_mag * np.exp(1j * phi_q1)
    Omega_q2 = Omega_q2_mag * np.exp(1j * phi_q2)
    Omega_r = 0.0
    
    epsilon = 1j * Omega_r - (Omega_q1 * chi_1 * sigma_z1 / g_1 + Omega_q2 * chi_2 * sigma_z2 / g_2)
    delta_eff = delta_resonator + chi_1 * sigma_z1 + chi_2 * sigma_z2
    decay = np.exp(-(1j * delta_eff + kappa / 2) * t)
    alpha_ss = epsilon / (1j * kappa / 2 + delta_eff)
    return alpha_ss + (0 - alpha_ss) * decay

# steady state snr
def calculate_snr(state1, state2, params, chi_1, chi_2, delta_r1, delta_r2, kappa, g_1, g_2, delta_resonator, tau):
    alpha_1 = calculate_steady_state(*state1, params, chi_1, chi_2, delta_r1, delta_r2, kappa, g_1, g_2, delta_resonator)
    alpha_2 = calculate_steady_state(*state2, params, chi_1, chi_2, delta_r1, delta_r2, kappa, g_1, g_2, delta_resonator)

    return np.abs(alpha_1 - alpha_2) * np.sqrt(2 * kappa * tau)

# time-integrated snr
def integrated_snr(state1, state2, params, chi_1, chi_2, delta_r1, delta_r2, kappa, g_1, g_2, delta_resonator, tau, eta):
    t = np.linspace(0, tau, 1000)  # GHz time

    alpha_1 = alpha_traj(t, *state1, params, chi_1, chi_2, kappa, g_1, g_2, delta_resonator)
    alpha_2 = alpha_traj(t, *state2, params, chi_1, chi_2, kappa, g_1, g_2, delta_resonator)
    
    W_t = alpha_1 - alpha_2 # np.ones_like(t)

    numerator = np.abs(np.trapz(W_t * (alpha_2 - alpha_1), t))**2
    denominator = 0.5 * np.trapz(np.abs(W_t)**2, t)

    return eta * kappa * numerator / denominator

# # ─── Time-integrated SNR over τ ────────────────────────────────────────────────
# def integrated_snr(state1, state2, params, chi_1, chi_2, delta_r1, delta_r2, kappa, g_1, g_2, delta_resonator):
#     """
#     SNR(τ) = 4 η κ τ · [separation/noise] · (1 – e^{-κτ/2})²
#     """
#     ss_snr = calculate_snr(state1, state2, params,
#                             chi_1, chi_2,
#                             delta_r1, delta_r2,
#                             kappa, g_1, g_2, delta_resonator)
#     ringup = (1 - np.exp(-kappa * tau / 2))**2
#     return 4 * eta * kappa * tau * ss_snr * ringup
def generate_states(energy_levels):
    states = []
    for s1 in range(energy_levels):
        for s2 in range(energy_levels):
            states.append((s1 if s1 != 0 else -1, s2 if s2 != 0 else -1))
    return states

def get_state_pairs(optimization_case, energy_levels):

    if optimization_case == 'all_states':
        states = generate_states(energy_levels)
        
        pairs = []
        for i, s1 in enumerate(states):
            for j, s2 in enumerate(states[i+1:], i+1):
                pairs.append((s1, s2))
        return pairs
    
    elif optimization_case == 'binary_states':
        binary_states = [
            (-1, -1),  # |00⟩
            (-1,  1),  # |01⟩
            ( 1, -1),  # |10⟩
            ( 1,  1)   # |11⟩
        ]
        pairs = []
        for i, s1 in enumerate(binary_states):
            for j, s2 in enumerate(binary_states[i+1:], i+1):
                pairs.append((s1, s2))
        return pairs
    
    elif optimization_case == 'adjacent_states':
        return [((-1, -1), (1, 1))]  # |01⟩ vs |10⟩
    
    else:
        raise ValueError(f"Unknown optimization case: {optimization_case}")

def objective_function(params, state_pairs, chi_1, chi_2, delta_r1, delta_r2, kappa, g_1, g_2, delta_resonator, tau):
    snrs = [
        calculate_snr(s1, s2, params, chi_1, chi_2, delta_r1, delta_r2, kappa, g_1, g_2, delta_resonator, tau)
        for s1, s2 in state_pairs
    ]
    return -min(snrs)

def optimize_parameters(delta_r1, delta_r2,
                       g_1, g_2,
                       kappa, delta_resonator, tau,
                       optimization_case='binary_states',
                       energy_levels=2):
    chi_1, chi_2 = compute_chis(g_1, g_2, delta_r1, delta_r2)

    state_pairs = get_state_pairs(optimization_case, energy_levels)

    initial = [2.0, np.pi, 2.0, np.pi/2]
    bounds  = [(0.001, 2.0),
               (0, 2*np.pi),
               (0.001, 2.0),
               (0, 2*np.pi)]

    res = minimize(
        objective_function,
        initial,
        args=(state_pairs,
              chi_1, chi_2,
              delta_r1, delta_r2,
              kappa, g_1, g_2, delta_resonator, tau),
        bounds=bounds
    )

    opt = res.x
    final_snrs = [
        calculate_snr(s1, s2, opt,
                     chi_1, chi_2,
                     delta_r1, delta_r2,
                     kappa, g_1, g_2, delta_resonator, tau)
        for s1, s2 in state_pairs
    ]

    return {
        'delta_r1': delta_r1,
        'delta_r2': delta_r2,
        'g_1'      : g_1,
        'g_2'      : g_2,
        'kappa'    : kappa,
        'delta_resonator': delta_resonator,
        'Omega_q1_mag': opt[0],
        'phi_q1'     : opt[1],
        'Omega_q2_mag': opt[2],
        'phi_q2'     : opt[3],
        'min_snr'    : np.min(final_snrs),
        'avg_snr'    : np.mean(final_snrs),
        'max_snr'    : np.max(final_snrs),
        'std_snr'    : np.std(final_snrs),
        'optimization_case': optimization_case,
        'energy_levels': energy_levels
    }
