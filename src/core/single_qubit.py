import numpy as np
from scipy.optimize import minimize
from scipy import integrate

# dispersive shift calculation
def compute_chi(g, delta_r):
    return g**2 / delta_r

# steady-state pointer amplitude calculation
def calculate_steady_state(sigma_z, params, chi, delta_r, kappa, g, delta_resonator):
    Omega_q_mag, phi_q = params
    Omega_q = Omega_q_mag * np.exp(1j * phi_q)
    Omega_r = 0.0

    epsilon = (1j * Omega_r - (Omega_q * chi * sigma_z / g))
    delta_eff = chi * sigma_z

    return epsilon / (1j * kappa/2 + delta_eff + delta_resonator)

def alpha_traj(t, sigma_z, params, chi, kappa, g, delta_resonator):
    Omega_q_mag, phi_q = params
    Omega_q = Omega_q_mag * np.exp(1j * phi_q)
    Omega_r = 0.0
    
    epsilon = 1j * Omega_r - (Omega_q * chi * sigma_z / g)
    delta_eff = delta_resonator + chi * sigma_z
    decay = np.exp(-(1j * delta_eff + kappa / 2) * t)
    alpha_ss = epsilon / (1j * kappa / 2 + delta_eff)
    return alpha_ss + (0 - alpha_ss) * decay

# steady state snr
def calculate_snr(state1, state2, params, chi, delta_r, kappa, g, delta_resonator, tau):
    alpha_1 = calculate_steady_state(state1, params, chi, delta_r, kappa, g, delta_resonator)
    alpha_2 = calculate_steady_state(state2, params, chi, delta_r, kappa, g, delta_resonator)
    return np.abs(alpha_1 - alpha_2) * np.sqrt(2 * kappa * tau)

# time-integrated snr
def integrated_snr(state1, state2, params, chi, delta_r, kappa, g, delta_resonator, tau, eta):
    t = np.linspace(0, tau, 1000)  # GHz time
    alpha_1 = alpha_traj(t, state1, params, chi, kappa, g, delta_resonator)
    alpha_2 = alpha_traj(t, state2, params, chi, kappa, g, delta_resonator)
    W_t = alpha_1 - alpha_2
    numerator = np.abs(np.trapz(W_t * (alpha_2 - alpha_1), t))**2
    denominator = 0.5 * np.trapz(np.abs(W_t)**2, t)
    return eta * kappa * numerator / denominator

def generate_states(energy_levels):
    states = []
    for s in range(energy_levels):
        states.append(s if s != 0 else -1)
    return states

def get_state_pairs(optimization_case, energy_levels):
    if optimization_case == 'all_states':
        states = generate_states(energy_levels)
        pairs = []
        for i, s1 in enumerate(states):
            for s2 in states[i+1:]:
                pairs.append((s1, s2))
        return pairs
    elif optimization_case == 'binary_states':
        binary_states = [
            -1,  # |0⟩
             1   # |1⟩
        ]
        return [(binary_states[0], binary_states[1])]
    elif optimization_case == 'adjacent_states':
        return [(-1, 1)]  # |0⟩ vs |1⟩
    else:
        raise ValueError(f"Unknown optimization case: {optimization_case}")

def objective_function(params, state_pairs, chi, delta_r, kappa, g, delta_resonator, tau):
    snrs = [
        calculate_snr(s1, s2, params, chi, delta_r, kappa, g, delta_resonator, tau)
        for s1, s2 in state_pairs
    ]
    return -min(snrs)

def optimize_parameters(delta_r, g, kappa, delta_resonator, tau,
                       optimization_case='binary_states',
                       energy_levels=2):
    chi = compute_chi(g, delta_r)
    state_pairs = get_state_pairs(optimization_case, energy_levels)
    initial = [2.0, np.pi]
    bounds  = [(0.001, 2.0), (0, 2*np.pi)]
    res = minimize(
        objective_function,
        initial,
        args=(state_pairs, chi, delta_r, kappa, g, delta_resonator, tau),
        bounds=bounds
    )
    opt = res.x
    final_snrs = [
        calculate_snr(s1, s2, opt, chi, delta_r, kappa, g, delta_resonator, tau)
        for s1, s2 in state_pairs
    ]
    return {
        'delta_r': delta_r,
        'g'      : g,
        'kappa'    : kappa,
        'delta_resonator': delta_resonator,
        'Omega_q_mag': opt[0],
        'phi_q'     : opt[1],
        'min_snr'    : np.min(final_snrs),
        'avg_snr'    : np.mean(final_snrs),
        'max_snr'    : np.max(final_snrs),
        'std_snr'    : np.std(final_snrs),
        'optimization_case': optimization_case,
        'energy_levels': energy_levels
    }

# def integrated_snr(state1, state2, params, chi, delta_r, kappa, g, delta_resonator, tau, eta):

#     t = np.linspace(0, tau, 1000)
#     alpha1 = alpha_traj(t, state1, params, chi, kappa, g, delta_resonator)
#     alpha2 = alpha_traj(t, state2, params, chi, kappa, g, delta_resonator)
#     W = alpha1 - alpha2

#     signal  = np.trapz(np.abs(W)**2, t)
#     numerator   = signal**2
#     denominator = 0.5 * signal # 0.5 because of the 2-photon state

#     return eta * kappa * numerator / denominator


# t1  = around 50-300 microseconds