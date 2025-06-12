import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

energy_levels = 3
t = np.linspace(0, 1000, 1000)

# setup params (fixed)
kappa = 0.01
g_1 = 0.08
g_2 = 0.08
delta_resonator = -0.1
delta_r1 = 0.8
delta_r2 = 0.8

# drive params (for manual plotting)
Omega_q1_mag = 3.0
phi_q1 = np.pi
Omega_q2_mag = 2.0
phi_q2 = 1 * np.pi / 2

optimize = True
optimize_type = 'min'  # 'min', 'avg', 'spacing'
initial_params = [0.1, 0, 0.1, np.pi/2]  
bounds = [
    (0.001, 2.0),    # Omega_q1_mag
    (0, 2*np.pi),   # phi_q1
    (0.001, 2.0),    # Omega_q2_mag 
    (0, 2*np.pi)    # phi_q2
]

chi_1 = g_1**2 / delta_r1
chi_2 = g_2**2 / delta_r2

def calculate_steady_state(sigma_z1, sigma_z2, params):
    Omega_q1_mag, phi_q1, Omega_q2_mag, phi_q2 = params
    Omega_q1 = Omega_q1_mag * np.exp(1j * phi_q1)
    Omega_q2 = Omega_q2_mag * np.exp(1j * phi_q2)
    Omega_r = 0.0
    
    epsilon = 1j * Omega_r - (Omega_q1 * chi_1 * sigma_z1 / g_1 + Omega_q2 * chi_2 * sigma_z2 / g_2)
    delta_eff = delta_resonator + chi_1 * sigma_z1 + chi_2 * sigma_z2
    return epsilon / (1j * kappa / 2 + delta_eff)

def calculate_snr(state1, state2, params, n_avg=1):
    alpha1 = calculate_steady_state(*state1, params)
    alpha2 = calculate_steady_state(*state2, params)
    distance_squared = np.abs(alpha1 - alpha2)**2
    return n_avg * distance_squared

def calculate_all_snrs(params):
    snrs = []
    for i, state1 in enumerate(states):
        for j, state2 in enumerate(states):
            if i != j:
                snrs.append(calculate_snr(state1, state2, params))
    return np.array(snrs)

def calculate_spacing_metric(params):
    steady_states = []
    for state in states:
        alpha = calculate_steady_state(*state, params)
        steady_states.append(alpha)
    steady_states = np.array(steady_states)
    
    distances = []
    for i in range(len(steady_states)):
        for j in range(i + 1, len(steady_states)):
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

def objective_function(params): 
    if optimize_type == 'spacing':
        return calculate_spacing_metric(params)
    else:
        snrs = calculate_all_snrs(params)
        if optimize_type == 'min':
            return -np.min(snrs)
        else:  # 'avg'
            return -np.mean(snrs)

states = []
for s1 in range(energy_levels):
    for s2 in range(energy_levels):
        sigma_z1 = 2 * s1 - 1
        sigma_z2 = 2 * s2 - 1
        states.append((sigma_z1, sigma_z2))

if optimize:
    result = minimize(
        objective_function,
        initial_params,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 10000}
    )
    params = result.x
    print("\nOptimized Parameters:")
    print(f"Omega_q1_mag = {params[0]:.4f}")
    print(f"phi_q1 = {params[1]:.4f}")
    print(f"Omega_q2_mag = {params[2]:.4f}")
    print(f"phi_q2 = {params[3]:.4f}")
    
    snrs = calculate_all_snrs(params)
    print(f"\nSNR Statistics:")
    print(f"Minimum SNR = {np.min(snrs):.4f}")
    print(f"Average SNR = {np.mean(snrs):.4f}")
    print(f"Maximum SNR = {np.max(snrs):.4f}")
    print(f"SNR Standard Deviation = {np.std(snrs):.4f}")
    
    if optimize_type == 'spacing':
        steady_states = []
        for state in states:
            alpha = calculate_steady_state(*state, params)
            steady_states.append(alpha)
        steady_states = np.array(steady_states)
        
        distances = []
        for i in range(len(steady_states)):
            for j in range(i+1, len(steady_states)):
                dist = np.abs(steady_states[i] - steady_states[j])
                distances.append(dist)
        distances = np.array(distances)
        
        print(f"\nSpacing Statistics:")
        print(f"Minimum Distance = {np.min(distances):.4f}")
        print(f"Maximum Distance = {np.max(distances):.4f}")
        print(f"Average Distance = {np.mean(distances):.4f}")
        print(f"Distance Standard Deviation = {np.std(distances):.4f}")
        print(f"Uniformity Metric = {np.std(distances)/np.mean(distances):.4f}")
else:
    params = [Omega_q1_mag, phi_q1, Omega_q2_mag, phi_q2]
    print("\nUsing manual parameters:")
    print(f"Omega_q1_mag = {params[0]:.4f}")
    print(f"phi_q1 = {params[1]:.4f}")
    print(f"Omega_q2_mag = {params[2]:.4f}")
    print(f"phi_q2 = {params[3]:.4f}")

    snrs = calculate_all_snrs(params)
    print(f"\nSNR Statistics:")
    print(f"Minimum SNR = {np.min(snrs):.4f}")
    print(f"Average SNR = {np.mean(snrs):.4f}")
    print(f"Maximum SNR = {np.max(snrs):.4f}")
    print(f"SNR Standard Deviation = {np.std(snrs):.4f}")

# SNR matrix
n_states = len(states)
snr_matrix = np.zeros((n_states, n_states))
for i, state1 in enumerate(states):
    for j, state2 in enumerate(states):
        if i != j:
            snr_matrix[i, j] = calculate_snr(state1, state2, params)

print("\nSNR Matrix:")
header = "         " + "    ".join([f"|{s1}{s2}⟩" for s1, s2 in [(s1, s2) for s1 in range(energy_levels) for s2 in range(energy_levels)]])
print(header)
print("-" * (len(header) + 10))
for i, state1 in enumerate(states):
    s1 = (state1[0] + 1) // 2
    s2 = (state1[1] + 1) // 2
    row = f"|{s1}{s2}⟩ |"
    for j, state2 in enumerate(states):
        if i == j:
            row += "   -    "
        else:
            row += f" {snr_matrix[i,j]:6.2f} "
    print(row)

def alpha_traj(t, sigma_z1, sigma_z2, params):
    Omega_q1_mag, phi_q1, Omega_q2_mag, phi_q2 = params
    Omega_q1 = Omega_q1_mag * np.exp(1j * phi_q1)
    Omega_q2 = Omega_q2_mag * np.exp(1j * phi_q2)
    Omega_r = 0.0
    
    epsilon = 1j * Omega_r - (Omega_q1 * chi_1 * sigma_z1 / g_1 + Omega_q2 * chi_2 * sigma_z2 / g_2)
    delta_eff = delta_resonator + chi_1 * sigma_z1 + chi_2 * sigma_z2
    decay = np.exp(-(1j * delta_eff + kappa / 2) * t)
    alpha_ss = epsilon / (1j * kappa / 2 + delta_eff)
    return alpha_ss + (0 - alpha_ss) * decay

labels = []
colors = {}
color_list = ['blue', 'red', 'darkred', 'green', 'purple', 'magenta', 'orange', 'brown', 'black']
for i, (s1, s2) in enumerate(states):
    state = f"{s1}{s2}"
    labels.append(r"$|%d%d\rangle$" % ((s1 + 1) // 2, (s2 + 1) // 2))
    colors[state] = color_list[i % len(color_list)]

plt.figure(figsize=(10, 8))
for (s1, s2), label in zip(states, labels):
    alpha = alpha_traj(t, s1, s2, params)
    state = f"{s1}{s2}"
    plt.plot(alpha.real, alpha.imag, label=label, color=colors[state], linewidth=1.5)
    alpha_ss = calculate_steady_state(s1, s2, params)
    plt.plot(alpha_ss.real, alpha_ss.imag, 'o', color=colors[state], markersize=4)

plt.xlabel("Real")
plt.ylabel("Imag")
plt.title(f"Pointer State Trajectories (n={energy_levels})")
plt.grid(True)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
plt.axis("equal")
plt.tight_layout()
plt.show()


# n_states = len(states)
# snr_matrix = np.zeros((n_states, n_states))
# for i, state1 in enumerate(states):
#     for j, state2 in enumerate(states):
#         if i != j:
#             snr_matrix[i, j] = calculate_snr(state1, state2, result.x)
# print("\nSNR Matrix:")
# header = "           " + " ".join([f"|{(3-s1)//2}{(3-s2)//2}⟩" for s1, s2 in states])
# print(header)
# print("-" * (len(header) + 10))
# for i, state1 in enumerate(states):
#     row = f"|{(3-state1[0])//2}{(3-state1[1])//2}⟩ |"
#     for j, state2 in enumerate(states):
#         if i == j:
#             row += "   -    "
#         else:
#             row += f" {snr_matrix[i,j]:6.2f} "
#     print(row)


# phase response graphs

# omega_probe = np.linspace(-30, 30, 300)
# epsilon = 0.5 * kappa * np.sqrt(2.0)
# alpha_phases = {}

# states = {'00': (-1, -1), '01': (-1, +1), '10': (+1, -1), '11': (+1, +1)}

# for label, (sz1_val, sz2_val) in states.items():
#     delta_eff = omega_probe - (chi_1 * sz1_val + chi_2 * sz2_val)
#     alpha = epsilon / (kappa / 2 + 1j * delta_eff)
#     alpha_phases[label] = np.angle(alpha)

# plt.figure(figsize=(8, 6))
# for label, phase in alpha_phases.items():
#     plt.plot(omega_probe, phase, label=f"|{label}⟩")
# plt.title("Phase Response vs Probe Frequency")
# plt.xlabel("ω - ω_r (MHz)")
# plt.ylabel("Arg(α) [rad]")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
