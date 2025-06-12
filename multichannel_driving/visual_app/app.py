import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from pathlib import Path
import sys

pointer_sim_path = Path(__file__).resolve().parents[1] / "param_sweeps" / "v3"
sys.path.append(str(pointer_sim_path))
from pointer_sim import (
    compute_chis,
    calculate_steady_state,
    alpha_traj,
    calculate_snr,
    integrated_snr,
    get_state_pairs,
    generate_states
)

st.set_page_config(layout="wide", page_title="Qubit Pointer State Visualization")

tau = 200
eta = 0.4
energy_levels = 3
t = np.linspace(0, 1000, 1000)

# --- Sidebar Inputs ---
st.sidebar.title("Simulation Parameters")

energy_levels = st.sidebar.number_input("Energy Levels", value=3, min_value=2, max_value=4, step=1)
manual = not st.sidebar.toggle("Optimize Parameters", value=True)
optimize_type = st.sidebar.selectbox("Optimization Metric", ['min', 'avg', 'spacing'])
optimization_case = st.sidebar.selectbox("Optimization Case", ['all', 'only two level', '10/01'])

st.sidebar.markdown("### System Parameters")
kappa = st.sidebar.number_input("κ", value=0.01, help="Decay rate")
g_1 = st.sidebar.number_input("g₁", value=0.08, help="Coupling strength 1")
g_2 = st.sidebar.number_input("g₂", value=0.08, help="Coupling strength 2")
delta_resonator = st.sidebar.number_input("δᵣ", value=-0.1, help="Resonator detuning")
delta_r1 = st.sidebar.number_input("Δ₁", value=0.8, help="Qubit 1 detuning")
delta_r2 = st.sidebar.number_input("Δ₂", value=0.8, help="Qubit 2 detuning")

chi_1, chi_2 = compute_chis(g_1, g_2, delta_r1, delta_r2)

if manual:
    st.sidebar.markdown("### Drive Parameters")
    Omega_q1_mag = st.sidebar.number_input("|Ω₁|", value=2.0, help="Drive amplitude 1")
    phi_q1 = st.sidebar.number_input("φ₁ (radians)", value=np.pi, help="Drive phase 1")
    Omega_q2_mag = st.sidebar.number_input("|Ω₂|", value=2.0, help="Drive amplitude 2")
    phi_q2 = st.sidebar.number_input("φ₂ (radians)", value=np.pi / 2, help="Drive phase 2")
    params = [Omega_q1_mag, phi_q1, Omega_q2_mag, phi_q2]
else:
    initial_params = [0.1, 0, 0.1, np.pi/2]
    bounds = [(0.001, 2.0), (0, 2*np.pi), (0.001, 2.0), (0, 2*np.pi)]

use_integrated_snr = st.sidebar.toggle("Use Integrated SNR", value=False)

def calculate_snr_wrapper(state1, state2, params):
    if use_integrated_snr:
        return integrated_snr(state1, state2, params, chi_1, chi_2, delta_r1, delta_r2, kappa, g_1, g_2, delta_resonator, tau, eta)
    return calculate_snr(state1, state2, params, chi_1, chi_2, delta_r1, delta_r2, kappa, g_1, g_2, delta_resonator, tau)

def calculate_all_snrs(params):
    return np.array([calculate_snr_wrapper(s1, s2, params) for i, s1 in enumerate(states) for j, s2 in enumerate(states) if i != j])

def calculate_spacing_metric(params):
    alphas = [calculate_steady_state(*state, params, chi_1, chi_2, delta_r1, delta_r2, kappa, g_1, g_2, delta_resonator) for state in states]
    distances = [np.abs(a1 - a2) for i, a1 in enumerate(alphas) for j, a2 in enumerate(alphas) if i < j]
    distances = np.array(distances)
    if distances.min() < 0.1:
        return float('inf')
    return -(distances.min() * (1 - np.std(distances) / np.mean(distances)))

def objective_function(params):
    if optimize_type == 'spacing':
        return calculate_spacing_metric(params)
    snrs = calculate_all_snrs(params)
    return -np.min(snrs) if optimize_type == 'min' else -np.mean(snrs)


# --- Run Optimization or Manual ---
if not manual:
    optimization_case = 'all_states' if optimization_case == 'all' else 'binary_states' if optimization_case == 'only two level' else 'adjacent_states' if optimization_case == '10/01' else ''

    state_pairs = get_state_pairs(optimization_case, energy_levels)
    states = generate_states(energy_levels)
    
    def optimization_objective(params):
        if optimize_type == 'spacing':
            return calculate_spacing_metric(params)
        snrs = [calculate_snr_wrapper(s1, s2, params) for s1, s2 in state_pairs]
        return -np.min(snrs) if optimize_type == 'min' else -np.mean(snrs)
    
    result = minimize(optimization_objective, initial_params, method='L-BFGS-B', bounds=bounds)
    params = result.x
    st.sidebar.write("### Optimized Parameters")
    st.sidebar.write(f"Omega_q1_mag = {params[0]:.4f}")
    st.sidebar.write(f"phi_q1 = {params[1]:.4f}")
    st.sidebar.write(f"Omega_q2_mag = {params[2]:.4f}")
    st.sidebar.write(f"phi_q2 = {params[3]:.4f}")
else:
    states = generate_states(energy_levels)

# Calculate SNRs for all state pairs for visualization
snrs = calculate_all_snrs(params)
st.sidebar.markdown("### SNR Statistics")
st.sidebar.write(f"Minimum: {np.min(snrs):.3f}")
st.sidebar.write(f"Average: {np.mean(snrs):.3f}")
st.sidebar.write(f"Maximum: {np.max(snrs):.3f}")
st.sidebar.write(f"Std Dev: {np.std(snrs):.3f}")

# --- Plot Pointer State Trajectories ---
st.title("Pointer State Trajectories")
fig, ax = plt.subplots(figsize=(10, 8))
colors = plt.cm.get_cmap('tab10', len(states))

for i, (s1, s2) in enumerate(states):
    alpha = alpha_traj(t, s1, s2, params, chi_1, chi_2, kappa, g_1, g_2, delta_resonator)
    alpha_ss = calculate_steady_state(s1, s2, params, chi_1, chi_2, delta_r1, delta_r2, kappa, g_1, g_2, delta_resonator)
    ax.plot(alpha.real, alpha.imag, label=f"|{s1 if s1 != -1 else 0}{s2 if s2 != -1 else 0}⟩", color=colors(i))
    ax.plot(alpha_ss.real, alpha_ss.imag, 'o', color=colors(i))

ax.set_xlabel("Re(α)")
ax.set_ylabel("Im(α)")
ax.set_title("Pointer State Trajectories")
ax.grid(True)
ax.legend(loc='upper right')
ax.axis("equal")
st.pyplot(fig, use_container_width=True)
