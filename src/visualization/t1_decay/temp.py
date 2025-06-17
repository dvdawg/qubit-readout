import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy import integrate

# Dispersive shift calculation
def compute_chi(g, delta_r):
    return g**2 / delta_r

# Steady-state pointer amplitude
def calculate_steady_state(sigma_z, params, chi, delta_r, kappa, g, delta_resonator):
    Omega_q_mag, phi_q = params
    Omega_q = Omega_q_mag * np.exp(1j * phi_q)
    Omega_r = 0.0
    epsilon = (1j * Omega_r - (Omega_q * chi * sigma_z / g))
    delta_eff = chi * sigma_z
    return epsilon / (1j * kappa / 2 + delta_eff + delta_resonator)

# Time-dependent pointer trajectory
def alpha_traj(t, sigma_z, params, chi, kappa, g, delta_resonator):
    Omega_q_mag, phi_q = params
    Omega_q = Omega_q_mag * np.exp(1j * phi_q)
    Omega_r = 0.0
    epsilon = 1j * Omega_r - (Omega_q * chi * sigma_z / g)
    delta_eff = delta_resonator + chi * sigma_z
    decay = np.exp(-(1j * delta_eff + kappa / 2) * t)
    alpha_ss = epsilon / (1j * kappa / 2 + delta_eff)
    return alpha_ss + (0 - alpha_ss) * decay

# Simulate T1 decay
def alpha_traj_with_t1(t, params, chi, kappa, g, delta_resonator, T1, seed=None):
    rng = np.random.default_rng(seed)
    t_jump = -T1 * np.log(rng.uniform())

    t_jump = t_jump if t_jump < t[-1] else np.inf  # no jump if beyond time window

    alpha = np.zeros_like(t, dtype=complex)
    for i, ti in enumerate(t):
        if ti < t_jump:
            alpha[i] = alpha_traj(np.array([ti]), +1, params, chi, kappa, g, delta_resonator)[0]
        else:
            t_rel = ti - t_jump
            alpha_ss = calculate_steady_state(-1, params, chi, delta_r, kappa, g, delta_resonator)
            delta_eff = delta_resonator - chi
            decay = np.exp(-(1j * delta_eff + kappa / 2) * t_rel)
            alpha[i] = alpha_ss + (alpha[i - 1] - alpha_ss) * decay
    return alpha, t_jump

# Visualization
t = np.linspace(0, 1000, 1000)  # ns
tau = 200
delta_resonator = -0.1
g = 0.08
kappa = 0.01
delta_r = 0.8
T1 = 30000  # ns = 30 µs
chi = compute_chi(g, delta_r)
params = [2.0, np.pi]

fig, ax = plt.subplots(figsize=(10, 8))

# |0⟩ trajectory
alpha_0 = alpha_traj(t, -1, params, chi, kappa, g, delta_resonator)
alpha_ss_0 = calculate_steady_state(-1, params, chi, delta_r, kappa, g, delta_resonator)
ax.plot(alpha_0.real, alpha_0.imag, label=r"$|0⟩$", lw=2)
ax.plot(alpha_ss_0.real, alpha_ss_0.imag, 'o', label=r"SS $|0⟩$")

# |1⟩ with T1 decay
alpha_1_t1, t_jump = alpha_traj_with_t1(t, params, chi, kappa, g, delta_resonator, T1, seed=42)
alpha_ss_1 = calculate_steady_state(+1, params, chi, delta_r, kappa, g, delta_resonator)
ax.plot(alpha_1_t1.real, alpha_1_t1.imag, label=r"$|1⟩$ w/ T₁ decay", lw=2)
ax.plot(alpha_ss_1.real, alpha_ss_1.imag, 'o', label=r"SS $|1⟩$")

# Add T1 jump marker if within window
if t_jump < t[-1]:
    idx_jump = np.searchsorted(t, t_jump)
    ax.axvline(x=alpha_1_t1.real[idx_jump], color='gray', linestyle='--', label=f"T₁ jump @ {t_jump:.1f} ns")
else:
    ax.plot([], [], ' ', label="No T₁ jump (t > end)")

ax.set_xlabel("Re(α)")
ax.set_ylabel("Im(α)")
ax.set_title("Pointer State Trajectories with T₁ Decay")
ax.grid(True)
ax.legend()
ax.axis("equal")
plt.show()
