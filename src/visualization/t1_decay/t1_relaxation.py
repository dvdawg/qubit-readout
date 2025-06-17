import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy import integrate

def compute_chi(g, delta_r):
    return g**2 / delta_r

def calculate_steady_state(sigma_z, params, chi, delta_r, kappa, g, delta_resonator):
    Omega_q_mag, phi_q = params
    Omega_q = Omega_q_mag * np.exp(1j * phi_q)
    Omega_r = 0.0
    epsilon = (1j * Omega_r - (Omega_q * chi * sigma_z / g))
    delta_eff = chi * sigma_z
    return epsilon / (1j * kappa / 2 + delta_eff + delta_resonator)

def alpha_traj(t, sigma_z, params, chi, kappa, g, delta_resonator):
    Omega_q_mag, phi_q = params
    Omega_q = Omega_q_mag * np.exp(1j * phi_q)
    Omega_r = 0.0
    epsilon = 1j * Omega_r - (Omega_q * chi * sigma_z / g)
    delta_eff = delta_resonator + chi * sigma_z
    decay = np.exp(-(1j * delta_eff + kappa / 2) * t)
    alpha_ss = epsilon / (1j * kappa / 2 + delta_eff)
    return alpha_ss + (0 - alpha_ss) * decay

def alpha_traj_with_t1(t, params, chi, kappa, g, delta_resonator, T1, seed=None):
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    t_jump = -T1 * np.log(rng.uniform())
    t_jump = t_jump if t_jump < t[-1] else np.inf 

    alpha = np.zeros_like(t, dtype=complex)

    if t_jump == np.inf:
        alpha = alpha_traj(t, +1, params, chi, kappa, g, delta_resonator)
    else:
        idx_jump = np.searchsorted(t, t_jump)
        t_pre = t[:idx_jump]
        t_post = t[idx_jump:]
        
        alpha[:idx_jump] = alpha_traj(t_pre, +1, params, chi, kappa, g, delta_resonator)

        alpha_jump = alpha_traj(np.array([t_jump]), +1, params, chi, kappa, g, delta_resonator)[0]
        alpha_ss = calculate_steady_state(-1, params, chi, delta_r, kappa, g, delta_resonator)
        delta_eff = delta_resonator - chi
        decay = np.exp(-(1j * delta_eff + kappa / 2) * (t_post - t_jump))
        alpha[idx_jump:] = alpha_ss + (alpha_jump - alpha_ss) * decay

    return alpha, t_jump

t = np.linspace(0, 10000, 30000)  # ns
tau = 200
delta_resonator = -0.08
g = 0.08
kappa = 0.05
delta_r = 0.85
T1 = 30000  # ns
chi = compute_chi(g, delta_r)
params = [2.0, 0]

fig, ax = plt.subplots(figsize=(10, 8))

alpha_0 = alpha_traj(t, -1, params, chi, kappa, g, delta_resonator)
line0, = ax.plot(alpha_0.real, alpha_0.imag, label=r"$|0⟩$", lw=2)
ax.plot(alpha_0.real[-1], alpha_0.imag[-1], 'o', color=line0.get_color(), label=r"steady state $|0⟩$")

alpha_1_t1, t_jump = alpha_traj_with_t1(t, params, chi, kappa, g, delta_resonator, T1)
line1, = ax.plot(alpha_1_t1.real, alpha_1_t1.imag, label=r"$|1⟩$", lw=2)
ax.plot(alpha_1_t1.real[-1], alpha_1_t1.imag[-1], 'o', color=line1.get_color(), label=r"steady state $|1⟩$")

if t_jump < t[-1]:
    idx_jump = np.searchsorted(t, t_jump)
    ax.plot(alpha_1_t1.real[idx_jump], alpha_1_t1.imag[idx_jump], 'o', color='red', markersize=6, label=f"T1 jump: {t_jump:.1f} ns")
else:
    ax.plot([], [], ' ', label="No T1")

ax.set_xlabel("Re(α)")
ax.set_ylabel("Im(α)")
ax.set_title("Pointer State Trajectories")
ax.grid(True)
ax.legend()
ax.axis("equal")
plt.show()
