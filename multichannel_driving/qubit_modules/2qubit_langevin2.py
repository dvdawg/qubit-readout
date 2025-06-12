import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 10, 1000)

# params
kappa = 5.0

g_1 = 0.2
g_2 = 0.1

delta_r1 = 1.4
delta_r2 = 1.2

chi_1 = g_1**2 / delta_r1
chi_2 = g_2**2 / delta_r2

phi_q1 = 0
phi_q2 = 0 * np.pi / 3

Omega_q1_mag = 0.1
Omega_q1 = Omega_q1_mag * np.exp(1j * phi_q1)

ratio = chi_1 * g_2 / (chi_2 * g_1)
Omega_q2_mag = Omega_q1_mag * ratio
Omega_q2 = Omega_q2_mag * np.exp(1j * phi_q2)

Omega_r1 = 0.0
Omega_r2 = 0.0


def epsilon_eff(sigma_z1, sigma_z2):
    return 1j * (Omega_r1 + Omega_r2) - (
        Omega_q1 * chi_1 * sigma_z1 / g_1 + Omega_q2 * chi_2 * sigma_z2 / g_2)

def alpha_traj(t, sigma_z1, sigma_z2, alpha0=0):
    chi_total = chi_1 * sigma_z1 + chi_2 * sigma_z2
    decay = np.exp(-(1j * (delta_r1 + delta_r2 + chi_total) + kappa / 2) * t)
    alpha_ss = epsilon_eff(sigma_z1, sigma_z2) / (-1j * kappa / 2 + (delta_r1 + delta_r2 + chi_total))
    return alpha_ss + (alpha0 - alpha_ss) * decay

states = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
labels = [r"$|%d%d\rangle$" % ((1 - s1) // 2, (1 - s2) // 2) for s1, s2 in states]

plt.figure(figsize=(10, 8))
colors = {'00': 'blue', '01': 'red', '10': 'green', '11': 'purple'}
for (s1, s2), label in zip(states, labels):
    alpha = alpha_traj(t, s1, s2)
    state = f"{(1 - s1) // 2}{(1 - s2) // 2}"
    plt.plot(alpha.real, alpha.imag, label=label, color=colors[state])
    alpha_ss = epsilon_eff(s1, s2) / (-1j * kappa / 2 + (delta_r1 + delta_r2 + chi_1 * s1 + chi_2 * s2))
    plt.plot(alpha_ss.real, alpha_ss.imag, 'o', color=colors[state])

plt.xlabel("Real")
plt.ylabel("Imag")
plt.title(f"Pointer State Trajectories")
plt.grid(True)
plt.legend()
plt.axis("equal")
plt.tight_layout()
plt.show()

omega_probe = np.linspace(-30, 30, 300)
epsilon = 0.5 * kappa * np.sqrt(2.0)
alpha_phases = {}

states = {'00': (-1, -1), '01': (-1, +1), '10': (+1, -1), '11': (+1, +1)}

for label, (sz1_val, sz2_val) in states.items():
    delta_eff = omega_probe - (chi_1 * sz1_val + chi_2 * sz2_val)
    alpha = epsilon / (kappa / 2 + 1j * delta_eff)
    alpha_phases[label] = np.angle(alpha)

plt.figure(figsize=(8, 6))

for label, phase in alpha_phases.items():
    plt.plot(omega_probe, phase, label=f"|{label}⟩")
plt.title("Phase Response")
plt.xlabel("omega - omega_0 (MHz)")
plt.ylabel("Arg (rad)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
