import numpy as np
import matplotlib.pyplot as plt

g = 0.08e9 * 2 * np.pi # Hz
Delta_r = -1.2e9 * 2 * np.pi # Hz
omega_r = 5.0e9 * 2 * np.pi # Hz
delta_resonator = -0.1e9 * 2 * np.pi
omega_d = omega_r + delta_resonator
omega_q = omega_r + Delta_r
Delta = omega_q - omega_d

chi = -2.0e6 * 2 * np.pi
E_C = -chi * Delta_r**2 / g**2
E_J = ((omega_q + E_C)**2) / (8 * E_C)

def transmon_energies(n_levels, E_C, omega_q):
    return np.array([n * omega_q - (E_C / 2) * n * (n - 1) for n in range(n_levels)])

def transmon_ladder_operator(n_levels):
    a = np.zeros((n_levels, n_levels))
    for n in range(n_levels - 1):
        a[n, n+1] = np.sqrt(n + 1)
    return a + a.T

n_levels = 8
energies = transmon_energies(n_levels, E_C, omega_q)
n_op = transmon_ladder_operator(n_levels)

n_r_vals = np.linspace(0, 150, 300)
quasienergies = []
    
for n_r in n_r_vals:
    A = 2 * g * np.sqrt(n_r)
    H_rot = np.diag(energies - energies[0] - np.arange(n_levels) * omega_d)
    H_drive = 0.5 * A * n_op

    H = H_rot + H_drive

    evals = np.linalg.eigvalsh(H)
    folded = ((evals + omega_d / 2) % omega_d) - omega_d / 2
    quasienergies.append(np.sort(folded))

quasienergies = np.array(quasienergies)

plt.figure(figsize=(9, 6))
for i in range(n_levels):
    plt.plot(n_r_vals, quasienergies[:, i] / (2 * np.pi * 1e9), label=f"|{i}⟩")
plt.xlabel("Resonator photon number")
plt.ylabel("Floquet quasienergies (GHz)")
plt.title("Floquet Quasienergies vs. Resonator Photon Number")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
