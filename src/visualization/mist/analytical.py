import numpy as np
import matplotlib.pyplot as plt

# Constants
g = 0.08e9 * 2 * np.pi # Hz
Delta_r = -0.8e9 * 2 * np.pi # Hz
omega_r = 5.0e9 * 2 * np.pi # Hz
delta_resonator = -0.1e9 * 2 * np.pi # Hz
omega_d = omega_r + delta_resonator
omega_q = omega_r + Delta_r
Delta = omega_q - omega_d

chi = -2.0e6 * 2 * np.pi 
E_C = -chi * Delta_r**2 / g**2
E_J = ((omega_q + E_C)**2) / (8 * E_C)

def transmon_energies(n_levels, E_C, omega_q):
    return np.array([n * omega_q - (E_C / 2) * n * (n - 1) for n in range(n_levels)])

def dipole_matrix(n_levels):
    mat = np.zeros((n_levels, n_levels))
    for n in range(n_levels - 1):
        mat[n, n+1] = mat[n+1, n] = np.sqrt(n + 1)
    return mat

n_levels = 8
energies = transmon_energies(n_levels, E_C, omega_q)
n_op = dipole_matrix(n_levels)

n_r_vals = np.linspace(0, 150, 300)
quasienergies = []

for n_r in n_r_vals:
    epsilon = 2 * g * np.sqrt(n_r) 
    H = np.diag(energies - energies[0] - np.arange(n_levels) * omega_d)
    H += 0.5 * epsilon * n_op
    evals = np.linalg.eigvalsh(H)
    folded = ((evals + omega_d / 2) % omega_d) - omega_d / 2
    quasienergies.append(np.sort(folded))

quasienergies = np.array(quasienergies)

plt.figure(figsize=(9, 6))
for i in range(n_levels):
    plt.plot(n_r_vals, quasienergies[:, i] / (2 * np.pi * 1e9), label=f"|{i}⟩")
plt.xlabel(r"Resonator photon number $\bar{n}_r = |\alpha|^2$")
plt.ylabel("Floquet quasienergies (GHz)")
plt.title("Floquet Quasienergies vs. Resonator Photon Number")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
