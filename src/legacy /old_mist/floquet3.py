import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

# params
g = 0.12e9 * 2 * np.pi
Delta_r = -1.2e9 * 2 * np.pi
omega_r = 7.4e9 * 2 * np.pi
delta_resonator = 0.01e9 * 2 * np.pi
omega_d = omega_r + delta_resonator
omega_q = omega_r + Delta_r

# transmon params
chi = g**2 / Delta_r
E_C = -chi * Delta_r**2 / g**2
E_J = ((omega_q + E_C)**2) / (8 * E_C)

# Floquet settings
n_levels = 8
n_r_max = 150
n_r_vals = np.linspace(0, n_r_max, 300)
m_max = 1  
M = 2 * m_max + 1

def transmon_energies(n_levels, E_C, E_J):
    return np.array([
        -E_J + np.sqrt(8 * E_J * E_C) * (n + 0.5) - (E_C / 12) * (6 * n**2 + 6 * n + 3)
        for n in range(n_levels)
    ])

energies = transmon_energies(n_levels, E_C, E_J)
quasienergies = []

for n_r in n_r_vals:
    # Define bare H0 and drive operator V_op
    H0 = np.diag(energies)
    V_op = np.zeros((n_levels, n_levels), dtype=complex)
    for i in range(n_levels - 1):
        V_op[i, i+1] = V_op[i+1, i] = g * np.sqrt(n_r)

    # Build Floquet Hamiltonian in extended Hilbert space
    H_F = np.zeros((n_levels * M, n_levels * M), dtype=complex)
    for m in range(-m_max, m_max + 1):
        idx = m + m_max
        H_F[idx*n_levels:(idx+1)*n_levels, idx*n_levels:(idx+1)*n_levels] = H0 + m * omega_d * np.eye(n_levels)
        if idx + 1 < M:
            H_F[idx*n_levels:(idx+1)*n_levels, (idx+1)*n_levels:(idx+2)*n_levels] = 0.5 * V_op
        if idx - 1 >= 0:
            H_F[idx*n_levels:(idx+1)*n_levels, (idx-1)*n_levels:idx*n_levels] = 0.5 * V_op

    evals = eigh(H_F, eigvals_only=True)
    folded = ((evals + omega_d/2) % omega_d) - omega_d/2
    quasi = np.sort(folded)[:n_levels]
    quasienergies.append(quasi.real)

quasienergies = np.array(quasienergies)


plt.figure(figsize=(9, 6))
for i in range(n_levels):
    plt.plot(n_r_vals, quasienergies[:, i] / (2 * np.pi * 1e9), label=f"|{i}>")
plt.xlabel(r"Resonator photon number $\bar{n}_r$")
plt.ylabel("Floquet quasienergies (GHz)")
plt.title("Floquet Quasienergies vs. Resonator Photon Number")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
