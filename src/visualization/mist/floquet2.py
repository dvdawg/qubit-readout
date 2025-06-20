import numpy as np
import matplotlib.pyplot as plt

# params
g = 0.12e9 * 2 * np.pi
Delta_r = -1.2e9 * 2 * np.pi
omega_r = 7.4e9 * 2 * np.pi
delta_resonator = 0.01e9 * 2 * np.pi
omega_d = omega_r + delta_resonator
omega_q = omega_r + Delta_r
Delta = omega_q - omega_d

chi = g**2 / Delta_r
E_C = -chi * Delta_r**2 / g**2
E_J = ((omega_q + E_C)**2) / (8 * E_C)

def transmon_energies(n_levels, E_C, E_J):
    return np.array([
        -E_J + np.sqrt(8 * E_J * E_C) * (n + 0.5) - (E_C / 12) * (6 * n**2 + 6 * n + 3)
        for n in range(n_levels)
    ])

n_levels = 8                     
n_r_max = 150                    
energies = transmon_energies(n_levels, E_C, E_J)

n_r_vals = np.linspace(0, n_r_max, 300)
quasienergies = []

for n_r in n_r_vals:
    H = np.diag(energies)
    m = -1 * n_levels + 1
    for i in range(n_levels - 1):
        coupling = g * np.sqrt(n_r)  
        H[i, i+1] = H[i+1, i] = coupling
        H[i, i] += m * omega_d
        m += 1
    
    evals = np.linalg.eigvalsh(H)

    folded = ((evals + omega_d / 2) % omega_d) - omega_d / 2
    print(folded)
    print("a!!")
    quasienergies.append(np.sort(folded))

quasienergies = np.array(quasienergies)

plt.figure(figsize=(9, 6))
for i in range(n_levels):
    plt.plot(n_r_vals, quasienergies[:, i] / (omega_d * 2 * np.pi * 1e9), label=f"|{i}⟩")
plt.xlabel(r"Resonator photon number $\bar{n}_r$")
plt.ylabel("Floquet quasienergies (GHz)")
plt.title("Floquet Quasienergies vs. Resonator Photon Number")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
