import numpy as np
import matplotlib.pyplot as plt

# Physical constants
g = 0.12e9 * 2 * np.pi                      # Coupling (Hz)
Delta_r = -1.2e9 * 2 * np.pi                # Qubit-resonator detuning
omega_r = 7.4e9 * 2 * np.pi                 # Resonator frequency
delta_resonator = 0.01e9 * 2 * np.pi        # Resonator drive detuning
omega_d = omega_r + delta_resonator         # Drive frequency
omega_q = omega_r + Delta_r                 # Qubit frequency
Delta = omega_q - omega_d

# Transmon parameters
chi = g**2 / Delta_r
E_C = -chi * Delta_r**2 / g**2
E_J = ((omega_q + E_C)**2) / (8 * E_C)

# Transmon levels and Floquet parameters
n_levels = 5
N_floquet = 5          # Number of Floquet zones (odd number!)
M = N_floquet // 2     # Floquet index runs from -M to M

# Bare transmon energy levels
def transmon_energies(n_levels, E_C, E_J):
    return np.array([
        -E_J + np.sqrt(8 * E_J * E_C) * (n + 0.5) - (E_C / 12) * (6 * n**2 + 6 * n + 3)
        for n in range(n_levels)
    ])

# Approximate number operator in transmon basis
def number_operator(n_levels):
    return np.diag(np.arange(n_levels))

# Set up
energies = transmon_energies(n_levels, E_C, E_J)
Ht = np.diag(energies)
n_op = number_operator(n_levels)

# Sweep resonator photon number -> converts to drive amplitude
n_r_vals = np.linspace(0, 150, 200)
quasienergies = []

for n_r in n_r_vals:
    epsilon = 2 * g * np.sqrt(n_r)   # Classical drive amplitude scaling

    dim = n_levels * N_floquet
    HF = np.zeros((dim, dim), dtype=complex)

    for m in range(-M, M + 1):
        i = m + M
        # Diagonal block: Ht + m * ω_d
        HF[i*n_levels:(i+1)*n_levels, i*n_levels:(i+1)*n_levels] = (
            Ht + m * omega_d * np.eye(n_levels)
        )

        # Off-diagonal blocks: (ε / 2) * n̂
        if i < N_floquet - 1:
            HF[i*n_levels:(i+1)*n_levels, (i+1)*n_levels:(i+2)*n_levels] = (epsilon/2) * n_op
            HF[(i+1)*n_levels:(i+2)*n_levels, i*n_levels:(i+1)*n_levels] = (epsilon/2) * n_op

    # Diagonalize
    evals = np.linalg.eigvalsh(HF)

    # Fold quasienergies into Floquet Brillouin zone
    folded = ((evals + omega_d/2) % omega_d) - omega_d/2
    quasienergies.append(np.sort(folded[:n_levels]))  # Select lowest dressed levels

# Convert to array
quasienergies = np.array(quasienergies)

# Plot
plt.figure(figsize=(9, 6))
for i in range(n_levels):
    plt.plot(n_r_vals, quasienergies[:, i] / (2 * np.pi * 1e9), label=f"|ϕ_{i}⟩")

plt.xlabel(r"Resonator photon number $\bar{n}_r$")
plt.ylabel("Floquet quasienergies (GHz)")
plt.title("Floquet Quasienergies vs. Resonator Photon Number")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
