import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

g = 0.08e9 * 2 * np.pi # Hz
Delta_r = -0.8e9 * 2 * np.pi # Hz
omega_r = 7.0e9 * 2 * np.pi # Hz
delta_resonator = -0.1e9 * 2 * np.pi # Hz
omega_d = omega_r + delta_resonator
omega_q = omega_r + Delta_r
Delta = omega_q - omega_d

chi = -2.0e6 * 2 * np.pi 
E_C = -chi * Delta_r**2 / g**2
E_J = ((omega_q + E_C)**2) / (8 * E_C)

def transmon_energies(n_levels, E_C, omega_q):
    return np.array([n * omega_q - (E_C / 2) * n * (n - 1) for n in range(n_levels)])

n_levels = 6  # transmon energy levels
floquet_sectors = 3  # number of Floquet sectors (m = -1, 0, +1)
dim = n_levels * floquet_sectors

def build_transmon_operators(n_charge=25, n_levels=8):
    n = np.arange(-n_charge, n_charge + 1)
    dim = len(n)
    n_op_charge = np.diag(n)

    cos_phi = np.zeros((dim, dim))
    for i in range(dim - 1):
        cos_phi[i, i + 1] = 0.5
        cos_phi[i + 1, i] = 0.5

    H_charge = 4 * E_C * np.diag(n**2) - E_J * cos_phi

    evals, U = eigh(H_charge)

    U_trunc = U[:, :n_levels]
    E_n = evals[:n_levels]

    n_op_transmon = U_trunc.T @ n_op_charge @ U_trunc

    return E_n, n_op_transmon

# Rebuild transmon energies and operator
E_n, n_op = build_transmon_operators(n_charge=25, n_levels=n_levels)

# Build H0 from exact transmon energies
H0 = np.diag(E_n)
Id = np.eye(n_levels)

# Floquet zones: m = -1, 0, +1
m_vals = np.arange(-(floquet_sectors//2), floquet_sectors//2 + 1)
Id = np.eye(n_levels)

def build_floquet_matrix(A):
    Hf = np.zeros((dim, dim), dtype=np.complex128)  
    for i, m in enumerate(m_vals):
        # Diagonal blocks: H0 + m * ħω_d * I
        Hf[i*n_levels:(i+1)*n_levels, i*n_levels:(i+1)*n_levels] = H0 + m * omega_d * Id

        # Off-diagonal blocks: drive coupling
        if i < floquet_sectors - 1:
            block = 0.5 * A * n_op
            j = i + 1
            Hf[i*n_levels:(i+1)*n_levels, j*n_levels:(j+1)*n_levels] = block
            Hf[j*n_levels:(j+1)*n_levels, i*n_levels:(i+1)*n_levels] = block.T.conj()
    
    return Hf

# Sweep over different photon numbers
n_r_vals = np.linspace(0, 180, 100)
quasienergies = []

for n_r in n_r_vals:
    A = 2 * g * np.sqrt(n_r)
    Hf = build_floquet_matrix(A)
    eigvals, _ = eigh(Hf)

    folded = np.mod(eigvals + omega_d/2, omega_d) - omega_d/2
    quasienergies.append(np.sort(folded))  
    
quasienergies = np.array(quasienergies)

plt.figure(figsize=(8,6))
for i in range(n_levels * floquet_sectors):
    # Calculate which transmon state and Floquet sector this curve represents
    transmon_state = i % n_levels
    floquet_sector = m_vals[i // n_levels]
    label = f"|{transmon_state}⟩ (m={floquet_sector})"
    plt.plot(n_r_vals, quasienergies[:, i] / (2 * np.pi * 1e9), lw=1, label=label)
plt.xlabel(r"$\bar{n}_r$")
plt.ylabel("Quasienergy (GHz)")
plt.title("Floquet Quasienergies vs. Average Photon Number")
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
