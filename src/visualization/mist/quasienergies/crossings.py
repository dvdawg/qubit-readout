import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt

chi = -2.0e6 * 2 * np.pi # disp (Hz)
Delta_r = -1.2e9 * 2 * np.pi # qubit-resonator detuning (Hz)
g = 0.08e9 * 2 * np.pi # Hz
omega_r = 5.0e9 * 2 * np.pi # Hz
delta_resonator = -0.1e9 * 2 * np.pi # resonator detuning (Hz)
kappa = 0.008e9 * 2 * np.pi # linewidth (Hz)
Omega_r = 7.0e6 * 2 * np.pi # drive amp (Hz)

E_C = -chi * Delta_r**2 / g**2
omega_q = omega_r + Delta_r
E_J = ((omega_q + E_C)**2) / (8 * E_C)

print(f"E_C / 2pi = {E_C / (2*np.pi*1e9):.3f} GHz")
print(f"E_J / 2pi = {E_J / (2*np.pi*1e9):.3f} GHz")

ng = 0.0
N_charge = 15
N_F = 5

n = np.arange(-N_charge, N_charge + 1)
dim = len(n)
H0 = np.diag(4 * E_C * (n - ng) ** 2)
cos_phi = np.zeros((dim, dim))
for i in range(dim - 1):
    cos_phi[i, i+1] = cos_phi[i + 1, i] = -0.5 * E_J
H0 += cos_phi

Evals, U = eigh(H0)
n_op = np.diag(n)
n_mat = U.T @ n_op @ U

def build_HF(Evals, n_mat, epsilon_t, omega_d, N_F):
    d = len(Evals)
    D = d * (2 * N_F + 1)
    HF = np.zeros((D, D))
    for m in range(-N_F, N_F + 1):
        for i in range(d):
            idx = (m + N_F) * d + i
            HF[idx, idx] = Evals[i] + m * omega_d
    for m in range(-N_F, N_F):
        for i in range(d):
            for j in range(d):
                idx1 = (m + N_F) * d + i
                idx2 = (m + 1 + N_F) * d + j
                HF[idx1, idx2] += 0.5 * epsilon_t * n_mat[i, j]
                HF[idx2, idx1] += 0.5 * epsilon_t * n_mat[j, i]
    return HF

omega_d = omega_r + delta_resonator
alpha_ss = (1j * Omega_r) / (kappa / 2 + 1j * delta_resonator)
n_r_vals = np.linspace(0, 200, 150)
quasienergies = []

for n_r in n_r_vals:
    epsilon_t = 2 * g * np.sqrt(n_r)
    HF = build_HF(Evals[:10], n_mat[:10, :10], epsilon_t, omega_d, N_F)
    eigvals, _ = eigh(HF)
    folded = ((eigvals + omega_d / 2) % omega_d) - omega_d / 2
    quasienergies.append(np.sort(folded[:10]))

quasienergies = np.array(quasienergies)

plt.figure(figsize=(10, 6))
for i in range(10):
    plt.plot(n_r_vals, quasienergies[:, i] / (2 * np.pi * 1e9), label=f"|{i}⟩")
plt.xlabel(r"Resonator photon number $\bar{n}_r = |\alpha|^2$")
plt.ylabel("Floquet quasienergies (GHz)")
plt.title("Floquet Quasienergies vs. Resonator Photon Number")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
