import numpy as np
import matplotlib.pyplot as plt
from qutip import *

kappa = 3.0  # resonator linewidth  (κ/2π)
chi = 2.5          # dispersive shift     (χ/2π)  — choose χ = κ/2 for near-optimal SNR
n_bar = 2.0          # desired steady-state photons when qubit = |g>

delta_r = 0.0        # probe on bare cavity resonance  (ω_d = ω_r)
epsilon = 0.5 * kappa * np.sqrt(n_bar)     # drive amplitude so that |α_g|^2 ≈ n_bar

Ncav = 12 # Fock-space truncation for the cavity
Nq = 4 # Number of qubit levels
t_final = 8.0 / kappa # integration window  (≈ 8/κ  is plenty)
Nt = 4000
tlist = np.linspace(0.0, t_final, Nt)

# Operators
a = tensor(destroy(Ncav), qeye(Nq))            # cavity annihilation
sm = tensor(qeye(Ncav), destroy(Nq))           # qubit lowering
sz = 2*sm.dag()*sm - tensor(qeye(Ncav), qeye(Nq))

# Hamiltonian
H = chi * sz * a.dag() * a + epsilon * (a + a.dag())

# Collapse operators
c_ops = [np.sqrt(kappa) * a]

# Initial states for each qubit level
vac = basis(Ncav, 0)
psi_0 = tensor(vac, basis(Nq, 0))  # |0⟩
psi_1 = tensor(vac, basis(Nq, 1))  # |1⟩
psi_2 = tensor(vac, basis(Nq, 2))  # |2⟩
psi_3 = tensor(vac, basis(Nq, 3))  # |3⟩

# observ
e_ops = [a, (a + a.dag())/2]     # α(t) and X(t) = (a + a†)/2

# initial state
res_0 = mesolve(H, psi_0, tlist, c_ops, e_ops, progress_bar=None)
res_1 = mesolve(H, psi_1, tlist, c_ops, e_ops, progress_bar=None)
res_2 = mesolve(H, psi_2, tlist, c_ops, e_ops, progress_bar=None)
res_3 = mesolve(H, psi_3, tlist, c_ops, e_ops, progress_bar=None)

# expect values
alpha_0 = res_0.expect[0]
alpha_1 = res_1.expect[0]
alpha_2 = res_2.expect[0]
alpha_3 = res_3.expect[0]

X0 = res_0.expect[1]
X1 = res_1.expect[1]
X2 = res_2.expect[1]
X3 = res_3.expect[1]

# Calculate weights and integrated signals
weight = np.abs(X3 - X0)  # Using |h⟩ and |g⟩ as reference states
M_0 = np.trapz(weight * X0, tlist)
M_1 = np.trapz(weight * X1, tlist)
M_2 = np.trapz(weight * X2, tlist)
M_3 = np.trapz(weight * X3, tlist)

# Vacuum-noise variance (single quadrature, quantum-limited amp, η = 1)
variance = 0.5 * np.trapz(weight**2, tlist)
SNR = np.abs(M_3 - M_0) / np.sqrt(2 * variance)

# Create two separate figures
plt.figure(figsize=(6, 5))
plt.plot(np.real(alpha_0), np.imag(alpha_0), label='|0⟩')
plt.plot(np.real(alpha_1), np.imag(alpha_1), label='|1⟩')
plt.plot(np.real(alpha_2), np.imag(alpha_2), label='|2⟩')
plt.plot(np.real(alpha_3), np.imag(alpha_3), label='|3⟩')
plt.scatter(np.real(alpha_0[-1]), np.imag(alpha_0[-1]), s=25)
plt.scatter(np.real(alpha_1[-1]), np.imag(alpha_1[-1]), s=25)
plt.scatter(np.real(alpha_2[-1]), np.imag(alpha_2[-1]), s=25)
plt.scatter(np.real(alpha_3[-1]), np.imag(alpha_3[-1]), s=25)
plt.xlabel(r'Real')
plt.ylabel(r'Imag')
plt.title('Pointer state trajectories')
plt.legend(frameon=False)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 5))
plt.plot(tlist, X0, label='X  |0⟩')
plt.plot(tlist, X1, label='X  |1⟩')
plt.plot(tlist, X2, label='X  |2⟩')
plt.plot(tlist, X3, label='X  |3⟩')
plt.xlabel('Time (µs)')
plt.ylabel(r'$\langle X\rangle$')
plt.title('X-quadrature during read-out')
plt.legend(frameon=False)
plt.tight_layout()
plt.show()

print(f"Integrated homodyne signals:")
print(f"M_g = {M_0:.3f}")
print(f"M_e = {M_1:.3f}")
print(f"M_f = {M_2:.3f}")
print(f"M_h = {M_3:.3f}")
print(f"Vacuum-limited SNR: {SNR:.2f}") 