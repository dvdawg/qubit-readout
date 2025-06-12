import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# System parameters
kappa = 5.0          # resonator linewidth (κ/2π)
chi = 2.5           # dispersive shift (χ/2π)
n_bar = 2.0         # steady-state photons for ground state
delta_r = 0.0       # probe on bare cavity resonance

# Calculate drive amplitude for desired photon number
epsilon = 0.5 * kappa * np.sqrt(n_bar)

# Hilbert space dimensions
Ncav = 12           # Cavity Fock space truncation
Nq = 4              # Number of qubit levels

# Time parameters
t_final = 8.0 / kappa
Nt = 4000
tlist = np.linspace(0.0, t_final, Nt)

# Define operators
a = tensor(destroy(Ncav), qeye(Nq))            # cavity annihilation
sm = tensor(qeye(Ncav), destroy(Nq))           # qubit lowering
sz = 2*sm.dag()*sm - tensor(qeye(Ncav), qeye(Nq))

# Hamiltonian
H = chi * sz * a.dag() * a + epsilon * (a + a.dag())

# Collapse operators
c_ops = [np.sqrt(kappa) * a]

# Initial states
vac = basis(Ncav, 0)
psi_0 = tensor(vac, basis(Nq, 0))  # |0⟩
psi_1 = tensor(vac, basis(Nq, 1))  # |1⟩
psi_2 = tensor(vac, basis(Nq, 2))  # |2⟩
psi_3 = tensor(vac, basis(Nq, 3))  # |3⟩

# Observables
e_ops = [a]  # We only need the cavity field for pointer states

# Solve for each initial state
res_0 = mesolve(H, psi_0, tlist, c_ops, e_ops, progress_bar=None)
res_1 = mesolve(H, psi_1, tlist, c_ops, e_ops, progress_bar=None)
res_2 = mesolve(H, psi_2, tlist, c_ops, e_ops, progress_bar=None)
res_3 = mesolve(H, psi_3, tlist, c_ops, e_ops, progress_bar=None)

# Extract expectation values
alpha_0 = res_0.expect[0]
alpha_1 = res_1.expect[0]
alpha_2 = res_2.expect[0]
alpha_3 = res_3.expect[0]

# Create figure for pointer state trajectories
plt.figure(figsize=(10, 8))

# Plot trajectories
plt.plot(np.real(alpha_0), np.imag(alpha_0), label='|0⟩', linewidth=2)
plt.plot(np.real(alpha_1), np.imag(alpha_1), label='|1⟩', linewidth=2)
plt.plot(np.real(alpha_2), np.imag(alpha_2), label='|2⟩', linewidth=2)
plt.plot(np.real(alpha_3), np.imag(alpha_3), label='|3⟩', linewidth=2)

# Mark final states
plt.scatter(np.real(alpha_0[-1]), np.imag(alpha_0[-1]), s=100, marker='o')
plt.scatter(np.real(alpha_1[-1]), np.imag(alpha_1[-1]), s=100, marker='o')
plt.scatter(np.real(alpha_2[-1]), np.imag(alpha_2[-1]), s=100, marker='o')
plt.scatter(np.real(alpha_3[-1]), np.imag(alpha_3[-1]), s=100, marker='o')

# Add labels and title
plt.xlabel('Real')
plt.ylabel('Imag')
plt.title('Pointer State Trajectories')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# Make the plot square and add equal aspect ratio
plt.axis('equal')
plt.tight_layout()

# Show the plot
plt.show()

# Print some information about the simulation
print("\nSimulation Parameters:")
print(f"Cavity linewidth (κ/2π): {kappa:.1f} MHz")
print(f"Dispersive shift (χ/2π): {chi:.1f} MHz")
print(f"Steady-state photons (n̄): {n_bar:.1f}")
print(f"Integration time: {t_final:.1f} μs")
print(f"Number of levels: {Nq}") 