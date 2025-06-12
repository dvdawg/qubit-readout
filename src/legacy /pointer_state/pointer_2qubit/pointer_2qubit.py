import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# params
kappa = 5.0
chi_1 = 2.5
chi_2 = 1.5
n_bar = 2.0
epsilon = 0.5 * kappa * np.sqrt(n_bar)
delta_r = 0.0 
omega_r = 0.0

Ncav = 12  # cavity cutoff
Nq = 2     # 2-level qubits
t_final = 8.0 / kappa
Nt = 4000
tlist = np.linspace(0, t_final, Nt)

# operators
a = tensor(destroy(Ncav), qeye(2), qeye(2))  # cavity
sz1 = tensor(qeye(Ncav), sigmaz(), qeye(2))
sz2 = tensor(qeye(Ncav), qeye(2), sigmaz())

# dispersive
H_disp = (omega_r-chi_1 * sz1 - chi_2 * sz2) * a.dag() * a
H_drive = epsilon * (a + a.dag())
H = H_disp + H_drive

# Collapse operators (cavity loss)
c_ops = [np.sqrt(kappa) * a]

# Initial states: |00>, |01>, |10>, |11>
qubit_states = {
    '00': tensor(basis(Ncav, 0), basis(2, 0), basis(2, 0)),
    '01': tensor(basis(Ncav, 0), basis(2, 0), basis(2, 1)),
    '10': tensor(basis(Ncav, 0), basis(2, 1), basis(2, 0)),
    '11': tensor(basis(Ncav, 0), basis(2, 1), basis(2, 1)),
}

# Simulate trajectories
results = {}
for label, state in qubit_states.items():
    result = mesolve(H, state, tlist, c_ops, [a])
    alpha_t = result.expect[0]
    results[label] = alpha_t

# Plot x-p phase space
plt.figure(figsize=(8, 6))
for label, alpha_t in results.items():
    x_t = np.real(alpha_t)
    p_t = np.imag(alpha_t)
    plt.plot(x_t, p_t, label=f"|{label}>")

plt.xlabel("x quadrature (Re α)")
plt.ylabel("p quadrature (Im α)")
plt.title("Pointer State Trajectories for 2-Qubit System")
plt.legend()
plt.grid(True)
plt.axis("equal")
plt.tight_layout()
plt.show()
