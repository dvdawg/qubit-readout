import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# params
kappa = 5.0
g1 = 1.0  
g2 = 1.0 
Delta1 = 0.1
Delta2 = 0.1
chi1 = g1**2 / Delta1
chi2 = g2**2 / Delta2
J = 0.05  
Omega_q1 = 0.2 
Omega_q2 = 0.2 
Omega_r = 0.0

Ncav = 12
t_final = 8.0 / kappa
Nt = 4000
tlist = np.linspace(0.0, t_final, Nt)

# operators
a = tensor(destroy(Ncav), qeye(2), qeye(2))
sm1 = tensor(qeye(Ncav), sigmam(), qeye(2))
sp1 = sm1.dag()
sz1 = tensor(qeye(Ncav), sigmaz(), qeye(2))
sm2 = tensor(qeye(Ncav), qeye(2), sigmam())
sp2 = sm2.dag()
sz2 = tensor(qeye(Ncav), qeye(2), sigmaz())

# hamiltonian
H = ((Delta1 + chi1) * sp1 * sm1 + (Delta2 + chi2) * sp2 * sm2
     - chi1 * sz1 * a.dag() * a - chi2 * sz2 * a.dag() * a
     + ((Omega_q1 + 1j * Omega_r * chi1 / g1) * sp1 + ((Omega_q1 + 1j * Omega_r * chi1 / g1) * sp1).dag())
     + ((Omega_q2 + 1j * Omega_r * chi2 / g2) * sp2 + ((Omega_q2 + 1j * Omega_r * chi2 / g2) * sp2).dag())
     + ((1j * Omega_r - Omega_q1 * chi1 / g1) * sz1 * a.dag() + ((1j * Omega_r - Omega_q1 * chi1 / g1) * sz1 * a.dag()).dag())
     + ((1j * Omega_r - Omega_q2 * chi2 / g2) * sz2 * a.dag() + ((1j * Omega_r - Omega_q2 * chi2 / g2) * sz2 * a.dag()).dag()))
     # + J * (sp1 * sm2 + sm1 * sp2)  # optional coupling

# collapse operator
c_ops = [np.sqrt(kappa) * a]

# initial states
init_00 = tensor(basis(Ncav, 0), basis(2, 0), basis(2, 0))
init_01 = tensor(basis(Ncav, 0), basis(2, 0), basis(2, 1))
init_10 = tensor(basis(Ncav, 0), basis(2, 1), basis(2, 0))
init_11 = tensor(basis(Ncav, 0), basis(2, 1), basis(2, 1))

# simulation
res_00 = mesolve(H, init_00, tlist, c_ops, [a])
res_01 = mesolve(H, init_01, tlist, c_ops, [a])
res_10 = mesolve(H, init_10, tlist, c_ops, [a])
res_11 = mesolve(H, init_11, tlist, c_ops, [a])

plt.figure(figsize=(8, 7))
plt.plot(np.real(res_00.expect[0]), np.imag(res_00.expect[0]), label='|00⟩', color='blue')
plt.plot(np.real(res_01.expect[0]), np.imag(res_01.expect[0]), label='|01⟩', color='red')
plt.plot(np.real(res_10.expect[0]), np.imag(res_10.expect[0]), label='|10⟩', color='green')
plt.plot(np.real(res_11.expect[0]), np.imag(res_11.expect[0]), label='|11⟩', color='purple')
plt.scatter(np.real(res_00.expect[0][-1]), np.imag(res_00.expect[0][-1]), color='blue')
plt.scatter(np.real(res_01.expect[0][-1]), np.imag(res_01.expect[0][-1]), color='red')
plt.scatter(np.real(res_10.expect[0][-1]), np.imag(res_10.expect[0][-1]), color='green')
plt.scatter(np.real(res_11.expect[0][-1]), np.imag(res_11.expect[0][-1]), color='purple')
plt.xlabel("Real")
plt.ylabel("Imag")
plt.title("Pointer State Trajectories (IQ Plane)")
plt.axis("equal")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

omega_probe = np.linspace(-30, 30, 300)  # MHz
epsilon = 0.5 * kappa * np.sqrt(2.0)     # keep n̄ ~ 2
alpha_phases = {}

states = {'00': (-1, -1), '01': (-1, +1), '10': (+1, -1), '11': (+1, +1)}

for label, (sz1_val, sz2_val) in states.items():
    delta_eff = omega_probe - (chi1 * sz1_val + chi2 * sz2_val)
    alpha = epsilon / (kappa/2 + 1j * delta_eff)
    alpha_phases[label] = np.angle(alpha)

plt.figure(figsize=(8, 6))
for label, phase in alpha_phases.items():
    plt.plot(omega_probe, phase, label=f"|{label}⟩")
plt.title("Phase Response")
plt.xlabel("omega - omega_0 (MHz)")
plt.ylabel("Arg (rad)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
