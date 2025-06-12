import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# parameters
kappa = 3.0
g = 1.0
Delta = 0.2
chi = g**2 / Delta
Omega_q = 0.2
Omega_r = 0.0

Ncav = 12
Nt = 4000
t_final = 8.0 / kappa
tlist = np.linspace(0.0, t_final, Nt)

# operators
a = tensor(destroy(Ncav), qeye(2))
sm = tensor(qeye(Ncav), sigmam())
sp = sm.dag()
sz = tensor(qeye(Ncav), sigmaz())

# collapse
c_ops = [np.sqrt(kappa) * a]

# frequency sweep
omega_0 = 0.0  # bare cavity
omega_probe = np.linspace(-10, 10, 300)  # MHz detuning
alpha_vals = {0: [], 1: []}  # pointer states

for qubit_state in [0, 1]:
    sz_val = 1 if qubit_state == 1 else -1
    alpha_ss = []
    for delta in omega_probe:
        denom = kappa/2 + 1j * (delta - chi * sz_val)
        epsilon = 0.5 * kappa * np.sqrt(2.0)  # coherent drive to maintain |alpha|^2 ~ 2
        alpha = epsilon / denom
        alpha_ss.append(alpha)
    alpha_vals[qubit_state] = np.array(alpha_ss)

# plot pointer state trajectories (IQ plane)
init_g = tensor(basis(Ncav, 0), basis(2, 0))
init_e = tensor(basis(Ncav, 0), basis(2, 1))

H = ((Delta + chi) * sp * sm
     - chi * sz * a.dag() * a
     + ((Omega_q + 1j * Omega_r * chi / g) * sp + ((Omega_q + 1j * Omega_r * chi / g) * sp).dag())
     + ((1j * Omega_r - Omega_q * chi / g) * sz * a.dag() + ((1j * Omega_r - Omega_q * chi / g) * sz * a.dag()).dag()))

res_g = mesolve(H, init_g, tlist, c_ops, [a])
res_e = mesolve(H, init_e, tlist, c_ops, [a])

# plot
fig1, ax1 = plt.subplots(figsize=(8, 6))
ax1.plot(np.real(res_g.expect[0]), np.imag(res_g.expect[0]), label='|0⟩', color='blue')
ax1.plot(np.real(res_e.expect[0]), np.imag(res_e.expect[0]), label='|1⟩', color='red')
ax1.scatter(-Omega_q/g, 0, label='vo', color='green')
ax1.scatter(np.real(res_g.expect[0][-1]), np.imag(res_g.expect[0][-1]), color='blue')
ax1.scatter(np.real(res_e.expect[0][-1]), np.imag(res_e.expect[0][-1]), color='red')
ax1.set_xlabel("Real")
ax1.set_ylabel("Imag")
ax1.set_title("Pointer State Trajectories")
ax1.axis("equal")
ax1.legend()
ax1.grid(True)

fig2, ax2 = plt.subplots(figsize=(8, 6))
phase_0 = np.angle(alpha_vals[0])
phase_1 = np.angle(alpha_vals[1])
ax2.plot(omega_probe, phase_0, label="|0⟩")
ax2.plot(omega_probe, phase_1, label="|1⟩")
ax2.set_title("Phase")
ax2.set_xlabel("omega - omega_0 (MHz)")
ax2.set_ylabel("Arg (rad)")
ax2.legend()
ax2.grid(True)

plt.show()
