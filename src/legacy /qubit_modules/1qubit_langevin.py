import numpy as np
import matplotlib.pyplot as plt

# params
chi = 1.0
kappa = 3.0
g = 1.0
Omega_q = 0.2
Omega_r = 0.0
t = np.linspace(0, 10, 1000)

def alpha(t, sigma_z, alpha0=0):
    decay = np.exp(-(1j * chi * sigma_z + kappa / 2) * t)
    alpha_ss = -(Omega_r + 1j * Omega_q * chi / g * sigma_z) / (kappa / 2 + 1j * chi * sigma_z)
    return alpha_ss + (alpha0 - alpha_ss) * decay

# plot
plt.figure(figsize=(10, 8))
for sz, label in [(-1, r"$|0⟩$"), (1, r"$|1⟩$")]:
    traj = alpha(t, sz)
    color = 'blue' if sz == -1 else 'red'
    plt.plot(traj.real, traj.imag, label=label, color=color)

plt.scatter(-Omega_q/g, 0, label='vo', color='green')
plt.xlabel("Real")
plt.ylabel("Imag")
plt.title("Pointer State Trajectories")
plt.grid(True)
plt.legend()
plt.axis('equal')
plt.show()

# Phase response plot
omega_probe = np.linspace(-30, 30, 300)  # MHz
epsilon = 0.5 * kappa * np.sqrt(2.0)     # keep n̄ ~ 2
alpha_phases = {}

states = {'0': -1, '1': 1}

for label, sz_val in states.items():
    delta_eff = omega_probe - chi * sz_val
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
