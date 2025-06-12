import numpy as np
import matplotlib.pyplot as plt

# Parameters (all in units where ħ = 1)
kappa = 5.0                # Cavity linewidth  κ
chi_base = 2.5             # Base dispersive shift χ (so χ_j = j * χ_base)
n_bar = 2.0                # Desired steady‑state photon number for |g⟩
epsilon = 0.5 * kappa * np.sqrt(n_bar)   # Drive amplitude  ε
levels = range(4)          # j = 0,1,2,3  (|g⟩ … |3⟩)

# Time grid
t_final = 8.0 / kappa
Nt = 400
t = np.linspace(0, t_final, Nt)

# Storage for trajectories
trajectories = {}

for j in levels:
    Delta_j = -chi_base * j        # Probe on bare cavity resonance ⇒ Δ_j = -χ_j
    gamma_j = kappa / 2 - 1j * Delta_j
    alpha_ss = -1j * epsilon / gamma_j

    # Analytical solution for coherent‑state amplitude α_j(t)
    alpha_t = alpha_ss * (1 - np.exp(-gamma_j * t))
    trajectories[j] = alpha_t

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
for j, alpha_t in trajectories.items():
    ax.plot(alpha_t.real, alpha_t.imag, label=f"|{j}⟩")
    # mark steady‑state with a filled marker
    ax.plot(alpha_t.real[-1], alpha_t.imag[-1], marker='o')

# Axes styling to mimic phase‑space diagram
ax.axhline(0, linestyle='--', linewidth=0.8)
ax.axvline(0, linestyle='--', linewidth=0.8)
ax.grid(True, linestyle='--', alpha=0.3)
ax.set_xlabel(r"$x=\mathrm{Re}\{\alpha\}$", fontsize=12)
ax.set_ylabel(r"$p=\mathrm{Im}\{\alpha\}$", fontsize=12)
ax.set_title("Pointer‑state trajectories up to |3⟩", fontsize=14, pad=15)
ax.set_aspect('equal', 'box')
ax.legend(loc='upper center', ncol=4, frameon=True, fontsize=10)

plt.tight_layout()
plt.show()