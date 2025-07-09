import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import qutip as qt
import scqubits as scq
import floquet as ft

g = 0.120
EC = 0.220
omega_r = 5.7
omega_d = omega_r - 0.033

kappa_vals = np.linspace(0.005, 0.05, 20)
nbar_vals = np.linspace(0, 180, 91)
drive_amps = 2.0*g*np.sqrt(nbar_vals)

transition_prob_matrix = np.zeros((len(kappa_vals), len(nbar_vals)))

Delta = 1.0
omega_q = omega_r + Delta
EJ = (omega_q+EC)**2/(8*EC)

num_states = 12
qubit_params = dict(EJ=EJ, EC=EC, ng=0.2, ncut=31)
tmon = scq.Transmon(**qubit_params, truncated_dim=num_states)
hs = scq.HilbertSpace([tmon])
hs.generate_lookup()

evals = hs["evals"][0][:num_states]
H0 = 2*np.pi*qt.Qobj(np.diag(evals-evals[0]))
H1 = hs.op_in_dressed_eigenbasis(tmon.n_operator)

options = ft.Options(
    num_cpus=4,
    nsteps=1000,
    fit_range_fraction=1.0,
    overlap_cutoff=0.8,
    floquet_sampling_time_fraction=0.0,
    save_floquet_modes=True
)

for i, kappa in enumerate(kappa_vals):
    print(f"κ = {kappa:.3f} GHz")
    model = ft.Model(
        H0,
        H1,
        omega_d_values=np.array([2*np.pi*omega_d]),
        drive_amplitudes=2*np.pi*drive_amps
    )
    fa = ft.FloquetAnalysis(model, state_indices=list(range(num_states)), options=options)
    data = fa.run()
    floq_modes = data["floquet_modes"][0] 

    for j in range(len(nbar_vals)):
        overlaps=np.abs(floq_modes[j, :, 0])**2
        ground_branch_idx=int(np.argmax(overlaps))
        transition_prob_matrix[i, j]=1-overlaps[ground_branch_idx]

plt.figure(figsize=(12, 8))
im = plt.imshow(
    transition_prob_matrix.T,
    extent=[kappa_vals[0], kappa_vals[-1], nbar_vals[0], nbar_vals[-1]],
    aspect='auto',
    origin='lower',
    cmap='viridis',
)
cbar = plt.colorbar(im)
cbar.set_label('Transition Probability', fontsize=12)

plt.xlabel('Resonator linewidth κ (GHz)', fontsize=12)
plt.ylabel('Photon number $\overline{n}$', fontsize=12)
plt.title('Ground‐state leakage probability vs κ', fontsize=14)
plt.yscale('log')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Maximum transition probability: {transition_prob_matrix.max():.3f}")
print(f"Drive amplitudes: {drive_amps[0]:.3f} – {drive_amps[-1]:.3f} GHz")
