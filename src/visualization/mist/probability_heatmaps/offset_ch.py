import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import qutip as qt
import scqubits as scq
import floquet as ft

g = 0.120
EC = 0.220
omega_r = 6.7
omega_d = omega_r - 0.033

Delta = 1.0
omega_q = omega_r + Delta
EJ = (omega_q + EC)**2 / (8 * EC)

ng_vals   = np.linspace(0.0, 0.5, 50)
nbar_vals = np.linspace(20, 180, 81)
drive_amps = 2.0 * g * np.sqrt(nbar_vals)

transition_prob_matrix = np.zeros((len(ng_vals), len(nbar_vals)))

options = ft.Options(
    num_cpus=4,
    nsteps=500,
    fit_range_fraction=1.0,
    overlap_cutoff=0.8,
    floquet_sampling_time_fraction=[0.0, 0.25, 0.5, 0.75, 1.0],
    save_floquet_modes=True
)

for i, ng in enumerate(ng_vals):
    num_states = 12
    qubit_params = dict(EJ=EJ, EC=EC, ng=ng, ncut=31)
    tmon = scq.Transmon(**qubit_params, truncated_dim=num_states)
    hs = scq.HilbertSpace([tmon])
    hs.generate_lookup()

    evals = hs["evals"][0][:num_states]
    H0 = 2 * np.pi * qt.Qobj(np.diag(evals - evals[0]))
    H1 = hs.op_in_dressed_eigenbasis(tmon.n_operator)

    model = ft.Model(
        H0,
        H1,
        omega_d_values=np.array([2*np.pi * omega_d]),
        drive_amplitudes=2*np.pi * drive_amps
    )

    fa = ft.FloquetAnalysis(model, state_indices=list(range(num_states)), options=options)
    data = fa.run()
    bare = data["bare_state_overlaps"][0]

    transition_prob_matrix[i, :] = 1.0 - bare[:, 0]

plt.figure(figsize=(12, 8))
im = plt.imshow(
    transition_prob_matrix.T,
    extent=[ng_vals[0], ng_vals[-1], nbar_vals[0], nbar_vals[-1]],
    aspect='auto',
    origin='lower',
    cmap='viridis',
    vmin=0.0,
    vmax=1.0
)
cbar = plt.colorbar(im)
cbar.set_label('Transition Probability', fontsize=12)
cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

plt.xlabel('Gate charge $n_g$', fontsize=12)
plt.ylabel('Photon number $\\,\\overline{n}$', fontsize=12)
plt.title('Ground‐state Leakage vs. Gate Charge', fontsize=14)
plt.yscale('log')
plt.ylim(nbar_vals[0], nbar_vals[-1])
plt.xlim(ng_vals[0], ng_vals[-1])
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Maximum transition probability: {transition_prob_matrix.max():.3f}")
print(f"Gate charge sweep: {ng_vals[0]:.3f} – {ng_vals[-1]:.3f}")
