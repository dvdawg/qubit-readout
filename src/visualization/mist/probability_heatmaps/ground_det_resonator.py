import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import qutip as qt
import scqubits as scq
import floquet as ft

g = 0.120
EC = 0.220
omega_q = 6.289177
EJ = (omega_q + EC)**2/(8*EC)

Delta_vals = np.linspace(0.75, 1.5, 100)
nbar_vals = np.linspace(5, 100, 96)
drive_amps = 2.0 * g * np.sqrt(nbar_vals)
transition_prob_matrix = np.zeros((len(Delta_vals), len(nbar_vals)))

options = ft.Options(num_cpus=8, nsteps=1000, fit_range_fraction=1.0, overlap_cutoff=0.8, floquet_sampling_time_fraction=0.0, save_floquet_modes=True)

transmon_dim = 12
osc_dim = 12

for i, Delta in enumerate(Delta_vals):
    omega_r = omega_q - Delta
    omega_d = omega_r

    tmon = scq.Transmon(EJ=EJ, EC=EC, ng=0.2, ncut=25, truncated_dim=transmon_dim)
    reson = scq.Oscillator(E_osc=omega_r, truncated_dim=osc_dim)

    hs = scq.HilbertSpace([tmon, reson])
    hs.generate_lookup()

    H0 = hs.hamiltonian() 

    esys_t = tmon.eigensys(evals_count=transmon_dim)
    n_sub = tmon.n_operator(energy_esys=esys_t)  
    n_op = scq.identity_wrap(n_sub, tmon, [tmon, reson], op_in_eigenbasis=True)
    n_op.dims = H0.dims

    model = ft.Model(H0, n_op, omega_d_values=np.array([omega_d]), drive_amplitudes=drive_amps)
    fa = ft.FloquetAnalysis(model, state_indices=list(range(transmon_dim * osc_dim)), options=options)

    data = fa.run()
    bare = data["bare_state_overlaps"][0]
    P_ground = np.abs(bare[:, 0])
    transition_prob_matrix[i, :] = 1.0 - P_ground

plt.figure(figsize=(12, 8))
im = plt.imshow(transition_prob_matrix.T, extent=[Delta_vals[0], Delta_vals[-1], nbar_vals[0], nbar_vals[-1]], aspect='auto', origin='lower', cmap='viridis')
cbar = plt.colorbar(im); cbar.set_label('Transition Probability', fontsize=12)
plt.xlabel('Transmon–resonator detuning Δ (GHz)', fontsize=12)
plt.ylabel('Photon number $\\overline{n}$', fontsize=12)
plt.title('Ground-state leakage probability', fontsize=14)
plt.tight_layout(); plt.show()

print(f"Maximum transition probability: {transition_prob_matrix.max():.3f}")
print(f"Drive amplitudes: {drive_amps[0]:.3f} – {drive_amps[-1]:.3f} GHz")
