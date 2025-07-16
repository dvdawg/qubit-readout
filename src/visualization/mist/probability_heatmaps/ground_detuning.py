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

Delta_vals = np.linspace(0.75, 1.2, 80)
nbar_vals = np.linspace(5, 50, 81)
drive_amps = 2.0 * g * np.sqrt(nbar_vals)

transition_prob_matrix = np.zeros((len(Delta_vals), len(nbar_vals)))

def transmon_levels(n_levels, EC, EJ):
    return np.array([
        -EJ + np.sqrt(8*EJ*EC)*(n + 0.5) - (EC/12)*(6*n**2 + 6*n + 3)
        for n in range(n_levels)
    ])

def quantum_critical_n(omega_qs, EC, omega_r, g):
    n_crits = []
    for omega_q in omega_qs:
        EJ = (omega_q + EC)**2 / (8*EC)
        levels = transmon_levels(10, EC, EJ)
        mnc = float('inf')
        for k in range(10):
            for l in range(10):
                if abs(k-l) != 1:
                    continue
                gkl = g * np.sqrt(min(k, l) + 1)
                ωkl = levels[k] - levels[l]
                n_crit = abs((ωkl - omega_d)/(2*gkl))**2
                if k == 0 or l == 0:
                    mnc = min(mnc, n_crit)
        n_crits.append(mnc)
    return np.array(n_crits)

omega_qs_ana = omega_r + Delta_vals
n_crit_ana = quantum_critical_n(omega_qs_ana, EC, omega_r, g)

options = ft.Options(
    num_cpus=4,
    nsteps=1000,
    fit_range_fraction=1.0,
    overlap_cutoff=0.8,
    floquet_sampling_time_fraction=0.0,
    save_floquet_modes=True
)

for i, Delta in enumerate(Delta_vals):
    omega_q = omega_r + Delta
    EJ = (omega_q + EC) ** 2 / (8 * EC)
    num_states = 12
    qubit_params = dict(EJ=EJ, EC=EC, ng=0.2, ncut=31)
    tmon = scq.Transmon(**qubit_params, truncated_dim=num_states)
    hs = scq.HilbertSpace([tmon])
    hs.generate_lookup()
    evals = hs["evals"][0][:num_states]
    H0 = 2*np.pi * qt.Qobj(np.diag(evals - evals[0]))
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
    extent=[Delta_vals[0], Delta_vals[-1], nbar_vals[0], nbar_vals[-1]],
    aspect='auto',
    origin='lower',
    cmap='viridis')

cbar = plt.colorbar(im)
cbar.set_label('Transition Probability', fontsize=12)
plt.plot(Delta_vals, n_crit_ana, 'r--', linewidth=2, alpha=0.8, label='JC‐like')
plt.xlabel('Transmon–resonator detuning Δ (GHz)', fontsize=12)
plt.ylabel('Photon number $\overline{n}$', fontsize=12)
plt.title('Ground‐state leakage probability', fontsize=14)
# plt.yscale('log')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Maximum transition probability: {transition_prob_matrix.max():.3f}")
print(f"Drive amplitudes: {drive_amps[0]:.3f} – {drive_amps[-1]:.3f} GHz")
