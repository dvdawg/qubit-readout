import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
import scqubits as scq
import floquet as ft

omega_r_GHz = 7.0
E_C = 0.28
ng = 0.25
g_GHz = 0.11 
N_levels = 10
ncut = 41
Delta_min = 0.5
Delta_max = 2.0
N_Delta = 20
Delta_vals = np.linspace(Delta_min, Delta_max, N_Delta)
nbar_min = 0.1
nbar_max = 500
N_nbar = 200
nbar_vals = np.unique(np.concatenate([
    np.linspace(nbar_min, 10, 100),
    np.logspace(np.log10(10), np.log10(nbar_max), 100)
]))
nbar_vals.sort()

ncrit_list = []

for Delta in Delta_vals:
    omega_q_GHz = omega_r_GHz + Delta
    E_J = ((omega_q_GHz + E_C) ** 2) / (8 * E_C)
    qubit_params = {"EJ": E_J, "EC": E_C, "ng": ng, "ncut": ncut}
    tmon = scq.Transmon(**qubit_params, truncated_dim=N_levels)
    hilbert_space = scq.HilbertSpace([tmon])
    hilbert_space.generate_lookup()
    evals = hilbert_space["evals"][0][:N_levels]
    H0 = 2.0 * np.pi * qt.Qobj(np.diag(evals - evals[0]))
    H1 = hilbert_space.op_in_dressed_eigenbasis(tmon.n_operator)
    omega_d = 2.0 * np.pi * omega_r_GHz

    A_vals_GHz = 2.0 * g_GHz * np.sqrt(nbar_vals)
    drive_amplitudes = 2.0 * np.pi * A_vals_GHz

    model = ft.Model(H0, H1, omega_d_values=[omega_d], drive_amplitudes=drive_amplitudes)
    floquet_analysis = ft.FloquetAnalysis(model, state_indices=[0])
    data_vals = floquet_analysis.run()

    if "avg_excitation" not in data_vals:
        print(f"Warning: no 'avg_excitation' in data_vals for Δ={Delta}. Keys: {list(data_vals.keys())}")
        ncrit_list.append(np.nan)
        continue

    pops = np.array(data_vals["avg_excitation"])
    pops_vs_nbar = pops.squeeze()
    idx = np.where(pops_vs_nbar >= 2.0)[0]
    if idx.size > 0:
        ncrit = nbar_vals[idx[0]]
    else:
        ncrit = np.nan
    ncrit_list.append(ncrit)

ncrit_array = np.array(ncrit_list)

plt.figure(figsize=(6,4))
plt.plot(Delta_vals, ncrit_array, marker='o', linestyle='-')
plt.yscale('log')
plt.xlabel('Detuning Δ = ω_q - ω_r (GHz)')
plt.ylabel('Critical photon number n₍crit₎')
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.tight_layout()
plt.show()
