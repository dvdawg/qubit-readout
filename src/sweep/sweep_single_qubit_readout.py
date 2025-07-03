import numpy as np
import pandas as pd
from pathlib import Path
import sys

core_path = Path(__file__).resolve().parents[1] / "core"
sys.path.insert(0, str(core_path))
from single_qubit import compute_chi, integrated_snr

import qutip as qt
import scqubits as scq
import floquet as ft

photon_numbers = np.arange(20, 101, 1)
kappa = 0.01 # GHz
g = 0.12 # GHz
delta_resonator = 0.0 # GHz
eta = 0.15

EC = 0.220  # GHz
omega_r = 5.3 # GHz
omega_d = omega_r - 0.033  # GHz

options = ft.Options(
    num_cpus = 4,
    nsteps = 1000,
    fit_range_fraction = 1.0,
     floquet_sampling_time_fraction = 0.0,
    save_floquet_modes = True
)

num_states = 12  # level truncation

# sweep setup
delta_r_min, delta_r_max, ndelta_r = 0.05, 5.0, 100
delta_r_vals = np.linspace(delta_r_min, delta_r_max, ndelta_r)
tau_factors  = np.linspace(3, 5, 10)

results = []

for tau_factor in tau_factors:
    tau = tau_factor / kappa # ns
    for n_target in photon_numbers:
        for delta_r in delta_r_vals:
            chi = compute_chi(g, delta_r)
            Omega_q_mag = 2 * chi * np.sqrt(n_target)
            params = [Omega_q_mag, 0.0]

            snr = integrated_snr(-1, 1, params, chi, delta_r, kappa, g, delta_resonator, tau, eta)
            passes_snr = (snr >= 5)

            Delta_q = delta_r
            omega_q = omega_r + Delta_q
            EJ = (omega_q + EC) ** 2 / (8 * EC)

            tmon = scq.Transmon(EJ=EJ, EC=EC, ng=0.2, ncut=31, truncated_dim=num_states)
            hs = scq.HilbertSpace([tmon])
            hs.generate_lookup()
            evals= hs["evals"][0][:num_states]

            H0 = 2 * np.pi * qt.Qobj(np.diag(evals - evals[0]))
            H1 = hs.op_in_dressed_eigenbasis(tmon.n_operator)

            drive_amp = Omega_q_mag
            model = ft.Model(
                H0,
                H1,
                omega_d_values   = np.array([2*np.pi*omega_d]),
                drive_amplitudes= 2*np.pi*np.array([drive_amp])
            )

            analysis = ft.FloquetAnalysis(
                model,
                state_indices=list(range(num_states)),
                options=options
            )

            try:
                data = analysis.run()
                fm = data["floquet_modes"][0]
                psi_nt = fm[:, :, 0]
                levels = np.arange(psi_nt.shape[1])
                avg_lvl = np.sum(np.abs(psi_nt)**2 * levels, axis=1)
                mist_flag = np.any(avg_lvl >= 2.0)
                passes_mist = not mist_flag
            except Exception as e:
                passes_mist = False

            passes_check = passes_snr and passes_mist

            results.append({
                'tau_factor':   tau_factor,
                'tau_ns':       tau,
                'n_target':     n_target,
                'delta_r':      delta_r,
                'snr':          snr,
                'passes_snr':   passes_snr,
                'passes_mist':  passes_mist,
                'passes_check': passes_check,
            })

            if passes_check:
                break

# save
df = pd.DataFrame(results)
df.to_csv('single_qubit_min_detuning_vs_photon_number.csv', index=False)
print("Done. Results in single_qubit_min_detuning_vs_photon_number.csv")
