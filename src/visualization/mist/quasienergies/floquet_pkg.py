import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
import scqubits as scq
import floquet as ft
from pathlib import Path
import sys

conversion_path = Path(__file__).resolve().parents[3] / "core"
sys.path.insert(0, str(conversion_path))
from conversion import solve_EC, get_EJ

g = 0.120
EC = 0.220
omega_r = 5.7
Delta = 0.81
omega_d = omega_r - 0.033
omega_q = omega_r + Delta
EJ = (omega_q + EC) ** 2 / (8 * EC)

num_states = 20
tmon = scq.Transmon(EJ=EJ, EC=EC, ng=0.2, ncut=41, truncated_dim=num_states)
hilbert = scq.HilbertSpace([tmon])
hilbert.generate_lookup()

evals = hilbert["evals"][0][:num_states]
H0 = qt.Qobj(np.diag(evals - evals[0]))
H1 = hilbert.op_in_dressed_eigenbasis(tmon.n_operator)

nbar_vals = np.linspace(0, 180, 181)
drive_amps = 2 * g * np.sqrt(nbar_vals)[:, None]

model = ft.Model(H0, H1, omega_d_values=np.array([omega_d]), drive_amplitudes=drive_amps)

options = ft.Options(
    num_cpus=4,
    nsteps=10000,
    fit_range_fraction=1.0,
    overlap_cutoff=0.8,
    floquet_sampling_time_fraction=0.0,
    save_floquet_modes=True
)

analysis = ft.FloquetAnalysis(model, state_indices=list(range(num_states)), options=options)
out = analysis.run()
E_q = out["quasienergies"][0]

fold = lambda x: ((x + omega_d / 2) % omega_d) - omega_d / 2
E_z = fold(E_q)

E_branch = np.zeros_like(E_z)
E_branch[0] = E_z[0]

# for i in range(1, len(nbar_vals)):
#     prev, cur = E_branch[i - 1], E_z[i].copy()
#     taken = np.zeros_like(cur, dtype=bool)
#     for k, p in enumerate(prev):
#         idx = np.argmin(np.abs(cur - p) + 1e6 * taken)
#         E_branch[i, k] = cur[idx]
#         taken[idx] = True
#     if np.any(~taken):
#         E_branch[i, ~taken] = cur[~taken]

n_show = 12
plt.figure(figsize=(9, 6))
for m in range(n_show):
    plt.plot(nbar_vals, E_z[:, m] / omega_d, lw=1, label=fr'$|{m}\rangle$')
plt.xlabel(r'Resonator photon number $\bar{n}_r$')
plt.ylabel(r'Floquet quasienergies')
plt.title(fr'Floquet Quasienergies vs. Resonator Photon Number')
plt.xlim(nbar_vals[0], nbar_vals[-1])
plt.legend(ncol=2, fontsize=9)
plt.tight_layout()
plt.show()
