import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
import scqubits as scq
import floquet as ft

# --- fixed physical parameters ------------------------------------------------
g       = 0.120               # coupling (GHz)
EC      = 0.220               # charging energy (GHz)
omega_r = 5.700               # bare resonator (GHz)
omega_d = omega_r - 0.033     # drive (GHz)   ← fixed

# --- choose ONE qubit detuning -------------------------------------------------
Delta   = 0.81                # GHz; you may change this to inspect other Δ
omega_q = omega_r + Delta
EJ      = (omega_q + EC)**2 / (8*EC)          # transmon EJ (GHz)

# --- build truncated transmon --------------------------------------------------
num_states   = 20
tmon         = scq.Transmon(EJ=EJ, EC=EC, ng=0.2, ncut=41,
                            truncated_dim=num_states)
hilbert      = scq.HilbertSpace([tmon])
hilbert.generate_lookup()

evals = hilbert["evals"][0][:num_states]          # dressed eigen-values
H0    = qt.Qobj(np.diag(evals - evals[0]))        # zero the ground energy
H1    = hilbert.op_in_dressed_eigenbasis(tmon.n_operator)

# --- sweep photon number through the drive amplitude --------------------------
nbar_vals   = np.linspace(0, 180, 181)
drive_amps  = 2.0 * g * np.sqrt(nbar_vals)        # Jaynes–Cummings relation
drive_amps  = drive_amps[:, None]                 # (N,1) so Floquet sees N amps

model = ft.Model(
    H0, H1,
    omega_d_values = np.array([omega_d]),
    drive_amplitudes = drive_amps
)

options = ft.Options(
    num_cpus                   = 4,
    nsteps                     = 10000,
    fit_range_fraction         = 1.0,
    overlap_cutoff             = 0.8,
    floquet_sampling_time_fraction = 0.0,
    save_floquet_modes         = True
)

floquet = ft.FloquetAnalysis(
    model,
    state_indices = list(range(num_states)),
    options       = options
)

data = floquet.run()
E_q  = data["quasienergies"][0]                   # shape (N, num_states)

fold = lambda x: ((x + omega_d/2) % omega_d) - omega_d/2
E_fold = fold(E_q)

num_plot = 12
plt.figure(figsize=(9,6))
for k in range(num_plot):
    plt.plot(nbar_vals, E_fold[:, k] / omega_d, lw=1, label=f"|{k}⟩")
plt.xlabel(r"Resonator photon number $\bar{n}_r$")
plt.ylabel(r"Folded quasienergy $E/\hbar\omega_d$")
plt.title(fr"Floquet branches at fixed detuning $\Delta={Delta}$ GHz")
plt.xlim(nbar_vals[0], nbar_vals[-1])
plt.legend(ncol=2, fontsize=9)
plt.tight_layout()
plt.show()
