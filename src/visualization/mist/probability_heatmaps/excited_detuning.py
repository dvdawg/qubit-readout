import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import qutip as qt
import scqubits as scq
import floquet as ft

# system parameters
g       = 0.120         # coupling strength (GHz)
EC      = 0.220         # charging energy (GHz)
omega_r = 5.7           # resonator frequency (GHz)
omega_d = omega_r - 0.033  # drive frequency (GHz)

# sweep parameters
Delta_vals = np.linspace(0.75, 1.5, 20)
nbar_vals  = np.linspace(0, 180, 91)
drive_amps = 2.0 * g * np.sqrt(nbar_vals) # Effective Rabi freq ~ 2*g*sqrt(n)

# storage for the heatmap
leakage_prob_matrix = np.zeros((len(Delta_vals), len(nbar_vals)))

# analytical JC‐like critical photon numbers (for overlay)
def transmon_levels(n_levels, EC, EJ):
    """Standard anharmonic oscillator energy levels for a transmon."""
    return np.array([
        -EJ + np.sqrt(8*EJ*EC)*(n + 0.5)
        - (EC/12)*(6*n**2 + 6*n + 3)
        for n in range(n_levels)
    ])

def quantum_critical_n_excited(omega_qs, EC, omega_d_ana, g):
    """
    Calculates the critical photon number for leakage from the first excited state |1>.
    It considers transitions to the two adjacent levels, |0> and |2>, and
    returns the minimum n_crit required for either to occur.
    """
    n_crits = []
    for omega_q in omega_qs:
        EJ = (omega_q + EC)**2 / (8*EC)
        levels = transmon_levels(3, EC, EJ)

        # Transition 1 -> 0
        omega_10 = levels[1] - levels[0]
        g_10 = g * np.sqrt(1)
        detuning_10 = omega_10 - omega_d_ana
        n_crit_10 = (detuning_10 / (2 * g_10))**2

        # Transition 1 -> 2
        omega_21 = levels[2] - levels[1]
        g_21 = g * np.sqrt(2)
        detuning_21 = omega_21 - omega_d_ana
        n_crit_21 = (detuning_21 / (2 * g_21))**2

        n_crits.append(min(n_crit_10, n_crit_21))
        
    return np.array(n_crits)

omega_qs_ana = omega_r + Delta_vals
n_crit_ana   = quantum_critical_n_excited(omega_qs_ana, EC, omega_d, g)

# Floquet options
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
    EJ      = (omega_q + EC)**2 / (8*EC)

    num_states   = 12
    qubit_params = dict(EJ=EJ, EC=EC, ng=0.2, ncut=31)
    tmon = scq.Transmon(**qubit_params, truncated_dim=num_states)
    hs   = scq.HilbertSpace([tmon])
    hs.generate_lookup()

    evals = hs["evals"][0][:num_states]
    H0    = 2 * np.pi * qt.Qobj(np.diag(evals - evals[0]))
    H1    = hs.op_in_dressed_eigenbasis(tmon.n_operator)

    print(f"Δ = {Delta:.3f} GHz → ωq = {omega_q:.3f}, EJ = {EJ:.3f}")

    model = ft.Model(
        H0,
        H1,
        omega_d_values  = np.array([2 * np.pi * omega_d]),
        drive_amplitudes = 2 * np.pi * drive_amps
    )
    fa       = ft.FloquetAnalysis(model, state_indices=list(range(num_states)), options=options)
    data     = fa.run()
    
    # FIX: Check for an empty numpy array by using the .size attribute.
    if data["floquet_modes"].size == 0:
        print(f"Warning: Floquet analysis failed for Δ = {Delta:.3f} and returned no modes. Skipping.")
        # Fill this row with NaNs so it doesn't show up on the plot
        leakage_prob_matrix[i, :] = np.nan 
        continue
        
    floq_modes = data["floquet_modes"][0]

    for j in range(len(nbar_vals)):
        overlaps = np.abs(floq_modes[j, :, 1])**2
        
        if len(overlaps) == 0:
            leakage_prob_matrix[i, j] = np.nan
            continue
        
        excited_branch_idx = np.argmax(overlaps)
        leakage_prob_matrix[i, j] = 1 - overlaps[excited_branch_idx]

# plot the heatmap
plt.figure(figsize=(12, 8))
im = plt.imshow(
    leakage_prob_matrix.T,
    extent=[Delta_vals[0], Delta_vals[-1], nbar_vals[0], nbar_vals[-1]],
    aspect='auto',
    origin='lower',
    cmap='viridis',
)
cbar = plt.colorbar(im)
cbar.set_label('Transition probability', fontsize=12)

plt.plot(Delta_vals, n_crit_ana, 'r--', linewidth=2.5, alpha=0.9, label='Analytical Estimate (from |1⟩)')

plt.xlabel('Transmon–resonator detuning Δ (GHz)', fontsize=12)
plt.ylabel('Photon number $\overline{n}$', fontsize=12)
plt.title('Excited-state Leakage probability', fontsize=14)
plt.ylim(nbar_vals[0], nbar_vals[-1])
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Maximum leakage probability from |1>: {np.nanmax(leakage_prob_matrix):.3f}")
print(f"Drive amplitudes: {drive_amps[0]:.3f} – {drive_amps[-1]:.3f} GHz")