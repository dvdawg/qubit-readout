import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
import scqubits as scq
import floquet as ft


g = 0.11
EC = 0.28
omega_r = 6.2
omega_d = omega_r
kappa = 0.1
ng = 0.2

Delta_vals = np.linspace(0.75, 1.5, 31)
nbar_max = 200
nbar_vals = np.arange(0, nbar_max + 1)


drive_amps = 2.0 * g * np.sqrt(nbar_vals)  # GHz

def get_transmon_hamiltonian(EJ, EC, ng, num_states=12):
    qubit_params = dict(EJ=EJ, EC=EC, ng=ng, ncut=41)
    tmon = scq.Transmon(**qubit_params, truncated_dim=num_states)
    hs = scq.HilbertSpace([tmon])
    hs.generate_lookup()
    
    evals = hs["evals"][0][:num_states]
    evecs = hs["evecs"][0][:num_states]
    
    H0 = 2.0 * np.pi * qt.Qobj(np.diag(evals - evals[0]))
    
    H1 = hs.op_in_dressed_eigenbasis(tmon.n_operator)
    
    return H0, H1, evals, evecs

def find_critical_photon_number(Delta, drive_amps, options):
    omega_q = omega_r - Delta
    EJ = (omega_q + EC)**2 / (8 * EC)
    
    print(f"Delta = {Delta:.3f} GHz → ωq = {omega_q:.3f} GHz, EJ = {EJ:.3f} GHz")
    
    H0, H1, evals, evecs = get_transmon_hamiltonian(EJ, EC, ng)
    num_states = H0.shape[0]
    
    model = ft.Model(
        H0,
        H1,
        omega_d_values=np.array([2 * np.pi * omega_d]),
        drive_amplitudes=2.0 * np.pi * drive_amps
    )
    
    analysis = ft.FloquetAnalysis(
        model, 
        state_indices=list(range(num_states)), 
        options=options
    )
    
    try:
        data = analysis.run()
    except Exception as e:
        print(f"Failed at Delta = {Delta:.3f} GHz: {e}")
        return np.nan
    
    psi0 = data["floquet_modes"][0][:, 0, :]
    
    energy_levels = np.arange(num_states)
    avg_energy_levels = np.zeros(len(drive_amps))
    
    for i in range(len(drive_amps)):
        probs = np.abs(psi0[i, :])**2
        avg_energy_levels[i] = np.sum(probs * energy_levels)
    
    idx = np.where(avg_energy_levels >= 2.0)[0]
    if idx.size > 0:
        return nbar_vals[idx[0]]
    else:
        print(f"No ⟨N⟩ ≥ 2 found for Delta = {Delta:.3f} GHz")
        return np.nan

options = ft.Options(
    num_cpus=4,
    nsteps=2000,
    fit_range_fraction=0.8,
    overlap_cutoff=0.1,
    floquet_sampling_time_fraction=0.1,
    save_floquet_modes=True
)

n_crit_values = np.full_like(Delta_vals, np.nan, dtype=float)

for i, Delta in enumerate(Delta_vals):
    n_crit_values[i] = find_critical_photon_number(Delta, drive_amps, options)
    print(f"Delta = {Delta:.3f} GHz → n_crit = {n_crit_values[i]:.2f}")

def analytical_critical_n(Delta, g, EC):
    omega_q = omega_r - Delta
    EJ = (omega_q + EC)**2 / (8 * EC)
    
    alpha = -EC
    
    omega_01 = omega_q
    omega_12 = omega_q + alpha
    
    g_01 = g
    g_12 = g * np.sqrt(2)
    
    n_crit_01 = abs((omega_01 - omega_r) / (2 * g_01))**2
    n_crit_12 = abs((omega_12 - omega_r) / (2 * g_12))**2
    
    return min(n_crit_01, n_crit_12)

n_crit_analytical = np.array([analytical_critical_n(Delta, g, EC) for Delta in Delta_vals])

plt.figure(figsize=(12, 8))

plt.plot(Delta_vals, n_crit_analytical, '--', 
         label='Analytical (simplified)', linewidth=2, alpha=0.8, color='red')

mask = ~np.isnan(n_crit_values)
if np.any(mask):
    plt.plot(Delta_vals[mask], n_crit_values[mask], 'o-', 
             label='Numerical (Floquet)', linewidth=2, markersize=6, color='blue')

plt.xlabel('Detuning Δ (GHz)')
plt.ylabel('Critical photon number $n_{\\rm crit}$')
plt.yscale('log')
plt.title('Critical photon number vs. detuning\n(Average energy level = 2)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.figtext(0.02, 0.02, 
           f'Parameters: g = {g} GHz, EC = {EC} GHz, ωr = {omega_r} GHz, ng = {ng}',
           fontsize=10, ha='left')

plt.show()

print(f"\nSummary:")
print(f"Parameters: g = {g} GHz, EC = {EC} GHz, ωr = {omega_r} GHz")
print(f"Detuning range: {Delta_vals[0]:.2f} to {Delta_vals[-1]:.2f} GHz")
print(f"Successful calculations: {np.sum(mask)}/{len(Delta_vals)}")

if np.any(mask):
    print(f"Critical photon number range: {np.nanmin(n_crit_values):.2f} to {np.nanmax(n_crit_values):.2f}")