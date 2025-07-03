import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
import scqubits as scq
import floquet as ft

g = 0.120 # GHz
EC = 0.220 # GHz
omega_r = 5.3 # GHz
omega_d = omega_r - 0.033

Delta_vals = np.linspace(0.75, 1.5, 50) 
nbar_vals = np.linspace(0, 180, 181)
drive_amps = 2.0 * g * np.sqrt(nbar_vals)

def transmon_levels(n_levels, EC, EJ):
    return np.array([
        -EJ + np.sqrt(8 * EJ * EC) * (n + 0.5) - (EC / 12) * (6 * n**2 + 6 * n + 3)
        for n in range(n_levels)
    ])

def quantum_critical_n(omega_qs, EC, omega_r, g):
    n_crits = []
    for omega_q in omega_qs:
        EJ = (omega_q + EC)**2 / (8 * EC)
        total_ns = 10
        e_levels = transmon_levels(total_ns, EC, EJ)
        min_n_crit = float('inf')
        for k in range(total_ns):
            for l in range(total_ns):
                if abs(k - l) != 1:
                    continue
                g_kl = g * np.sqrt(min(k, l) + 1)
                omega_kl = e_levels[k] - e_levels[l]
                n_crit = abs((omega_kl - omega_r) / (2 * g_kl))**2
                if k == 0 or l == 0:
                    min_n_crit = min(min_n_crit, n_crit)
        n_crits.append(min_n_crit)
    return np.array(n_crits)

omega_qs_analytical = omega_r + Delta_vals
n_crit_analytical = quantum_critical_n(omega_qs_analytical, EC, omega_r, g)

crit_nbar_numerical = np.full_like(Delta_vals, np.nan, dtype=float)

options = ft.Options(
    num_cpus = 4,
    nsteps = 1000,
    fit_range_fraction = 1.0,
    overlap_cutoff = 0.8,
    floquet_sampling_time_fraction = 0.0,
    save_floquet_modes = True
)

for i, Delta in enumerate(Delta_vals):
    omega_q = omega_r + Delta 
    EJ = (omega_q + EC)**2 / (8 * EC)

    num_states = 12
    qubit_params = dict(EJ=EJ, EC=EC, ng=0.2, ncut=31)
    tmon = scq.Transmon(**qubit_params, truncated_dim=num_states)
    hs = scq.HilbertSpace([tmon])
    hs.generate_lookup()

    evals = hs["evals"][0][:num_states] 
    H0 = 2.0 * np.pi * qt.Qobj(np.diag(evals - evals[0])) 
    H1 = hs.op_in_dressed_eigenbasis(tmon.n_operator)

    print(f"Delta = {Delta:.3f} GHz → ωq = {omega_q:.3f} GHz, EJ = {EJ:.3f} GHz")

    model = ft.Model(
        H0,
        H1,
        omega_d_values=np.array([2.0 * np.pi * omega_d]),
        drive_amplitudes=2.0 * np.pi * drive_amps
    )

    analysis = ft.FloquetAnalysis(model, state_indices=list(range(num_states)), options=options)

    try:
        data = analysis.run()
    except Exception as e:
        print(f"Failed w/ Delta = {Delta:.3f} GHz: {e}")
        continue

    psi0 = data["floquet_modes"][0][:, 0, :] # middle index chooses the energy level 
    levels = np.arange(psi0.shape[1]) 
    avg_levels = np.sum(np.abs(psi0)**2 * levels, axis=1)

    idx = np.where(avg_levels >= 2.0)[0] # threshold for population

    if idx.size:
        crit_nbar_numerical[i] = nbar_vals[idx[0]]
    else:
        print(f"No N >= 2, Delta = {Delta:.3f} GHz")

plt.figure(figsize=(10, 6))
plt.plot(Delta_vals, n_crit_analytical, '--', label='Analytical (JC-like)', linewidth=2, alpha=0.8)

mask = ~np.isnan(crit_nbar_numerical)
plt.plot(Delta_vals[mask], crit_nbar_numerical[mask], 'o-', label='Numerical (Floquet)', linewidth=2, markersize=6)

plt.xlabel('Transmon–resonator detuning Δ (GHz)')
plt.ylabel('Critical photon number $\\bar{n}_{\\rm crit}$')
plt.yscale('log')
plt.title('Critical photon number vs. detuning')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Drive amplitudes: {drive_amps[0]:.3f} – {drive_amps[-1]:.3f} GHz")
