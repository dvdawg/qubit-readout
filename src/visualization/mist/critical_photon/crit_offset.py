import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
import scqubits as scq
import floquet as ft

g       = 0.120 
EC      = 0.220 
omega_r = 5.3 
omega_d = omega_r - 0.033
n_g     = 0.20   

Delta_vals = np.linspace(0.75, 1.5, 50)
nbar_vals  = np.linspace(0, 180, 181)
drive_amps = 2.0 * g * np.sqrt(nbar_vals)

def transmon_levels(n_levels, EC, EJ):
    return np.array([
        -EJ + np.sqrt(8 * EJ * EC) * (n + 0.5)
        - (EC / 12) * (6 * n**2 + 6 * n + 3)
        for n in range(n_levels)
    ])

def quantum_critical_n(omega_qs, EC, omega_r, g):
    n_crits = []
    for omega_q in omega_qs:
        EJ = (omega_q + EC)**2 / (8 * EC)
        e_lvls = transmon_levels(10, EC, EJ)
        min_n = np.min([
            abs((e_lvls[k]-e_lvls[l]-omega_r)/(2*g*np.sqrt(min(k,l)+1)))**2
            for k in range(10) for l in range(10)
            if abs(k-l)==1 and 0 in (k,l)
        ])
        n_crits.append(min_n)
    return np.array(n_crits)

omega_qs_analytical = omega_r + Delta_vals
n_crit_analytical   = quantum_critical_n(omega_qs_analytical, EC, omega_r, g)
crit_nbar_numerical = np.full_like(Delta_vals, np.nan)

options = ft.Options(
    num_cpus                       = 4,
    nsteps                         = 1000,
    fit_range_fraction             = 1.0,
    overlap_cutoff                 = 0.8,
    floquet_sampling_time_fraction = 0.0,
    save_floquet_modes             = True
)

for i, Delta in enumerate(Delta_vals):
    omega_q = omega_r + Delta
    EJ      = (omega_q + EC)**2 / (8 * EC)

    num_states   = 12
    qubit_params = dict(EJ=EJ, EC=EC, ng=n_g, ncut=31)
    tmon         = scq.Transmon(**qubit_params, truncated_dim=num_states)
    hs           = scq.HilbertSpace([tmon])
    hs.generate_lookup()

    evals = hs["evals"][0][:num_states]
    H0    = 2*np.pi * qt.Qobj(np.diag(evals - evals[0]))

    n_op_dressed = hs.op_in_dressed_eigenbasis(tmon.n_operator)  # Qobj
    H1           = n_op_dressed - n_g * qt.qeye(num_states)

    model = ft.Model(
        H0,
        H1,
        omega_d_values   = [2*np.pi * omega_d],
        drive_amplitudes = 2*np.pi * drive_amps
    )

    try:
        data = ft.FloquetAnalysis(
            model,
            state_indices = list(range(num_states)),
            options       = options
        ).run()
    except Exception as e:
        print(f"Δ={Delta:.3f} GHz failed: {e}")
        continue

    psi0       = data["floquet_modes"][0][:, 1, :]
    levels     = np.arange(psi0.shape[1])
    avg_levels = np.sum(np.abs(psi0)**2 * levels, axis=1)

    idx = np.where(avg_levels >= 3.0)[0]
    if idx.size:
        crit_nbar_numerical[i] = nbar_vals[idx[0]]
    else:
        print(f"No ⟨N⟩≥3 for Δ={Delta:.3f} GHz")

plt.figure(figsize=(10,6))
plt.plot(Delta_vals, n_crit_analytical, '--', label='Analytical (JC-like)', lw=2, alpha=.8)
mask = ~np.isnan(crit_nbar_numerical)
plt.plot(Delta_vals[mask], crit_nbar_numerical[mask], 'o-', label='Numerical (Floquet)', lw=2, ms=6)
plt.yscale('log')
plt.xlabel('Detuning Δ (GHz)')
plt.ylabel('Critical photon number $\\bar{n}_{\\mathrm{crit}}$')
plt.title('Critical photon number vs. detuning (with $n_g$ offset)')
plt.grid(alpha=.3)
plt.legend()
plt.tight_layout()
plt.show()

print(f"Drive amplitudes span {drive_amps[0]:.3f} – {drive_amps[-1]:.3f} GHz")
