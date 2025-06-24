import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
import scqubits as scq
import floquet as ft

g = 0.120 # GHz
EC = 0.220 # GHz
omega_r = 5.3 # GHz
omega_d = omega_r - 0.033

Delta_vals = np.linspace(0.75, 1.5, 31)
nbar_vals = np.linspace(0, 180, 181)

drive_amps = 2.0 * g * np.sqrt(nbar_vals)
drive_amplitudes_rad = 2.0 * np.pi * drive_amps

options = ft.Options(
    num_cpus=4,
    nsteps=1000,
    fit_range_fraction=1.0,
    overlap_cutoff=0.0,
    floquet_sampling_time_fraction=0.0,
    save_floquet_modes=True
)

def transmon_levels(n_levels, EC, EJ):
    return np.array([
        -EJ + np.sqrt(8 * EJ * EC) * (n + 0.5) - (EC / 12.0) * (6 * n**2 + 6 * n + 3)
        for n in range(n_levels)
    ])

def quantum_critical_n_JC(omega_qs, EC, omega_r, g):
    n_crits = []
    total_ns = 10
    for omega_q in omega_qs:
        EJ = (omega_q + EC)**2 / (8.0 * EC)
        e_levels = transmon_levels(total_ns, EC, EJ)
        min_n_crit = np.inf
        for k in range(total_ns):
            for l in range(total_ns):
                if abs(k - l) != 1:
                    continue
                g_kl = g * np.sqrt(min(k, l) + 1)
                omega_kl = e_levels[k] - e_levels[l]
                n_crit = abs((omega_kl - omega_r) / (2.0 * g_kl))**2
                if n_crit < min_n_crit:
                    min_n_crit = n_crit
        n_crits.append(min_n_crit)
    return np.array(n_crits)

omega_qs_analytical = omega_r + Delta_vals
n_crit_analytical = quantum_critical_n_JC(omega_qs_analytical, EC, omega_r, g)

crit_nbar_numerical = np.full_like(Delta_vals, np.nan, dtype=float)

print("Running Floquet simulations...")
for i, Delta in enumerate(Delta_vals):
    omega_q = omega_r + Delta
    EJ = (omega_q + EC)**2 / (8.0 * EC)

    num_states = 8
    qubit_params = dict(EJ=EJ, EC=EC, ng=0.2, ncut=41)
    tmon = scq.Transmon(**qubit_params, truncated_dim=num_states)
    hs = scq.HilbertSpace([tmon])
    hs.generate_lookup()

    evals = hs["evals"][0][:num_states]
    H0 = 2.0 * np.pi * qt.Qobj(np.diag(evals - evals[0]))
    H1 = hs.op_in_dressed_eigenbasis(tmon.n_operator)

    omega_d_rad = 2.0 * np.pi * omega_d

    print(f"  Δ = {Delta:.3f} GHz → ω_q = {omega_q:.3f} GHz, EJ = {EJ:.3f} GHz ...", end='', flush=True)

    model = ft.Model(
        H0,
        H1,
        omega_d_values=np.array([omega_d_rad]),
        drive_amplitudes=drive_amplitudes_rad
    )
    analysis = ft.FloquetAnalysis(model,
                                  state_indices=list(range(num_states)),
                                  options=options)
    try:
        data = analysis.run()
    except Exception as e:
        print(f" Failed: {e}")
        continue

    psi0 = data["floquet_modes"][0]
    try:
        psi0_vecs = psi0[:, 0, :]
    except Exception:
        psi0_vecs = psi0

    levels = np.arange(psi0_vecs.shape[0])
    avg_levels = np.sum(np.abs(psi0_vecs)**2 * levels[:, None], axis=0)

    idx = np.where(avg_levels >= 2.0)[0]
    if idx.size > 0:
        crit_nbar_numerical[i] = nbar_vals[idx[0]]
        print(f" critical n̄ ≈ {crit_nbar_numerical[i]:.1f}")
    else:
        print(" no ⟨n⟩≥2 found")

plt.figure(figsize=(8, 5))
plt.plot(Delta_vals, n_crit_analytical, '--', label='Analytical JC-like', linewidth=2, alpha=0.8)
mask = ~np.isnan(crit_nbar_numerical)
plt.plot(Delta_vals[mask], crit_nbar_numerical[mask], 'o-', label='Numerical (Floquet)', linewidth=2, markersize=5)
plt.xlabel('Transmon–resonator detuning Δ (GHz)')
plt.ylabel('Critical photon number $\\bar{n}_\\mathrm{crit}$')
plt.yscale('log')
plt.title('Critical photon number vs. detuning (Floquet vs JC-like estimate)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\nParameters used:")
print(f"  g = {g:.3f} GHz")
print(f"  E_C = {EC:.3f} GHz")
print(f"  ω_r = {omega_r:.3f} GHz, ω_d = {omega_d:.3f} GHz")
print(f"  Detuning sweep: {Delta_vals[0]:.2f} → {Delta_vals[-1]:.2f} GHz")
print(f"  Photon numbers tested: {nbar_vals[0]:.1f} → {nbar_vals[-1]:.1f}")
