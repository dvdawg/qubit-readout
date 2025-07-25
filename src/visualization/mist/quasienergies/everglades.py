import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import qutip as qt
import scqubits as scq
import floquet as ft

g = 0.120
EC = 0.220
# omega_r = 5.7
omega_q = 5.7 # 6.289177
# omega_d = omega_r - 0.033

Delta_vals = np.linspace(0.75, 1.5, 50)
nbar_vals = np.linspace(5, 180, 176)
drive_amps = 2.0 * g * np.sqrt(nbar_vals)
transition_prob_matrix = np.zeros((len(Delta_vals), len(nbar_vals)))

def transmon_levels(n_levels,EC,EJ):
    return np.array([-EJ+np.sqrt(8*EJ*EC)*(n+0.5)-(EC/12)*(6*n**2+6*n+3) for n in range(n_levels)])

def quantum_critical_n(omega_qs, omega_r, g):
    return ((omega_qs-omega_r)/(2*g))**2

# omega_qs = omega_r + Delta_vals
# n_crit_ana = quantum_critical_n(omega_qs, omega_r, g)

options = ft.Options(num_cpus = 8, nsteps = 1000, fit_range_fraction = 1.0, overlap_cutoff = 0.8, floquet_sampling_time_fraction = 0.0, save_floquet_modes = True)

transmon_dim = 12
osc_dim = 10 # int(nbar_vals.max())

Delta = 0.9
# omega_q = omega_r + Delta
omega_r = omega_q - Delta
omega_d = omega_r
EJ = (omega_q + EC)**2/(8*EC)
tmon = scq.Transmon(EJ=EJ, EC=EC, ng=0.2, ncut=25, truncated_dim=transmon_dim)
reson = scq.Oscillator(E_osc=omega_r * 2.0 * np.pi, truncated_dim=osc_dim)
hs = scq.HilbertSpace([tmon,reson])
hs.generate_lookup()
# evals = hs["evals"][0][:transmon_dim]
# H0 = 2 * np.pi * qt.Qobj(np.diag(evals - evals[0]))
H0 = hs.hamiltonian()
n_op = hs.op_in_dressed_eigenbasis(tmon.n_operator)
n_op.dims = H0.dims
model = ft.Model(H0, n_op, omega_d_values=np.array([omega_d]), drive_amplitudes=drive_amps)
fa = ft.FloquetAnalysis(model, state_indices=list(range(transmon_dim)), options=options)

data = fa.run()
E_q = data["quasienergies"][0]

fold = lambda x: ((x + omega_d / 2) % omega_d) - omega_d / 2
E_z = fold(E_q)

E_branch = np.zeros_like(E_z)
E_branch[0] = E_z[0]

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
