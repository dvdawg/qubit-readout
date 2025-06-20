import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
import scqubits as scq
import floquet as ft

num_states = 20       
qubit_params = dict(EJ=20.0, EC=0.20, ng=0.25, ncut=41) 
tmon = scq.Transmon(**qubit_params, truncated_dim=num_states)

hilbert_space = scq.HilbertSpace([tmon])
hilbert_space.generate_lookup()

evals = hilbert_space["evals"][0][:num_states]
H0 = 2.0*np.pi * qt.Qobj(np.diag(evals - evals[0]))     
H1 = hilbert_space.op_in_dressed_eigenbasis(tmon.n_operator)

omega_q = (evals[1]-evals[0]) * 2.0*np.pi               
omega_d = 2.0 * np.pi * 7.40                               
Delta = omega_q - omega_d                               

g = 0.12 * 2.0 * np.pi                                     

nbar_vals = np.linspace(0, 180, 181)
drive_amps = 2.0 * g * np.sqrt(nbar_vals)
drive_amps = drive_amps[:, None]                        
omega_d_vals = np.array([omega_d])

model = ft.Model(H0, H1,
                 omega_d_values      = omega_d_vals,
                 drive_amplitudes    = drive_amps)

options = ft.Options(
    num_cpus                        = 4,        
    nsteps                          = 30_000,   
    fit_range_fraction              = 1.0,      
    overlap_cutoff                  = 0.0,      
    floquet_sampling_time_fraction = 0.0,   
    save_floquet_modes              = True)

floquet_analysis = ft.FloquetAnalysis(model,
                                      state_indices = list(range(10)),
                                      options       = options)

data = floquet_analysis.run()
E_q = data["quasienergies"][0, :, :]               

fold = lambda x: ((x + omega_d/2) % omega_d) - omega_d/2
E_fold = fold(E_q)

plt.figure(figsize=(9,6))
num_plotted_states = 9
for idx in range(num_plotted_states):
    plt.plot(nbar_vals,
             E_fold[:, idx] / (2.0 * np.pi),   
             lw=1,
             label=rf'$|{idx}\rangle$')

plt.xlabel(r"Resonator photon number $\bar{n}_r$")
plt.ylabel('Floquet quasienergies')
plt.title('Floquet Quasienergies vs. Resonator Photon Number')
plt.xlim(nbar_vals[0], nbar_vals[-1])
plt.legend(loc='upper right', ncol=2, fontsize=10)
plt.tight_layout()
plt.show()
