import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
import scqubits as scq
import floquet as ft
import sys
from pathlib import Path

conversion_path = Path(__file__).resolve().parents[3] / "core"
sys.path.insert(0, str(conversion_path))

from conversion import(
    solve_EC,
    get_EJ
)

g = 0.120
EC = 0.220
omega_r = 5.0
omega_d = 7.0535 # omega_r - 0.033

Delta = 1.2
omega_q = omega_r + Delta
EJ = (omega_q + EC)**2 / (8 * EC)

num_states = 20       
qubit_params = dict(EJ=EJ, EC=EC, ng=0.2, ncut=41) 
tmon = scq.Transmon(**qubit_params, truncated_dim=num_states)

hilbert_space = scq.HilbertSpace([tmon])
hilbert_space.generate_lookup()

evals = hilbert_space["evals"][0][:num_states]
H0 = 2.0*np.pi * qt.Qobj(np.diag(evals - evals[0]))     
H1 = hilbert_space.op_in_dressed_eigenbasis(tmon.n_operator)

# omega_q = (evals[1] - evals[0]) * 2.0*np.pi               
# omega_d = 2.0 * np.pi * 6.76                               
# Delta = omega_q - omega_d           

nbar_vals = np.linspace(0, 180, 181)
drive_amps = 2.0 * g * np.sqrt(nbar_vals)
drive_amps = drive_amps[:, None]

# floquet sim
model = ft.Model(H0, H1,
    omega_d_values=np.array([2*np.pi * omega_d]),   
    drive_amplitudes=2*np.pi * drive_amps)

options = ft.Options(
    num_cpus = 4,        
    nsteps = 10000,   
    fit_range_fraction = 1.0,      
    overlap_cutoff = 0.8,      
    floquet_sampling_time_fraction = 0.0,   
    save_floquet_modes = True)

floquet_analysis = ft.FloquetAnalysis(model,
    state_indices = list(range(num_states)),
    options = options)

data = floquet_analysis.run()
E_q = data["quasienergies"][0, :, :]               

fold = lambda x: ((x + omega_d/2) % omega_d) - omega_d/2
E_fold = fold(E_q)

num_plotted_states = 12
plt.figure(figsize=(9,6))

# coloring
# highlighted_states = [0, 1, 7, 10, 11]
# highlighted_colors = ['red', 'blue', 'green', 'orange', 'purple']

for idx in range(num_plotted_states):
    # if idx in highlighted_states:
    #     color_idx = highlighted_states.index(idx)
    #     color = highlighted_colors[color_idx]
    # else:
    #     color = 'lightgray'
    
    plt.plot(nbar_vals,
             E_fold[:, idx] / omega_d,   
             lw=1,
             label=rf'$|{idx}\rangle$')
    
# colors = plt.cm.tab20(np.linspace(0, 1, num_plotted_states))

plt.xlabel(r"Resonator photon number $\bar{n}_r$")
plt.ylabel('Floquet quasienergies')
plt.title('Floquet Quasienergies vs. Resonator Photon Number')
plt.xlim(nbar_vals[0], nbar_vals[-1])
plt.legend(loc='upper right', ncol=2, fontsize=10)
plt.tight_layout()
plt.show()

print(EJ)