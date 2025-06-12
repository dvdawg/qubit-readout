import numpy as np
import pandas as pd
from itertools import product
import os
import time

from pointer_sim import optimize_parameters, compute_chis, calculate_snr

output_dir = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(output_dir, exist_ok=True)

start_time = time.time()

tau = 200
target_snr = 5.0   # e.g. SNR≥6 for ≈99% single-shot fidelity

delta_r1_vals       = np.linspace(0.1, 1.0, 10)   # 100–1000 MHz
delta_r2_vals       = np.linspace(0.1, 1.0, 10)
g_1_vals            = np.linspace(0.05,0.2, 5)   # 50–200 MHz
g_2_vals            = np.linspace(0.05,0.2, 5)
kappa_vals          = np.linspace(0.001,0.02,5)   # 1–20 MHz
delta_resonator_vals= np.linspace(-0.05,0.05, 5)   # –50 to +50 MHz detuning

results = []
total = (len(delta_r1_vals) * len(delta_r2_vals) *
         len(g_1_vals)      * len(g_2_vals)      *
         len(kappa_vals)    * len(delta_resonator_vals))

pos = 0

for delta_r1, delta_r2, g_1, g_2, kappa, delta_resonator in product(
        delta_r1_vals, delta_r2_vals,
        g_1_vals, g_2_vals,
        kappa_vals, delta_resonator_vals):

    res = optimize_parameters(delta_r1, delta_r2, g_1, g_2, kappa, delta_resonator, tau)

    # 2) compute the true 200 ns integrated SNR
    chi_1, chi_2 = compute_chis(g_1, g_2, delta_r1, delta_r2)
    state_pairs = [
        ((-1, -1), (1,  1)),
        ((-1,  1), (1, -1)),
        ((-1, -1), (1, -1)),
        ((-1,  1), (1,  1))
    ]
    params = [res['Omega_q1_mag'], res['phi_q1'],
              res['Omega_q2_mag'], res['phi_q2']]

    int_snrs = [
        calculate_snr(s1, s2, params, chi_1, chi_2, delta_r1, delta_r2, kappa, g_1, g_2, delta_resonator, tau)
        for s1, s2 in state_pairs
    ]
    res['snr_200ns']  = np.min(int_snrs)
    res['meets_spec'] = res['snr_200ns'] >= target_snr

    results.append(res)

    print(f"δr1={delta_r1:.3f}, δr2={delta_r2:.3f}, g1={g_1:.3f}, g2={g_2:.3f}, κ={kappa:.3f}, δr={delta_resonator:.3f}")
    print(f"  → Min SNR(200 ns) = {res['snr_200ns']:.3f}  Meets spec? {res['meets_spec']}")
    print("  → Drives: Ωq1={:.3f}, φq1={:.3f}, Ωq2={:.3f}, φq2={:.3f}".format(
          res['Omega_q1_mag'], res['phi_q1'],
          res['Omega_q2_mag'], res['phi_q2']))
    print(f"Progress: {pos}/{total} ({(pos/total)*100:.1f}%)")
    elapsed_time = time.time() - start_time
    if pos > 0:
        avg_time_per_iter = elapsed_time / pos
        remaining_time = avg_time_per_iter * (total - pos)
        print(f"Est. time remaining: {remaining_time/60:.1f} minutes")
    else:
        print("Est. time remaining: calculating...")
    print("---")
    pos += 1

df = pd.DataFrame(results)
output_path = os.path.join(output_dir, "optimized_sweep_results.csv")
df.to_csv(output_path, index=False)
print("\nSweep complete. Results written to", output_path)

print(f"→ {df['meets_spec'].sum()} / {len(df)} combos meet SNR≥{target_snr}")
print("Top 5 by worst-case SNR(200 ns):")
print(df.nlargest(5, 'snr_200ns')[[
    'delta_r1','delta_r2','g_1','g_2','kappa','delta_resonator','snr_200ns']]) 
