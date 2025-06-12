import numpy as np
import pandas as pd
from itertools import product
import os
import time
import argparse

from pointer_sim import optimize_parameters, compute_chis, calculate_snr

def parse_args():
    parser = argparse.ArgumentParser(description='Parameter sweep for qubit readout optimization')
    parser.add_argument('--optimization-case', type=str, default='binary_states',
                      choices=['binary_states', 'adjacent_states', 'all_states'],
                      help='Which states to optimize between')
    parser.add_argument('--energy-levels', type=int, default=2,
                      help='Number of energy levels per qubit (only used for all_states case)')
    parser.add_argument('--target-snr', type=float, default=5.0,
                      help='Target SNR threshold')
    return parser.parse_args()


args = parse_args()

output_dir = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(output_dir, exist_ok=True)

start_time = time.time()

tau = 200
target_snr = args.target_snr   # e.g. SNR≥6 for ≈99% single-shot fidelity

delta_r1_vals       = np.linspace(0.1, 1.0, 10)   # 100–1000 MHz
delta_r2_vals       = np.linspace(0.1, 1.0, 10)
g_1_vals            = np.linspace(0.05,0.2, 5)   # 50–200 MHz
g_2_vals            = np.linspace(0.05,0.2, 5)
kappa_vals          = np.linspace(0.001,0.02,5)   # 1–20 MHz
delta_resonator_vals= np.linspace(-0.05,0.05, 5)   # –50 to +50 MHz detuning

results = []
total = (len(delta_r1_vals) * len(delta_r2_vals) * len(g_1_vals) * len(g_2_vals) * len(kappa_vals) * len(delta_resonator_vals))
pos = 0

for delta_r1, delta_r2, g_1, g_2, kappa, delta_resonator in product(delta_r1_vals, delta_r2_vals, g_1_vals, g_2_vals, kappa_vals, delta_resonator_vals):

    res = optimize_parameters(delta_r1, delta_r2, g_1, g_2, kappa, delta_resonator, tau,
                            optimization_case=args.optimization_case,
                            energy_levels=args.energy_levels)

    # 2) compute the true 200 ns integrated SNR
    chi_1, chi_2 = compute_chis(g_1, g_2, delta_r1, delta_r2)
    params = [res['Omega_q1_mag'], res['phi_q1'],
                res['Omega_q2_mag'], res['phi_q2']]

    # Get state pairs for final SNR calculation
    from pointer_sim import get_state_pairs
    state_pairs = get_state_pairs(args.optimization_case, args.energy_levels)

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
output_filename = f"optimized_sweep_results_{args.optimization_case}_n{args.energy_levels}.csv"
output_path = os.path.join(output_dir, output_filename)
df.to_csv(output_path, index=False)
print("\nSweep complete. Results written to", output_path)

print(f"→ {df['meets_spec'].sum()} / {len(df)} combos meet SNR≥{target_snr}")
print("Top 5 by worst-case SNR(200 ns):")
print(df.nlargest(5, 'snr_200ns')[[
    'delta_r1','delta_r2','g_1','g_2','kappa','delta_resonator','snr_200ns']])

