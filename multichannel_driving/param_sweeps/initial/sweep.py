import numpy as np
import pandas as pd
from itertools import product
from pointer_sim import optimize_parameters, compute_chis

# Parameter sweep ranges
delta_r1_vals = np.linspace(0.1, 1.0, 4) # 100 to 1000 MHz
delta_r2_vals = np.linspace(0.1, 1.0, 4) # 100 to 1000 MHz
g_1_vals = np.linspace(0.05, 0.2, 4) # 50 to 200 MHz    
g_2_vals = np.linspace(0.05, 0.2, 4) # 50 to 200 MHz
kappa_vals = np.linspace(0.001, 0.02, 4) # 1 to 20 MHz
delta_resonator_vals = np.linspace(-0.05, 0.05, 4) # -50 to 50 MHz for resonator vs drive detuning

results = []

total_combinations = len(delta_r1_vals) * len(delta_r2_vals) * len(g_1_vals) * len(g_2_vals) * len(kappa_vals) * len(delta_resonator_vals)
current = 0

for delta_r1, delta_r2, g_1, g_2, kappa, delta_resonator in product(delta_r1_vals, delta_r2_vals, g_1_vals, g_2_vals, kappa_vals, delta_resonator_vals):
    current += 1
    print(f"Processing combination {current}/{total_combinations}")
    print(f"Parameters: delta_r1={delta_r1:.2f}, delta_r2={delta_r2:.2f}, g_1={g_1:.2f}, g_2={g_2:.2f}, kappa={kappa:.2f}, delta_resonator={delta_resonator:.2f}")
    
    chi_1, chi_2 = compute_chis(g_1, g_2, delta_r1, delta_r2)
    result = optimize_parameters(chi_1, chi_2, delta_r1, delta_r2, kappa, g_1, g_2, delta_resonator)
    results.append(result)
    
    print(f"Optimized parameters:")
    print(f"  Omega_q1_mag = {result['Omega_q1_mag']:.4f}")
    print(f"  phi_q1 = {result['phi_q1']:.4f}")
    print(f"  Omega_q2_mag = {result['Omega_q2_mag']:.4f}")
    print(f"  phi_q2 = {result['phi_q2']:.4f}")
    print(f"  Min SNR = {result['min_snr']:.4f}")
    print(f"  Avg SNR = {result['avg_snr']:.4f}")
    print("---")

df_results = pd.DataFrame(results)
df_results.to_csv("optimized_sweep_results.csv", index=False)
print("\nFinished sweep")

print("\nSummary Statistics:")
print(f"Best minimum SNR: {df_results['min_snr'].max():.4f}")
print(f"Best average SNR: {df_results['avg_snr'].max():.4f}")
print("\nTop 5 combinations by minimum SNR:")
print(df_results.nlargest(5, 'min_snr')[['delta_r1', 'delta_r2', 'g_1', 'g_2', 'kappa', 'min_snr', 'avg_snr']])
