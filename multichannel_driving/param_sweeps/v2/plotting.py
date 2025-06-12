import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Load results
df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'output', "optimized_sweep_results.csv"))

# Remove invalid entries
df = df[df['snr_200ns'].notnull()]

# Convert GHz → MHz
for col in ['delta_r1', 'delta_r2', 'g_1', 'g_2', 'kappa', 'delta_resonator']:
    df[col] *= 1e3

# Compute dispersive shifts
df['chi_1'] = df['g_1']**2 / df['delta_r1']
df['chi_2'] = df['g_2']**2 / df['delta_r2']

# Define parameter pairs to plot
plot_pairs = [
    ('chi_1', 'kappa'),
    ('delta_resonator', 'kappa'),
    ('chi_1', 'delta_resonator'),
    ('chi_1', 'chi_2')
]

# Plot heatmaps showing the max SNR for each coordinate
for x, y in plot_pairs:
    pivot = df.pivot_table(index=y, columns=x, values='snr_200ns', aggfunc='max')
    if pivot.shape[0] < 2 or pivot.shape[1] < 2:
        continue

    # Round axes to 2 significant figures
    pivot.index = [float(f"{v:.2g}") for v in pivot.index]
    pivot.columns = [float(f"{v:.2g}") for v in pivot.columns]

    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, cmap="viridis", annot=False, cbar_kws={'label': 'SNR (200 ns)'})
    plt.title(f'SNR (200 ns) vs {x} vs {y}')
    plt.xlabel(f'{x} (MHz)')
    plt.ylabel(f'{y} (MHz)')
    plt.tight_layout()
    plt.show()
