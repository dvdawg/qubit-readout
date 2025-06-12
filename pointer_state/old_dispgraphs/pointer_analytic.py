import numpy as np
import matplotlib.pyplot as plt

# Parameters (example values)
g = 1.0
Delta = 5.0
alpha = 0.5
delta_r = 0.0
epsilon = 1.0
kappa = 1.0

# Functions for chi_n

def chi_0():
    return -g**2 / Delta

def chi_1():
    return g**2 / Delta - 2 * g**2 / (Delta - alpha)

def chi_2():
    return 2 * g**2 / (Delta - alpha) - 3 * g**2 / (Delta - 2 * alpha)

def chi_3():
    return 3 * g**2 / (Delta - 2 * alpha) - 4 * g**2 / (Delta - 3 * alpha)

chi_funcs = [chi_0, chi_1, chi_2, chi_3]

# Function for alpha_n
def alpha_n(chi_n):
    denom = delta_r + chi_n - 1j * kappa / 2
    return -epsilon / denom

# Compute alphas
alphas = [alpha_n(chi_func()) for chi_func in chi_funcs]

# Plot
plt.figure(figsize=(6, 6))
colors = ['r', 'g', 'b', 'm']
labels = [r'$n=0$', r'$n=1$', r'$n=2$', r'$n=3$']

for i, a in enumerate(alphas):
    plt.plot(a.real, a.imag, 'o', color=colors[i], label=labels[i], markersize=10)
    plt.text(a.real, a.imag, f'  {labels[i]}', fontsize=12, va='center')

plt.axhline(0, color='gray', lw=0.5)
plt.axvline(0, color='gray', lw=0.5)
plt.xlabel('Re($\\alpha_n$)')
plt.ylabel('Im($\\alpha_n$)')
plt.title('Pointer-State Trajectories for $n=0,1,2,3$')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
