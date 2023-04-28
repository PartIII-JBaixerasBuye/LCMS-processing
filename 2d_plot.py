import matplotlib.pyplot as plt
import numpy as np


def error(K, EM):
    return np.power(1 + K * EM / 1000, -2) / 2


K = np.logspace(1.5, 4, 50)
EM = np.logspace(-1.5, 2, 50)
slices = np.linspace(0, 0.5, 11)
Ks, EMs = np.meshgrid(K, EM)

error = error(Ks, EMs)
fig, ax = plt.subplots(1, 1)

plt.contourf(Ks, EMs, error, slices, cmap='inferno_r')
plt.colorbar()
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_ylabel('EM / mM')
ax.set_xlabel('K / M-1')
plt.scatter(100, 3, color='c', marker='o')
ax.annotate('A', (100, 3), xytext=(110, 3.3), color='c', weight='bold', size=20)
plt.scatter(2000, 3, color='c', marker='o')
ax.annotate('D', (2000, 3), xytext=(2200, 3.3), color='c', weight='bold', size=20)
plt.plot(np.full((50), 100), EM, color='c', linestyle='dashdot')
plt.scatter(100, 40, color='c', marker='o')
ax.annotate('B', (100, 40), xytext=(110, 44), color='c', weight='bold', size=20)
plt.scatter(400, 3, color='c', marker='o')
ax.annotate('C', (400, 3), xytext=(440, 3.3), color='c', weight='bold', size=20)
plt.plot(K, np.full((50), 3), color='c', linestyle='dashed')
plt.savefig('2d_plot.png')
plt.show()
