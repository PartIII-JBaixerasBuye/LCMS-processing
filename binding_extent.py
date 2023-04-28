import matplotlib.pyplot as plt
import numpy as np

KEM = np.logspace(-2, 2, 100)
bonds_0 = 1 / (2 * KEM ** 2 + 4 * KEM + 1)
bonds_1 = bonds_0 * KEM * 4
bonds_2 = bonds_1 * KEM / 2

plt.plot(KEM, bonds_0, color='k', linestyle='dashdot')
plt.plot(KEM, bonds_1, color='k', linestyle='dashed')
plt.plot(KEM, bonds_2, color='k')
plt.legend(['0', '1', '2'])

ax = plt.gca()
ax.fill_between((0.12, 0.4), (-2, -2), y2=(2, 2), alpha=0.2)
ax.set_ylim(0, 1)
ax.set_xscale('log')
ax.set_xlabel('K·EM ')
ax.set_ylabel('Equilibrium composition')

plt.savefig('Hydrogenbonds.png')
plt.show()
