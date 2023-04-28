import matplotlib.pyplot as plt
import numpy as np

y = np.loadtxt('IT_long_data')
x = np.loadtxt('IT_long_t')

plt.plot(x, y, color='k')
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel('Time / min')
ax.set_ylabel('Absorbance / A.U.')

plt.savefig('imine_trimer.png')
plt.show()