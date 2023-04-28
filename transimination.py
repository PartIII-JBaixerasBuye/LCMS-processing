import matplotlib.pyplot as plt
import numpy as np

# Import data
N = 2
t = np.loadtxt('IT_t')
data = np.loadtxt('IT_data')[:2]
names = ['1 mM', '0.1 mM']

# Peak locations
peaks = [[(1.35, 1.375), (1.375, 1.415), (1.445, 1.489), (1.4885, 1.525)],
         [(1.35, 1.375), (1.375, 1.425), (1.445, 1.489), (1.4885, 1.525)]]
peak_colors = np.array([[(238, 126, 76), (51, 153, 204), (52, 153, 103), (144, 32, 255)]] * 2) / 255

# Align peaks
max1 = np.argmax(data[0])
max2 = np.argmax(data[1])
offsets = [0, (max2 - max1) / 3601 * 3]

# Plot
fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)
for ax, values, conc, off, peak_list, colors in zip(axs, data, names, offsets, peaks, peak_colors):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(0.7, 2)
    ax.set_ylim(0, 2)
    ax.set_ylabel('Absorbance / A.U.')
    ax.text(0.95, 0.75, conc, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)

    ax.plot(t - off, values, color='k')
    for (x1, x2), color in zip(peak_list, colors):
        ax.fill_between(t - off, values, where=np.logical_and(t - off > x1, t - off <= x2), color=color)

ax.set_xlabel('Time / min')
plt.savefig('Transimination.png')
plt.show()
