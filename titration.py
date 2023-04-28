import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

def line(x, m, c):
    return m * x + c

# import data
N = 9
IT_t = np.loadtxt('IT_t')
data = np.loadtxt('IT_data')[2:11]
data = data / np.max(data[:, 2200:], axis=-1)[:, np.newaxis] * 2
concs = np.array([0.1, 0.15, 0.3, 0.4, 0.6, 0.8, 0.95, 1.2, 1.9])
names = [f'{val * 10} mM' for val in concs]

# adjust alignment
maxs = [np.argmax(values[2400:]) for values in data]
offsets = [(max_i - maxs[0]) / 3601 * 3 for max_i in maxs]

# Peak details
peaks = [[(1.77, 1.94), (2.32, 2.5), (2.70, 2.75)]] * N
peak_colors = np.array([[(52, 153, 103), (144, 32, 255), (51, 153, 204)]] * N) / 255

integrations = []
fig, axs = plt.subplots(N, 1, sharex=True, sharey=True, figsize=(8.27, 11.69), dpi=100)
for ax, values, conc, off, peak_list, colors in zip(axs, data, names, offsets, peaks, peak_colors):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(0.95, 0.75, conc, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)

    ax.plot(IT_t - off, values, color='k')
    for (x1, x2), color in zip(peak_list, colors):
        # Colour peak
        ax.fill_between(IT_t - off, values, where=np.logical_and(IT_t - off > x1, IT_t - off <= x2), color=color)

        # Integrate
        mask_integration = np.logical_and(IT_t - off >= x1, IT_t - off <= x2)
        integrations.append(np.sum(values[mask_integration]) * 3 / 3601)

axs[len(axs) // 2].set_ylabel('Absorbance / A.U.')
ax.set_xlabel('Retention time / min')
plt.savefig('Titration_traces.png')
plt.show()

# Process integrations
integrations = np.array(integrations).reshape((N, 3))
fraction_absorbance = integrations[:, 2] / (integrations[:, 2] + integrations[:, 1])
# fraction_absorbance = integrations[:, 1] / integrations[:, 2]
np.savetxt('Titration_concs', concs)
np.savetxt('Titration_integrations', fraction_absorbance)

# Plot
plt.scatter(concs, fraction_absorbance, marker='+', c='k')
ax = plt.gca()
ax.set_xlabel('Eq. of iBuNH2')
ax.set_ylabel('% of total absorbance')
plt.savefig('Titration_plot.png')
plt.show()