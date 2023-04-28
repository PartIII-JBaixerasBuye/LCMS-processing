import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.optimize import curve_fit
import pandas as pd
import seaborn as sns

# Useful functions for peak deconvolution
def gauss(x, mu, sigma, abs):
    return stats.norm.pdf(x, mu, sigma) * np.sqrt(2 * np.pi) * sigma * abs


def double_gauss(x, mu1, sigma1, abs1, mu2, sigma2, abs2):
    return gauss(x, mu1, sigma1, abs1) + gauss(x, mu2, sigma2, abs2)


def gauss_area(mu, sigma, abs):
    return np.sqrt(2 * np.pi) * sigma * abs


# Import data
N = 3
t = np.loadtxt('IT_t')
data = np.loadtxt('IT_data')[-3:][np.array([1, 0, 2])]
names = ['a)', 'b)', 'c)']

# Peak details
fit_region = (2.3, 2.5)
fit_colors = ['r', 'm']
peaks = [[(1.8, 1.9), (2.711, 2.773), (2.49, 2.58)],
         [(1.8, 1.9), (2.688, 2.746)],
         [(1.8, 1.9), (2.705, 2.764)]]
peak_colors = np.array([[(52, 153, 103), (51, 153, 204), (238, 126, 76),]] * N) / 255


# Align peaks
maxs = [np.argmax(values[2000:2400]) for values in data]
offsets = [(max_i - maxs[1]) / 3601 * 3 for max_i in maxs]
data = data / np.max(data[:, 2400:], axis=-1)[:, np.newaxis] * 2

# Figure details
integrations = []
fig = plt.figure(1)
ax1 = plt.subplot2grid((3, 4), (0, 0), 1, 3)
ax2 = plt.subplot2grid((3, 4), (1, 0), 1, 3)
ax3 = plt.subplot2grid((3, 4), (2, 0), 1, 3)
ax4 = plt.subplot2grid((3, 4), (0, 3), 3, 1)
axs = [ax1, ax2, ax3]

# Process LCMS data
for ax, values, conc, off, peak_list, colors in zip(axs, data, names, offsets, peaks, peak_colors):
    # Peak deconvolution by double gaussian fitting
    # Define fit region
    fit_mask = np.logical_and(t >= fit_region[0], t <= fit_region[1])
    x = (t - off)[fit_mask]
    y = values[fit_mask]
    # Fit double gaussian
    params, cov = curve_fit(double_gauss, x, y, p0=(2.366, 0.02, 0.4, 2.4, 0.02, 0.3))
    errors = np.sqrt(np.diag(cov))
    mu1, sigma1, abs1, mu2, sigma2, abs2 = params

    # Plot details
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(0.03, 0.85, conc, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
    ax.set_xlim(1.7, 2.9)
    ax.set_ylim(0, 3.5)

    # Plot
    ax.plot(t - off, values, color='k')
    ax.fill_between(x, gauss(x, mu2, sigma2, abs2) + gauss(x, mu1, sigma1, abs1), color=fit_colors[0])
    ax.fill_between(x, gauss(x, mu2, sigma2, abs2), color=fit_colors[1])
    for (x0, x1), color in zip(peak_list, colors):
        ax.fill_between(t - off, values, where=np.logical_and(t - off >= x0, t - off <= x1), color=color)

    # Integrate
    mask_integration = np.logical_and(t - off >= peak_list[1][0], t - off <= peak_list[1][1])
    integrations.append([gauss_area(mu2, sigma2, abs2), np.sum(values[mask_integration]) * 3 / 3601])

# Format graph
ax.set_xlabel('Retention time / min')
axs[N // 2].set_ylabel('Absorbance / A.U.')

# Process integrals
integrations = np.array(integrations)
# Normalize peak areas
integrations = integrations / np.sum(integrations, axis=-1)[:, np.newaxis]
print(integrations / (integrations[:, 0] + integrations[:, 1])[:, np.newaxis])
# Adjust for relative absorbance
ratio = integrations[0, 0] / integrations[0, 1]
integrations[:, 0] = integrations[:, 0] / ratio
integrations = integrations / np.sum(integrations, axis=1)[:, np.newaxis]
print(integrations)

# Plot grouped bar chart
# This is not the best way to do this but I am not very familiar with pandas
df = pd.DataFrame([['25', 'a)', integrations[0, 0]],
                   ['25', 'b)', integrations[1, 0]],
                   ['25', 'c)', integrations[2, 0]],
                   ['24', 'a)', integrations[0, 1]],
                   ['24', 'b)', integrations[1, 1]],
                   ['24', 'c)', integrations[2, 1]]],
                  columns=['Compound', 'column', 'val'])
sns.barplot(df, x='val', y='column', hue='Compound', ax=ax4, palette=[fit_colors[1], peak_colors[0, 1]], width=0.3)
ax4.plot([0.5, 0.5], [-2, 3], color='k', linestyle='dashed')
ax4.set_ylim(2.3, -0.5)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.set(yticklabels=[])
ax4.set_ylabel('')
ax4.set_xlabel('Composition')
ax4.get_legend().remove()

plt.tight_layout()
plt.savefig('calibration.png')
plt.show()
