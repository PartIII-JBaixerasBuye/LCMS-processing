import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.optimize import curve_fit


# Useful functions for peak deconvolution
def gauss(x, mu, sigma, abs):
    return stats.norm.pdf(x, mu, sigma) * np.sqrt(2 * np.pi) * sigma * abs


def double_gauss(x, mu1, sigma1, abs1, mu2, sigma2, abs2):
    return gauss(x, mu1, sigma1, abs1) + gauss(x, mu2, sigma2, abs2)


def gauss_area(mu, sigma, abs):
    return np.sqrt(2 * np.pi) * sigma * abs


def line(x, m, c):
    return m * x + c


# import data
N = 9
IT_t = np.loadtxt('IT_t')
data = np.loadtxt('IT_data')[2:11]
data = data / np.max(data[:, 2200:], axis=-1)[:, np.newaxis] * 2
concs = np.array([0.1, 0.15, 0.3, 0.4, 0.6, 0.8, 0.95, 1.2, 1.9])
names = [f'{val} mM' for val in concs]

# adjust alignment
maxs = [np.argmax(values[2400:]) for values in data]
offsets = [(max_i - maxs[0]) / 3601 * 3 for max_i in maxs]

# Peak details
fit_region = (2.32, 2.5)
fit_colors = ['b', 'm']
peaks = [[(1.77, 1.94), (2.70, 2.75)]] * N
peak_colors = [['r', 'b']] * N


integrations = []
fig, axs = plt.subplots(N, 1, sharex=True, sharey=True, figsize=(8.27, 11.69), dpi=100)
for ax, values, conc, off, peak_list, colors in zip(axs, data, names, offsets, peaks, peak_colors):
    # Peak deconvolution by double gaussian fitting
    fit_mask = np.logical_and(IT_t >= fit_region[0], IT_t <= fit_region[1])
    x = (IT_t - off)[fit_mask]
    y = values[fit_mask]
    params, cov = curve_fit(double_gauss, x, y, p0=(2.34, 0.02, 0.4, 2.4, 0.02, 0.3))
    errors = np.sqrt(np.diag(cov))
    mu1, sigma1, abs1, mu2, sigma2, abs2 = params

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel('Absorbance / A.U.')
    ax.text(0.95, 0.75, conc, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)

    # Plot and colour peaks
    ax.plot(IT_t - off, values, color='k')
    ax.fill_between(x, gauss(x, mu2, sigma2, abs2) + gauss(x, mu1, sigma1, abs1), color=fit_colors[0])
    ax.fill_between(x, gauss(x, mu2, sigma2, abs2), color=fit_colors[1])
    for (x1, x2), color in zip(peak_list, colors):
        # color
        ax.fill_between(IT_t - off, values, where=np.logical_and(IT_t - off > x1, IT_t - off <= x2), color=color)

        # Integrate
        mask_integration = np.logical_and(IT_t - off >= x1, IT_t - off <= x2)
        integrations.append(np.sum(values[mask_integration]) * 3 / 3601)

ax.set_xlabel('Retention time / min')
plt.show()

# Process integrations
integrations = np.array(integrations).reshape((N, 3))
fraction_absorbance = integrations[:, 2] / (integrations[:, 2] + integrations[:, 1])
fraction_absorbance = integrations[:, 1] / integrations[:, 2]
np.savetxt('Titration_concs', concs)
np.savetxt('Titration_integrations', fraction_absorbance)

# Plot
plt.scatter(concs, fraction_absorbance, marker='+', c='k')
ax = plt.gca()
ax.set_xlabel('Eq. of iBuNH2')
ax.set_ylabel('% of total absorbance')
plt.show()
