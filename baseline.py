import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pybaselines.whittaker import airpls
from os import listdir
from os.path import isfile, join

# Find excel files to extract data from
mypath = 'All Data'
names = [f for f in listdir(mypath) if isfile(join(mypath, f))]
paths = [join(mypath, f) for f in names]
N = len(paths)

# Cutoff the first 0.7 min because weird baseline
cutoff_time = 0.7
cutoff_long = 0.7
cutoff = int(3600 * cutoff_time / 3)
cutoff_long = int(6000 * cutoff_long / 5)

# Extract data from files
data = np.full((N - 1, 3601 - cutoff), np.nan)
for i, path in enumerate(paths):
    df = pd.read_excel(path, sheet_name=0, header=None, names=[0, 1, 2, 3])
    if i < N - 1:
        data[i] = df[1].to_numpy()[cutoff:3601]
    else:
        long_data = df[1].to_numpy()[cutoff_long:6001]

# Time on spreadsheet is rounded weirdly so define it here
t = (np.arange(3601) / 3601 * 3 - 0.02)[cutoff:]
t_long = (np.arange(6001) / 6001 * 5 - 0.02)[cutoff_long:]

# Scale data
data[np.isnan(data)] = 0
data = data / np.max(data, axis=-1)[:, np.newaxis] * 2
long_data = long_data / np.max(long_data) * 2

# Get baselines from data and visual inspection
# Long experiment done later
fig, axs = plt.subplots(N // 2, 2, sharex=True, sharey=True, figsize=(8.27, 11.69), dpi=100)
baselines = []
for ax, values, conc in zip(np.ravel(axs), data, names):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(0.95, 0.75, conc, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)

    ax.set_ylabel('Absorbance / A.U')

    baseline = airpls(values)[0]
    baselines.append(baseline)
    ax.plot(t, values, color='k')
    ax.plot(t, baseline, color='r', linestyle=':')

# Display
plt.savefig('baselines.png')
ax.set_xlabel('Retention time / min')
plt.show()


# Save processed data
baselines = np.array(baselines)
np.savetxt('IT_data', data - baselines)
np.savetxt('IT_t', t)

# Process long
baseline = airpls(long_data, lam=1e8)[0]
np.savetxt('IT_long_data', long_data - baseline)
np.savetxt('IT_long_t', t_long)
