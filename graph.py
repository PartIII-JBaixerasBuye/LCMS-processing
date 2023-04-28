import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from functools import partial


ratio = 1.5
x_data = np.array([0.1, 0.15, 0.3, 0.4, 0.6, 0.795563303, 0.954675964, 1.209256221, 1.909351928])
y_data = np.array([0.174418605, 0.2275, 0.416498994, 0.396124193, 0.5, 0.735520589, 0.768358626, 0.796044275, 0.825662483])
y_data = y_data * ratio / (y_data * ratio + (1 - y_data))

def equation(x, iBuNH2_0, CHO_0, PO_0, K):
    # Vars
    iBuNH2, iBuNH_CHO, PO_NH2, PO_CHO = x

    # Boundary conditions
    eq1 = iBuNH2 + iBuNH_CHO - iBuNH2_0
    eq2 = PO_NH2 + PO_CHO - PO_0
    eq3 = iBuNH_CHO + PO_CHO - CHO_0

    # equilibrium
    eq4 = PO_NH2 * iBuNH_CHO / iBuNH2 / PO_CHO - K

    return [eq1, eq2, eq3, eq4]


def solve_eq(PO_0, CHO_0, ratio, iBuNH2_0, K):
    values = np.full((len(iBuNH2_0)), np.nan)
    for i, conc in enumerate(iBuNH2_0):
        iBuNH2, iBuNH_CHO, PO_NH2, PO_CHO = fsolve(equation, (conc, CHO_0 / 10, PO_0, CHO_0 / 10), args=(conc, CHO_0, PO_0, 1 / K))
        values[i] = iBuNH_CHO / (iBuNH_CHO + PO_CHO)
    return values


iBuNH2_0 = 0.15
PO_0 = 0.15
CHO_0 = 0.1
k1 = 1.5
k3 = 3.0

func = partial(solve_eq, PO_0, CHO_0, ratio)
K, K_err = curve_fit(func, x_data, y_data, 2, absolute_sigma=False)

x = np.linspace(0.1, 2, 50)
y1 = solve_eq(PO_0, CHO_0, ratio, x, k1)
y2 = solve_eq(PO_0, CHO_0, ratio, x, K[0])
y3 = solve_eq(PO_0, CHO_0, ratio, x, k3)

y1 = np.array(y1)
y2 = np.array(y2)
y3 = np.array(y3)

plt.plot(x / 0.15, 1 - y3, color='k', linestyle='dashdot', alpha=0.5, label=f'K = {k3}')
plt.plot(x / 0.15, 1 - y2, color='k', label=f'K = {np.round(K[0], 1)}')
plt.plot(x / 0.15, 1 - y1, color='k', linestyle='dashed', alpha=0.5, label=f'K = {k1}')
plt.legend()

ax = plt.gca()
plt.scatter(x_data / 0.15, 1 - y_data, c='k', marker='+')

ax.set_xlabel('Equiv of iBuNH2')
ax.set_ylabel('% duplex at equilibrium')

plt.savefig('titration_fit.png')
plt.show()