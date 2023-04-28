import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


def equation(x, iBuNH2_0, CHO_0, PO_0, K):
    # Species in equilibrium
    iBuNH2, iBuNH_CHO, PO_NH2, PO_CHO = x

    # Boundary conditions
    eq1 = iBuNH2 + iBuNH_CHO - iBuNH2_0
    eq2 = PO_NH2 + PO_CHO - PO_0
    eq3 = iBuNH_CHO + PO_CHO - CHO_0

    # equilibrium
    eq4 = PO_NH2 * iBuNH_CHO / (iBuNH2 * PO_CHO) - K

    return [eq1, eq2, eq3, eq4]


# returns a function which can be queried for equilibrium distributions (iBuNH2, iBuNH_CHO, PO_NH2, PO_CHO)
# given initial conditions
def make_system(iBuNH2_0, CHO_0, PO_0):
    def find_eq(K):
        return fsolve(equation, (iBuNH2_0, 0.1 * CHO_0, PO_0, 0.1 * CHO_0), args=(iBuNH2_0, CHO_0, PO_0, K))
    return find_eq

K = 2
iBuNH2_0, CHO_0, PO_0 = 0.15, 0.1, 0.15
sim = make_system(K, CHO_0, PO_0)

N = 200
x = np.linspace()
