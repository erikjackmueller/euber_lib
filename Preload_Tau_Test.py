import euber_libV3 as eb3
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import time


# Platin
rho_pt = 21450  # kg/m³
E_pt = 126.8e9  # Pa
# Scandiumaluminiumnitrit
rho_scaln = 3318  # kg/m³
E_scaln = 315.2e9  # Pa
# Nickel
rho_ni = 8908  # kg/m³
E_ni = 200e9  # Pa

parameters = {'b1': 0.03, 'L1': 0.45, 'L2': 0.45, 'L3': 0.45,
                  'dA': 0.0135, 'dB': 0.0135, 'dC': 0.0135,
                  'EA': E_pt, 'EB': E_scaln, 'EC': E_ni,
                  'rhoA': rho_pt, 'rhoB': rho_scaln, 'rhoC': rho_ni}
Ns = np.logspace(0, 5, num=100, base=10)

def get_ekfs(mode=None, tau_alt=None):

    ekfs = np.zeros(len(Ns))
    prec = 2000
    precs = np.zeros(len(Ns))


    for i in range(len(Ns)):

        beam = eb3.euber(parameters=parameters, N=Ns[i], mode=mode, tau_alt=tau_alt)

        try:
            beam.prec = prec
            beam.lyrd_beam_reduced_params()
            ekf = beam.get_first_n_ekf_bruteforce(n=1, om=80, stepsize=200)
            kappa = beam.kappa

        except TypeError:

            rec = prec * 2
            beam.prec = prec
            beam.lyrd_beam_reduced_params()
            ekf = beam.get_first_n_ekf_bruteforce(n=1, om=80, stepsize=200)
            kappa = beam.kappa

        while kappa > prec:
            prec = prec * 2
            beam.prec = prec
            beam.lyrd_beam_reduced_params()
            ekf = beam.get_first_n_ekf_bruteforce(n=1, om=80, stepsize=200)
            kappa = beam.kappa
            print("-----increasing precision-----")

        ekfs[i] = ekf[0] / 2 / np.pi

        diff = 0
        if ekfs[1] != 0:
            diff = derivative(ekfs, a=(i-1), method='forward', h=1)
            # print(f"Diff = {diff}")

        if ((diff < 0) and tau_alt == 11):
            tau_alt = 0
            mode = "Alt_tau"


        precs[i] = prec

        # print(f"iteration:{i + 1}, precision: {prec}")
        if (i + 1) % 20 == 0:
            print(f"progress: {(i + 1)}%")
    return ekfs, precs

def derivative(f, a, method='central', h=0.01):
    '''Compute the difference formula for f'(a) with step size h.

    Parameters
    ----------
    f : function
        Vectorized function of one variable
    a : number
        Compute derivative at x = a
    method : string
        Difference formula: 'forward', 'backward' or 'central'
    h : number
        Step size in difference formula

    Returns
    -------
    float
        Difference formula:
            central: f(a+h) - f(a-h))/2h
            forward: f(a+h) - f(a))/h
            backward: f(a) - f(a-h))/h
    '''
    if method == 'central':
        return (f(a + h) - f(a - h)) / (2 * h)
    elif method == 'forward':
        return (f[a + h] - f[a]) / h
    elif method == 'backward':
        return (f(a) - f(a - h)) / h
    else:
        raise ValueError("Method must be 'central', 'forward' or 'backward'.")

ekfs_0, precs_0 = get_ekfs(mode="Alt_tau")
ekfs, precs = get_ekfs(tau_alt=11)

fig, ax = plt.subplots()
ax.plot(Ns, ekfs)
ax.plot(Ns, ekfs_0, c="k", linestyle=":")
ax.set_xscale("log")
ax.set_ylabel("Eigenfrequenzen")
ax.set_xlabel("Vorspannungen")
ax.legend(["Diff Switch"])
plt.show()