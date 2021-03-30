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
combs = []
start_list = [11, 21, 12, 22, 13, 23]
combs.append(start_list)
for i in range(5):
    list = range(6)
    for j in combinations(list, (i + 1)):
        list_c = start_list.copy()
        for k in j:
            list_c[k] = 0
        combs.append(list_c)

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
        precs[i] = prec
        # print(f"iteration:{i + 1}, precision: {prec}")
        if (i+1) % 20 == 0:
            print(f"progress: {(i + 1)}%")
    return ekfs, precs

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

print("creating initial set")
ekfs_0, precs_0 = get_ekfs(mode="Alt_tau")
plt.plot(Ns, ekfs_0, c="k", linestyle=":")
plt.xscale("log")
plt.show()

for l in range(len(combs)):
    start = time.time()
    print(f"creating plot {l + 1} of {len(combs)}")
    ekfs, precs = get_ekfs(tau_alt=combs[l])
    perm_str = "_".join([str(elem) for elem in combs[l]])
    fig, ax = plt.subplots()
    ax.plot(Ns, ekfs)
    ax.plot(Ns, ekfs_0, c="k", linestyle=":")
    ax.set_xscale("log")
    ax.set_ylabel("Eigenfrequenzen")
    ax.set_xlabel("Vorspannungen")
    ax.legend(["Combination", "Alt_Taus"])
    ax.text(2, 128, 'RMSE = {0:.3g}'.format(rmse(ekfs, ekfs_0)), verticalalignment='top')
    # plt.savefig("euber_preload_" + perm_str)

    print(f"time: {(time.time() - start) / 60}min")


# print(precs, ekfs, Ns)
# fig, ax = plt.subplots(1, 2, figsize=[12, 6])
# ax[0].plot(Ns, precs)
# ax[1].plot(Ns, ekfs)
# ax[0].set_xlabel("Vorspannungen")
# ax[0].set_ylabel("Präzisionen")
# ax[0].set_yscale("log")
# ax[0].set_xscale("log")
# ax[1].set_xscale("log")
# ax[1].set_ylabel("Eigenfrequenzen")
# ax[1].set_xlabel("Vorspannungen")

# plt.show()

# print(precs)