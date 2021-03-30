import euber_libV3 as eb3
import numpy as np
import time
import matplotlib.pyplot as plt

startTime = time.time()

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
scales = np.logspace(0, 6, num=150, base=10)
scales = scales[::-1]
ekfs = np.zeros(len(scales))
prec = 50
precs = np.zeros(len(scales))

for i in range(len(scales)):

    scale = scales[i]

    # still failing geometry
    # parameters = {'b1': 5e-6 * scale, 'L1': 0.3e-3 * scale, 'L2': 0.7e-3 * scale, 'L3': 1e-3 * scale,
    #               'dA': 1e-7 * scale, 'dB': 5e-7 * scale, 'dC': 5e-7 * scale,
    #               'EA': 2e11, 'EB': 7e10, 'EC': 2e11,
    #               'rhoA': 8700, 'rhoB': 2700, 'rhoC': 8700}

    parameters = {'b1': 2e-6 * scale, 'L1': 1e-6 * scale, 'L2': 1e-6 * scale, 'L3': 1e-6 * scale,
                  'dA': 1e-7 * scale, 'dB': 1e-7 * scale, 'dC': 1e-7 * scale,
                  'EA': E_pt, 'EB': E_scaln, 'EC': E_ni,
                  'rhoA': rho_pt, 'rhoB': rho_scaln, 'rhoC': rho_ni}

    beam = eb3.euber(parameters=parameters, N=100)

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
    print(f"iteration:{i + 1}, precision: {prec}")


fig, ax = plt.subplots(1, 2, figsize=[12, 6])
ax[0].plot(scales, precs)
ax[1].plot(scales, ekfs)
ax[0].set_xlabel("Größenordnung")
ax[0].set_ylabel("Präzisionen")
ax[0].set_yscale("log")
ax[0].set_xscale("log")
ax[1].set_xscale("log")
ax[1].set_ylabel("Eigenfrequenzen")
ax[1].set_xlabel("Größenordnung")
plt.show()

# print(f"ekfs: {ekfs}")
# print(precs, ekfs, scales)