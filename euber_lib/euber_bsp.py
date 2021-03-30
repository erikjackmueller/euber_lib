import euber_libV3 as eb3
import numpy as np
import time
import matplotlib.pyplot as plt

startTime = time.time()

# Platin
rho_pt = float(21450)  # kg/m³
E_pt = float(126.8e9)  # Pa
# Scandiumaluminiumnitrit
rho_scaln = float(3318)  # kg/m³
E_scaln = float(315.2e9)  # Pa
# Nickel
rho_ni = float(8908)  # kg/m³
E_ni = float(200e9)  # Pa

# Geometry parameters
lb = np.array([0.01, 0.2, 0.2, 0.2, 0.007, 0.007, 0.007])
ub = np.array([0.05, 0.7, 0.7, 0.7, 0.02, 0.02, 0.02])
bound_means = (lb + ub) / 2
bndmiddl = bound_means.tolist()

# parameters = {'b1': bound_means[0], 'L1': bound_means[1], 'L2': bound_means[2], 'L3': bound_means[3],
#               'dA': bound_means[4], 'dB': bound_means[5], 'dC': bound_means[6],
#               'EA': E_pt, 'EB': E_scaln, 'EC': E_ni,
#               'rhoA': rho_pt, 'rhoB': rho_scaln, 'rhoC': rho_ni}
scale = 1e4
#
#failing geometric
# parameters = {'b1': 5e-6 * scale, 'L1': 0.3e-3 * scale, 'L2': 0.7e-3 * scale, 'L3': 1e-3 * scale,
#               'dA': 1e-7 * scale, 'dB': 5e-7 * scale, 'dC': 5e-7 * scale,
#               'EA': 2e11, 'EB': 7e10, 'EC': 2e11,
#               'rhoA': 8700, 'rhoB': 2700, 'rhoC': 8700}

parameters = {'b1': 2e-6 * scale, 'L1': 1e-6 * scale, 'L2': 1e-6 * scale, 'L3': 1e-6 * scale,
                  'dA': 1e-7 * scale, 'dB': 1e-7 * scale, 'dC': 1e-7 * scale,
                  'EA': E_pt, 'EB': E_scaln, 'EC': E_ni,
                  'rhoA': rho_pt, 'rhoB': rho_scaln, 'rhoC': rho_ni}

# b 10, 100 um
# dA 100, 500 nm
# L 10, 1000 um
#

time1 = time.time() - startTime
# beam = eb3.euber(*bndmiddl, *mat_param, N=100)
beam = eb3.euber(parameters=parameters, N=100)#, mode="symbolic")
# beam = eb3.euber(parameters=parameters, N=2e-2)
time2 = time.time() - startTime - time1
time21 = beam.time_init1
time22 = beam.time_init2
time23 = beam.time_init3

beam.prec = 1000
beam.lyrd_beam_reduced_params()

time3 = (time.time() - time2 - time1)
ekf = beam.get_first_n_ekf_bruteforce(n=1, om=1000, stepsize=200)
time4 = ekf[3]
time41 = sum(ekf[4])
time42 = sum(ekf[5])
time_t = time.time() - startTime

print(f" Set up: {time1} \n Initialisation: {time2} (Parameters: {time21}, Taus: {time22} , Misc: {time23})"
      f" \n Matrix Computations: {time41} \n Determinant Computation:"
      f" {time42} \n Root finding: {time4 - (time41 + time42)} \n Total time: {time_t}")

value = ekf[0] / 2 / np.pi
print(ekf[0] / 2 / np.pi)  # cmp to MAMM2020 Paper

executionTime = (time.time() - startTime)
print(executionTime)

"""
 Set up: 0.0 
 Initialisation: 0.0 (Parameters: 0.0, Taus: 0.0 , Misc: 0.0) 
 Matrix Computations: 4.287548303604126 
 Determinant Computation: 0.3720064163208008 
 additional Computation: 83.04391241073608 
 Total time: 87.70446467399597]"""