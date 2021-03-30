# Version 3 with steiner implementation for non-mid section 2
import sympy as sp
from sympy import Matrix as syma
from sympy import diff
import numpy as np
import scipy.optimize
import pdb
import mpmath as mp
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import scipy as sc
from scipy.sparse.linalg import splu
import time
import warnings


class euber(object):

    def __init__(self, parameters, b1=0.03, b2=0.05, b3=0.03, L1=0.3, L2=0.7, L3=1.0, dA=0.02, dB=0.01, dC=0.02,
                 EA=200.0e9,
                 EB=70.0e9, EC=200.0e9, rhoA=8700.0, rhoB=2700.0, rhoC=8700.0, N=100, mode="default", tau_alt=[]):
        eigenvalues_of_simple_bridge = [4.7300407449, 7.8532046241, 10.9956078382, 14.1371654913, 17.2787596574,
                                        20.4203522456]
        init_start = time.time()
        self.parameters = parameters.copy()
        self.N = N
        self.mode = mode
        self.kappa = 0
        self.tau_alt = tau_alt
        if type(self.tau_alt) != list:
            self.tau_alt = [self.tau_alt]

        if self.mode == "symbolic":
            # sympy stuff
            self.x = sp.symbols('x')
            tau1, tau2 = sp.symbols('tau1 tau2')
            tau11, tau12, tau13 = sp.symbols('tau11 tau12 tau13')
            tau21, tau22, tau23 = sp.symbols('tau21 tau22 tau23')

            self.Lsym_1, self.Lsym_2, self.Lsym_3 = sp.symbols('Lsym_1 Lsym_2 Lsym_3')
            self.Esym_1, self.Esym_2, self.Esym_3 = sp.symbols('Esym_1 Esym_2 Esym_3')
            self.Isym_1, self.Isym_2, self.Isym_3 = sp.symbols('Isym_1 Isym_2 Isym_3')

            self.smbls = ['tau11', 'tau21', 'tau12', 'tau22', 'tau13', 'tau23',
                          'Esym_1', 'Esym_2', 'Esym_3',
                          'Isym_1', 'Isym_2', 'Isym_3',
                          'Lsym_1', 'Lsym_2', 'Lsym_3']


            # trancendet Ansatz functions for E-B-Beam with 3 sections and axial tension
            TRAN_axt = ([sp.cos(self.x * tau1), sp.sin(self.x * tau1), sp.cosh(self.x * tau2), sp.sinh(self.x * tau2)])
            self.X1_axt = [i.subs([(tau1, tau11)
                                      , (tau2, tau21)]) for i in TRAN_axt]
            self.X2_axt = [i.subs([(tau1, tau12)
                                      , (tau2, tau22)]) for i in TRAN_axt]
            self.X3_axt = [i.subs([(tau1, tau13)
                                      , (tau2, tau23)]) for i in TRAN_axt]

            # Calculating lambda initially to avoid multiple calculations
            self.BCMatrix_3sec_axt_lambda = self.get_BCMatrix_3sec_axt_lambda()

        # mpmath
        # calculation of taus for normal case
        # tau_sec_axt_mp_alt is used for alternative omitted tau
        tau_start = time.time() - init_start
        self.tau1_sec1_axt_mp = lambda omega, k1, p1: mp.sqrt(2) * mp.sqrt(
            (mp.sqrt(4 * k1 ** 4 * omega ** 2 + p1 ** 4) - p1 ** 2) / k1 ** 4) / 2  # sin&cos = LOI2
        self.tau1_sec2_axt_mp = lambda lambda_1, k1, k2, p1, p2: ((1 / 2) * mp.sqrt(-2 * p2 ** 4 + 2 * mp.sqrt(
            4 * k1 ** 4 * k2 ** 4 * lambda_1 ** 4 + 4 * k2 ** 4 * lambda_1 ** 2 * p1 ** 4 + p2 ** 8)) / k2 ** 2)
        self.tau1_sec3_axt_mp = lambda lambda_1, k1, k3, p1, p3: ((1 / 2) * mp.sqrt(-2 * p3 ** 4 + 2 * mp.sqrt(
            4 * k1 ** 4 * k3 ** 4 * lambda_1 ** 4 + 4 * k3 ** 4 * lambda_1 ** 2 * p1 ** 4 + p3 ** 8)) / k3 ** 2)

        self.tau2_sec1_axt_mp = lambda omega, k1, p1: mp.sqrt(2) * mp.sqrt(
            (p1 ** 2 + mp.sqrt(4 * k1 ** 4 * omega ** 2 + p1 ** 4)) / k1 ** 4) / 2  # sinh&cosh = LOI1
        self.tau2_sec2_axt_mp = lambda lambda_1, k1, k2, p1, p2: ((1 / 2) * mp.sqrt(2 * p2 ** 4 + 2 * mp.sqrt(
            4 * k1 ** 4 * k2 ** 4 * lambda_1 ** 4 - 4 * k2 ** 4 * lambda_1 ** 2 * p1 ** 4 + p2 ** 8)) / k2 ** 2)
        self.tau2_sec3_axt_mp = lambda lambda_1, k1, k3, p1, p3: ((1 / 2) * mp.sqrt(2 * p3 ** 4 + 2 * mp.sqrt(
            4 * k1 ** 4 * k3 ** 4 * lambda_1 ** 4 - 4 * k3 ** 4 * lambda_1 ** 2 * p1 ** 4 + p3 ** 8)) / k3 ** 2)
        self.tau2_23_alt = lambda lambda_1, k1, k2, p1, p2: ((1 / 2) * mp.sqrt(2 * p2 ** 4 - 2 * mp.sqrt(
            4 * k1 ** 4 * k2 ** 4 * lambda_1 ** 4 - 4 * k2 ** 4 * lambda_1 ** 2 * p1 ** 4 + p2 ** 8)) / k2 ** 2)
        tau_time = time.time() - init_start

        # calculation of taus with LOI1 and LOI2 inverted (this was used to revert false imaginary roots to real values)

        self.tau1_sec1_axt_mp_alt = lambda omega, k1, p1: mp.sqrt(2) * mp.sqrt(
            (mp.sqrt(4 * k1 ** 4 * omega ** 2 + p1 ** 4) + p1 ** 2) / k1 ** 4) / 2  # sin&cos = LOI2
        self.tau1_sec2_axt_mp_alt = lambda lambda_1, k1, k2, p1, p2: ((1 / 2) * mp.sqrt(2 * p2 ** 4 + 2 * mp.sqrt(
            4 * k1 ** 4 * k2 ** 4 * lambda_1 ** 4 + 4 * k2 ** 4 * lambda_1 ** 2 * p1 ** 4 + p2 ** 8)) / k2 ** 2)
        self.tau1_sec3_axt_mp_alt = lambda lambda_1, k1, k3, p1, p3: ((1 / 2) * mp.sqrt(2 * p3 ** 4 + 2 * mp.sqrt(
            4 * k1 ** 4 * k3 ** 4 * lambda_1 ** 4 + 4 * k3 ** 4 * lambda_1 ** 2 * p1 ** 4 + p3 ** 8)) / k3 ** 2)

        self.tau2_sec1_axt_mp_alt = lambda omega, k1, p1: mp.sqrt(2) * mp.sqrt(
            (-p1 ** 2 + mp.sqrt(4 * k1 ** 4 * omega ** 2 + p1 ** 4)) / k1 ** 4) / 2  # sinh&cosh = LOI1
        self.tau2_sec2_axt_mp_alt = lambda lambda_1, k1, k2, p1, p2: ((1 / 2) * mp.sqrt(-2 * p2 ** 4 + 2 * mp.sqrt(
            4 * k1 ** 4 * k2 ** 4 * lambda_1 ** 4 - 4 * k2 ** 4 * lambda_1 ** 2 * p1 ** 4 + p2 ** 8)) / k2 ** 2)
        self.tau2_sec3_axt_mp_alt = lambda lambda_1, k1, k3, p1, p3: ((1 / 2) * mp.sqrt(-2 * p3 ** 4 + 2 * mp.sqrt(
            4 * k1 ** 4 * k3 ** 4 * lambda_1 ** 4 - 4 * k3 ** 4 * lambda_1 ** 2 * p1 ** 4 + p3 ** 8)) / k3 ** 2)

        # numpy

        self.tau1_sec1_axt_np = lambda omega, k1, p1: np.sqrt(2) * np.sqrt(
            (np.sqrt(4 * k1 ** 4 * omega ** 2 + p1 ** 4) - p1 ** 2) / k1 ** 4) / 2  # sin&cos = LOI2
        self.tau1_sec2_axt_np = lambda lambda_1, k1, k2, p1, p2: ((1 / 2) * np.sqrt(-2 * p2 ** 4 + 2 * np.sqrt(
            4 * k1 ** 4 * k2 ** 4 * lambda_1 ** 4 + 4 * k2 ** 4 * lambda_1 ** 2 * p1 ** 4 + p2 ** 8)) / k2 ** 2)
        self.tau1_sec3_axt_np = lambda lambda_1, k1, k3, p1, p3: ((1 / 2) * np.sqrt(-2 * p3 ** 4 + 2 * np.sqrt(
            4 * k1 ** 4 * k3 ** 4 * lambda_1 ** 4 + 4 * k3 ** 4 * lambda_1 ** 2 * p1 ** 4 + p3 ** 8)) / k3 ** 2)

        self.tau2_sec1_axt_np = lambda omega, k1, p1: np.sqrt(2) * np.sqrt(
            (p1 ** 2 + np.sqrt(4 * k1 ** 4 * omega ** 2 + p1 ** 4)) / k1 ** 4) / 2  # sinh&cosh = LOI1
        self.tau2_sec2_axt_np = lambda lambda_1, k1, k2, p1, p2: ((1 / 2) * np.sqrt(2 * p2 ** 4 + 2 * np.sqrt(
            4 * k1 ** 4 * k2 ** 4 * lambda_1 ** 4 - 4 * k2 ** 4 * lambda_1 ** 2 * p1 ** 4 + p2 ** 8)) / k2 ** 2)
        self.tau2_sec3_axt_np = lambda lambda_1, k1, k3, p1, p3: ((1 / 2) * np.sqrt(2 * p3 ** 4 + 2 * np.sqrt(
            4 * k1 ** 4 * k3 ** 4 * lambda_1 ** 4 - 4 * k3 ** 4 * lambda_1 ** 2 * p1 ** 4 + p3 ** 8)) / k3 ** 2)

        self.time_init1 = tau_start
        self.time_init2 = tau_time - tau_start
        self.time_init3 = time.time() - init_start - tau_start - self.time_init2

    def define_simple_beam(self):

        self.A_1, self.I_1 = self.get_area_and_inertia_rectangle(self.parameters['b_1'], self.h_1)
        self.A_2, self.I_2 = self.get_area_and_inertia_rectangle(self.parameters['b_2'], self.h_2)
        self.A_3, self.I_3 = self.get_area_and_inertia_rectangle(self.parameters['b_3'], self.h_3)
        self.k_1, self.p_1 = self.get_k_and_p_value(self.A_1, self.I_1, self.parameters['EA'], self.parameters['rhoA'],
                                                    self.N)
        self.k_2, self.p_2 = self.get_k_and_p_value(self.A_2, self.I_2, self.parameters['EB'], self.parameters['rhoB'],
                                                    self.N)
        self.k_3, self.p_3 = self.get_k_and_p_value(self.A_3, self.I_3, self.parameters['EC'], self.parameters['rhoC'],
                                                    self.N)

    def define_layered_beam(self):  # uniform

        # Layers
        self.b_2 = self.parameters['b1']  # !!
        self.b_3 = self.parameters['b1']  # !!
        self.h_1 = self.parameters['dA'] + self.parameters['dB'] + self.parameters['dC']
        self.h_2 = self.parameters['dA'] + self.parameters['dB']
        self.h_3 = self.h_1

        # initialize parameter arrays
        # Sections 123
        self.b = np.array([self.parameters['b1'], self.b_2, self.b_3])
        helper = self.parameters['dA'] + self.parameters['dB'] + self.parameters['dC']
        self.height = np.array([helper, helper - self.parameters['dC'], helper])  # only if A_1  = A3

        # Layers abc
        self.d = np.array([self.parameters['dA'], self.parameters['dB'], self.parameters['dC']])
        self.ys = self.get_centerpoints_of_layers_rectangle(*self.d)  # center points of each Layer
        self.A_layers, self.I_layers = self.get_area_and_inertia_rectangle(self.b,
                                                                           self.d)  # only if b = [b1,b1,b1]=[bA,bB,bC]
        self.E_layers = np.array([self.parameters['EA'], self.parameters['EB'], self.parameters['EC']])

        neutral_layer = self.get_neutral_layer(self.ys[[0, 1, 2]], self.A_layers[[0, 1, 2]], self.E_layers[[0, 1, 2]])
        offset_to_neutral_layer = self.ys[[0, 1, 2]] - neutral_layer
        self.I_layers123_steiner = np.array(
            self.I_layers[[0, 1, 2]] + offset_to_neutral_layer ** 2 * self.A_layers[[0, 1, 2]])

        neutral_layer = self.get_neutral_layer(self.ys[[0, 1]], self.A_layers[[0, 1]], self.E_layers[[0, 1]])
        offset_to_neutral_layer = self.ys[[0, 1]] - neutral_layer
        self.I_layers12_steiner = np.array(self.I_layers[[0, 1]] + offset_to_neutral_layer ** 2 * self.A_layers[[0, 1]])

        self.rho_layers = np.array([self.parameters['rhoA'], self.parameters['rhoB'], self.parameters['rhoC']])
        self.rho_layers_weighted = self.rho_layers * self.A_layers

        # self.get_section_E([EA, EB, EC], [dA, dB, dC])
        self.I_1 = np.sum(self.I_layers123_steiner)
        self.I_2 = np.sum(self.I_layers12_steiner)
        dist_of_neutral_layers = self.h_1 / 2 - self.h_2 / 2

        self.I_3 = self.I_1

        self.E_1 = np.sum(self.I_layers123_steiner * self.E_layers) / self.I_1
        self.E_2 = np.sum(self.I_layers12_steiner * self.E_layers[:2]) / self.I_2
        self.E_3 = self.E_1  # Seciton1 and 3 have same layers
        self.I_2 = self.I_2 + dist_of_neutral_layers ** 2 * np.sum(self.A_layers[:1])

        self.A_1 = np.sum(self.A_layers)
        self.A_2 = np.sum(self.A_layers[:2])
        self.A_3 = self.A_1  # Seciton1 and 3 have same layers
        self.rho_1 = np.sum(self.rho_layers_weighted) / self.A_1
        self.rho_2 = np.sum(self.rho_layers_weighted[:2]) / self.A_2
        self.rho_3 = self.rho_1  # Seciton1 and 3 have same layers

        self.k_1, self.p_1 = self.get_k_and_p_mp(self.A_1, self.I_1, self.E_1, self.rho_1, self.N)
        self.k_2, self.p_2 = self.get_k_and_p_mp(self.A_2, self.I_2, self.E_2, self.rho_2, self.N)
        self.k_3, self.p_3 = self.get_k_and_p_mp(self.A_3, self.I_3, self.E_3, self.rho_3, self.N)

    # Systemmatrix for beam with 3 sections and axial tension
    # #######################################################
    def BCMatrix_3sec_axt(self):

        """
        Function that creates the symbolic BC Matrix using predefined Sympy expressions
        :return: BC, sympy array of dimension 12x12
        """

        BC = sp.zeros(12, 12)
        # Boundary Conditions left x==|--|==|
        # sets x=0 for cos, sin, cosh, sinh functions
        BC[0, :4] = syma([[i.subs(self.x, 0) for i in self.X1_axt]])
        BC[1, :4] = syma([[diff(i, self.x).subs(self.x, 0) for i in self.X1_axt]])
        # Transition Conditions left |==x--|==| Lsym_1
        BC[2, :8] = syma([[i.subs(self.x, self.Lsym_1) for i in self.X1_axt] +
                          [i.subs(self.x, self.Lsym_1) * (-1) for i in self.X2_axt]])
        BC[3, :8] = syma([[diff(i, self.x).subs(self.x, self.Lsym_1) for i in self.X1_axt] +
                          [diff(i, self.x).subs(self.x, self.Lsym_1) * (-1) for i in self.X2_axt]])
        BC[4, :8] = syma([[self.Esym_1 * self.Isym_1 * diff(i, self.x, self.x).subs(self.x, self.Lsym_1) * (-1)
                           for i in self.X1_axt] + [self.Esym_2 * self.Isym_2 * diff(i, self.x, self.x).subs(self.x,
                                                                                                             self.Lsym_1)
                                                    for i in self.X2_axt]])
        BC[5, :8] = syma([[self.Esym_1 * self.Isym_1 * diff(i, self.x, self.x, self.x).subs(self.x, self.Lsym_1)
                           for i in self.X1_axt] + [
                              self.Esym_2 * self.Isym_2 * diff(i, self.x, self.x, self.x).subs(self.x,
                                                                                               self.Lsym_1) * (-1) for i
                              in self.X2_axt]])
        # Transition Conditions right |==|--x==|
        BC[6, 4:] = syma([[i.subs(self.x, self.Lsym_2) for i in self.X2_axt] + [i.subs(self.x, self.Lsym_2) * (-1)
                                                                                for i in self.X3_axt]])
        BC[7, 4:] = syma([[diff(i, self.x).subs(self.x, self.Lsym_2) for i in self.X2_axt] +
                          [diff(i, self.x).subs(self.x, self.Lsym_2) * (-1) for i in self.X3_axt]])
        BC[8, 4:] = syma([[self.Esym_2 * self.Isym_2 * diff(i, self.x, self.x).subs(self.x, self.Lsym_2) * (-1) for i in
                           self.X2_axt] + [self.Esym_3 * self.Isym_3 * diff(i, self.x, self.x).subs(self.x, self.Lsym_2)
                                           for i in self.X3_axt]])
        BC[9, 4:] = syma(
            [[self.Esym_2 * self.Isym_2 * diff(i, self.x, self.x, self.x).subs(self.x, self.Lsym_2) for i in
              self.X2_axt] + [self.Esym_3 * self.Isym_3 * diff(i, self.x, self.x, self.x).subs(self.x,
                                                                                               self.Lsym_2) * (-1) for i
                              in self.X3_axt]])
        # Boundary Conditions right |==|--|==x
        BC[10, 8:] = syma([[i.subs(self.x, self.Lsym_3) for i in self.X3_axt]])
        BC[11, 8:] = syma([[diff(i, self.x).subs(self.x, self.Lsym_3) for i in self.X3_axt]])
        return BC

    def get_BCMatrix_3sec_axt_lambda(self):

        smbls = ['tau11', 'tau21', 'tau12', 'tau22', 'tau13', 'tau23',
                 'Esym_1', 'Esym_2', 'Esym_3',
                 'Isym_1', 'Isym_2', 'Isym_3',
                 'Lsym_1', 'Lsym_2', 'Lsym_3']
        symdet = self.BCMatrix_3sec_axt()
        return sp.utilities.lambdify(smbls, symdet, modules='mpmath')

    def get_BCMatrix_3sec_axt_numpy(self):

        self.matrix_lambda = sp.utilities.lambdify(self.smbls, self.BCMatrix_3sec_axt(), modules='numpy')

    def calc_taus(self, omega):
        mp.mp.dps = self.prec

        # Section 1
        self.tau11 = mp.fabs(self.tau1_sec1_axt_mp(omega, self.k_1, self.p_1))
        self.tau21 = mp.fabs(self.tau2_sec1_axt_mp(omega, self.k_1, self.p_1))
        # Section 2
        self.tau12 = mp.fabs(self.tau1_sec2_axt_mp(self.tau11, self.k_1, self.k_2, self.p_1, self.p_2))
        self.tau22 = mp.fabs(self.tau2_sec2_axt_mp(self.tau21, self.k_1, self.k_2, self.p_1, self.p_2))
        # Section 3
        self.tau13 = mp.fabs(self.tau1_sec3_axt_mp(self.tau11, self.k_1, self.k_3, self.p_1, self.p_3))
        self.tau23 = mp.fabs(self.tau2_sec3_axt_mp(self.tau21, self.k_1, self.k_3, self.p_1, self.p_3))

        if 11 in self.tau_alt:
            self.tau11 = mp.fabs(self.tau1_sec1_axt_mp_alt(omega, self.k_1, self.p_1))
        if 21 in self.tau_alt:
            self.tau21 = mp.fabs(self.tau2_sec1_axt_mp_alt(omega, self.k_1, self.p_1))
        # Section 2
        if 12 in self.tau_alt:
            self.tau12 = mp.fabs(self.tau1_sec2_axt_mp_alt(self.tau11, self.k_1, self.k_2, self.p_1, self.p_2))
        if 22 in self.tau_alt:
            self.tau22 = mp.fabs(self.tau2_sec2_axt_mp_alt(self.tau21, self.k_1, self.k_2, self.p_1, self.p_2))
        # Section 3
        if 13 in self.tau_alt:
            self.tau13 = mp.fabs(self.tau1_sec3_axt_mp_alt(self.tau11, self.k_1, self.k_3, self.p_1, self.p_3))
        if 23 in self.tau_alt:
            self.tau23 = mp.fabs(self.tau2_sec3_axt_mp_alt(self.tau21, self.k_1, self.k_3, self.p_1, self.p_3))

        if self.mode != "symbolic":
            self.taus = [self.tau11, self.tau21, self.tau12, self.tau22, self.tau13, self.tau23]

    def get_BCMatrix_3sec_axt_det(self):

        mp.mp.dps = self.prec

        #use symbolic mode to create symbolic BCMatrix that is then transformed with sympy lambdify into a mp.math matrix
        if self.mode == "symbolic":
            matrix_start = time.time()
            mat = self.BCMatrix_3sec_axt_lambda(self.tau11, self.tau21,
                                                self.tau12, self.tau22,
                                                self.tau13, self.tau23,
                                                self.E_1, self.E_2, self.E_3,
                                                self.I_1, self.I_2, self.I_3,
                                                self.parameters['L1'], self.parameters['L2'], self.parameters['L3'])
            self.matrix_time = time.time() - matrix_start
        else:
            matrix_start = time.time()
            self.get_BC_Matrix()
            mat = self.BCM
            self.matrix_time = time.time() - matrix_start
        det_start = time.time()
        det = mp.det(mat)
        self.det_time = time.time() - det_start
        self.kappa = 0
        # calculate condition number kappa from stretching factors in sigma by using SVD
        # sigma = mp.svd_r(mat, compute_uv=False)
        # zeros in sigma represent a matrix without full rank, meaning the BCMatrix is singular due to low precision
        # if min(sigma) == 0:
        #     kappa = max(sigma) / min(sigma)
        #     kappa_float = np.float(mp.log10(kappa))
        #     self.kappa = kappa_float

        if det == 0:
            warnings.warn("BCMatrix is singular, a much higher precision is needed!")
            self.kappa = np.inf
            return -np.inf
        else:

        # check legitemacy of log transform around the zero transition (log transform predicts +-1 transition?)
            return np.float64(mp.log(mp.fabs(det)) * mp.sign(det))
        # return det

    def get_BC_Matrix(self):

        """
        Function that creates the BC Matrix using mpmath and the previous calculated variables.
        The resulting Matrix is stored as self.BCM.
        """

        self.BCM = mp.matrix([[1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, self.taus[0], 0, self.taus[1], 0, 0, 0, 0, 0, 0, 0, 0],
                              [mp.cos(self.parameters['L1'] * self.taus[0]),
                               mp.sin(self.parameters['L1'] * self.taus[0]),
                               mp.cosh(self.parameters['L1'] * self.taus[1]),
                               mp.sinh(self.parameters['L1'] * self.taus[1]),
                               -mp.cos(self.parameters['L1'] * self.taus[2]),
                               -mp.sin(self.parameters['L1'] * self.taus[2]),
                               -mp.cosh(self.parameters['L1'] * self.taus[3]),
                               -mp.sinh(self.parameters['L1'] * self.taus[3]),
                               0, 0, 0, 0],
                              [-self.taus[0] * mp.sin(self.parameters['L1'] * self.taus[0]),
                               self.taus[0] * mp.cos(self.parameters['L1'] * self.taus[0]),
                               self.taus[1] * mp.sinh(self.parameters['L1'] * self.taus[1]),
                               self.taus[1] * mp.cosh(self.parameters['L1'] * self.taus[1]),
                               self.taus[2] * mp.sin(self.parameters['L1'] * self.taus[2]),
                               -self.taus[2] * mp.cos(self.parameters['L1'] * self.taus[2]),
                               -self.taus[3] * mp.sinh(self.parameters['L1'] * self.taus[3]),
                               -self.taus[3] * mp.cosh(self.parameters['L1'] * self.taus[3]),
                               0, 0, 0, 0],
                              [self.E_1 * self.I_1 * self.taus[0] ** 2 * mp.cos(self.parameters['L1'] * self.taus[0]),
                               self.E_1 * self.I_1 * self.taus[0] ** 2 * mp.sin(self.parameters['L1'] * self.taus[0]),
                               -self.E_1 * self.I_1 * self.taus[1] ** 2 * mp.cosh(self.parameters['L1'] * self.taus[1]),
                               -self.E_1 * self.I_1 * self.taus[1] ** 2 * mp.sinh(self.parameters['L1'] * self.taus[1]),
                               -self.E_2 * self.I_2 * self.taus[2] ** 2 * mp.cos(self.parameters['L1'] * self.taus[2]),
                               -self.E_2 * self.I_2 * self.taus[2] ** 2 * mp.sin(self.parameters['L1'] * self.taus[2]),
                               self.E_2 * self.I_2 * self.taus[3] ** 2 * mp.cosh(self.parameters['L1'] * self.taus[3])
                                  ,
                               self.E_2 * self.I_2 * self.taus[3] ** 2 * mp.sinh(self.parameters['L1'] * self.taus[3]),
                               0, 0, 0, 0],
                              [self.E_1 * self.I_1 * self.taus[0] ** 3 * mp.sin(self.parameters['L1'] * self.taus[0]),
                               -self.E_1 * self.I_1 * self.taus[0] ** 3 * mp.cos(self.parameters['L1'] * self.taus[0]),
                               self.E_1 * self.I_1 * self.taus[1] ** 3 * mp.sinh(self.parameters['L1'] * self.taus[1]),
                               self.E_1 * self.I_1 * self.taus[1] ** 3 * mp.cosh(self.parameters['L1'] * self.taus[1]),
                               -self.E_2 * self.I_2 * self.taus[2] ** 3 * mp.sin(self.parameters['L1'] * self.taus[2]),
                               self.E_2 * self.I_2 * self.taus[2] ** 3 * mp.cos(self.parameters['L1'] * self.taus[2]),
                               -self.E_2 * self.I_2 * self.taus[3] ** 3 * mp.sinh(self.parameters['L1'] * self.taus[3]),
                               -self.E_2 * self.I_2 * self.taus[3] ** 3 * mp.cosh(self.parameters['L1'] * self.taus[3]),
                               0, 0, 0, 0],
                              [0, 0, 0, 0,
                               mp.cos(self.parameters['L2'] * self.taus[2]),
                               mp.sin(self.parameters['L2'] * self.taus[2]),
                               mp.cosh(self.parameters['L2'] * self.taus[3]),
                               mp.sinh(self.parameters['L2'] * self.taus[3]),
                               -mp.cos(self.parameters['L2'] * self.taus[4]),
                               -mp.sin(self.parameters['L2'] * self.taus[4]),
                               -mp.cosh(self.parameters['L2'] * self.taus[5]),
                               -mp.sinh(self.parameters['L2'] * self.taus[5])],
                              [0, 0, 0, 0,
                               -self.taus[2] * mp.sin(self.parameters['L2'] * self.taus[2]),
                               self.taus[2] * mp.cos(self.parameters['L2'] * self.taus[2]),
                               self.taus[3] * mp.sinh(self.parameters['L2'] * self.taus[3]),
                               self.taus[3] * mp.cosh(self.parameters['L2'] * self.taus[3]),
                               self.taus[4] * mp.sin(self.parameters['L2'] * self.taus[4]),
                               -self.taus[4] * mp.cos(self.parameters['L2'] * self.taus[4]),
                               -self.taus[5] * mp.sinh(self.parameters['L2'] * self.taus[5]),
                               -self.taus[5] * mp.cosh(self.parameters['L2'] * self.taus[5])],
                              [0, 0, 0, 0,
                               self.E_2 * self.I_2 * self.taus[2] ** 2 * mp.cos(self.parameters['L2'] * self.taus[2]),
                               self.E_2 * self.I_2 * self.taus[2] ** 2 * mp.sin(self.parameters['L2'] * self.taus[2]),
                               -self.E_2 * self.I_2 * self.taus[3] ** 2 * mp.cosh(self.parameters['L2'] * self.taus[3]),
                               -self.E_2 * self.I_2 * self.taus[3] ** 2 * mp.sinh(self.parameters['L2'] * self.taus[3]),
                               -self.E_3 * self.I_3 * self.taus[4] ** 2 * mp.cos(self.parameters['L2'] * self.taus[4]),
                               -self.E_3 * self.I_3 * self.taus[4] ** 2 * mp.sin(self.parameters['L2'] * self.taus[4]),
                               self.E_3 * self.I_3 * self.taus[5] ** 2 * mp.cosh(self.parameters['L2'] * self.taus[5]),
                               self.E_3 * self.I_3 * self.taus[5] ** 2 * mp.sinh(self.parameters['L2'] * self.taus[5])],
                              [0, 0, 0, 0,
                               self.E_2 * self.I_2 * self.taus[2] ** 3 * mp.sin(self.parameters['L2'] * self.taus[2]),
                               -self.E_2 * self.I_2 * self.taus[2] ** 3 * mp.cos(self.parameters['L2'] * self.taus[2]),
                               self.E_2 * self.I_2 * self.taus[3] ** 3 * mp.sinh(self.parameters['L2'] * self.taus[3]),
                               self.E_2 * self.I_2 * self.taus[3] ** 3 * mp.cosh(self.parameters['L2'] * self.taus[3]),
                               -self.E_3 * self.I_3 * self.taus[4] ** 3 * mp.sin(self.parameters['L2'] * self.taus[4]),
                               self.E_3 * self.I_3 * self.taus[4] ** 3 * mp.cos(self.parameters['L2'] * self.taus[4]),
                               -self.E_3 * self.I_3 * self.taus[5] ** 3 * mp.sinh(self.parameters['L2'] * self.taus[5]),
                               -self.E_3 * self.I_3 * self.taus[5] ** 3 * mp.cosh(
                                   self.parameters['L2'] * self.taus[5])],
                              [0, 0, 0, 0, 0, 0, 0, 0,
                               mp.cos(self.parameters['L3'] * self.taus[4]),
                               mp.sin(self.parameters['L3'] * self.taus[4]),
                               mp.cosh(self.parameters['L3'] * self.taus[5]),
                               mp.sinh(self.parameters['L3'] * self.taus[5])],
                              [0, 0, 0, 0, 0, 0, 0, 0,
                               -self.taus[4] * mp.sin(self.parameters['L3'] * self.taus[4]),
                               self.taus[4] * mp.cos(self.parameters['L3'] * self.taus[4]),
                               self.taus[5] * mp.sinh(self.parameters['L3'] * self.taus[5]),
                               self.taus[5] * mp.cosh(self.parameters['L3'] * self.taus[5])]])

    def peek_det(self, oms):
        mp.mp.dps = self.prec
        det = []
        for om in oms:
            self.calc_taus(om)
            det.append(self.get_BCMatrix_3sec_axt_det())
        return np.array(det)

    def peek_det_scalar(self, om):
        mp.mp.dps = self.prec
        if self.mode == "Alt_tau":
            self.calc_taus_alt(om)
        else:
            self.calc_taus(om)
        return self.get_BCMatrix_3sec_axt_det()

    def lyrd_beam_reduced_params(self):  # , b1,
        # l1, l2, l3,
        # dA, dB, dC,
        # EA=200.0e9, EB=70.0e9, EC=200.0e9,    #Layers
        # rhoA=8700.0, rhoB=2700.0, rhoC=8700.0,#Layers
        # N=1000):
        L1, L2, L3 = self.parameters['L1'], self.parameters['L2'], self.parameters['L3']
        self.parameters['L2'] = L1 + L2
        self.parameters['L3'] = L1 + L2 + L3
        self.define_layered_beam()

    # search angular eigenfrequencies
    def get_first3_ekf(self):
        # does not work on random structures
        det = []
        om = 100.0
        while True:
            det.append(self.peek_det_scalar(om))
            if len(det) > 1 and np.sign(det[-2]) != np.sign(det[-1]):
                om1 = scipy.optimize.brentq(self.peek_det_scalar, om - 100, om)
                break
            om += 100
        om2 = scipy.optimize.brentq(self.peek_det_scalar, om1 * 2, om1 * 3, xtol=1e-2)
        om3 = scipy.optimize.brentq(self.peek_det_scalar, om1 * 3, om2 * 3, xtol=1e-2)
        return np.array([om1, om2, om3])

    def get_first_n_ekf_bruteforce(self, n=3, om=100.0, stepsize=100.0, brentq_xtol=1e-2):
        count = 0
        ekf_start = time.time()
        det_times = []
        matrix_times = []
        det = []
        oms = []
        ekfs = []
        while True:
            det.append(self.peek_det_scalar(om))
            det_times.append(self.det_time)
            matrix_times.append(self.matrix_time)
            oms.append(om)
            if len(det) > 1 and np.sign(det[-2]) != np.sign(det[-1]):
                ekfs.append(scipy.optimize.brentq(self.peek_det_scalar, oms[-2], oms[-1], xtol=brentq_xtol))
                stepsize = ekfs[0]
                if len(ekfs) == n:
                    return np.array(ekfs), det, oms, time.time() - ekf_start, matrix_times, det_times

            om += stepsize
            count += 1
            if len(det) == 3 and max(det) == -np.inf:
            # if max(det) == -np.inf:
                return np.zeros(3)


    ### alternative eigenvalue calc
    def calc_taus_alt(self, omega):
        mp.mp.dps = self.prec

        # Section 1
        self.tau11 = mp.fabs(self.tau1_sec1_axt_mp(omega, self.k_1, self.p_1))
        self.tau21 = mp.fabs(self.tau2_sec1_axt_mp(omega, self.k_1, self.p_1))
        # Section 2
        self.tau12 = mp.fabs(self.tau1_sec2_axt_mp(self.tau11, self.k_1, self.k_2, self.p_1, self.p_2))
        self.tau22 = mp.fabs(self.tau2_sec2_axt_mp(self.tau21, self.k_1, self.k_2, self.p_1, self.p_2))
        # Section 3
        self.tau13 = mp.fabs(self.tau1_sec3_axt_mp(self.tau11, self.k_1, self.k_3, self.p_1, self.p_3))
        self.tau23 = mp.fabs(self.tau2_23_alt(self.tau21, self.k_1, self.k_3, self.p_1, self.p_3))

        if self.mode != "symbolic":
            self.taus = [self.tau11, self.tau21, self.tau12, self.tau22, self.tau13, self.tau23]

    def calc_taus_alt_1(self, omega):
        mp.mp.dps = self.prec

        # Section 1
        self.tau11 = mp.fabs(self.tau1_sec1_axt_mp(omega, self.k_1, self.p_1))
        self.tau21 = mp.fabs(self.tau2_sec1_axt_mp(omega, self.k_1, self.p_1))
        # Section 2
        self.tau12 = mp.fabs(self.tau1_sec2_axt_mp_alt(self.tau11, self.k_1, self.k_2, self.p_1, self.p_2))
        self.tau22 = mp.fabs(self.tau2_sec2_axt_mp_alt(self.tau21, self.k_1, self.k_2, self.p_1, self.p_2))
        # Section 3
        self.tau13 = mp.fabs(self.tau1_sec3_axt_mp(self.tau11, self.k_1, self.k_3, self.p_1, self.p_3))
        self.tau23 = mp.fabs(self.tau2_23_alt(self.tau21, self.k_1, self.k_3, self.p_1, self.p_3))

        if self.mode != "symbolic":
            self.taus = [self.tau11, self.tau21, self.tau12, self.tau22, self.tau13, self.tau23]

    def get_first_n_ekf_bruteforcEAlt(self, n=3, om=100.0, stepsize=100.0, brentq_xtol=1e-2):
        det = []
        oms = []
        ekfs = []
        while True:
            det.append(self.peek_det_scalar_alt(om))
            oms.append(om)
            if len(det) > 1 and np.sign(det[-2]) != np.sign(det[-1]):
                if self.mode == "alt_taus":
                    ekfs.append(scipy.optimize.brentq(self.peek_det_scalar_alt, oms[-2], oms[-1], xtol=brentq_xtol))
                elif self.mode == "alt_taus_1":
                    ekfs.append(scipy.optimize.brentq(self.peek_det_scalar_alt, oms[-2], oms[-1], xtol=brentq_xtol))
                elif self.mode == "alt_taus_2":
                    ekfs.append(scipy.optimize.brentq(self.peek_det_scalar_alt, oms[-2], oms[-1], xtol=brentq_xtol))
                else:
                    ekfs.append(scipy.optimize.brentq(self.peek_det_scalar, oms[-2], oms[-1], xtol=brentq_xtol))
                stepsize = ekfs[0]
                if len(ekfs) == n:
                    return np.array(ekfs), det, oms

            om += stepsize

    def peek_det_scalar_alt(self, om):
        mp.mp.dps = self.prec
        self.calc_taus_alt(om)
        return self.get_BCMatrix_3sec_axt_det()

    def peek_det_scalar_alt_1(self, om):
        mp.mp.dps = self.prec
        self.calc_taus_alt(om)
        return self.get_BCMatrix_3sec_axt_det()

    def peek_det_scalar_alt_2(self, om):
        mp.mp.dps = self.prec
        self.calc_taus_alt(om)
        return self.get_BCMatrix_3sec_axt_det()

    def get_k_and_p_mp(self, A, Iz, E, rho, N):
        k = mp.mpmathify((E * Iz / rho / A) ** 0.25)
        p = mp.mpmathify((N / rho / A) ** 0.5)
        return k, p

    ### Static Methods
    @staticmethod
    def get_approx_ekf(fist_3_ekfs, order, poly_deg=2):
        coef = np.polyfit(range(1, poly_deg + 2), fist_3_ekfs, poly_deg)
        return np.polyval(coef, order)

    @staticmethod
    def get_area_and_inertia_rectangle(b, h):
        A = b * h
        Iz = b * h ** 3 / 12
        return A, Iz

    @staticmethod
    def get_k_and_p_value(A, Iz, E, rho, N):
        k = (E * Iz / rho / A) ** 0.25
        p = (N / rho / A) ** 0.5
        return k, p

    @staticmethod
    def get_centerpoints_of_layers_rectangle(*args):
        return np.array([d / 2 + sum(args[:i]) for i, d in enumerate(args)])

    @staticmethod
    def get_neutral_layer(ys, A_layers, E_layers):
        helper = np.array(A_layers * E_layers)
        return np.sum(ys * helper) / np.sum(helper)


def plot_beam_geometry(geom_param, hight_factor=10, colors=['red', 'green', 'orange']):
    # Plotting Geometry
    b1, l1, l2, l3, hA, hB, hC = geom_param
    hA, hB, hC = hA * hight_factor, hB * hight_factor, hC * hight_factor
    rects = []
    # layerA
    rects.append(Rectangle((0, 0), l1 + l2 + l3, hA, color=colors[0], edgecolor=None, linewidth=0))
    # layerB
    rects.append(Rectangle((0, hA), l1 + l2 + l3, hB, color=colors[1], edgecolor=None, linewidth=0))
    # layerC
    rects.append(Rectangle((0, hA + hB), l1, hC, color=colors[2], edgecolor=None, linewidth=0))
    rects.append(Rectangle((l1 + l2, hA + hB), l3, hC, color=colors[2], edgecolor=None, linewidth=0))

    fig, ax = plt.subplots(1, figsize=cm2inch((l1 + l2 + l3) * 3.2, (hA + hB + hC) * 1.2),
                           subplot_kw={'xlim': (0, l1 + l2 + l3), 'ylim': (0, hA + hB + hC)})
    [ax.add_patch(p) for p in rects]
    ax.axis('off')

    return fig


def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i / inch for i in tupl[0])
    else:
        return tuple(i / inch for i in tupl)





















































