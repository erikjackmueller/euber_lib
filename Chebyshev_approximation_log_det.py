import numpy as np
import mpmath as mp
import time
import random

#input 12x12 Matrix
mp.mp.dps = 50000
A = mp.matrix(
[['1.0', '0.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0'],
 ['0.0', '0.49062853002881556', '0.0', '0.49062866702173223', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0'],
 ['0.88203688184375317', '0.47118036787131636', '1.1227920515867501', '0.51055067437658265', '-0.85084783450537795', '-0.52541218344982159', '-1.1569572051744931', '-0.5818504744392446', '0.0', '0.0', '0.0', '0.0'],
 ['-0.2311745312671405', '0.43275245877020069', '0.2504907968164292', '0.55087396761260321', '0.2906577462388459', '-0.4706885789091225', '-0.32187947109667221', '-0.64002830562602743', '0.0', '0.0', '0.0', '0.0'],
 ['157946997.87865532', '84374617.543281037', '-201059316.76444157', '-91424738.551285949', '-88143244.788399123', '-54429867.271798491', '119854592.2077822', '60276604.033337957', '0.0', '0.0', '0.0', '0.0'],
 ['41396594.577003489', '-77493303.391669111', '44855597.608227799', '98645464.576438188', '-30110574.226551734', '48760797.110106846', '-33344995.45515151', '-66303516.870942348', '0.0', '0.0', '0.0', '0.0'],
 ['0.0', '0.0', '0.0', '0.0', '0.44788407496498205', '0.89409163702204616', '1.6770999492103481', '1.3463521974733625', '-0.55597812186530204', '-0.83119692492641679', '-1.5213239822127666', '-1.146484478244564'],
 ['0.0', '0.0', '0.0', '0.0', '-0.49461102793907139', '0.2477692370032844', '0.74480154656609981', '0.92777108267950682', '0.4078089254411196', '-0.2727787286589548', '-0.56249815132223653', '-0.74640515750124306'],
 ['0.0', '0.0', '0.0', '0.0', '46398373.546327205', '92623069.401249781', '-173738604.68242844', '-139474902.67964052', '-99559414.172426251', '-148843049.12926965', '272424764.68263935', '205301939.52756113'],
 ['0.0', '0.0', '0.0', '0.0', '51239033.752749977', '-25667556.080590032', '77157465.496696528', '96112132.99799967', '-73026646.399300336', '48846689.025947514', '-100727016.92738359', '-133659399.1599524'],
 ['0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.098749536123081272', '0.99511231984910943', '2.2934688986468432', '2.0639766445045748'],
 ['0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '-0.48823049470113306', '0.048449339749094783', '1.0126461098572671', '1.1252415885989009']])

# referrence mpmath determinant computation
start_1 = time.time()

det_1 = np.float64(mp.log(mp.fabs(mp.det(A))) * mp.sign(mp.det(A)))

time_1 = time.time() - start_1

# chebyshev approximation
start_2 = time.time()

sv_vector = mp.matrix(12, 2)
A_abs = mp.matrix(12)
for i in range(12):
    for j in range(12):
        A_abs[i, j] = abs(A[i, j])

for i in range(12):
    sv_vector[i, 0] = A_abs[i, i] - sum(A_abs[i, :])
    sv_vector[i, 1] = A_abs[i, i] - sum(A_abs[:, i])

s_min = min(max(sv_vector[:, 0]), max(sv_vector[:, 1]))
s_max = mp.sqrt(mp.norm(A, p=1) * mp.norm(A, p=10))
# S = mp.svd_r(A, compute_uv=False)
# delta = max(s_max, abs(1 - s_max))
m = 2
dim = 12
delta = s_min ** 2/ (s_max ** 2 + s_min ** 2)
# delta_2 = min(S) ** 2/ (max(S) ** 2 + min(S) ** 2)
# s_min, s_max = min(S), max(S)
interval = [-1, 1]
func = lambda x : mp.log(1 -  ((1 - 2 * delta) * x + 1) / 2)
coeffs = mp.chebyfit(func, interval, N=m)
# B = mp.eye(12) - A
B = 1/(s_max ** 2 + s_min ** 2) * A.T * A

def Han_algorithm(B, m, dim):
    """
    From Insu Han, Dmitry Malioutov and Jinwoo Shin:  Large-scale Log-determinant Computation
    through Stochastic Chebyshev Expansions, Proceedings of the 32 nd International Conference on Machine
    Learning, Lille, France, 2015. JMLR: W&CP volume 37

    :param B: Square input Matrix
    :param m: Order of chebyshev approximation
    :param dim: Dimension of input Matrix
    :return G: log-determinant of B for cases, where all eigenvalues of B lie within (0, 1)
    """
    B = mp.eye(12) - B
    G = 0
    v = mp.matrix(dim, 1)
    w = mp.matrix(dim, 3)

    for i in range(m):
        for j in range(dim):
            v[j] = random.choice([-1, 1])

        u = coeffs[0] * v
        n = len(coeffs)

        if m > 1:
            w[:, 0] = v
            w[:, 1] = B * v
            u = u + coeffs[1] * v
            for k in range(1, n, 1):
                w[:, 2] = 2 * B * w[:, 1] - w[:, 0]
                u = u + coeffs[k] * w[:, 2]
                w[:, 0] = w[:, 1]
                w[:, 1] = w[:, 2]

        G = G + v.T * u/m

        return G


G = Han_algorithm(B, m, dim)[0]
G = (G + dim * mp.log(s_min ** 2 + s_max ** 2)) / 2
det_2 = np.float(G)

time_2 = time.time() - start_2

print(det_1, time_1)
print(det_2, time_2)
