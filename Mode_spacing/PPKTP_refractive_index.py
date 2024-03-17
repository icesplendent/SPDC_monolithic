from scipy.optimize import fsolve
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

A = [2.12725, 2.19229, 3.0333]    # Ref1(n_e), Ref2(n_o), Ref3(n_y for n_o)
B = [1.18431, 0.83547, 0.04154]
C = [5.14852e-2, 0.0497, -0.04547]
D = [0.6603, 0, -0.01408]
E = [100.00567, 0]
F = [9.68956e-3, 0.01621]
a_0 = [-6.1537e-6, -12.101e-6]
a_1 = [64.505e-6, 59.129e-6]
a_2 = [-56.447e-6, 44/414e-6]
a_3 = [17.169e-6, 12.415e-6]
b_0 = -0.96751e-8
b_1 = 13.192e-8
b_2 = -11.78e-8
b_3 = 3.6292e-8

T_0 = 24.5
lam_p = 0.532

def delta(lam, T):
    return (a_0[0] + (a_1[0] / lam) + (a_2[0] / (lam ** 2)) + (a_3[0] / (lam ** 3))) * (T - 25) + \
            (b_0 + (b_1 / lam) + (b_2 / (lam ** 2)) + (b_3 / (lam ** 3))) * ((T - 25) ** 2)

def delta_o(lam, T):
    return (a_0[1] + (a_1[1] / lam) + (a_2[1] / (lam ** 2)) + (a_3[1] / (lam ** 3))) * T

# along z direction, Ref1 and Ref2
def n_e(lam, T):
    return np.sqrt(A[0] + 
                   (B[0] / (1 - (C[0] / (lam ** 2)))) +
                   (D[0] / (1 - (E[0] / (lam ** 2)))) - 
                   F[0] * (lam ** 2)) + delta(lam, T)

# along y direction, Ref2
def n_o(lam, T):
    return np.sqrt(A[1] + 
                   (B[1] / (1 - (C[1] / (lam ** 2)))) +
                   (D[1] / (1 - (E[1] / (lam ** 2)))) - 
                   F[1] * (lam ** 2)) + delta_o(lam, T)

# along y direction, Ref3
def n_o_3(lam, T):
    return np.sqrt(
        A[2] + B[2] / ((lam ** 2) + C[2]) + D[2] * (lam ** 2)
    ) + 1.3e-5 * T