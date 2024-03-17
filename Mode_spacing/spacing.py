from scipy.optimize import fsolve
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from PPKTP_refractive_index import *

# light velocity in vacuum
c = 3e8

# pump wavelength (um)
lam_p = 0.532
lam_s = 1.064
lam_i = 1.064

# crystal length (mm)
# L = 2  # 2mm

def prop_vector(lam, T):
    return 2 * math.pi * n_e(lam, T) / lam

# group velocity
def vG(n):
    return c / n

# signal: extraordinary; idler: ordinary
def delta_wG(L, T):
    vGs = vG(n_e(lam_s, T))
    vGi = vG(n_o_3(lam_i, T))
    return 1.77 * math.pi / abs(1 / vGs - 1 / vGi) / (L * 1e-3)  # crystal length in mm

# cluster spacing
def OmegaC(L, T):
    vGs = vG(n_e(lam_s, T))
    vGi = vG(n_o_3(lam_i, T))
    return (math.pi / (L * 1e-3)) * vGs * vGi / abs((vGs - vGi))

print(n_o_3(0.532, 0))
print(n_o_3(0.532, 20))
print(n_o_3(0.532, 25))
print(n_o_3(0.532, 30))
print(n_o_3(0.532, 35))

plt.subplot(1, 2, 1)
T = np.linspace(0, 1000, 5000)  # Desired temperature range (Celcius)
yT1 = delta_wG(2, T)
yT2 = 2 * OmegaC(2, T)

plt.plot(T, yT1, label="biphoton bandwidth delta_wG, L=2mm")
plt.plot(T, yT2, label="2 times Cluster spacing OmegaC, L=2mm")
plt.legend()

plt.xlabel('T (Celcius)')  # Label for the x-axis
plt.ylabel('Value')  # Label for the y-axis
plt.title('delta_wG V.S. 2 * OmegaC')

plt.subplot(1, 2, 2)
l = np.linspace(1, 20, 5000)  # 1 ~ 20mm
yL1 = delta_wG(l, 25)
yL2 = 2 * OmegaC(l, 25)
# yL1_100 = delta_wG(l, 100)
# yL2_100 = OmegaC(l, 100)

plt.plot(l, yL1, label="biphoton bandwidth delta_wG, T=25 degree")
plt.plot(l, yL2, label="2 times Cluster spacing OmegaC, T=25 degree")
# plt.plot(l, yL1_100, label="biphoton bandwidth delta_wG, T=100 degree")
# plt.plot(l, yL2_100, label="Cluster spacing OmegaC, T=100 degree")
plt.legend()

plt.xlabel('L (mm)')  # Label for the x-axis
plt.ylabel('Value')  # Label for the y-axis
plt.title('delta_wG V.S. 2 * OmegaC')

plt.show()