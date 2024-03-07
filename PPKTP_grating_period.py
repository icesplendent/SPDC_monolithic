import numpy as np
import math
import matplotlib.pyplot as plt

A_0 = 9.96e-6
A_1 = 9.92e-6
A_2 = -8.96e-6
A_3 = 4.101e-6
B_0 = -1.1882e-8
B_1 = 10.459e-8
B_2 = -9.8136e-8
B_3 = 3.1481e-8

T_0 = 24.5
lam_p = 1.064

def F(T):
    return (T - T_0) * (T + 570.82)

def n_z(lam):
    return 3.3134 + 0.05694 / (lam ** 2 - 0.05658) - 0.01682 * lam ** 2

print (n_z(1.064))

def n_e(lam, T):
    return n_z(1.064) + (A_0 + A_1 / lam + A_2 / (lam ** 2) + A_3 / (lam ** 3)) * (T - 25.0) + (B_0 + B_1 / lam + B_2 / (lam ** 2) + B_3 / (lam ** 3)) * ((T - 25.0) ** 2)


def prop_vector(lam, T):
    return 2 * math.pi * n_e(lam, T) / lam

def lam_1(lam):
    return lam * lam_p / (lam - lam_p)

# return period
def mom_con(lam_2, T):
    return 2 * math.pi / (prop_vector(lam_p, T) - prop_vector(lam_2, T) - prop_vector(lam_1(lam_2), T))


# Grating period range
y = np.linspace(1.35, 5, 5000)  # Desired wavelength range (μm)
x_25 = mom_con(y, 25)
# x_200 = mom_con(y, 200)
x_lim25 = np.clip(x_25, 10, 50)
# x_lim200 = np.clip(x_200, 15, 32)


# Plotting
plt.plot(x_lim25, y, label="25 deg. C")
# plt.plot(x_lim200, y, label="200 deg. C")
plt.xlabel('Grating Period (μm)')
plt.ylabel('Wavelength (μm)')
plt.title('Parametric Tuning Curve for PPKTP')
plt.legend

plt.legend()

# Add text annotations for function names
plt.text(0.5, 1.5, '25 deg. C', color='blue')
plt.text(1.5, 0.5, '200 deg. C', color='orange')

plt.ylim([1.54, 1.57])
plt.show()