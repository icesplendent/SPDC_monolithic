import numpy as np
import math
import matplotlib.pyplot as plt
from PPKTP_refractive_index import *

lam_p = 0.532

def prop_vector_e(lam, T):
    return 2 * math.pi * n_e(lam, T) / lam

def prop_vector_o(lam, T):
    return 2 * math.pi * n_o_3(lam, T) / lam

def lam_1(lam):
    return lam * lam_p / (lam - lam_p)

# return period Type0 and 2
def mom_con_0(lam_2, T):
    return 2 * math.pi / (prop_vector_e(lam_p, T) - prop_vector_e(lam_2, T) - prop_vector_e(lam_1(lam_2), T))

def mom_con_2(lam_2, T):
    return 2 * math.pi / (prop_vector_o(lam_p, T) - prop_vector_e(lam_2, T) - prop_vector_o(lam_1(lam_2), T))


# Interested points for type0 and type2
y_interested = [1.064, 1.25, 1.5]
x_interested_0 = [mom_con_0(y_interested[0], 25), mom_con_0(y_interested[1], 25) , mom_con_0(y_interested[2], 25)]
x_interested_2 = [mom_con_2(y_interested[0], 25), mom_con_2(y_interested[1], 25) , mom_con_2(y_interested[2], 25)]

# ==========================================
### Plotting (Type0)
# Grating period range
y = np.linspace(1, 5, 5000)  # Desired wavelength range (μm)
x_25 = mom_con_0(y, 25)
x_lim25 = np.clip(x_25, 8, 50)

plt.subplot(1, 2, 1)
plt.scatter(x_interested_0, y_interested, color='blue', label='interested')

# Annotate each point with its value
for i, (x1, y1) in enumerate(zip(x_interested_0, y_interested)):
    plt.text(x1, y1, f'({x1:.2f}, {y1:.2f})', fontsize=10, ha='left')

plt.plot(x_lim25, y, label="25 deg. C")
# plt.plot(x_lim200, y, label="200 deg. C")
plt.xlabel('Grating Period (μm)')
plt.ylabel('Wavelength (μm)')
plt.title('Type0 Parametric Tuning Curve for PPKTP')

plt.legend()

# Add text annotations for function names
plt.text(0.5, 1.5, '25 deg. C', color='blue')

# ==========================================
### Plotting (Type2)
# Grating period range
y_2 = np.linspace(1, 5, 5000)  # Desired wavelength range (μm)
x_25_2 = mom_con_2(y_2, 25)
# x_lim25_2 = np.clip(x_25_2, 8, 50)

plt.subplot(1, 2, 2)
plt.scatter(x_interested_2, y_interested, color='blue', label='interested')

# Annotate each point with its value
for i, (x1, y1) in enumerate(zip(x_interested_2, y_interested)):
    plt.text(x1, y1, f'({x1:.2f}, {y1:.2f})', fontsize=10, ha='left')

plt.plot(x_25_2, y_2, label="25 deg. C")
# plt.plot(x_lim200, y, label="200 deg. C")
plt.xlabel('Grating Period (μm)')
plt.ylabel('Wavelength (μm)')
plt.xlim(0, 1000)
plt.title('Type2 Parametric Tuning Curve for PPKTP')

plt.legend()

# Add text annotations for function names
plt.text(0.5, 1.5, '25 deg. C', color='blue')

# plt.ylim([1.54, 1.57])
plt.show()