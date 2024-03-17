from scipy.optimize import fsolve
import numpy as np
import math
import cmath
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from PPKTP_refractive_index import *
from scipy.signal import argrelextrema

# speed of light
c = 299792458

# Monochromatic light
T0 = 30           # Temperature (Celcius)
N_e1064 = 1.8718  # 1064, Nz
N_o1064 = 1.7768  # 1064, Ny 
N_e532 = 2.0587   # 532,  Nz
N_o532 = 1.9127   # 532,  Ny

# Mirrors for 1064nm
Rin = 0.997
Rout = 0.97
rin = math.sqrt(Rin)
rout = math.sqrt(Rout)

# Suppose loseless mirror
tin = math.sqrt(1 - Rin)
tout = math.sqrt(1 - Rout)

# crystal (cavity length in mm)
L = 10e-3

# incident optical path for normal incident (i.e. \theta_t = 0), omega = 2pi * freq
def delta_i(n, d, omega):
    return (n * d * omega) / c # Assuming θ_t = 0, cos(0) = 1

# optical path for a circular loop
def delta(n, d, omega):
    return 2 * n * d * omega  / c # Assuming θ_t = 0, cos(0) = 1

# transmission spectrum w.r.t to frequency and crystal length
def transmission(n, d, omega):
    return (abs(-1 * tin * tout * np.exp(-1j * delta_i(n, d, omega))) \
            / abs(1 - rin * rout * np.exp(-1j * delta(n, d, omega)))) ** 2

num_lengths = 2
precision = 100000
crystal_length = np.linspace(10e-3, 20e-3, num_lengths)
wavelength = np.linspace(1064.7e-9, 1064.8e-9, precision)
# freq = np.linspace(280e12, 283e12, 100000) # 276THz ~ 285THz

y_532o = np.array([np.linspace(0, 1, precision) for _ in range(num_lengths)])
y_1064o = np.array([np.linspace(0, 1, precision) for _ in range(num_lengths)])
y_1064e = np.array([np.linspace(0, 1, precision) for _ in range(num_lengths)])

# Type 2 o = o + e, from refractive_index.py
for i in range (0, num_lengths):
    y_1064o[i] = transmission(n_o_3(wavelength * 1e6, T0), crystal_length[i], 2 * np.pi * c / wavelength)
    y_1064e[i] = transmission(n_e(wavelength * 1e6, T0), crystal_length[i], 2 * np.pi * c / wavelength)

# Find local maxima indices
# max_indices_532o = argrelextrema(y_532o, np.greater)
# max_indices_1064o = argrelextrema(y_1064o, np.greater)
# max_indices_1064e = argrelextrema(y_1064e, np.greater)


# Get corresponding x (frequency) values
# max_wavelength_532o = wavelength[max_indices_532o]
# max_wavelength_1064o = wavelength[max_indices_1064o[0]]
# max_wavelength_1064e = wavelength[max_indices_1064e[0]]
# max_532o = transmission(N_o532, L, 2 * np.pi * c / max_wavelength_532o)
# max_1064o = transmission(N_o1064, L, 2 * np.pi * c / max_wavelength_1064o)
# max_1064e = transmission(N_e1064, L, 2 * np.pi * c / max_wavelength_1064e)


fig, axs = plt.subplots(num_lengths, 1, figsize=(10, 8))


# Plot the function
for i in range(0, num_lengths):
    # axs[i].plt.plot(wavelength, y_532o, color='red', label='Transmission of 532 o')
    axs[i].plot(wavelength, y_1064o[i], color='blue', label='Transmission of 1064 o')
    axs[i].plot(wavelength, y_1064e[i], color='green', label='Transmission of 1064 e')

    max_indices_1064o = argrelextrema(y_1064o[i], np.greater)
    max_indices_1064e = argrelextrema(y_1064e[i], np.greater)
    max_wavelength_1064o = wavelength[max_indices_1064o]
    max_wavelength_1064e = wavelength[max_indices_1064e]
    max_1064o = transmission(n_o_3(max_wavelength_1064o * 1e6, T0), L, 2 * np.pi * c / max_wavelength_1064o)
    max_1064e = transmission(n_e(max_wavelength_1064e * 1e6, T0), L, 2 * np.pi * c / max_wavelength_1064e)
    # axs[0].scatter(max_wavelength_532o, max_532o, color='red', label='Local Maxima 532o')
    axs[i].scatter(max_wavelength_1064o, max_1064o, color='blue', label='Local Maxima 1064o')
    axs[i].scatter(max_wavelength_1064e, max_1064e, color='green', label='Local Maxima 1064e')

    for freq, value in zip(max_wavelength_1064o, max_1064o):
        axs[i].annotate(f'({freq:.14f}, {value:.1f})', xy=(freq, value), xytext=(-20, 10), textcoords='offset points', color='blue')

    for freq, value in zip(max_wavelength_1064e, max_1064e):
        axs[i].annotate(f'({freq:.14f}, {value:.1f})', xy=(freq, value), xytext=(-20, -10), textcoords='offset points', color='green')

    axs[i].set_title(f'Crystal length = {crystal_length[i]}')
    axs[i].set_xlabel('wavelength')
    axs[i].set_ylabel('Transmission')
    axs[i].legend(loc='upper right', bbox_to_anchor=(1, 1))


# Annotate the local maxima points
# for freq, value in zip(max_wavelength_532o, max_532o):
#     axs[0].plt.annotate(f'({freq:.9f}, {value:.1f})', xy=(freq, value), xytext=(-20, 10), textcoords='offset points', color='red')

# for freq, value in zip(max_wavelength_1064o, max_1064o):
#     plt.annotate(f'({freq:.9f}, {value:.1f})', xy=(freq, value), xytext=(-20, 10), textcoords='offset points', color='blue')

# for freq, value in zip(max_wavelength_1064e, max_1064e):
#     plt.annotate(f'({freq:.9f}, {value:.1f})', xy=(freq, value), xytext=(-20, 10), textcoords='offset points', color='green')


# plt.title('Transmission spectrum')
# plt.xlabel('$f$')
# plt.ylabel('T')
# plt.grid(True)
# plt.legend()
plt.show()