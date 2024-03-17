from scipy.optimize import fsolve
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from PPKTP_refractive_index import *

lam_p = 0.532

def prop_vector_e(lam, T):
    return 2 * math.pi * n_e(lam, T) / lam

def prop_vector_o(lam, T):
    return 2 * math.pi * n_o_3(lam, T) / lam

def lam_1(lam):
    return lam * lam_p / (lam - lam_p)

# Type0 e = e + e
def equation_to_solve0(vars, T, grating):
    lam_2 = vars[0]
    # Define the equation to solve
    return 2 * math.pi / (prop_vector_e(lam_p, T) - prop_vector_e(lam_2, T) - prop_vector_e(lam_1(lam_2), T)) - grating

# Type2, o = e + o
def equation_to_solve2(vars, T, grating):
    lam_2 = vars[0]
    # Define the equation to solve
    return 2 * math.pi / (prop_vector_o(lam_p, T) - prop_vector_e(lam_2, T) - prop_vector_o(lam_1(lam_2), T)) - grating

# Initial guess for the solution 532 pump Type0
period_7_lam_s_0 = np.zeros(191)
period_7_lam_i_0 = np.zeros(191)
period_89_lam_s_0 = np.zeros(191)
period_89_lam_i_0 = np.zeros(191)
period_9_lam_s_0 = np.zeros(191)
period_9_lam_i_0 = np.zeros(191)
period_95_lam_s_0 = np.zeros(191)
period_95_lam_i_0 = np.zeros(191)
period_10_lam_s_0 = np.zeros(191)
period_10_lam_i_0 = np.zeros(191)
period_11_lam_s_0 = np.zeros(191)
period_11_lam_i_0 = np.zeros(191)
desired = np.zeros(191)

# Initial guess for the solution 532 pump Type2
period_7_lam_s_2 = np.zeros(191)
period_7_lam_i_2 = np.zeros(191)
period_89_lam_s_2 = np.zeros(191)
period_89_lam_i_2 = np.zeros(191)
period_9_lam_s_2 = np.zeros(191)
period_9_lam_i_2 = np.zeros(191)
period_95_lam_s_2 = np.zeros(191)
period_95_lam_i_2 = np.zeros(191)
period_10_lam_s_2 = np.zeros(191)
period_10_lam_i_2 = np.zeros(191)
period_11_lam_s_2 = np.zeros(191)
period_11_lam_i_2 = np.zeros(191)
desired = np.zeros(191)

# Use fsolve to find the numerical solution (Type0)
for i in range (0, 191):
    initial_guess = [0.955]
    period_89_lam_s_0[i] = fsolve(equation_to_solve0, initial_guess, args=(i, 8.9))
    period_89_lam_i_0[i] = lam_1(period_89_lam_s_0[i])

    initial_guess = [0.9]
    period_9_lam_s_0[i] = fsolve(equation_to_solve0, initial_guess, args=(i, 9))
    period_9_lam_i_0[i] = lam_1(period_9_lam_s_0[i])

    initial_guess = [0.855]
    period_95_lam_s_0[i] = fsolve(equation_to_solve0, initial_guess, args=(i, 9.5))
    period_95_lam_i_0[i] = lam_1(period_95_lam_s_0[i])

    initial_guess = [0.755]
    period_10_lam_s_0[i] = fsolve(equation_to_solve0, initial_guess, args=(i, 10))
    period_10_lam_i_0[i] = lam_1(period_10_lam_s_0[i])

    initial_guess = [0.655]
    period_11_lam_s_0[i] = fsolve(equation_to_solve0, initial_guess, args=(i, 11))
    period_11_lam_i_0[i] = lam_1(period_11_lam_s_0[i])

    desired[i] = 1.064

# Use fsolve to find the numerical solution (Type2)
for i in range (0, 191):
    # initial_guess = [1]
    # period_89_lam_s_2[i] = fsolve(equation_to_solve2, initial_guess, args=(i, 482))
    # period_89_lam_i_2[i] = lam_1(period_89_lam_s_2[i])

    # initial_guess = [1]
    # period_9_lam_s_2[i] = fsolve(equation_to_solve2, initial_guess, args=(i, 483))
    # period_9_lam_i_2[i] = lam_1(period_9_lam_s_2[i])

    initial_guess = [1.064]
    period_95_lam_s_2[i] = fsolve(equation_to_solve2, initial_guess, args=(i, 484.98))
    period_95_lam_i_2[i] = lam_1(period_95_lam_s_2[i])

    # initial_guess = [1]
    # period_10_lam_s_2[i] = fsolve(equation_to_solve2, initial_guess, args=(i, 485))
    # period_10_lam_i_2[i] = lam_1(period_10_lam_s_2[i])

    # initial_guess = [1]
    # period_11_lam_s_2[i] = fsolve(equation_to_solve2, initial_guess, args=(i, 486))
    # period_11_lam_i_2[i] = lam_1(period_11_lam_s_2[i])


print('hello')
print(n_e(1.36, 30))


# Plot with lam V.S. T
x = np.linspace(0, 354, 355)                        # Temperature
x2 = np.linspace(0, 190, 191)  
plt.subplot(1, 2, 1)
plt.plot(x2, period_89_lam_s_0, label="8.9μm", color='green')
plt.plot(x2, period_89_lam_i_0, color='green')
plt.plot(x2, period_9_lam_s_0, label="9μm", color='black')
plt.plot(x2, period_9_lam_i_0, color='black')
plt.plot(x2, period_95_lam_s_0, label="9.5μm", color='red')
plt.plot(x2, period_95_lam_i_0, color='red')
plt.plot(x2, period_10_lam_s_0, label="10μm", color='blue')
plt.plot(x2, period_10_lam_i_0, color='blue')
plt.plot(x2, period_11_lam_s_0, label="11μm", color='purple')
plt.plot(x2, period_11_lam_i_0, color='purple')
plt.plot(x2, desired, color='orange')

plt.legend()

plt.xlabel('Temperature (°C)')
plt.ylabel('Wavelength (μm)')
plt.title('Type0 PPKTP Wavelength-tuning curve for 532')

plt.xlim([0, 200])
plt.ylim([0.5, 2])

plt.subplot(1, 2, 2)
# plt.plot(x2, period_89_lam_s_2, label="8.9μm", color='green')
# plt.plot(x2, period_89_lam_i_2, color='green')
# plt.plot(x2, period_9_lam_s_2, label="9μm", color='black')
# plt.plot(x2, period_9_lam_i_2, color='black')
plt.plot(x2, period_95_lam_s_2, label="484.98μm", color='red')
plt.plot(x2, period_95_lam_i_2, color='red')
# plt.plot(x2, period_10_lam_s_2, label="10μm", color='blue')
# plt.plot(x2, period_10_lam_i_2, color='blue')
# plt.plot(x2, period_11_lam_s_2, label="11μm", color='purple')
# plt.plot(x2, period_11_lam_i_2, color='purple')
plt.plot(x2, desired, color='orange')

plt.legend()

plt.xlabel('Temperature (°C)')
plt.ylabel('Wavelength (μm)')
plt.title('Type2 PPKTP Wavelength-tuning curve for 532')

plt.xlim([0, 200])
# plt.ylim([0.6, 2])


# # Set different precision for y axis
# y_precision = 0.1  # Precision for y-axis

# # Formatter function for y axis
# def y_formatter(y, pos):
#     return "{:.1f}".format(y)

# # Set formatter for y axis
# plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(y_formatter))

# # Set the increment of the y-axis ticks to achieve precision of 0.2
# plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(y_precision))

plt.show()