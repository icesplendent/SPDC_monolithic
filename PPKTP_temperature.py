from scipy.optimize import fsolve
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

A = 2.12725
B = 1.18431
C = 5.14852e-2
D = 0.6603
E = 100.00567
F = 9.68956e-3
a_0 = -6.1537e-6
a_1 = 64.505e-6
a_2 = -56.447e-6
a_3 = 17.169e-6
b_0 = -0.96751e-8
b_1 = 13.192e-8
b_2 = -11.78e-8
b_3 = 3.6292e-8

T_0 = 24.5
lam_p = 0.532

def delta(lam, T):
    return (a_0 + (a_1 / lam) + (a_2 / (lam ** 2)) + (a_3 / (lam ** 3))) * (T - 25) + \
            (b_0 + (b_1 / lam) + (b_2 / (lam ** 2)) + (b_3 / (lam ** 3))) * ((T - 25) ** 2)

def n_e(lam, T):
    return np.sqrt(A + 
                   (B / (1 - (C / (lam ** 2)))) +
                   (D / (1 - (E / (lam ** 2)))) - 
                   F * (lam ** 2)) + delta(lam, T)

def prop_vector(lam, T):
    return 2 * math.pi * n_e(lam, T) / lam

def lam_1(lam):
    return lam * lam_p / (lam - lam_p)

def equation_to_solve(vars, T, grating):
    lam_2 = vars[0]
    # Define the equation to solve
    return 2 * math.pi / (prop_vector(lam_p, T) - prop_vector(lam_2, T) - prop_vector(lam_1(lam_2), T)) - grating

# Initial guess for the solution 1064 pump
period_385_lam_s = np.zeros(191)
period_385_lam_i = np.zeros(191)
period_378_lam_s = np.zeros(191)
period_378_lam_i = np.zeros(191)
period_356_lam_s = np.zeros(191)
period_356_lam_i = np.zeros(191)
initial_guess = 0.0

# Initial guess for the solution 532 pump
period_9_lam_s = np.zeros(191)
period_9_lam_i = np.zeros(191)

# Use fsolve to find the numerical solution
for i in range (0, 191):
    initial_guess = [1.855]
    period_385_lam_s[i] = fsolve(equation_to_solve, initial_guess, args=(i, 38.5))
    period_385_lam_i[i] = lam_1(period_385_lam_s[i])

    initial_guess = [1.755]
    period_378_lam_s[i] = fsolve(equation_to_solve, initial_guess, args=(i, 37.8))
    period_378_lam_i[i] = lam_1(period_378_lam_s[i])

    initial_guess = [1.555]
    period_356_lam_s[i] = fsolve(equation_to_solve, initial_guess, args=(i, 35.6))
    period_356_lam_i[i] = lam_1(period_356_lam_s[i])

    initial_guess = [0.855]
    period_9_lam_s[i] = fsolve(equation_to_solve, initial_guess, args=(i, 9))
    period_9_lam_i[i] = lam_1(period_9_lam_s[i])




# print(period_32_lam_s)
# print(n_e(1.516, 25))
print('hello')
print(n_e(1.36, 30))


# Plot with lam V.S. T
x = np.linspace(0, 354, 355)                        # Temperature
x2 = np.linspace(0, 190, 191)  
# plt.plot(x2, period_385_lam_s, label="38.5μm", color='brown')
# plt.plot(x2, period_385_lam_i, color='brown')
# plt.plot(x2, period_378_lam_s, label="37.8μm", color='red')
# plt.plot(x2, period_378_lam_i, color='red')
# plt.plot(x2, period_356_lam_s, label="35.6μm", color='orange')
# plt.plot(x2, period_356_lam_i, color='orange')
plt.plot(x2, period_9_lam_s, label="9μm", color='green')
plt.plot(x2, period_9_lam_i, color='green')

plt.legend()

# Add text annotations for function names
# plt.text(0.5, 1.5, '25 deg. C', color='blue')
# plt.text(1.5, 0.5, '200 deg. C', color='orange')

plt.xlabel('Temperature (°C)')
plt.ylabel('Wavelength (μm)')
plt.title('Wavelength-tuning curve')

plt.xlim([40, 200])
plt.ylim([0.85, 1.5])


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