from scipy.optimize import fsolve
import numpy as np
import math
import matplotlib.pyplot as plt

A_1 = 5.35583
A_2 = 0.100473
A_3 = 0.20692
A_4 = 100.0
A_5 = 11.34927
A_6 = 1.5334e-2
B_1 = 4.629e-7
B_2 = 3.862e-8
B_3 = -0.89e-8
B_4 = 2.657e-5
T_0 = 24.5
lam_p = 1.064

def F(T):
    return (T - T_0) * (T + 570.82)

def n_e(lam, T):
    return np.sqrt(A_1 + B_1 * F(T) + 
                   (A_2 + B_2 * F(T)) / (lam ** 2 - (A_3 + B_3 * F(T)) ** 2 ) +
                   (A_4 + B_4 * F(T)) / (lam ** 2 - A_5 ** 2) - (A_6 * lam ** 2))

def prop_vector(lam, T):
    return 2 * math.pi * n_e(lam, T) / lam

def lam_1(lam):
    return lam * lam_p / (lam - lam_p)

def equation_to_solve(vars, T, grating):
    lam_2 = vars[0]
    # Define the equation to solve
    return 2 * math.pi / (prop_vector(lam_p, T) - prop_vector(lam_2, T) - prop_vector(lam_1(lam_2), T)) - grating

# Initial guess for the solution
period_30_lam_s = np.zeros(355)
period_30_lam_i = np.zeros(355)
period_29_lam_s = np.zeros(355)
period_29_lam_i = np.zeros(355)
period_28_lam_s = np.zeros(355)
period_28_lam_i = np.zeros(355)
period_27_lam_s = np.zeros(355)
period_27_lam_i = np.zeros(355)
period_26_lam_s = np.zeros(355)
period_26_lam_i = np.zeros(355)
period_31_lam_s = np.zeros(191)
period_31_lam_i = np.zeros(191)
period_32_lam_s = np.zeros(355)
period_32_lam_i = np.zeros(355)
initial_guess = 0.0

# Use fsolve to find the numerical solution
for i in range (0, 191):
    initial_guess = [1.855]
    period_31_lam_s[i] = fsolve(equation_to_solve, initial_guess, args=(i, 31))
    period_31_lam_i[i] = lam_1(period_31_lam_s[i])

for i in range (0, 355):
    initial_guess = [1.515]
    period_30_lam_s[i] = fsolve(equation_to_solve, initial_guess, args=(i, 30))
    period_30_lam_i[i] = lam_1(period_30_lam_s[i])

    initial_guess = [1.455]
    period_29_lam_s[i] = fsolve(equation_to_solve, initial_guess, args=(i, 29))
    period_29_lam_i[i] = lam_1(period_29_lam_s[i])

    initial_guess = [1.455]
    period_28_lam_s[i] = fsolve(equation_to_solve, initial_guess, args=(i, 28))
    period_28_lam_i[i] = lam_1(period_28_lam_s[i])

    initial_guess = [1.455]
    period_27_lam_s[i] = fsolve(equation_to_solve, initial_guess, args=(i, 27))
    period_27_lam_i[i] = lam_1(period_27_lam_s[i])

    initial_guess = [1.405]
    period_26_lam_s[i] = fsolve(equation_to_solve, initial_guess, args=(i, 26))
    period_26_lam_i[i] = lam_1(period_26_lam_s[i])

    initial_guess = [1.855]
    period_32_lam_s[i] = fsolve(equation_to_solve, initial_guess, args=(i, 32))   # out of range
    period_32_lam_i[i] = lam_1(period_32_lam_s[i])

# print(period_32_lam_s)
# print(n_e(1.516, 25))
print('hello')
print(n_e(1.36, 30))


# Plot with lam V.S. T
x = np.linspace(0, 354, 355)                        # Temperature
x2 = np.linspace(0, 190, 191)  
plt.plot(x2, period_31_lam_s, label="31μm", color='red')
plt.plot(x2, period_31_lam_i, color='red')
plt.plot(x, period_30_lam_s, label="30μm", color='orange')
plt.plot(x, period_30_lam_i, color='orange')
plt.plot(x, period_29_lam_s, label="29μm", color='yellow')
plt.plot(x, period_29_lam_i, color='yellow')
plt.plot(x, period_28_lam_s, label="28μm", color='green')
plt.plot(x, period_28_lam_i, color='green')
plt.plot(x, period_27_lam_s, label="27μm", color='blue')
plt.plot(x, period_27_lam_i, color='blue')
plt.plot(x, period_26_lam_s, label="26μm", color='purple')
plt.plot(x, period_26_lam_i, color='purple')
# plt.plot(x, period_32_lam_s, label="32μm signal")
# plt.plot(x, period_32_lam_i, label="32μm idler")

plt.legend()

# Add text annotations for function names
# plt.text(0.5, 1.5, '25 deg. C', color='blue')
# plt.text(1.5, 0.5, '200 deg. C', color='orange')

plt.xlabel('Temperature (°C)')
plt.ylabel('Wavelength (μm)')
plt.title('Wavelength-tuning curve')

plt.ylim([1, 5])
plt.show()