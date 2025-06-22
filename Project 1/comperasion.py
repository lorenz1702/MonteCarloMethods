import math
import matplotlib.pyplot as plt
import numpy as np
import random

def delta_x(b, a, N):
    return (b - a) / N

def calculate_sine(x):
    return math.sin(x)

# --- Trapezoidal Rule ---
def calculate_trapezoidal(x, delta_x):
    return 0.5 * (calculate_sine(x) + calculate_sine(x + delta_x)) * delta_x

def approximate_integral_trapezoidal(a, b, N):
    delta_x_val = delta_x(b, a, N)
    total_sum = 0
    x = a
    for _ in range(N):
        total_sum += calculate_trapezoidal(x, delta_x_val)
        x += delta_x_val
    return total_sum

# --- Midpoint Rule ---
def approximate_integral_midpoint(a, b, N):
    delta_x_val = delta_x(b, a, N)
    total_sum = 0
    x = a + delta_x_val / 2
    for _ in range(N):
        total_sum += calculate_sine(x) * delta_x_val
        x += delta_x_val
    return total_sum

# --- Monte Carlo Method ---
def approximate_integral_monte_carlo(a, b, N):
    total_sum = 0
    for _ in range(N):
        random_x = random.uniform(a, b)  # Generate a random number between a and b
        total_sum += calculate_sine(random_x)
    return (b - a) * (total_sum / N)

#alles richtig

# --- Main part of the script ---

a = 0
b = math.pi
theoretical_integral = 2  # Integral of sin(x) from 0 to pi

# Values of N to be tested
N_values = np.logspace(1, 5, 50, dtype=int)
# verstehe ich nicht
# Lists to store the results
errors_trapezoidal = []
errors_midpoint = []
errors_monte_carlo = []

# Calculate the approximations and errors for each N
for N in N_values:
    # Trapezoidal Rule
    integral_approximation_trapezoidal = approximate_integral_trapezoidal(a, b, N)
    errors_trapezoidal.append(abs(integral_approximation_trapezoidal - theoretical_integral))

    # Midpoint Rule
    integral_approximation_midpoint = approximate_integral_midpoint(a, b, N)
    errors_midpoint.append(abs(integral_approximation_midpoint - theoretical_integral))

    # Monte Carlo Method
    integral_approximation_monte_carlo = approximate_integral_monte_carlo(a, b, N)
    errors_monte_carlo.append(abs(integral_approximation_monte_carlo - theoretical_integral))


# --- Plotting the errors in a log-log plot ---

plt.figure(figsize=(10, 8))

plt.loglog(N_values, errors_trapezoidal, marker='o', linestyle='-', label='Trapezoidal Rule')
plt.loglog(N_values, errors_midpoint, marker='x', linestyle='--', label='Midpoint Rule')
plt.loglog(N_values, errors_monte_carlo, marker='.', linestyle=':', label='Monte Carlo Method')


plt.xlabel('N')
plt.ylabel('Absolute Error')
plt.title('Error Comparison of Integration Methods')
plt.grid(True)
plt.legend()
plt.show()