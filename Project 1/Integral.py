import math
import matplotlib.pyplot as plt
import numpy as np  # Import numpy
import matplotlib
matplotlib.use("TkAgg")

def delta_x(b, a, N):
    """Calculates the step size, delta_x."""
    return (b - a) / N

def calculate_sine(x):
    """Calculates the sine of x."""
    return math.sin(x)

def calculate_trapezoidal(x, delta_x):
    """Calculates the area of a single trapezoid."""
    return 0.5 * (calculate_sine(x) + calculate_sine(x + delta_x)) * delta_x

def approximate_integral(a, b, N):
    """Approximates the integral with the trapezoidal rule."""
    delta_x_val = delta_x(b, a, N)
    total_sum = 0
    x = a
    for _ in range(N):
        total_sum += calculate_trapezoidal(x, delta_x_val)
        x += delta_x_val
    return total_sum

# --- Main part of the script ---

a = 0
b = math.pi
theoretical_integral = 2  # Integral of sin(x) from 0 to pi

# Values of N to be tested - using numpy for a range of values
N_values = np.logspace(1, 5, 50, dtype=int)  # 50 values from 10^1 to 10^5

# Lists to store the results
errors_integral = []

# Calculate the approximations and errors for each N
for N in N_values:
    integral_approximation = approximate_integral(a, b, N)
    errors_integral.append(abs(integral_approximation - theoretical_integral))

# --- Plotting the error in a log-log plot ---

plt.figure(figsize=(8, 6))  # Create a figure

# Log-log plot of the error
plt.loglog(N_values, errors_integral, marker='o', linestyle='-', color='red', label='Trapezoidal Rule Error')

# Add a theoretical error line (optional, but illustrative)
# The error for the trapezoidal rule is proportional to 1/N^2
theoretical_error = [1/N**2 for N in N_values]  # Example: Error ~ 1/N^2
plt.loglog(N_values, theoretical_error, linestyle='--', color='blue', label='Theoretical Error (1/N^2)')


plt.xlabel('Number of Subintervals (N)')
plt.ylabel('Absolute Error')
plt.title('Error of Integral Approximation (Log-Log Plot)')
plt.grid(True)
plt.legend()
plt.show()


# --- Plot of Integral Approximation and Theoretical Value---

integral_approximations = []
for N in N_values:
    approximation = approximate_integral(a,b,N)
    integral_approximations.append(approximation)

plt.figure(figsize=(8,6))
plt.plot(N_values, integral_approximations, marker='o', linestyle='-', label='Approximation')
plt.axhline(y=theoretical_integral, color='r', linestyle='--', label='Theoretical Value')
plt.xscale('log')
plt.xlabel('Number of Subintervals (N)')
plt.ylabel('Approximated value')
plt.title('Approximation over N')
plt.grid(True)
plt.legend()
plt.show()



# --- Plot Approximated Integral - 2 ---
plt.figure(figsize=(8, 6))  # Create a figure
plt.plot(N_values, [x - 2 for x in integral_approximations], marker='o', linestyle='-', label='Approximation - 2')
plt.axhline(y=0, color='r', linestyle='--', label="Theoretical Value - 2") # Corrected y value to 0
plt.xscale('log')
plt.xlabel('Number of Subintervals (N)')
plt.ylabel('Approximated value - 2')
plt.title('Approximation - 2')
plt.grid(True)
plt.legend()
plt.show()