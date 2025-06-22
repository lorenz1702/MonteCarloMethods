import math
import matplotlib.pyplot as plt
import numpy as np
import random

def calculate_sine_squares(x):
    """Calculates the square of the sine of x."""
    return math.sin(x) ** 2

def delta_x(b, a, bins):
    """Calculates the width of a bin."""
    return (b - a) / bins

def begining_weights(bins):
    """Initializes the weights uniformly."""
    weights = [1.0 / bins for _ in range(bins)]
    weights = np.array(weights)
    return weights

def calculate_N_sample_size(N, bins, weights):
    """Calculates the number of samples per bin, proportional to the weights."""
    nx_bin = np.round(N * weights).astype(int)
    return nx_bin

def calculate_new_weights(f_average_values, total_f_average):
    """Calculates the new weights based on the f_average values."""
    if total_f_average == 0:  # Avoid division by zero.  Return zeros in this case.
        return np.zeros_like(f_average_values)
    new_weights = f_average_values**2 / total_f_average
    return new_weights


def calculate_f_average(N_sample_size_i, a, b):
    """Calculates the average of f(x) within a bin."""
    if N_sample_size_i == 0: # Avoid ZeroDivisionError
        return 0.0

    total_sum = 0
    for _ in range(N_sample_size_i):
        random_x = random.uniform(a, b)
        total_sum += calculate_sine_squares(random_x)
    return total_sum / N_sample_size_i


def run_iterations(N, bins, iterations, weights, grids_value, a, b):
    """Performs the VEGAS iterations."""

    f_average_values_history = []
    weights_history = []

    for _ in range(iterations):
        N_sample_size = calculate_N_sample_size(N, bins, weights)

        f_average_values = np.zeros(bins)
        for i in range(bins):
            bin_start = a + i * grids_value
            bin_end = a + (i + 1) * grids_value
            f_average_values[i] = calculate_f_average(N_sample_size[i], bin_start, bin_end)

        f_average_values_history.append(f_average_values)

        total_f_average = np.sum(f_average_values * N_sample_size) / N

        new_weights = calculate_new_weights(f_average_values, total_f_average)

        weights = new_weights / np.sum(new_weights)  # Normalize weights
        weights_history.append(weights)

    return weights, f_average_values_history, weights_history


def vegas(N, a, b, bins, iterations):
    """Implements the VEGAS algorithm for Monte Carlo integration."""

    if a > b:
        a, b = b, a  # Ensure a <= b

    grids_value = delta_x(b, a, bins)
    weights = begining_weights(bins)

    final_weights, f_average_values_history, weights_history = run_iterations(N, bins, iterations, weights, grids_value, a, b)

    N_sample_size = calculate_N_sample_size(N, bins, final_weights)
    final_integral = np.sum(f_average_values_history[-1] * N_sample_size) / N * (b - a)

    print(f"Final Weights: {final_weights}")  # Print final weights
    return final_integral, f_average_values_history, weights_history


# --- Main Program ---
N = 100
a = 0
b = 2 * math.pi
bins = 10
iterations = 3

integral_result, f_average_values_history, weights_history = vegas(N, a, b, bins, iterations)
print(f"The integral of sin^2(x) from {a} to {b} is (VEGAS): {integral_result}")
print(f"Exact value: { (b-a)/2 - (math.sin(2*b) - math.sin(2*a))/4}")

# --- Plotting ---

# 1. Plot of f_average values for each iteration
plt.figure(figsize=(10, 6))
for i, f_averages in enumerate(f_average_values_history):
    plt.plot(range(1, bins + 1), f_averages, label=f"Iteration {i+1}")
plt.xlabel("Bin")
plt.ylabel("f_average")
plt.title("f_average Values per Bin and Iteration")
plt.legend()
plt.grid(True)
plt.show()

# 2. Plot of weights for each iteration
plt.figure(figsize=(10, 6))
for i, weights in enumerate(weights_history):
    plt.plot(range(1, bins + 1), weights, label=f"Iteration {i+1}")
plt.xlabel("Bin")
plt.ylabel("Weight")
plt.title("Weights per Bin and Iteration")
plt.legend()
plt.grid(True)
plt.show()

# 3. Plot of the distribution of sample points after the last iteration
final_weights = weights_history[-1]
N_sample_size = calculate_N_sample_size(N, bins, final_weights)
samples = []
for i in range(bins):
    bin_start = a + i * delta_x(b, a, bins)
    bin_end = a + (i + 1) * delta_x(b, a, bins)
    for _ in range(N_sample_size[i]):
        samples.append(random.uniform(bin_start, bin_end))

plt.figure(figsize=(10, 6))
plt.hist(samples, bins=bins, range=(a, b), density=True)
plt.xlabel("x")
plt.ylabel("Density of Sample Points")
plt.title("Distribution of Sample Points after the Last Iteration")
plt.grid(True)

# Overlay the actual function (sin^2(x))
x_vals = np.linspace(a, b, 500)
y_vals = calculate_sine_squares(x_vals)
plt.plot(x_vals, y_vals, color='red', label='sin²(x)')
plt.legend()

plt.show()