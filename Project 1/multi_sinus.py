import math
import matplotlib.pyplot as plt
import numpy as np
import vegas



def calculate_sine_multidim(x):

    return np.prod(np.sin(x))


def approximate_integral_midpoint_multidim(a, b, N, d):

    delta_x = (b - a) / N
    points = np.array([a + delta_x] * d, dtype=float)
    total_sum = 0.0

    def increment_points(points, d, delta_x):
        for i in range(d):
            points[i] += delta_x
            if points[i] <= b:
                return True
            #points[i] = a + delta_x / 2
            points[i] = a + delta_x
        return False

    while True:
        total_sum += calculate_sine_multidim(points) * (delta_x ** d)
        if not increment_points(points, d, delta_x):
            break
    return total_sum

def approximate_integral_monte_carlo_multidim(a, b, N, d):

    total_sum = 0.0
    for _ in range(N):
        random_points = np.random.uniform(a, b, d)
        total_sum += calculate_sine_multidim(random_points)
    return (b - a) ** d * (total_sum / N)


def approximate_integral_vegas_algorithm(a, b, N, d):
    integ = vegas.Integrator(d*[[a, b]])
    # adapt grid
    training = integ(calculate_sine_multidim, nitn=N, neval=2000)
    # final analysis
    result = integ(calculate_sine_multidim, nitn=N, neval=10000)
    return result.itn_results[N-1].mean



# --- Main part of the script ---

a = 0
b = math.pi
N = 2
dimensions = range(1, 25)
theoretical_integrals = [2 ** d for d in dimensions]

errors_midpoint = []
errors_monte_carlo = []
errors_vegas = []

for d in dimensions:
    integral_vegas = approximate_integral_vegas_algorithm(a, b, N, d)
    errors_vegas.append(abs(integral_vegas - theoretical_integrals[d - 1]))

    integral_midpoint = approximate_integral_midpoint_multidim(a, b, N, d)
    errors_midpoint.append(abs(integral_midpoint - theoretical_integrals[d - 1]))

    integral_monte_carlo = approximate_integral_monte_carlo_multidim(a, b, N, d)
    errors_monte_carlo.append(abs(integral_monte_carlo - theoretical_integrals[d - 1]))
    print(d)

# --- Plotting the errors ---

plt.figure(figsize=(10, 8))
plt.semilogy(dimensions, errors_vegas, marker='o', linestyle='-', label='Vegas Rule')
plt.semilogy(dimensions, errors_midpoint, marker='x', linestyle='--', label='Midpoint Rule')
plt.semilogy(dimensions, errors_monte_carlo, marker='.', linestyle=':', label='Monte Carlo Method')

plt.xlabel('Dimensions (d)')
plt.ylabel('Absolute Error (log scale)')
plt.title(f'Error Comparison of Integration Methods (N={N})')
plt.grid(True)
plt.legend()
plt.show()