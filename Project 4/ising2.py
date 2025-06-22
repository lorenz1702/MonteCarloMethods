import numpy as np
import matplotlib.pyplot as plt
import time

# --- Simulation Parameters ---
L = 8
N = L * L

# Temperature range
temperatures = np.linspace(1.0, 4.0, 15) 
# temperatures = np.linspace(2.0, 2.5, 21) 

# Monte Carlo steps
n_th_steps = 2000 * N
n_mc_steps = 10000 * N
measurement_interval = N

def initial_state(L):
    return np.random.choice([-1, 1], size=(L, L))

def calculate_energy_change(spins, i, j, L):
    s_ij = spins[i, j]
    neighbor_sum = (
        spins[(i + 1) % L, j] +
        spins[(i - 1 + L) % L, j] +
        spins[i, (j + 1) % L] +
        spins[i, (j - 1 + L) % L]
    )

    delta_E = 2 * s_ij * neighbor_sum
    return delta_E

def calculate_total_magnetization(spins):
    return np.sum(spins)

def autocorr_func_1d(x):
    x = np.atleast_1d(x)
    n = len(x)
    var = np.var(x)
    x = x - np.mean(x)

    # Use numpy's correlate function in 'full' mode
    #r = np.correlate(x, x, mode='full')[-n:]
    # Result is r[k] = sum_{i=0}^{n-1-k} x_i * x_{i+k}
    x_centered = x - np.mean(x)
    r = np.zeros(n, dtype=float)
    for k in range(n):
        slice1 = x_centered[0: n - k]
        slice2 = x_centered[k: n]
        r[k] = np.sum(slice1 * slice2)


    return r / (var * (np.arange(n, 0, -1)))


def calculate_integrated_autocorrelation_time(time_series, c=5):
    time_series = np.asarray(time_series)
    n_samples = len(time_series)
    rho = autocorr_func_1d(time_series)

    tau_int_sum = 0.0
    for t in range(1, n_samples):
        tau_int_sum += rho[t]
        current_tau_est = 0.5 + tau_int_sum

    tau = 0.5 + tau_int_sum
    tau = max(0.5, tau)
    return tau


# --- Main Simulation Loop ---
results = {'T': [], 'M': [], 'M_err': [], 'tau': []}
start_time_total = time.time()

for T in temperatures:
    print(f"Simulating T = {T:.3f}...")
    start_time_temp = time.time()

    spins = initial_state(L)
    magnetization_timeseries = []

    # Thermalization phase
    for step in range(n_th_steps):
        i, j = np.random.randint(0, L, size=2)

        delta_E = calculate_energy_change(spins, i, j, L)

        if delta_E <= 0 or np.random.rand() < np.exp(-delta_E / T):
            spins[i, j] *= -1

    # Measurement phase
    accepted_flips = 0
    for step in range(n_mc_steps):
        i, j = np.random.randint(0, L, size=2)
        delta_E = calculate_energy_change(spins, i, j, L)

        if delta_E <= 0 or np.random.rand() < np.exp(-delta_E / T):
            spins[i, j] *= -1
            accepted_flips += 1


        if (step + 1) % measurement_interval == 0:
            M_current = np.abs(calculate_total_magnetization(spins)) / N
            magnetization_timeseries.append(M_current)

    magnetization_timeseries = np.array(magnetization_timeseries)
    n_meas = len(magnetization_timeseries)

    M_avg = np.mean(magnetization_timeseries)

    # Calculate integrated autocorrelation time (tau)
    tau = calculate_integrated_autocorrelation_time(magnetization_timeseries)

    # Calculate the statistical error
    variance_M = np.var(magnetization_timeseries)
    if n_meas > 1 and tau > 0:
        n_eff = n_meas / (2 * tau)
        M_err = np.sqrt(variance_M / n_eff)
    elif n_meas > 0: # Handle case with only 1 measurement or tau=0
         M_err = np.sqrt(variance_M / n_meas) # Naive error
    else:
         M_err = np.nan # No measurements

    results['T'].append(T)
    results['M'].append(M_avg)
    results['M_err'].append(M_err)
    results['tau'].append(tau)

    end_time_temp = time.time()
    acceptance_rate = accepted_flips / n_mc_steps if n_mc_steps > 0 else 0
    print(f"  -> Done T={T:.3f} in {end_time_temp - start_time_temp:.2f} s")
    print(f"     M = {M_avg:.4f} +/- {M_err:.4f}, tau = {tau:.2f}, N_meas = {n_meas}, Accept Rate: {acceptance_rate:.3f}")


end_time_total = time.time()
print(f"\nTotal simulation time: {end_time_total - start_time_total:.2f} s")

# --- Plotting ---
plt.figure(figsize=(10, 6))
temps = np.array(results['T'])
mag = np.array(results['M'])
mag_err = np.array(results['M_err'])


valid_indices = ~np.isnan(mag) & ~np.isnan(mag_err)
temps = temps[valid_indices]
mag = mag[valid_indices]
mag_err = mag_err[valid_indices]


plt.errorbar(temps, mag, yerr=mag_err, fmt='o-', capsize=5, label='Simulation Data $|M|/N$')

# Add theoretical critical temperature (for 2D Ising model)
Tc_theor = 2  / (np.log(1 + np.sqrt(2)))
plt.axvline(Tc_theor, color='r', linestyle='--', label=f'Theoretical $T_c \\approx {Tc_theor:.3f}$')

plt.xlabel('Temperature $T$ ($1/B$)')
plt.ylabel('Average Absolute Magnetization per Site $\langle |M| \\rangle / N$')
plt.title(f'Ising Model Magnetization vs. Temperature (Metropolis, L={L})')
plt.legend()
plt.grid(True)
plt.show()