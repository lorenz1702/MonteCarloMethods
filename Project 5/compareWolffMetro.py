import numpy as np
import matplotlib.pyplot as plt
import time

# --- Simulation Parameters ---
L = 8
N = L * L

# Temperature range
temperatures = np.linspace(1.0, 4.0, 15)

# Monte Carlo steps for Metropolis
n_th_steps_metro = 200 * N
n_mc_steps_metro = 1000 * N
measurement_interval_metro = N

# Monte Carlo steps for Wolff
n_th_steps_wolff = 2000
n_mc_steps_wolff = 10000
measurement_interval_wolff = 10


def initial_state(L_param):
    return np.random.choice([-1, 1], size=(L_param, L_param))


def calculate_energy_change(spins_param, i, j, L_param):
    s_ij = spins_param[i, j]
    neighbor_sum = (
            spins_param[(i + 1) % L_param, j] +
            spins_param[(i - 1 + L_param) % L_param, j] +
            spins_param[i, (j + 1) % L_param] +
            spins_param[i, (j - 1 + L_param) % L_param]
    )
    delta_E = 2 * s_ij * neighbor_sum
    return delta_E


def calculate_total_magnetization(spins_param):
    return np.sum(spins_param)


def autocorr_func_1d(x):
    x = np.atleast_1d(x)
    n = len(x)
    if n == 0:
        return np.array([])

    x_centered = x - np.mean(x)

    c_k_numerators = np.correlate(x_centered, x_centered, mode='full')[n - 1:]

    if len(c_k_numerators) == 0:
        return np.array([])

    c0 = c_k_numerators[0]

    if c0 < 1e-12:
        return np.ones(n) if n > 0 else np.array([])

    rho = c_k_numerators / c0
    return rho


def calculate_integrated_autocorrelation_time(time_series, c=5):
    time_series = np.asarray(time_series)
    n_samples = len(time_series)

    if n_samples < 2:
        return 0.5

    rho = autocorr_func_1d(time_series)

    if len(rho) < 2 or np.abs(rho[0] - 1.0) > 1e-6 or np.all(np.isnan(rho)):
        return 0.5

    tau_int_sum = 0.0
    for t in range(1, len(rho)):
        current_tau_est = 0.5 + tau_int_sum
        if t > c * current_tau_est:
            break
        if np.isnan(rho[t]):
            break
        tau_int_sum += rho[t]

    tau = 0.5 + tau_int_sum
    tau = max(0.5, tau)
    return tau


def wolff_cluster_step(spins_param, L_param, T_param):
    i_seed, j_seed = np.random.randint(0, L_param), np.random.randint(0, L_param)
    S_seed = spins_param[i_seed, j_seed]

    C_set = set()
    C_set.add((i_seed, j_seed))

    F_stack = [(i_seed, j_seed)]

    p = 1. - np.exp(-2.0 / T_param)

    while F_stack:
        i_curr, j_curr = F_stack.pop()

        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            i_neigh, j_neigh = (i_curr + dr + L_param) % L_param, \
                               (j_curr + dc + L_param) % L_param

            neighbour_coord_tuple = (i_neigh, j_neigh)

            if spins_param[i_neigh, j_neigh] == S_seed and \
                    neighbour_coord_tuple not in C_set:
                if np.random.rand() < p:
                    C_set.add(neighbour_coord_tuple)
                    F_stack.append(neighbour_coord_tuple)

    for i_cluster, j_cluster in C_set:
        spins_param[i_cluster, j_cluster] *= -1


# --- Metropolis Simulation ---
results_metro = {'T': [], 'M': [], 'M_err': [], 'tau': []}
print("--- Starting Metropolis Simulation ---")
start_time_total_metro = time.time()

# do the metro simulation for all the temperature
for T_metro in temperatures:
    print(f"Metropolis: Simulating T = {T_metro:.3f}...")
    start_time_temp_metro = time.time()

    spins = initial_state(L)
    magnetization_timeseries_metro = []

    # Thermalization steps
    for step in range(n_th_steps_metro):
        i, j = np.random.randint(0, L, size=2)
        delta_E = calculate_energy_change(spins, i, j, L)
        if delta_E <= 0 or np.random.rand() < np.exp(-delta_E / T_metro):
            spins[i, j] *= -1
    accepted_flips_metro = 0

    # Measurement Phase
    for step in range(n_mc_steps_metro):
        i, j = np.random.randint(0, L, size=2)
        delta_E = calculate_energy_change(spins, i, j, L)
        if delta_E <= 0 or np.random.rand() < np.exp(-delta_E / T_metro):
            spins[i, j] *= -1
            accepted_flips_metro += 1
        if (step + 1) % measurement_interval_metro == 0:
            M_current = np.abs(calculate_total_magnetization(spins)) / N
            magnetization_timeseries_metro.append(M_current)

    magnetization_timeseries_metro = np.array(magnetization_timeseries_metro)
    n_meas_metro = len(magnetization_timeseries_metro)

    #Calculate the autocorrection
    if n_meas_metro > 1:
        M_avg_metro = np.mean(magnetization_timeseries_metro)
        variance_M_metro = np.var(magnetization_timeseries_metro, ddof=0)
        tau_metro = calculate_integrated_autocorrelation_time(magnetization_timeseries_metro)
        n_eff_metro = n_meas_metro / (2 * tau_metro)
        if n_eff_metro < 1: n_eff_metro = 1
        M_err_metro = np.sqrt(variance_M_metro / n_eff_metro) if variance_M_metro >= 0 else np.nan
    elif n_meas_metro == 1:
        M_avg_metro = magnetization_timeseries_metro[0]
        M_err_metro = np.nan
        tau_metro = 0.5
    else:
        M_avg_metro = np.nan
        M_err_metro = np.nan
        tau_metro = np.nan

    results_metro['T'].append(T_metro)
    results_metro['M'].append(M_avg_metro)
    results_metro['M_err'].append(M_err_metro)
    results_metro['tau'].append(tau_metro)

    end_time_temp_metro = time.time()
    print(
        f"  -> Done Metropolis T={T_metro:.3f} in {end_time_temp_metro - start_time_temp_metro:.2f} s, tau={tau_metro:.2f}")

end_time_total_metro = time.time()
print(f"\nTotal Metropolis simulation time: {end_time_total_metro - start_time_total_metro:.2f} s")

# --- Wolff Simulation ---
results_wolff = {'T': [], 'M': [], 'M_err': [], 'tau': []}
print("\n--- Starting Wolff Simulation ---")
start_time_total_wolff = time.time()

for T_wolff in temperatures:
    print(f"Wolff: Simulating T = {T_wolff:.3f}...")
    start_time_temp_wolff = time.time()

    spins = initial_state(L)
    magnetization_timeseries_wolff = []

    # Thermalization steps
    for step in range(n_th_steps_wolff):
        wolff_cluster_step(spins, L, T_wolff)

    # Measurement Phase
    for step in range(n_mc_steps_wolff):
        wolff_cluster_step(spins, L, T_wolff)
        if step % measurement_interval_wolff == 0:
            M_current = np.abs(calculate_total_magnetization(spins)) / N
            magnetization_timeseries_wolff.append(M_current)

    magnetization_timeseries_wolff = np.array(magnetization_timeseries_wolff)
    n_meas_wolff = len(magnetization_timeseries_wolff)

    # Calculate the autocorrection
    if n_meas_wolff > 1:
        M_avg_wolff = np.mean(magnetization_timeseries_wolff)
        variance_M_wolff = np.var(magnetization_timeseries_wolff, ddof=0)
        tau_wolff = calculate_integrated_autocorrelation_time(magnetization_timeseries_wolff)
        n_eff_wolff = n_meas_wolff / (2 * tau_wolff)
        if n_eff_wolff < 1: n_eff_wolff = 1
        M_err_wolff = np.sqrt(variance_M_wolff / n_eff_wolff) if variance_M_wolff >= 0 else np.nan
    elif n_meas_wolff == 1:
        M_avg_wolff = magnetization_timeseries_wolff[0]
        M_err_wolff = np.nan;
        tau_wolff = 0.5
    else:
        M_avg_wolff = np.nan;
        M_err_wolff = np.nan;
        tau_wolff = np.nan

    results_wolff['T'].append(T_wolff)
    results_wolff['M'].append(M_avg_wolff)
    results_wolff['M_err'].append(M_err_wolff)
    results_wolff['tau'].append(tau_wolff)

    end_time_temp_wolff = time.time()
    print(
        f"  -> Done Wolff T={T_wolff:.3f} in {end_time_temp_wolff - start_time_temp_wolff:.2f} s, tau={tau_wolff:.2f}")

end_time_total_wolff = time.time()
print(f"\nTotal Wolff simulation time: {end_time_total_wolff - start_time_total_wolff:.2f} s")

# --- Plotting Comparison ---
plt.figure(figsize=(12, 7))

# Metropolis Data
temps_metro_plot = np.array(results_metro['T'])
mag_metro_plot = np.array(results_metro['M'])
mag_err_metro_plot = np.array(results_metro['M_err'])
valid_indices_metro = ~np.isnan(mag_metro_plot) & ~np.isnan(mag_err_metro_plot)
if np.any(valid_indices_metro):
    plt.errorbar(temps_metro_plot[valid_indices_metro], mag_metro_plot[valid_indices_metro],
                 yerr=mag_err_metro_plot[valid_indices_metro], fmt='o-', capsize=5, label='Metropolis $|M|/N$')

# Wolff Data
temps_wolff_plot = np.array(results_wolff['T'])
mag_wolff_plot = np.array(results_wolff['M'])
mag_err_wolff_plot = np.array(results_wolff['M_err'])
valid_indices_wolff = ~np.isnan(mag_wolff_plot) & ~np.isnan(mag_err_wolff_plot)
if np.any(valid_indices_wolff):
    plt.errorbar(temps_wolff_plot[valid_indices_wolff], mag_wolff_plot[valid_indices_wolff],
                 yerr=mag_err_wolff_plot[valid_indices_wolff], fmt='s--', capsize=5, label='Wolff $|M|/N$')

Tc_theor = 2 / (np.log(1 + np.sqrt(2)))
plt.axvline(Tc_theor, color='k', linestyle=':', label=f'Theoretical $T_c \\approx {Tc_theor:.3f}$')

plt.xlabel('Temperature $T$ ($J/k_B$)')
plt.ylabel('Average Absolute Magnetization per Site $\langle |M| \\rangle / N$')
plt.title(f'Ising Model: Metropolis vs. Wolff (L={L})')
plt.legend()
plt.grid(True)
plt.show()

# Plot Autocorrelation Times
plt.figure(figsize=(12, 7))

# Metropolis Tau
tau_metro_plot = np.array(results_metro['tau'])
valid_tau_metro = ~np.isnan(tau_metro_plot) & valid_indices_metro  # Use same T valid indices
if np.any(valid_tau_metro):
    plt.plot(temps_metro_plot[valid_tau_metro], tau_metro_plot[valid_tau_metro], 'o-', label='Metropolis $\\tau_{int}$')

# Wolff Tau
tau_wolff_plot = np.array(results_wolff['tau'])
valid_tau_wolff = ~np.isnan(tau_wolff_plot) & valid_indices_wolff  # Use same T valid indices
if np.any(valid_tau_wolff):
    plt.plot(temps_wolff_plot[valid_tau_wolff], tau_wolff_plot[valid_tau_wolff], 's--', label='Wolff $\\tau_{int}$')

plt.axvline(Tc_theor, color='k', linestyle=':', label=f'Theoretical $T_c \\approx {Tc_theor:.3f}$')
plt.xlabel('Temperature $T$ ($J/k_B$)')
plt.ylabel('Integrated Autocorrelation Time $\\tau_{int}$')
plt.title(f'Autocorrelation Time: Metropolis vs. Wolff (L={L})')
plt.legend()
plt.grid(True)
plt.yscale('log')  # Autocorrelation times can vary greatly
plt.show()
