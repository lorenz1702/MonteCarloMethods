import numpy as np
import matplotlib.pyplot as plt
import time

L = 8
N = L * L

temperatures = np.linspace(1.0, 4.0, 15)

n_th_steps = 2000
n_mc_steps = 10000
measurement_interval = N  # measurement_interval wird im Wolff-Teil nicht direkt verwendet


def initial_state(L_param):
    return np.random.choice([-1, 1], size=(L_param, L_param))


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


def wolff_cluster(spins_param, L_param, T_param):
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


results = {'T': [], 'M': [], 'M_err': [], 'tau': []}
start_time_total = time.time()

for T_current_loop in temperatures:
    T = T_current_loop  # Damit wolff_cluster die globale Variable T verwenden kann, falls es nicht als Parameter übergeben wird
    # In dieser Version wird T als Parameter übergeben, also ist diese Zeile nicht mehr nötig,
    # aber beibehalten, um minimale Änderungen am Hauptschleifenstil zu gewährleisten.
    # Besser wäre es, T_current_loop direkt an wolff_cluster zu übergeben.
    # Für Konsistenz mit der Bitte, Variablennamen nicht zu ändern, wird T hier gesetzt.

    print(f"Simulating T = {T:.3f}...")
    start_time_temp = time.time()

    spins = initial_state(L)
    magnetization_timeseries = []

    for step in range(n_th_steps):
        wolff_cluster(spins, L, T)

    for step in range(n_mc_steps):
        wolff_cluster(spins, L, T)

        if step % 10 == 0:
            M_current = np.abs(calculate_total_magnetization(spins)) / N
            magnetization_timeseries.append(M_current)

    magnetization_timeseries = np.array(magnetization_timeseries)
    n_meas = len(magnetization_timeseries)

    if n_meas > 1:
        M_avg = np.mean(magnetization_timeseries)
        variance_M = np.var(magnetization_timeseries, ddof=0)

        tau = calculate_integrated_autocorrelation_time(magnetization_timeseries)

        n_eff = n_meas / (2 * tau)
        if n_eff < 1: n_eff = 1

        if variance_M >= 0 and n_eff > 0:
            M_err = np.sqrt(variance_M / n_eff)
        else:
            M_err = np.nan
    elif n_meas == 1:
        M_avg = magnetization_timeseries[0]
        M_err = np.nan
        tau = 0.5
    else:
        M_avg = np.nan
        M_err = np.nan
        tau = np.nan

    results['T'].append(T)
    results['M'].append(M_avg)
    results['M_err'].append(M_err)
    results['tau'].append(tau)

    end_time_temp = time.time()
    print(f"  -> Done T={T:.3f} in {end_time_temp - start_time_temp:.2f} s")

end_time_total = time.time()
print(f"\nTotal simulation time: {end_time_total - start_time_total:.2f} s")

plt.figure(figsize=(10, 6))
temps = np.array(results['T'])
mag = np.array(results['M'])
mag_err = np.array(results['M_err'])

valid_indices = ~np.isnan(mag) & ~np.isnan(mag_err)
temps_valid = temps[valid_indices]
mag_valid = mag[valid_indices]
mag_err_valid = mag_err[valid_indices]

if len(temps_valid) > 0:
    plt.errorbar(temps_valid, mag_valid, yerr=mag_err_valid, fmt='o-', capsize=5, label='Simulation Data $|M|/N$')
else:
    print("Keine validen Daten zum Plotten.")  # Hinzugefügt für den Fall, dass alle Daten NaN sind

Tc_theor = 2 / (np.log(1 + np.sqrt(2)))
plt.axvline(Tc_theor, color='r', linestyle='--', label=f'Theoretical $T_c \\approx {Tc_theor:.3f}$')

plt.xlabel('Temperature $T$ ($J/k_B$)')  # Achsenbeschriftung angepasst
plt.ylabel('Average Absolute Magnetization per Site $\langle |M| \\rangle / N$')
plt.title(f'Ising Model Magnetization vs. Temperature (Wolff, L={L})')  # Titel angepasst
plt.legend()
plt.grid(True)
plt.show()
