import numpy as np
import matplotlib.pyplot as plt
import time
import random

L = 64
N = L * L
q_states = 5

temperatures = np.concatenate((np.linspace(0.5, 0.80, 7, endpoint=False), np.linspace(0.80, 1.0, 11, endpoint=False),
                               np.linspace(1.0, 2.0, 7, endpoint=True)))
temperatures = np.linspace(0.845, 0.862, 20)

n_th_steps_wolff = 200
n_mc_steps_wolff = 1000
measurement_interval_wolff = 10





def initial_state_hot(L_param, q_param):
    return np.random.randint(0, q_param, size=(L_param, L_param))


def initial_state_cool(L_param, q_param):
    return np.zeros((L_param, L_param), dtype=int)


def calculate_magnetization(spins_param, q_param):
    N_total = spins_param.size
    counts = np.bincount(spins_param.ravel(), minlength=q_param)
    N_max = np.max(counts)
    if q_param <= 1: return 1.0
    magnetization = (q_param * N_max - N_total) / ((q_param - 1.0) * N_total)
    return magnetization


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


def wolff_cluster_step_potts(spins_param, L_param, q_param, T_param):
    i_seed, j_seed = np.random.randint(0, L_param), np.random.randint(0, L_param)
    S_seed = spins_param[i_seed, j_seed]
    C_set = set()
    C_set.add((i_seed, j_seed))
    F_stack = [(i_seed, j_seed)]
    p_add_bond_potts = 1. - np.exp(-1.0 / T_param)
    while F_stack:
        i_curr, j_curr = F_stack.pop()
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            i_neigh, j_neigh = (i_curr + dr + L_param) % L_param, \
                               (j_curr + dc + L_param) % L_param
            neighbour_coord_tuple = (i_neigh, j_neigh)
            if spins_param[i_neigh, j_neigh] == S_seed and \
                    neighbour_coord_tuple not in C_set:
                if np.random.rand() < p_add_bond_potts:
                    C_set.add(neighbour_coord_tuple)
                    F_stack.append(neighbour_coord_tuple)
    possible_new_states = [s for s in range(q_param) if s != S_seed]
    if not possible_new_states:
        return
    new_cluster_state = random.choice(possible_new_states)
    for i_cluster, j_cluster in C_set:
        spins_param[i_cluster, j_cluster] = new_cluster_state


def run_simulation(start_type):
    results = {'T': [], 'M': [], 'M_err': [], 'tau': []}
    for T in temperatures:
        print(f"Simulating T = {T:.3f} (Start: {start_type})...")
        if start_type == 'hot':
            spins = initial_state_hot(L, q_states)
        else:  # 'cool'
            spins = initial_state_cool(L, q_states)

        magnetization_timeseries = []
        for step in range(n_th_steps_wolff):
            wolff_cluster_step_potts(spins, L, q_states, T)
        for step in range(n_mc_steps_wolff):
            wolff_cluster_step_potts(spins, L, q_states, T)
            if step % measurement_interval_wolff == 0:
                M_current = calculate_magnetization(spins, q_states)
                magnetization_timeseries.append(M_current)

        magnetization_timeseries = np.array(magnetization_timeseries)
        n_meas = len(magnetization_timeseries)
        if n_meas > 1:
            M_avg = np.mean(magnetization_timeseries)
            variance_M = np.var(magnetization_timeseries, ddof=0)
            tau = calculate_integrated_autocorrelation_time(magnetization_timeseries)
            n_eff = n_meas / (2 * tau)
            if n_eff < 1: n_eff = 1
            M_err = np.sqrt(variance_M / n_eff) if variance_M >= 0 else np.nan
        elif n_meas == 1:
            M_avg = magnetization_timeseries[0];
            M_err = np.nan;
            tau = 0.5
        else:
            M_avg = np.nan;
            M_err = np.nan;
            tau = np.nan

        results['T'].append(T)
        results['M'].append(M_avg)
        results['M_err'].append(M_err)
        results['tau'].append(tau)
    return results


start_time_total = time.time()
print("\n--- Starting Wolff Simulation (Hot Start) ---")
results_hot = run_simulation('hot')
print("\n--- Starting Wolff Simulation (Cool Start) ---")
results_cool = run_simulation('cool')
end_time_total = time.time()
print(f"\nTotal simulation time: {end_time_total - start_time_total:.2f} s")

plt.figure(figsize=(12, 7))

temps_hot = np.array(results_hot['T'])
mag_hot = np.array(results_hot['M'])
mag_err_hot = np.array(results_hot['M_err'])
valid_indices_hot = ~np.isnan(mag_hot) & ~np.isnan(mag_err_hot)
if np.any(valid_indices_hot):
    plt.errorbar(temps_hot[valid_indices_hot], mag_hot[valid_indices_hot],
                 yerr=mag_err_hot[valid_indices_hot], fmt='o-', capsize=5,
                 label=f'Wolff (Hot Start)')

temps_cool = np.array(results_cool['T'])
mag_cool = np.array(results_cool['M'])
mag_err_cool = np.array(results_cool['M_err'])
valid_indices_cool = ~np.isnan(mag_cool) & ~np.isnan(mag_err_cool)
if np.any(valid_indices_cool):
    plt.errorbar(temps_cool[valid_indices_cool], mag_cool[valid_indices_cool],
                 yerr=mag_err_cool[valid_indices_cool], fmt='s--', capsize=5,
                 label=f'Wolff (Cool Start)')

Tc_theor_potts_q5 = 1.0 / np.log(1 + np.sqrt(q_states))
plt.axvline(Tc_theor_potts_q5, color='k', linestyle=':',
            label=f'Theoretical $T_c (q={q_states}) \\approx {Tc_theor_potts_q5:.3f}$')

plt.xlabel('Temperature $T$ ($J/k_B$)')
plt.ylabel(f'Magnetization M')
plt.title(f'Potts Model (q={q_states}): Hot vs. Cool Start (L={L})')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 7))

tau_hot = np.array(results_hot['tau'])
valid_tau_hot = ~np.isnan(tau_hot) & valid_indices_hot
if np.any(valid_tau_hot):
    plt.plot(temps_hot[valid_tau_hot], tau_hot[valid_tau_hot], 'o-', label='Wolff $\\tau_{int}$ (Hot Start)')

tau_cool = np.array(results_cool['tau'])
valid_tau_cool = ~np.isnan(tau_cool) & valid_indices_cool
if np.any(valid_tau_cool):
    plt.plot(temps_cool[valid_tau_cool], tau_cool[valid_tau_cool], 's--', label='Wolff $\\tau_{int}$ (Cool Start)')

plt.axvline(Tc_theor_potts_q5, color='k', linestyle=':',
            label=f'Theoretical $T_c (q={q_states}) \\approx {Tc_theor_potts_q5:.3f}$')
plt.xlabel('Temperature $T$ ($J/k_B$)')
plt.ylabel('Integrated Autocorrelation Time $\\tau_{int}$')
plt.title(f'Autocorrelation Time (q={q_states} Potts, L={L})')
plt.legend()
plt.grid(True)
plt.yscale('log')
plt.show()
