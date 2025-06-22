import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors
import numpy.random as random
import time # Added for timing

# --- Correct Autocorrelation and Error Functions ---

def autocorr_func_1d(x, norm=True):
    """Calculates the autocorrelation function of a 1D array."""
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = len(x)
    if n < 2:
        return np.array([1.0]) if norm else np.array([np.var(x)]) # Handle short series

    # Variance calculation without assuming mean is 0
    var = np.var(x)
    if var == 0: # Handle constant data case
        return np.zeros(n)

    x = x - np.mean(x)
    # Use numpy's correlate function in 'full' mode
    # r[k] = sum_{i=0}^{n-1-k} x_i * x_{i+k}
    r = np.correlate(x, x, mode='full')[n-1:] # Get lags 0 to n-1

    if norm:
        # Autocorrelation: Normalize by variance * (N-k) (?) - No, just C(0) approx N*Var
        # Standard definition: rho[k] = C[k] / C[0]
        # C[0] = sum(x_i^2) approx N*Var(x)
        # Note: np.correlate result needs careful normalization depending on mode
        # Let's use C(t) / C(0) where C(t) = E[(X_i - mu)(X_{i+t} - mu)]
        # For finite samples: C_hat(t) = (1/(N-t)) * sum_{i=0}^{N-t-1} (x_i - x_bar)(x_{i+t} - x_bar)
        # rho_hat(t) = C_hat(t) / C_hat(0) = C_hat(t) / Var(x)

        # Simplified approach using np.correlate result directly:
        # r[0] = sum(x*x) approx N*var
        # r[k] = sum(x_i * x_{i+k}) for k > 0
        # Normalization factor should be r[0]
        if r[0] == 0:
           return np.zeros(n) # Should not happen if var!=0
        result = r / r[0]
        return result
        # Alternative using statsmodels definition (often more robust)
        # try:
        #    import statsmodels.tsa.stattools as smt
        #    return smt.acf(x, nlags=n-1, fft=True)
        # except ImportError:
        #    print("Warning: statsmodels not found, using numpy.correlate for ACF.")
        #    if r[0] == 0: return np.zeros(n)
        #    return r / r[0]

    else:
        # Autocovariance C(t): Need to divide r[k] by N-k ? No.
        # r[k] is sum(x_i*x_{i+k}). Covariance is E[..].
        # Estimate C(t) = (1/N) * r[k] is biased but common.
        # Unbiased est C(t) = (1/(N-k)) * r[k]
        # Let's return sum, often sufficient if used consistently.
        return r # Return the direct sum result

def calculate_integrated_autocorrelation_time(time_series, c=5):
    """
    Estimates the integrated autocorrelation time tau using the windowing method.

    Args:
        time_series (array-like): The time series data.
        c (float): Factor determining the window size (typically 4 to 10).

    Returns:
        float: Estimated integrated autocorrelation time tau.
    """
    time_series = np.asarray(time_series)
    if time_series.ndim > 1:
         raise ValueError("Time series must be 1D")
    n_samples = len(time_series)
    if n_samples < 2:
        return 0.5 # Minimal tau

    try:
        # Calculate normalized autocorrelation function rho(t) using the helper
        rho = autocorr_func_1d(time_series, norm=True)

        # Ensure rho is finite
        if not np.all(np.isfinite(rho)):
             print(f"Warning: Non-finite values found in autocorrelation function (N={n_samples}). Returning tau=0.5")
             return 0.5 # Return minimal tau if ACF calculation failed

        # Automated windowing procedure (Sokal 1997)
        tau_int_sum = 0.0
        # The summation goes from t=1 up to W (window size)
        # rho[0] is always 1
        for t in range(1, n_samples):
            # Stop if rho becomes statistically indistinguishable from 0 (or negative)
            # Heuristic: Stop when the sum would decrease by adding rho[t] and rho[t+1]?
            # Sokal: Sum up to W where W = c*tau(W)
            # We estimate tau(W) = 0.5 + sum_{t=1}^{W} rho(t) iteratively

            if rho[t] < 0: # Stop summation if ACF turns negative (or use noise threshold)
                 # Optional: Check if negative value is statistically significant
                 # noise_level = approx 1/sqrt(N) ?? Needs care.
                 # Let's use simple cutoff for now.
                 break

            current_tau_est = 0.5 + tau_int_sum # Estimate tau using sum up to t-1
            if t > c * current_tau_est and t > 10: # Ensure window is not too small
                break # Stop summation: Window W = t-1

            tau_int_sum += rho[t]

        tau = 0.5 + tau_int_sum
        # Ensure tau is at least 0.5 (uncorrelated samples) and finite
        tau = max(0.5, tau)
        if not np.isfinite(tau):
            print(f"Warning: Calculated tau is not finite ({tau}, N={n_samples}). Returning tau=0.5")
            tau = 0.5

    except Exception as e:
        print(f"Error calculating autocorrelation time (N={n_samples}): {e}. Returning tau=0.5")
        tau = 0.5 # Default value in case of errors

    return tau


def calculate_autocorrelation_error(time_series):
    """
    Calculates the standard error of the mean for a potentially autocorrelated time series.

    Args:
        time_series (array-like): The time series data (e.g., magnetization history).

    Returns:
        float: The estimated standard error of the mean.
    """
    time_series = np.asarray(time_series)
    N = len(time_series)

    if N <= 1:
        # Cannot compute variance or autocorrelation reliably
        print("Warning: Cannot compute error for time series with N <= 1.")
        return np.nan # Return Not a Number

    variance = np.var(time_series) # Variance of the samples

    if variance < 1e-15:
        # If the data is constant, the error is zero
        return 0.0

    # Estimate the integrated autocorrelation time
    tau = calculate_integrated_autocorrelation_time(time_series)

    # Calculate the effective number of independent samples
    # Factor of 2 comes from variance of the mean for autocorrelated series
    n_eff = N / (2.0 * tau)

    if n_eff <= 1:
        # If N_eff is too small, the error estimate is unreliable.
        # This can happen if tau is very large (>= N/2).
        print(f"Warning: Effective sample size N_eff={n_eff:.2f} (N={N}, tau={tau:.2f}) <= 1. Error estimate might be unreliable. Using naive error.")
        # Fallback to naive standard error (assumes independence)
        standard_error = np.sqrt(variance / N)
        return standard_error
    else:
        # Standard error corrected for autocorrelation
        standard_error = np.sqrt(variance / n_eff)
        # Equivalent to: standard_error = np.sqrt(variance * 2 * tau / N)
        return standard_error

# --- User's Ising Model Class (with modifications) ---

class IsingModelVisualizer:

    def __init__(self, size, temperature):
        self.size = size
        self.N = size * size # Store N
        self.temperature = temperature
        if self.temperature <= 0:
            raise ValueError("Temperature must be positive.")
        self.beta = 1.0 / temperature
        self.lattice = self._initialize_lattice()

    def _initialize_lattice(self):
        # Start from random state is generally better for exploring phase space
        return np.random.choice([-1, 1], size=(self.size, self.size))
        # return np.ones((self.size, self.size), dtype=int) # Cold start option

    # Not needed for Metropolis if using _calculate_energy_change
    # def _calculate_energy(self): ...

    def _calculate_energy_change(self, i, j):
        # Assuming J=1, H=0
        spin = self.lattice[i, j]
        neighbor_sum = (
            self.lattice[(i + 1) % self.size, j]
            + self.lattice[(i - 1) % self.size, j]
            + self.lattice[i, (j + 1) % self.size]
            + self.lattice[i, (j - 1) % self.size]
        )
        # Delta E for flipping spin at (i,j)
        return 2.0 * spin * neighbor_sum # Factor of J=1 implicitly included

    def _metropolis_step(self):
        """Performs one Metropolis update attempt."""
        # Select random spin
        i = random.randint(0, self.size)
        j = random.randint(0, self.size)

        # Calculate energy change if this spin is flipped
        delta_e = self._calculate_energy_change(i, j)

        # Metropolis acceptance condition
        if delta_e <= 0 or random.rand() < np.exp(-self.beta * delta_e):
            self.lattice[i, j] *= -1  # Flip the spin

    def run_simulation(self, num_sweeps, thermalization_sweeps=0):
        """
        Runs the simulation for a number of sweeps.
        Measures magnetization once per sweep after thermalization.

        Args:
            num_sweeps (int): Number of measurement sweeps.
            thermalization_sweeps (int): Number of sweeps to discard for equilibration.

        Returns:
            list: History of absolute magnetization per spin measured after each production sweep.
        """
        magnetizations = []
        num_steps_per_sweep = self.N # Perform N steps per sweep

        print(f"  Thermalizing ({thermalization_sweeps} sweeps)...")
        # Thermalization sweeps
        for sweep in range(thermalization_sweeps):
            for _ in range(num_steps_per_sweep):
                 self._metropolis_step() # Perform N steps

        print(f"  Measuring ({num_sweeps} sweeps)...")
        # Production sweeps
        for sweep in range(num_sweeps):
            for _ in range(num_steps_per_sweep):
                 self._metropolis_step() # Perform N steps
            # Measure AFTER the sweep
            magnetizations.append(self.get_magnetization())

        print(f"  Finished sweeps.")
        return magnetizations

    def get_magnetization(self):
        """Calculates the absolute magnetization per spin (|M|/N)."""
        # Using np.mean is equivalent to np.sum/N
        return abs(np.mean(self.lattice))

    def visualize_lattice_2d(self):
        # (Keep your visualization code as is)
        fig, ax = plt.subplots(figsize=(8, 8))
        colors = [(1, 1, 0), (1, 0.5, 0), (0.5, 0, 0.5), (0, 0, 0)]
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)
        norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
        im = ax.imshow(self.lattice, cmap=cmap, norm=norm, origin='lower', extent=[0, self.size, 0, self.size])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"2D Ising Model at T = {self.temperature:.2f} (β = {self.beta:.2f})")
        ax.set_xticks(np.arange(0.5, self.size, 1))
        ax.set_yticks(np.arange(0.5, self.size, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(color='black', linestyle='-', linewidth=0.5)
        cbar = fig.colorbar(im, ax=ax, ticks=[-1, -0.5, 0, 0.5, 1])
        cbar.set_label("Spin")
        plt.tight_layout()
        plt.show()

# --- Main Simulation Logic (Modified Error Handling) ---

def simulate_magnetization_vs_temperature(size, temperatures, num_sweeps, thermalization_sweeps, num_runs=1):
    """
    Simulates the Ising model for various temperatures and calculates magnetization.

    Args:
        size (int): Lattice size (L).
        temperatures (array-like): Temperatures to simulate.
        num_sweeps (int): Number of measurement sweeps per run.
        thermalization_sweeps (int): Number of thermalization sweeps per run.
        num_runs (int): Number of independent runs per temperature.

    Returns:
        tuple: (average magnetizations, standard errors) across runs (or from autocorrelation if num_runs=1).
    """
    avg_magnetizations = []
    std_err_magnetizations = []
    all_taus = [] # Store autocorrelation times for info

    total_start_time = time.time()

    for i, temp in enumerate(temperatures):
        print(f"\n--- Simulating T = {temp:.3f} ({i+1}/{len(temperatures)}) ---")
        temp_start_time = time.time()
        run_final_magnetizations = [] # Store the final average M for each run at this T

        if num_runs == 1:
            model = IsingModelVisualizer(size, temp)
            # Run simulation (returns magnetization history *after* each sweep)
            magnetization_history = model.run_simulation(num_sweeps, thermalization_sweeps)

            if not magnetization_history:
                 print("Warning: No measurements collected.")
                 avg_mag = np.nan
                 std_err = np.nan
                 tau_info = np.nan
            else:
                 # Calculate average magnetization from the single run history
                 avg_mag = np.mean(magnetization_history)
                 # Calculate standard error using autocorrelation
                 std_err = calculate_autocorrelation_error(magnetization_history)
                 # Get tau for info (optional, re-calculates ACF but ok)
                 tau_info = calculate_integrated_autocorrelation_time(magnetization_history)

            avg_magnetizations.append(avg_mag)
            std_err_magnetizations.append(std_err)
            all_taus.append(tau_info)
            # Visualize the final state of the single run
            # model.visualize_lattice_2d()

        else: # num_runs > 1
            run_means = [] # Store the mean magnetization of each independent run
            print(f"  Performing {num_runs} independent runs...")
            for r in range(num_runs):
                print(f"    Run {r+1}/{num_runs}...")
                model = IsingModelVisualizer(size, temp)
                magnetization_history = model.run_simulation(num_sweeps, thermalization_sweeps)
                if magnetization_history:
                    run_means.append(np.mean(magnetization_history))
                else:
                    run_means.append(np.nan) # Handle case of no measurements if needed

            run_means = np.array(run_means)
            valid_run_means = run_means[~np.isnan(run_means)] # Exclude NaNs if any

            if len(valid_run_means) > 0:
                 # Average across the means of the independent runs
                 avg_mag = np.mean(valid_run_means)
                 # Standard error of the mean across runs
                 if len(valid_run_means) > 1:
                     # Use sample standard deviation (ddof=1)
                     std_err = np.std(valid_run_means, ddof=1) / np.sqrt(len(valid_run_means))
                 else:
                     std_err = np.nan # Cannot compute std err from a single run mean
                 tau_info = np.nan # Autocorrelation not typically used here
            else:
                 avg_mag = np.nan
                 std_err = np.nan
                 tau_info = np.nan

            avg_magnetizations.append(avg_mag)
            std_err_magnetizations.append(std_err)
            all_taus.append(tau_info) # No tau calculated in this mode
            # Visualize final state of the *last* run? Optional.
            # if r == num_runs - 1: model.visualize_lattice_2d()

        temp_end_time = time.time()
        print(f"  -> T={temp:.3f}: M = {avg_mag:.4f} +/- {std_err:.4f} (tau={tau_info:.2f} if N_runs=1)")
        print(f"  -> Time for T={temp:.3f}: {temp_end_time - temp_start_time:.2f} s")

    total_end_time = time.time()
    print(f"\nTotal Simulation Time: {total_end_time - total_start_time:.2f} s")
    return avg_magnetizations, std_err_magnetizations, all_taus

# --- Visualization Function (Modified for Clarity) ---

def visualize_magnetization_vs_temperature(temperatures, avg_magnetizations, std_err_magnetizations=None, taus=None):

    # Ensure inputs are numpy arrays for filtering
    temperatures = np.asarray(temperatures)
    avg_magnetizations = np.asarray(avg_magnetizations)
    if std_err_magnetizations is not None:
        std_err_magnetizations = np.asarray(std_err_magnetizations)

    # Filter out NaN values that might occur
    valid_indices = ~np.isnan(temperatures) & ~np.isnan(avg_magnetizations)
    if std_err_magnetizations is not None:
        valid_indices &= ~np.isnan(std_err_magnetizations)

    temps_plot = temperatures[valid_indices]
    mags_plot = avg_magnetizations[valid_indices]
    errs_plot = std_err_magnetizations[valid_indices] if std_err_magnetizations is not None else None

    plt.figure(figsize=(10, 6))
    if errs_plot is not None:
        plt.errorbar(temps_plot, mags_plot, yerr=errs_plot, fmt='o-', capsize=5, label='Simulation $\langle |M| \\rangle / N$')
    else:
        plt.plot(temps_plot, mags_plot, 'o-', label='Simulation $\langle |M| \\rangle / N$')

    # Theoretical Tc for 2D Ising model
    Tc_theor = 2.0 / np.log(1 + np.sqrt(2)) # Assuming J=1, kB=1
    plt.axvline(Tc_theor, color='r', linestyle='--', label=f'Theoretical $T_c \\approx {Tc_theor:.3f}$')

    plt.xlabel("Temperature $T$ ($J/k_B$)")
    plt.ylabel("Average Absolute Magnetization per Spin")
    plt.title(f"Ising Model Magnetization vs. Temperature (L={lattice_size})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Optional: Plot Tau if available (e.g., if num_runs=1)
    if taus is not None:
        taus = np.asarray(taus)
        valid_taus = taus[valid_indices]
        # Check if there are any valid tau values to plot
        if np.any(~np.isnan(valid_taus)):
            plt.figure(figsize=(10, 6))
            plt.plot(temps_plot, valid_taus, 's-')
            plt.axvline(Tc_theor, color='r', linestyle='--', label=f'Theoretical $T_c \\approx {Tc_theor:.3f}$')
            plt.xlabel('Temperature $T$ ($J/k_B$)')
            plt.ylabel('Integrated Autocorrelation Time $\\tau$')
            plt.title(f'Autocorrelation Time vs. Temperature (L={lattice_size}, N_runs=1)')
            #plt.yscale('log') # Tau often varies significantly, log scale can be useful
            plt.grid(True, which="both", ls="-")
            plt.legend()
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    lattice_size = 10 # Start smaller for quicker tests (e.g., 10 or 16)
    temperatures_to_simulate = np.linspace(1.5, 3.5, 21) # Range around Tc
    # temperatures_to_simulate = np.linspace(1.0, 4.0, 15)

    # --- SIMULATION PARAMETERS ---
    # Expressed in sweeps (1 sweep = N steps)
    # These need to be significantly larger near Tc or for larger L
    num_thermalization_sweeps = 1000 # Number of sweeps to discard
    num_measurement_sweeps = 5000  # Number of sweeps used for measurements

    # Choose error estimation strategy:
    # num_independent_runs = 1 # Use autocorrelation within one long run
    num_independent_runs = 5 # Use variance across multiple shorter runs

    print(f"Starting Ising simulation: L={lattice_size}, Runs/T={num_independent_runs}")
    print(f"Sweeps: Thermalization={num_thermalization_sweeps}, Measurement={num_measurement_sweeps}")

    avg_mags, std_err_mags, tau_results = simulate_magnetization_vs_temperature(
        lattice_size,
        temperatures_to_simulate,
        num_measurement_sweeps,
        num_thermalization_sweeps,
        num_independent_runs
    )

    visualize_magnetization_vs_temperature(
        temperatures_to_simulate,
        avg_mags,
        std_err_mags,
        taus = tau_results if num_independent_runs == 1 else None # Pass taus only if calculated
    )