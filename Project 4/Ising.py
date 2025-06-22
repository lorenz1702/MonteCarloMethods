import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors
import numpy.random as random

def autocorrelation_error(y):

    N = len(y)

    if N <= 1:
        return 0.0
    y = np.array(y)
    mean_y = np.mean(y)
    sigma2_y = np.var(y)

    if sigma2_y < 1e-15:
        return 0.0
    def autocorrelation(tau):

        N_inner = len(y)
        y_mean_inner = mean_y
        c_sum = 0
        c_zero_sum = 0


        if tau >= N_inner:
            return 0.0

        if N_inner - tau <= 0:
             return 0.0

        for i in range(0, N_inner - tau):
            term_i = (y[i] - y_mean_inner)
            term_i_tau = (y[i + tau] - y_mean_inner)
            c_sum += term_i * term_i_tau

            c_zero_sum += term_i * term_i


        if abs(c_zero_sum) < 1e-15:
            return 0.0
        c_sum /= c_zero_sum


        if N_inner - tau <= 0:
             return 0.0
        mean_c_sum = c_sum / (N_inner - tau)

        weight = 1 - ((tau * 1.0) / N_inner)

        return (weight * mean_c_sum)


    tau_int = 0.5
    for t_loop in range(1, N):
        auto_corr = autocorrelation(t_loop)
        if (auto_corr < 0):
            break
        tau_int += auto_corr

    sqrt_argument = 2 * tau_int / N
    if sqrt_argument < 0:
        sqrt_argument = 0.0

    sigma2_y_error = sigma2_y * np.sqrt(sqrt_argument)
    return sigma2_y_error

class IsingModelVisualizer:

    def __init__(self, size, temperature):
        self.size = size
        self.temperature = temperature
        self.beta = 1.0 / temperature
        self.lattice = self._initialize_lattice()

    def _initialize_lattice(self):
        return np.random.choice([-1, 1], size=(self.size, self.size))

    def _calculate_energy(self):
        energy = 0
        for i in range(self.size):
            for j in range(self.size):
                spin = self.lattice[i, j]
                neighbor_sum = (
                    self.lattice[(i + 1) % self.size, j]
                    + self.lattice[(i - 1) % self.size, j]
                    + self.lattice[i, (j + 1) % self.size]
                    + self.lattice[i, (j - 1) % self.size]
                )
                energy -= spin * neighbor_sum
        return energy / 2.0  # Divide by 2 to avoid double counting interactions

    def _calculate_energy_change(self, i, j):
        spin = self.lattice[i, j]
        neighbor_sum = (
            self.lattice[(i + 1) % self.size, j]
            + self.lattice[(i - 1) % self.size, j]
            + self.lattice[i, (j + 1) % self.size]
            + self.lattice[i, (j - 1) % self.size]
        )
        return 2 * spin * neighbor_sum


    def _metropolis_step(self):
        i = random.randint(0, self.size)
        j = random.randint(0, self.size)
        delta_e = self._calculate_energy_change(i, j)

        if delta_e <= 0:
            self.lattice[i, j] *= -1  # Flip the spin
        else:
            acceptance_probability = np.exp(-self.beta * delta_e)
            if random.rand() < acceptance_probability:
                self.lattice[i, j] *= -1  # Flip the spin

    def run_simulation(self, num_steps, thermalization_steps=0):
        magnetizations = []
        num_steps_per_sweep = self.size * self.size

        # Thermalization
        for _ in range(thermalization_steps):
            for _ in range(num_steps_per_sweep):
                self._metropolis_step()

        # Production run
        for _ in range(num_steps):
            for _ in range(num_steps_per_sweep):
                self._metropolis_step()
            magnetizations.append(self.get_magnetization())

        return magnetizations

    def get_magnetization(self):
        return abs(np.mean(self.lattice))


    def visualize_lattice_2d(self):
        fig, ax = plt.subplots(figsize=(8, 8))

        # Define custom colormap
        colors = [(1, 1, 0), (1, 0.5, 0), (0.5, 0, 0.5), (0, 0, 0)]  # Yellow, Orange, Purple, Black
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)
        norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)

        im = ax.imshow(self.lattice, cmap=cmap, norm=norm, origin='lower', extent=[0, self.size, 0, self.size])

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"2D Ising Model at β = {self.beta}")
        ax.set_xticks(np.arange(0.5, self.size, 1))
        ax.set_yticks(np.arange(0.5, self.size, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(color='black', linestyle='-', linewidth=0.5)

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, ticks=[-1, -0.5, 0, 0.5, 1])

        plt.tight_layout()
        plt.show()

def simulate_magnetization_vs_temperature(size, temperatures, num_steps, thermalization_steps, num_runs=5):
    avg_magnetizations = []
    std_err_magnetizations = []

    for temp in temperatures:
        all_magnetizations = []
        run_means = []
        run_SE_squared_list = []
        for _ in range(num_runs):
            model = IsingModelVisualizer(size, temp)
            magnetization_history = model.run_simulation(num_steps, thermalization_steps)
            #all_magnetizations.append(magnetization_history[num_steps:])# Average over the last half
            all_magnetizations.append(np.mean(magnetization_history[-int(0.5*num_steps):]))# Average over the last half
            #auto correlation
            data_series = np.array(magnetization_history)
            N = len(data_series)

            if N > 1:
                run_mean = np.mean(data_series)
                run_means.append(run_mean)
                error_term_k = autocorrelation_error(data_series)
                run_SE_squared_list.append(error_term_k)

        model.visualize_lattice_2d()
        avg_magnetization = np.mean(all_magnetizations)
        avg_magnetizations.append(avg_magnetization)
        std_err_magnetizations.append(run_SE_squared_list)
    return avg_magnetizations, std_err_magnetizations

def visualize_magnetization_vs_temperature(temperatures, avg_magnetizations, std_err_magnetizations=None):

    plt.figure(figsize=(10, 6))
    if std_err_magnetizations is not None:
        std_err_magnetizations = np.squeeze(std_err_magnetizations)
        plt.errorbar(temperatures, avg_magnetizations, yerr=std_err_magnetizations, fmt='o-', capsize=5, label='Magnetisierung')
    else:
        plt.plot(temperatures, avg_magnetizations, 'o-', label='Magnetization')

    plt.xlabel("Temperature")
    plt.ylabel("Magnetization per Spin")
    plt.title("Magnetization vs. Temperature")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    lattice_size = 15
    temperatures_to_simulate = np.linspace(1.0, 4.0, 15)  # Mehr Temperaturpunkte
    num_monte_carlo_steps = 1000
    num_thermalization_steps = 500
    #num_monte_carlo_steps = 100000
    #num_thermalization_steps = 50000
    num_independent_runs = 1

    avg_mags, std_err_mags = simulate_magnetization_vs_temperature(
        lattice_size, temperatures_to_simulate, num_monte_carlo_steps, num_thermalization_steps, num_independent_runs
    )
    visualize_magnetization_vs_temperature(temperatures_to_simulate, avg_mags, std_err_mags)