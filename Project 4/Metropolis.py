import numpy as np
import matplotlib.pyplot as plt

# Systemgröße
L = 100

# Kopplungskonstante J
J = 1.0

# Temperaturwerte
temperatures = [0.5, 1.0, 2.0, 5.0]

# Hilfsfunktion zur Energieberechnung mit periodischen Randbedingungen
def calculate_energy(spins):
    energy = 0
    for i in range(L):
        for j in range(L):
            S = spins[i, j]
            neighbors = spins[(i+1)%L, j] + spins[i, (j+1)%L] + \
                        spins[(i-1)%L, j] + spins[i, (j-1)%L]
            energy += -J * S * neighbors
    return energy / 2  # Jeder Nachbar wird doppelt gezählt

# Erzeuge zufällige Spin-Konfiguration
spins = np.random.choice([-1, 1], size=(L, L))

# Plot Heatmap der Spins
plt.figure(figsize=(4, 4))
plt.imshow(spins, cmap='coolwarm', interpolation='nearest')
plt.title('Zufällige Spin-Konfiguration')
plt.colorbar(label='Spin')
plt.axis('off')
plt.show()

# Liste von Energiezuständen erzeugen durch kleine Veränderungen
energies = []
for _ in range(100):
    # kleine Änderung: flippe zufälligen Spin
    i, j = np.random.randint(0, L, size=2)
    new_spins = np.copy(spins)
    new_spins[i, j] *= -1
    E = calculate_energy(new_spins)
    energies.append(E)

energies = np.array(energies)

# Boltzmann-Verteilungen plotten
plt.figure(figsize=(10, 6))

for T in temperatures:
    beta = 1 / T
    P_unnorm = np.exp(-beta * energies)
    Z = np.sum(P_unnorm)
    P = P_unnorm / Z
    plt.plot(energies, P, 'o', label=f'T = {T}')

plt.xlabel('Energie')
plt.ylabel('Boltzmann- P(E)')
plt.title('Boltzmann-Verteilung aus Spin-Konfigurationen')
plt.legend()
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 20                # Lattice size (LxL)
N = L * L
J = 1.0               # Coupling constant
steps_per_temp = 10000  # Total Monte Carlo steps per temperature
thermalization = 2000   # Thermalization steps
temps = np.linspace(1.5, 3.5, 20)  # Range of temperatures

def delta_energy(spins, i, j):
    """Calculate energy change if spin at (i, j) is flipped"""
    left = spins[i, (j - 1) % L]
    right = spins[i, (j + 1) % L]
    up = spins[(i - 1) % L, j]
    down = spins[(i + 1) % L, j]
    return 2 * J * spins[i, j] * (left + right + up + down)

def metropolis_step(spins, beta):
    """Perform one Metropolis sweep (N attempted spin flips)"""
    for _ in range(N):
        i, j = np.random.randint(0, L), np.random.randint(0, L)
        dE = delta_energy(spins, i, j)
        if dE <= 0 or np.random.rand() < np.exp(-beta * dE):
            spins[i, j] *= -1

def compute_autocorrelation(data):
    """Estimate integrated autocorrelation time"""
    data = data - np.mean(data)
    corr = np.correlate(data, data, mode='full') / len(data)
    corr = corr[len(data)-1:] / corr[len(data)-1]
    tau = 0.5
    for t in range(1, len(corr)):
        if corr[t] < 0:
            break
        tau += corr[t]
    return tau

magnetizations = []
errors = []

for T in temps:
    beta = 1.0 / T
    spins = np.random.choice([-1, 1], size=(L, L))

    # Thermalization
    for _ in range(thermalization):
        metropolis_step(spins, beta)

    mags = []
    for _ in range(steps_per_temp):
        metropolis_step(spins, beta)
        mag = np.abs(np.sum(spins)) / N
        mags.append(mag)

    mags = np.array(mags)
    tau_int = compute_autocorrelation(mags)
    N_eff = len(mags) / (2 * tau_int)
    mean_mag = np.mean(mags)
    std_error = np.std(mags) / np.sqrt(N_eff)

    magnetizations.append(mean_mag)
    errors.append(std_error)

    print(f"T={T:.2f} | <M>={mean_mag:.3f} ± {std_error:.3f} | τ_int ≈ {tau_int:.1f}")

# Plot
plt.figure(figsize=(8,6))
plt.errorbar(temps, magnetizations, yerr=errors, fmt='o-', capsize=4)
plt.xlabel('Temperature T')
plt.ylabel('Average Magnetization ⟨M⟩')
plt.title(f'Ising Model (L={L}x{L}) — Magnetization vs. Temperature')
plt.grid(True)
plt.show()

