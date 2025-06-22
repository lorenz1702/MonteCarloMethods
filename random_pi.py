import random
import math

def generate_random_number():
    """Generiert eine Zufallszahl zwischen 0 und Pi."""
    return random.uniform(0, math.pi)

def calculate_sine(x):
    """Berechnet den Sinus von x."""
    return math.sin(x)

def calculate_average_sine(num_samples):
    """
    Berechnet den Durchschnitt der Sinuswerte von zufälligen Zahlen
    zwischen 0 und Pi.

    Args:
        num_samples: Die Anzahl der Zufallszahlen, die generiert und
                     für die Berechnung des Durchschnitts verwendet werden sollen.

    Returns:
        Den Durchschnitt der Sinuswerte.  Gibt None zurück, wenn num_samples
        kleiner oder gleich 0 ist.
    """

    total_sine = 0
    for _ in range(num_samples):
        random_x = generate_random_number()
        total_sine += calculate_sine(random_x)
    return (total_sine / num_samples)* math.pi



# Hauptteil des Skripts
num_samples = 10  # Verwende eine große Anzahl für eine bessere Annäherung

average_sine = calculate_average_sine(num_samples)

if average_sine is not None:
    print(f"Der Durchschnitt von sin(x) für {num_samples} Zufallszahlen zwischen 0 und Pi ist: {average_sine}")

# Optional: Vergleich mit dem theoretischen Wert (für Interessierte)
# Der theoretische Wert des Integrals von sin(x) von 0 bis pi ist 2.
# Der Durchschnitt ist dann 2 / pi.
theoretical_average = 2 
print(f"Der theoretische Durchschnittswert ist: {theoretical_average}")

if average_sine is not None:
   print(f"Differenz zwischen berechnetem und theoretischem Durchschnitt: {abs(average_sine - theoretical_average)}")