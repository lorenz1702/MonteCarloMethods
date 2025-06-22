import numpy as np
import random
from collections import Counter

# --- Part 1: System Definition & Simulation Parameters ---
prop = np.array([0.5, 0.2, 0.05, 0.05, 0.2])
N = len(prop)
num_samples = 100000
burn_in_steps = 5000

# --- Part 2: Markov Chain Simulation (Accept/Reject Method) ---
def markov_step_metropolis(current_state, target_prop):
    N_states = len(target_prop)
    proposed_state = random.randint(0, N_states - 1)

    p_current = target_prop[current_state]
    p_proposed = target_prop[proposed_state]

    if p_current == 0:
        acceptance_prob = 1.0
    else:
        acceptance_prob = min(1, p_proposed / p_current)

    if random.random() < acceptance_prob:
        return proposed_state
    else:
        return current_state

def run_markov_chain(num_steps, start_state, target_prop):
    states = []
    current_state = start_state
    for _ in range(num_steps):
        current_state = markov_step_metropolis(current_state, target_prop)
        states.append(current_state)
    return states

# --- Part 3: Compute Observables from Simulation ---
print("--- Results from Markov Chain Simulation ---")
initial_s = random.randint(0, N - 1)
states_history = run_markov_chain(num_samples, initial_s, prop)
equilibrium_states = states_history[burn_in_steps:]

# 1. Verify Probability Distribution
print("Desired probability distribution:")
print(prop)

state_counts = Counter(equilibrium_states)
total_valid_samples = len(equilibrium_states)
simulated_prop = np.zeros(N)
for state, count in sorted(state_counts.items()):
    simulated_prop[state] = count / total_valid_samples

print("Distribution obtained from simulation:")
print(np.round(simulated_prop, 4))
print("")

# 2. Compute the Observable (Mean State)
print("--- Observable Calculation ---")
mean_state_from_simulation = np.mean(equilibrium_states)
print(f"Mean state from simulation: {mean_state_from_simulation:.4f}")

theoretical_mean_state = np.sum(np.arange(N) * prop)
print(f"Theoretical mean state: {theoretical_mean_state:.4f}")
print("")


# --- Part 4: Verification via Transition Matrix ---
print("--- Verification via Transition Matrix ---")

def acceptance(prop_param):
    N_param = len(prop_param)
    p_acc = np.zeros((N_param, N_param))
    for i in range(N_param):
        for j in range(N_param):
            if (prop_param[i] == 0):
                p_acc[i, j] = 1
            else:
                p_acc[i, j] = min(1, (prop_param[j] / prop_param[i]))
    return p_acc

def transition_matrix(prop_param, p_acc):
    N_param = len(prop_param)
    T = np.zeros((N_param, N_param))

    for i in range(N_param):
        for j in range(N_param):
            if i != j:
                T[i, j] = 1/N_param * p_acc[i, j]
            else:
                sum_val = 0
                for k in range(N_param):
                    sum_val += 1/N_param * (1-p_acc[i,k])
                T[i, i] = 1/N_param * p_acc[i, i] + sum_val
    return T

def check_correctness(start_state_param, transition_matrix_param, num_chains):
    for i in range(num_chains):
            next_prob = start_state_param.dot(transition_matrix_param)
            start_state_param = next_prob
    return start_state_param

p_acc_matrix = acceptance(prop)
T = transition_matrix(prop, p_acc_matrix)
start_state = np.array([0, 0, 1.0, 0, 0])
final_prop = check_correctness(start_state, T, 100)

print("Prop after using the computed transition matrix 100 times:")
print(final_prop)
