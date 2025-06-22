import numpy as np
import random
from collections import Counter



def markov_step(s, min_state=0, max_state=4):
    a = random.random()
    if a < (1/3):
        s_new = s
    elif (1/3) <= a < (2/3):
        s_new = s + 1
    else:
        s_new = s - 1

    if s_new < min_state:
        s_new = max_state
    elif s_new > max_state:
        s_new = min_state

    return s_new

def estimate_state_probabilities(num_samples, min_state=0, max_state=4):
    states = []
    current_state = random.randint(min_state, max_state)
    for _ in range(num_samples):
        current_state = markov_step(current_state, min_state, max_state)
        states.append(current_state)

    state_counts = Counter(states)
    total_samples = len(states)

    state_probabilities = {state: count / total_samples for state, count in state_counts.items()}
    return state_probabilities

def check_correctness(start_state, transition_matrix, num_chains):
    for i in range(num_chains):
            next_prob = start_state.dot(transition_matrix)
            start_state = next_prob
    return start_state

T = np.array([[1, 1, 0, 0, 1],
              [1, 1, 1, 0, 0],
              [0, 1, 1, 1, 0],
              [0, 0, 1, 1, 1],
              [1, 0, 0, 1, 1]])

T = 1/3 * T

num_samples = 10000
min_state = 0
max_state = 4

probabilities = estimate_state_probabilities(num_samples, min_state, max_state)

for state, probability in sorted(probabilities.items()):
    print(f"State {state}: {probability:.4f}")


start_state = np.array([1/2, 1/2, 0, 0, 0])
prop = check_correctness(start_state, T, 100)
print("Prop:")
print(prop)







