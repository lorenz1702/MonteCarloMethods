import numpy as np
# Step 1: Choose the derised probability distribution
prop = np.array([0.5, 0.2, 0.05, 0.05, 0.2])
print("Derised probability distribution:")
print(prop)
N = len(prop)

# Step 2: function to calculate the Acceptance Matrix
def acceptance(prop):
    N = len(prop)
    p_acc = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if (prop[i] == 0):
                p_acc[i, j] = 1
            else:
                p_acc[i, j] = min(1, (prop[j] / prop[i]))
    return p_acc


# Step 3: Calculate the transition out of the Acceptance Matrix
def transition_matrix(prop, p_acc):
    N = len(prop)
    T = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            if i != j:
                #T(i,j) = g(i,j)A(i,j), i=!j
                #g(i->j)= 1/N
                 T[i, j] = 1/N * p_acc[i, j]
            else:
                sum = 0

                for k in range(N):
                    sum += 1/N * (1-p_acc[i,k])
                # T(i,j) = g(i,j)A(i,j)+sum(g(i,k)*(1-A(i,k))
                T[i, j] = 1/N * p_acc[i, j] + sum
    return T

def check_correctness(start_state, transition_matrix, num_chains):
    for i in range(num_chains):
            next_prob = start_state.dot(transition_matrix)
            start_state = next_prob
    return start_state


p_acc_matrix = acceptance(prop)
print("Akzeptanzmatrix:")
print(p_acc_matrix)

T = transition_matrix(prop, p_acc_matrix)
print("Transtionmatrix T:")
print(T)

start_state = np.array([1/2, 1/2, 0, 0, 0])
prop = check_correctness(start_state, T, 100)
print("Prop after using the computed transition matrix 100 times:")
print(prop)

