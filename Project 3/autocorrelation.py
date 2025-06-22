import numpy as np

T = np.array([[1, 1, 0, 0, 1],
              [1, 1, 1, 0, 0],
              [0, 1, 1, 1, 0],
              [0, 0, 1, 1, 1],
              [1, 0, 0, 1, 1]])

T = 1/3 * T
print(T)

def generate_markov_chain(T, start_state, num_steps):
  num_states = T.shape[0]
  current_state = start_state
  chain = [current_state]

  for _ in range(num_steps):

    next_state = np.random.choice(num_states, p=T[current_state, :])
    chain.append(next_state)
    current_state = next_state

  return chain

def autocorrelation_error(y):
    N = len(y)
    y = np.array(y)
    mean_y = np.mean(y)
    sigma2_y = np.var(y)

    def autocorrelation(tau):
        """
        Calculate: (1 - t/N) * ⟨(y_i - ȳ)(y_{i+t} - ȳ)⟩
        """
        N = len(y)
        y_mean = np.mean(y)
        c_sum = 0
        c_zero_sum = 0
        for i in range(0, N-tau):
            c_sum += (y[i]-y_mean)* (y[i+tau]-y_mean)
            c_zero_sum += (y[i]-y_mean)* (y[i]-y_mean)
        c_sum /= c_zero_sum
        mean_c_sum = c_sum/(N-tau)

        weight = 1 - ((t*1.0) / N)

        return (weight * mean_c_sum)

    tau_int = 0.5
    for t in range(1, N):
        auto_corr = autocorrelation(t)
        if (auto_corr < 0) :
            print(f"The stop point of integrand is t = {t}")
            break

        tau_int += auto_corr


    sigma2_y_error = sigma2_y * np.sqrt(2 * tau_int) * (1/ np.sqrt(N))
    return sigma2_y_error

start_state = 0
num_steps = 10000


markov_chain = generate_markov_chain(T, start_state, num_steps)
print("\nGenerierte Markov-Kette:")
print(markov_chain)

error = autocorrelation_error(markov_chain)
print(f"Autokorrelationsfehler: {np.mean(markov_chain)}+-{error}")

