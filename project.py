import numpy as np
import matplotlib.pyplot as plt
import sys

def reward(bandits, choice):
    return bandits[choice, 1] * np.random.randn() + bandits[choice, 0]

def q_values(reward_sum, times_played):
    results = np.copy(reward_sum)
    results[times_played != 0] /= times_played[times_played != 0]
    return results

def random_selection(q_values):
    return np.random.randint(0, len(q_values))

def e_greedy(q_values, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(0, len(q_values))
    return np.argmax(q_values)

def ex1():
    bandits = np.array([
        [2.3, 0.9],
        [2.1, 0.6],
        [1.5, 0.4],
        [1.3, 2.0]
    ])

    reward_sum = np.zeros(len(bandits))
    times_played = np.zeros(len(bandits))

    time_steps = 1000
    results = np.zeros(time_steps)

    a = random_selection(q_values(reward_sum, times_played))
    for t in range(time_steps):
        r = reward(bandits, a)
        times_played[a] += 1
        reward_sum[a] += r
        Q = q_values(reward_sum, times_played)
        a = random_selection(Q)
        results[t] = np.mean(Q)

        print '\r', t,
        sys.stdout.flush()
    
    plt.plot(results)
    plt.show()

if __name__ == "__main__":
    ex1()

        
