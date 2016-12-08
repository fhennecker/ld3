import numpy as np

def reward(grid, a, b):
    return grid[b,a,0] + np.random.randn() * grid[b,a,1]

def boltzmann_choice(q_values, tau, player):
    n_actions = q_values.shape[0]
    q_values = q_values.flatten()
    distrib = np.exp(q_values/tau)/np.sum(np.exp(q_values/tau))
    if player == 'row':
        return np.random.choice(range(q_values.size), p=distrib) / n_actions
    return np.random.choice(range(q_values.size), p=distrib) % n_actions


def q_values(total_rewards, total_plays):
    results = np.copy(total_rewards)
    results[total_plays != 0] /= total_plays[total_plays != 0]
    return results

def run(sigmas):
    sigma, sigma0, sigma1 = np.square(sigmas)
    grid = np.array([
        [[11, sigma0], [-30, sigma], [0, sigma]],
        [[-30, sigma], [7, sigma1], [6, sigma]],
        [[0, sigma], [0, sigma], [5, sigma]]
    ])

    total_rewards = np.zeros_like(grid)
    total_plays = np.zeros_like(grid)
