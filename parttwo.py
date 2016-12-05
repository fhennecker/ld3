import numpy as np

def reward(grid, a, b):
    return np.random.randn(grid[b,a,0], grid[b,a,1])

def boltzmann(q_values, tau):
    q_values = q_values.flatten()
    distribution = np.exp(q_values/tau)/np.sum(np.exp(q_values/tau))
    return np.random.choice(range(q_values.size), p=distribution)

def run(sigmas):
    sigma, sigma0, sigma1 = np.square(sigmas)
    grid = np.array([
        [[11, sigma0], [-30, sigma], [0, sigma]],
        [[-30, sigma], [7, sigma1], [6, sigma]],
        [[0, sigma], [0, sigma], [5, sigma]]
    ])
