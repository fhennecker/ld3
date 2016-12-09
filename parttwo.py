import numpy as np
import matplotlib.pyplot as plt

def reward(grid, a, b):
    return grid[b,a,0] + np.random.randn() * grid[b,a,1]

def boltzmann_choice(total_rewards, total_plays, tau, player):
    if player not in ['row', 'col']: raise ValueError('Incorrect player type')

    Q = q_values(total_rewards, total_plays)
    P = np.ones(len(Q[0])) * 1./len(Q[0])
    if np.sum(total_plays) != 0:
        P = np.sum(total_plays, 0 if player == 'row' else 1) / np.sum(total_plays),
    if player == 'row': P = np.reshape(P, (1,-1))
    else:               P = np.reshape(P, (-1,1))
    EV = np.sum(Q * P, 1 if player == 'row' else 0)

    distribution = np.exp(EV/tau)/np.sum(np.exp(EV/tau))
    return np.random.choice(range(EV.size), p=distribution)


def q_values(total_rewards, total_plays):
    results = np.copy(total_rewards)
    results[total_plays != 0] /= total_plays[total_plays != 0]
    return results

def run(sigmas, tau=0.1):
    sigma, sigma0, sigma1 = np.square(sigmas)
    grid = np.array([
        [[11, sigma0], [-30, sigma], [0, sigma]],
        [[-30, sigma], [7, sigma1], [6, sigma]],
        [[0, sigma], [0, sigma], [5, sigma]]
    ])

    total_rewards = np.zeros((len(grid), len(grid)))
    total_plays = np.zeros((len(grid), len(grid)))
    rewards = np.zeros(5000)
    avg_rewards = np.zeros(5000)
    alpha = 0.1

    for t in range(5000):
        row_choice = boltzmann_choice(total_rewards, total_plays, tau, 'row')
        col_choice = boltzmann_choice(total_rewards, total_plays, tau, 'col')
        r = reward(grid, col_choice, row_choice)
        total_plays[row_choice, col_choice] += 1
        total_rewards[row_choice, col_choice] += r
        rewards[t] = r
        if t == 0:
            avg_rewards[0] = r
        else:
            avg_rewards[t] = (1-alpha) * avg_rewards[t-1] + alpha * r
    
    plt.plot(rewards)
    plt.plot(avg_rewards)
    plt.show()

if __name__ == '__main__':
    run([0.2, 0.2, 0.2], tau=0.1)
    #  run([0.1, 0.1, 4], tau=0.1)
    #  run([0.1, 4, 0.1], tau=0.1)
