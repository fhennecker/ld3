import numpy as np
import matplotlib.pyplot as plt

def reward(grid, a, b):
    return grid[b,a,0] + np.random.randn() * grid[b,a,1]

def boltzmann(total_rewards, total_plays, tau, player):
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

def optimistic_boltzmann(total_rewards, total_plays, tau, player):
    Q = q_values(total_rewards, total_plays)
    maxQ = np.max(Q, 1 if player == 'row' else 0)
    distribution = np.exp(maxQ/tau)/np.sum(np.exp(maxQ/tau))
    return np.random.choice(range(maxQ.size), p=distribution)

def q_values(total_rewards, total_plays):
    results = np.copy(total_rewards)
    results[total_plays != 0] /= total_plays[total_plays != 0]
    return results

def run(sigmas, func, tau=0.1):
    sigma, sigma0, sigma1 = np.square(sigmas)
    grid = np.array([
        [[11, sigma0], [-30, sigma], [0, sigma]],
        [[-30, sigma], [7, sigma1], [6, sigma]],
        [[0, sigma], [0, sigma], [5, sigma]]
    ])

    total_rewards = np.zeros((len(grid), len(grid)))
    total_plays = np.zeros((len(grid), len(grid)))
    n_steps = 5000
    rewards = np.zeros(n_steps)
    avg_rewards = np.zeros(n_steps)
    alpha = 0.02
    probas = np.zeros((n_steps,3))

    for t in range(n_steps):
        row_choice = func(total_rewards, total_plays, tau, 'row')
        col_choice = func(total_rewards, total_plays, tau, 'col')
        r = reward(grid, col_choice, row_choice)
        total_plays[row_choice, col_choice] += 1
        total_rewards[row_choice, col_choice] += r
        rewards[t] = r
        if t == 0:
            avg_rewards[0] = r
        else:
            avg_rewards[t] = (1-alpha) * avg_rewards[t-1] + alpha * r
    
    return avg_rewards

if __name__ == '__main__':
    def beautify(funcname):
        return funcname.replace('_', ' ').title()
    params = [[0.2, 0.2, 0.2]]#, [0.1, 0.1, 0.4], [0.1, 4, 0.1]]
    #  params = [[0.2, 0.2, 0.2], [0.1, 0.1, 0.4], [0.1, 4, 0.1]]
    taus = [1, 0.1]
    funcs = [boltzmann, optimistic_boltzmann]
    for index, param in enumerate(params):
        results, legends = [], []
        plt.figure(index)
        for func in funcs:
            for tau in taus:
                results.append(run(param, func, tau=tau))
                legends.append(beautify(func.__name__)+' '+str(tau))
        for res, color in zip(results, ['magenta', 'red', 'cyan', 'blue']):
            plt.plot(res, color=color)
        plt.legend(legends, loc=4)
        plt.ylim([-10, 12]);
        plt.ylabel('Smoothed reward'); plt.xlabel('Time')
        plt.grid(True)
        plt.savefig('fig%d.pdf'%index)
        plt.show()
