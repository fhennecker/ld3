import numpy as np
import matplotlib.pyplot as plt
import sys

def reward(bandits, choice):
    return bandits[choice, 1] * np.random.randn() + bandits[choice, 0]

def q_values(reward_sum, times_played):
    results = np.copy(reward_sum)
    results[times_played != 0] /= times_played[times_played != 0]
    return results

def random_selection(q_values, args={}):
    return np.random.randint(0, len(q_values))

def e_greedy(q_values, args={'epsilon':0.1, 't':None}):
    epsilon = args['epsilon']
    try: epsilon = 1/np.sqrt(args['t'])
    except: pass

    if np.random.rand() < epsilon:
        return np.random.randint(0, len(q_values))
    return np.argmax(q_values)

def softmax(q_values, args={'tau':0.1, 't':None}):
    tau = args['tau']
    try: tau = 4. * 1.*(1000.-args['t']) / 1000.
    except: pass

    distribution = np.exp(q_values/tau) / np.sum(np.exp(q_values/tau))
    return np.random.choice(np.arange(len(q_values)), p=distribution)

def get_avg_reward(bandits, time_steps, iterations, methods):
    results = np.zeros((time_steps, len(methods)))
    for m, (func, args) in enumerate(methods):

        for i in range(iterations):
            if 't' in args:
                args = {'epsilon':0, 'tau':0, 't':i}

            reward_sum = np.zeros(len(bandits))
            times_played = np.zeros(len(bandits))

            a = random_selection(q_values(reward_sum, times_played))
            for t in range(time_steps):
                r = reward(bandits, a)
                times_played[a] += 1
                reward_sum[a] += r
                Q = q_values(reward_sum, times_played)

                # getting next action with generic action selection method
                a = func(Q, args=args)
                results[t, m] += r
            print '\r' + func.__name__, i,
            sys.stdout.flush()
        results[:, m] /= iterations
        print '\r' + func.__name__, 'done'
    return results

def run(exercise):
    bandits = np.array([
        [2.3, 0.9],
        [2.1, 0.6],
        [1.5, 0.4],
        [1.3, 2.0]
    ])
    if exercise == 2 : bandits[:,1] *= 2
    print bandits

    algos = [
        (random_selection, {}),
        (e_greedy, {'epsilon':0}),
        (e_greedy, {'epsilon':0.1}),
        (e_greedy, {'epsilon':0.2}),
        (softmax, {'tau':0.1}),
        (softmax, {'tau':1}),
    ]
    if exercise == 3:
        algos += [
            (e_greedy, {'t':None}),
            (softmax, {'t':None}),
        ]


    results = get_avg_reward(bandits, 500, 1000, algos)
    
    plt.plot(results)
    plt.legend(['Random', 
        '$\epsilon=0$', '$\epsilon=0.1$', '$\epsilon=0.2$',
        '$\tau=0.1$', '$\tau=1$'])
    plt.show()

if __name__ == "__main__":
    run(3)
