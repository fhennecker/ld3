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
    Qai = np.zeros((time_steps, len(methods), len(bandits)))
    times_selected = np.zeros((len(methods), len(bandits)))
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
                Qai[t, m] += Q

                # getting next action with generic action selection method
                a = func(Q, args=args)
                results[t, m] += r

            times_selected[m] += times_played

            print '\r' + func.__name__, i,
            sys.stdout.flush()

        times_selected[m,:] /= iterations
        results[:, m] /= iterations
        Qai[:, m, :] /= iterations
        print '\r' + func.__name__, 'done'
    return results, Qai, times_selected

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
        (softmax, {'tau':1}),
        (softmax, {'tau':0.1}),
    ]
    if exercise == 3:
        algos += [
            (e_greedy, {'t':None}),
            (softmax, {'t':None}),
        ]


    results, Qai, times_selected = get_avg_reward(bandits, 1000, 3000, algos)
    

    plt.figure(figsize=(10, 6))
    plt.plot(results)
    legend = ['Random', 
        '$\epsilon$-greedy ($\epsilon=0$)', '$\epsilon$-greedy ($\epsilon=0.1$)', 
        '$\epsilon$-greedy ($\epsilon=0.2$)',
        'Softmax ($\\tau=1$)', 'Softmax ($\\tau=0.1$)'] 
    if exercise == 3:
        legend += ['$\epsilon$-greedy ($\epsilon(t)=1/\sqrt{t}$)', 
            'Softmax ($\\tau(t)=4*(1000-t)/1000)$']
    plt.legend(legend, loc=4, prop={'size':9})
    plt.grid(True)
    plt.xlabel('Time'); plt.ylabel('Average reward')
    plt.savefig('fig/ex1-%d.pdf'%exercise)
    plt.clf()
    #  plt.show()

    legend.append('$Q^*_{ai}$')
    for arm in range(len(bandits)):
        plt.plot(Qai[:,:,arm])
        plt.grid(True)
        plt.xlabel('Time'); plt.ylabel('$Q_{ai}$')
        plt.ylim([-0.5, 2.5])
        plt.plot([0, len(Qai)], [bandits[arm,0], bandits[arm,0]], linewidth=2)
        plt.legend(legend, loc=4, prop={'size':9})
        plt.savefig('fig/ex1-%d-q%d.pdf'%(exercise, arm))
        plt.clf()
        #  plt.show()

    for algo in range(len(algos)):
        plt.figure(figsize=(5,5))
        plt.bar(range(len(bandits)), times_selected[algo,:],
                tick_label=range(len(bandits)), align='center')
        plt.xlabel('Arm'); plt.ylabel('Times selected')
        plt.savefig('fig/ex1-%d-a%d.pdf'%(exercise, algo))
        plt.clf()
        #  plt.show()

if __name__ == "__main__":
    run(1)
    run(2)
    run(3)
