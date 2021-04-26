import numpy as np
import pandas as pd
import random as rand
import gym
from gym.envs.toy_text.frozen_lake import generate_random_map
from hiive.mdptoolbox.mdp import ValueIteration
from hiive.mdptoolbox.mdp import PolicyIterationModified
from hiive.mdptoolbox.mdp import PolicyIteration
import seaborn as sns
import matplotlib.pyplot as plt
import make_plots as myplots
from hiive.mdptoolbox.example import forest



def viz_optimal_policy_lake(P,R, size,gamma = 0.99,epsilon = 0.000001):

    vi = ValueIteration(P, R, gamma, epsilon=epsilon)
    vi.run()
    VI_stats = vi.run_stats
    print(VI_stats)
    myplots.plot_run_stats("final VI", "Frozen Lake", VI_stats, alg='VI', figsize=(4, 4))

    optimal_policy = vi.policy
    print(optimal_policy)
    optimal_policy = np.array(list(vi.policy)).reshape(size,size).astype(str)
    optimalV = np.round(np.array(list(vi.V)).reshape(size,size), 2)

    optimal_policy[optimal_policy == '0'] = '⬅'
    optimal_policy[optimal_policy == '1'] = '⬇'
    optimal_policy[optimal_policy == '2'] = '➤'
    optimal_policy[optimal_policy == '3'] = '⬆'

    vizpolicy = (np.asarray([a + " " + str(v) for a, v in zip(optimal_policy.flatten(), optimalV.flatten())])).reshape(
        size,size)

    vizpolicy_arrows_only = (np.asarray([a  for a in zip(optimal_policy.flatten())])).reshape(size,size)

    plt.figure(figsize=(size, size))
    plt.title("VI Optimal Policy, Frozen Lake")
    sns.heatmap(optimalV, cmap="Greens_r", annot=vizpolicy, fmt="")
    plt.show()

    plt.figure(figsize=(size, size))
    plt.title("VI Optimal Policy, Frozen Lake")
    sns.heatmap(optimalV, cmap="Greens_r", annot=vizpolicy_arrows_only, annot_kws={"size":14}, fmt="")
    plt.show()

    pi = PolicyIteration(P, R, gamma, max_iter=10000)  # PolicyIterationModified????
    # pi.setVerbose()
    pi.run()
    time = pi.time
    maxV = np.amax(pi.V)
    P_iters = pi.iter
    #PI_data.append([size, maxV, gamma, epsilon, time, P_iters])
    PI_stats = pi.run_stats
    #print(PI_stats)
    plt.close('all')
    myplots.plot_run_stats("final PI", "Frozen Lake", PI_stats, alg='PI', figsize=(4, 4))

    myplots.plot_run_stats_double("final", "Frozen Lake", VI_stats, PI_stats, alg1='VI',alg2='PI', figsize=(4, 4))

    optimal_policy = pi.policy
    print(optimal_policy)
    optimal_policy = np.array(list(pi.policy)).reshape(size, size).astype(str)
    optimalV = np.round(np.array(list(pi.V)).reshape(size, size), 2)

    optimal_policy[optimal_policy == '0'] = '⬅'
    optimal_policy[optimal_policy == '1'] = '⬇'
    optimal_policy[optimal_policy == '2'] = '➤'
    optimal_policy[optimal_policy == '3'] = '⬆'

    vizpolicy = (np.asarray([a + " " + str(v) for a, v in zip(optimal_policy.flatten(), optimalV.flatten())])).reshape(
        size, size)

    vizpolicy_arrows_only = (np.asarray([a  for a in zip(optimal_policy.flatten())])).reshape(size,size)


    plt.figure(figsize=(size, size))
    plt.title("PI Optimal Policy, Frozen Lake")
    sns.heatmap(optimalV, cmap="Blues_r", annot=vizpolicy, fmt="")
    plt.show()

    plt.figure(figsize=(size, size))
    plt.title("PI Optimal Policy, Frozen Lake")
    sns.heatmap(optimalV, cmap="Blues_r", annot=vizpolicy_arrows_only, annot_kws={"size":14}, fmt="")
    plt.show()

    return

def viz_optimal_policy_forest(P,R,size,gamma = 0.99,epsilon = 0.000001):
    #VI
    vi = ValueIteration(P, R, gamma, epsilon=epsilon)
    vi.run()
    VI_stats = vi.run_stats
    optimal_policy = np.array(list(vi.policy)).reshape(size).astype(str)
    optimalV = np.round(np.array(list(vi.V)).reshape(size), 2)

    optimal_policy[optimal_policy == '0'] = 'W'
    optimal_policy[optimal_policy == '1'] = 'C'
    vizpolicy = (np.asarray([a + " " + str(v) for a, v in zip(optimal_policy.flatten(), optimalV.flatten())])).reshape(size)

    print("Optimal Policy VI")
    row = ""
    for val in vizpolicy:
        row += (" {}, ".format(val))
    print(row)

    #PI
    pi = PolicyIteration(P, R, gamma)
    pi.run()
    PI_stats = pi.run_stats
    policyarr = np.array(list(pi.policy)).reshape(size).astype(str)
    valuearr = np.round(np.array(list(pi.V)).reshape(size), 2)
    policyarr[policyarr == '0'] = 'W'
    policyarr[policyarr == '1'] = 'C'
    policyviz = (np.asarray([a + " " + str(v) for a, v in zip(policyarr.flatten(), valuearr.flatten())])).reshape(size)

    print("")
    print("Policy Iteration: Optimal Policy")
    line = ""
    for val in policyviz:
        line += (" {}, ".format(val))
    print(line)

    myplots.plot_run_stats_double("final", "Forest ", VI_stats, PI_stats, alg1='VI', alg2='PI', figsize=(4, 4))


def run_VI_PI(exp, mdp_name, P, R, gammas, size, eps=[], num_games=100):
    print(exp)
    print(mdp_name)
    VI_data = []
    PI_data = []
    if exp=='gamma':
        epsilon = 9999
        i = 0
        for gamma in gammas:
            print(gamma)
            vi = ValueIteration(P, R, gamma, max_iter=10)
            # Convergence: difference(delta) in utility between subsequent iterations
            #vi.setVerbose()
            vi.run()

            time = vi.time
            maxV = np.amax(vi.V)
            V_iters = vi.iter
            VI_data.append([size, maxV, gamma, epsilon, time,  V_iters])
            VI_stats = vi.run_stats
            #print(VI_stats)
            myplots.plot_run_stats(exp, mdp_name, VI_stats, alg = 'VI', figsize=(4, 4))

            pi = PolicyIterationModified(P, R, gamma)  #PolicyIterationModified????
            #pi.setVerbose()
            pi.run()
            time = pi.time
            maxV = np.amax(pi.V)
            P_iters = pi.iter
            PI_data.append([size, maxV, gamma, epsilon, time, P_iters])
            PI_stats = pi.run_stats
            myplots.plot_run_stats(exp, mdp_name, PI_stats, alg = 'PI', figsize=(4, 4))

            i+=1

    if exp == 'epsilon':
        gamma = 0.99
        for epsilon in eps:
            print(epsilon)
            vi = ValueIteration(P, R, gamma, epsilon=epsilon, max_iter=10)
            vi.run()
            #vi.setVerbose()
            time = vi.time
            maxV = np.amax(vi.V)
            V_iters = vi.iter
            VI_data.append([size, maxV, gamma, epsilon, time, V_iters])

            pi = PolicyIterationModified(P, R, gamma, epsilon=epsilon)  # PolicyIterationModified????
            pi.run()
            time = pi.time
            maxV = np.amax(pi.V)
            P_iters = pi.iter
            PI_data.append([size, maxV, gamma, epsilon, time, P_iters])

    if exp == 'size':
        gamma = 0.99
        epsilon = 0.0001
        vi = ValueIteration(P, R, gamma, epsilon=epsilon, max_iter=10)
        vi.run()
        time = vi.time
        maxV = np.amax(vi.V)
        V_iters = vi.iter
        VI_data.append([size, maxV, gamma, epsilon, time, V_iters])

        pi = PolicyIterationModified(P, R, gamma, epsilon=epsilon)  # PolicyIterationModified????
        pi.run()
        time = pi.time
        maxV = np.amax(pi.V)
        P_iters = pi.iter
        PI_data.append([size, maxV, gamma, epsilon, time, P_iters])

        return VI_data, PI_data

    df_VI = pd.DataFrame(VI_data, columns=['Size', 'MaxV', 'Gamma', 'Epsilon', 'Time', 'Iterations'])
    df_PI = pd.DataFrame(PI_data, columns=['Size', 'MaxV', 'Gamma', 'Epsilon', 'Time', 'Iterations'])

    myplots.plot_vi_pi(exp, mdp_name, df_VI, df_PI)

    return df_VI, df_PI


if __name__ == '__main__':
    # create the environment
    np.random.seed(42)
    rand.seed(42)

    print("START")

    # forest_size = 10
    # P, R = forest(S=forest_size)
    # viz_optimal_policy_forest(P, R, forest_size, gamma=0.99, epsilon=0.000001)
    #
    # forest_size = 10000
    # P, R = forest(S=forest_size)
    # viz_optimal_policy_forest(P, R, forest_size, gamma=0.99, epsilon=0.000001)
    #
    #

    np.random.seed(9)
    rand.seed(9)

    size = 8
    p = 0.8
    random_map = generate_random_map(size, p)
    #
    env = gym.make("FrozenLake-v0", desc=random_map)
    env.seed(42)
    env.reset()
    env.render()

    num_states = env.observation_space.n
    num_actions = env.action_space.n

    P_lake = np.zeros((num_actions, num_states, num_states))
    R_lake = np.zeros((num_states, num_actions))

    for S in env.env.P:
        for A in env.env.P[S]:
            for val in env.env.P[S][A]:
                P_lake[A][S][val[1]] += val[0]
                R_lake[S][A] += val[2]


    gammas = np.arange(0.1, 0.99, 0.04)
    print(gammas)

    viz_optimal_policy_lake(P_lake, R_lake, size, gamma=0.99, epsilon=0.000001)

    np.random.seed(42)
    rand.seed(42)
    size = 25
    p = 0.8

    random_map = generate_random_map(size, p)

    env = gym.make("FrozenLake-v0", desc=random_map)
    env.seed(42)
    env.reset()
    env.render()

    num_states = env.observation_space.n
    num_actions = env.action_space.n

    P_lake = np.zeros((num_actions, num_states, num_states))
    R_lake = np.zeros((num_states, num_actions))

    for S in env.env.P:
        for A in env.env.P[S]:
            for val in env.env.P[S][A]:
                P_lake[A][S][val[1]] += val[0]
                R_lake[S][A] += val[2]

    gammas = np.arange(0.1, 0.99, 0.04)
    print(gammas)

    viz_optimal_policy_lake(P_lake, R_lake, size, gamma=0.99, epsilon=0.000001)








    #run_VI_PI(P_lake, R_lake, gammas, size, eps=0.0001, num_games=100)


#Jake Eichinger Today at 11:10 AM
# Question...if the reward is no longer changing in value iteration, but delta value between iterations is still changing what does this signify.? The values are still changing but the reward is not. Does this mean we have already gotten to an optimal policy (reward no longer changing) ?
# 10 replies
#
# Taneem  6 hours ago
# is that delta a really small number ? Like 1e-20 type value?
#
# Taneem  6 hours ago
# then yes you have already converged
#
# Taneem  6 hours ago
# you can set that whatever value you want (I just threw 1e-20 as an example)
#
# Jake Eichinger  6 hours ago
# No it’s on the order of like .001. That’s why  getting confused lol
#
# adnaan ahmed  6 hours ago
# I'll assume few things. The MDP is non-stochastic.
# So when you say reward is not changing, the reward is not changing for all the states when the same action is taken and you receive the same next state?
#
# Jake Eichinger  6 hours ago
# The MDP I'm using is stochastic, when I say reward I mean max v value.
#
# adnaan ahmed  6 hours ago
# yeah that can happen right.. we are looking for a equillibrium point where all the utility values are not changing for all the states.
#
# adnaan ahmed  6 hours ago
# So when only looking at max v value, there are some informations missing, there can be a particular states whose values are changing but is not max.. but you are only looking  at max (edited)
#
# adnaan ahmed  6 hours ago
# In value iteration we are not looking for max utility value as it's a process to push all states have max utility value at equilibrium (edited)
#
# Jake Eichinger  6 hours ago
# Ahhh yes that makes sense then. Thanks!