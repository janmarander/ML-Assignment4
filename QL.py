# I was not checking various sizes. They suggested that we do though, so assuming that that is valid, it could be
# an interesting comparison point for why QL is not really used

import numpy as np
import random as rand
import pandas as pd
import gym
from gym.envs.toy_text.frozen_lake import generate_random_map
from hiive.mdptoolbox.mdp import QLearning
import seaborn as sns
import matplotlib.pyplot as plt
import make_plots as myplots
from hiive.mdptoolbox.example import forest

    #   -Reward (Utility) vs epsilon (Q learning)
    #   -Reward (Utility) vs epsilon decay (Q learning)
    #   -Reward (Utility) vs alpha (Q learning)
    #   -Time vs epsilon (Q learning)
    #   -Time vs epsilon decay (Q learning)
    #   -Time vs alpha (Q learning)

def viz_optimal_policy_lake_QL(P,R, size,gamma = 0.99,alpha=0.5, epsilon_decay=.999, epsilon = 0.99):

    ql = QLearning(P, R, gamma, alpha=alpha, epsilon=epsilon, epsilon_decay=epsilon_decay,n_iter=5000000)
    ql.run()
    QL_stats = ql.run_stats
    #print(VI_stats)
    myplots.plot_run_stats("final QL", "Frozen Lake", QL_stats, alg='QL', figsize=(4, 4))

    optimal_policy = ql.policy
    print(optimal_policy)
    optimal_policy = np.array(list(ql.policy)).reshape(size,size).astype(str)
    optimalV = np.round(np.array(list(ql.V)).reshape(size,size), 2)

    optimal_policy[optimal_policy == '0'] = '⬅'
    optimal_policy[optimal_policy == '1'] = '⬇'
    optimal_policy[optimal_policy == '2'] = '➤'
    optimal_policy[optimal_policy == '3'] = '⬆'

    vizpolicy = (np.asarray([a + " " + str(v) for a, v in zip(optimal_policy.flatten(), optimalV.flatten())])).reshape(
        size,size)

    vizpolicy_arrows_only = (np.asarray([a  for a in zip(optimal_policy.flatten())])).reshape(size,size)

    plt.figure(figsize=(size, size))
    plt.title("Q Optimal Policy, Frozen Lake")
    sns.heatmap(optimalV, cmap="YlOrBr_r", annot=vizpolicy, fmt="")
    plt.show()

    plt.figure(figsize=(size, size))
    plt.title("Q Optimal Policy, Frozen Lake")
    sns.heatmap(optimalV, cmap="YlOrBr_r", annot=vizpolicy_arrows_only, annot_kws={"size":14}, fmt="")
    plt.show()
    return

def viz_optimal_policy_forest_QL(P,R,size,gamma = 0.99,alpha=0.5, epsilon_decay=.999, epsilon = 0.0000001):

    ql = QLearning(P, R, gamma=gamma, alpha=alpha, epsilon=epsilon, epsilon_decay=epsilon_decay, n_iter=2000000)
    # ql.setVerbose() #Convergence: difference(delta) in utility between subsequent iterations
    ql.run()

    optimal_policy = np.array(list(ql.policy)).reshape(size).astype(str)
    optimalV = np.round(np.array(list(ql.V)).reshape(size), 2)

    optimal_policy[optimal_policy == '0'] = 'W'
    optimal_policy[optimal_policy == '1'] = 'C'
    vizpolicy = (np.asarray([a + " " + str(v) for a, v in zip(optimal_policy.flatten(), optimalV.flatten())])).reshape(size)

    print("Optimal Policy QL")
    row = ""
    for val in vizpolicy:
        row += (" {}, ".format(val))
    print(row)



def run_QL(exp, mdp_name, P, R, gammas, alphas, epsilons, epsilon_decay_rates, size, num_games=100):
    print("QLearning")
    gamma0 = 0.99
    alpha0 = 0.1
    epsilon0 = 0.01 #99
    epsilon_decay0 = 0.99

    #
    # ql = QLearning(P, R, gamma0, alpha=0.1, epsilon=1.0, epsilon_decay=0.99, n_iter=10000)
    # #ql.setVerbose()  # Convergence: difference(delta) in utility between subsequent iterations
    # ql.run()
    # # run_stats.append(self._build_run_stat(i=self.iter, s=None, a=None, r=_np.max(policy_V),p=policy_next, v=policy_V, error=error))
    # Q_stats = ql.run_stats
    # myplots.plot_run_stats(exp, mdp_name, Q_stats, alg = 'QL', figsize=(4,4))

    #gammas
    # gamma is the discount factor. It quantifies how much importance we give for future rewards. It's also handy to
    # approximate the noise in future rewards. Gamma varies from 0 to 1. If Gamma is closer to zero, the agent will
    # tend to consider only immediate rewards.
    # mdptoolbox.mdp.QLearning(transitions, reward, discount, n_iter=10000, skip_check=False)

    Vmax_vs_gamma = []
    Time_vs_gamma = []
    print('gamma')
    for gamma in gammas:
        print(gamma)
        ql = QLearning(P, R, gamma = gamma, alpha = alpha0, epsilon = epsilon0, n_iter=2000000)
        #ql.setVerbose() #Convergence: difference(delta) in utility between subsequent iterations
        ql.run()
        time = ql.time
        maxV = np.amax(ql.V)
        #run_stats.append(self._build_run_stat(i=self.iter, s=None, a=None, r=_np.max(policy_V),p=policy_next, v=policy_V, error=error))
        Q_stats = ql.run_stats
        #print(ql.run_stats)
        Vmax_vs_gamma.append(maxV)
        Time_vs_gamma.append(time)
        myplots.plot_run_stats(exp + ' gamma='+str(gamma)+' ', mdp_name, Q_stats, alg = 'QL', figsize=(4, 4))
    myplots.plot_x_y(exp, mdp_name, gammas, Vmax_vs_gamma, 'Gamma', 'Reward')
    myplots.plot_x_y(exp, mdp_name, gammas, Time_vs_gamma, 'Gamma', 'Time')


    #alphas
    print("alpha")
    Vmax_vs_alpha = []
    Time_vs_alpha = []

    for alpha in alphas:
        print(alpha)
        ql = QLearning(P, R, gamma=gamma0, alpha = alpha, epsilon = epsilon0, epsilon_decay = epsilon_decay0, n_iter=2000000)
        #ql.setVerbose() #Convergence: difference(delta) in utility between subsequent iterations
        ql.run()
        time = ql.time
        maxV = np.amax(ql.V)
        #run_stats.append(self._build_run_stat(i=self.iter, s=None, a=None, r=_np.max(policy_V),p=policy_next, v=policy_V, error=error))
        Q_stats = ql.run_stats
        #print(ql.run_stats)
        Vmax_vs_alpha.append(maxV)
        Time_vs_alpha.append(time)
        myplots.plot_run_stats(exp + ' alpha='+str(alpha)+' ', mdp_name, Q_stats, alg = 'QL',figsize=(4, 4))
    myplots.plot_x_y(exp, mdp_name, alphas, Vmax_vs_alpha, 'Alpha', 'Reward')
    myplots.plot_x_y(exp, mdp_name, alphas, Time_vs_alpha, 'Alpha', 'Time')


    #epsilons
    print("epsilon")
    Vmax_vs_epsilon = []
    Time_vs_epsilon = []

    for epsilon in epsilons:
        print(epsilon)
        ql = QLearning(P, R, gamma=gamma0, alpha = alpha0, epsilon = epsilon, epsilon_decay = epsilon_decay0, n_iter=2000000)
        #ql.setVerbose() #Convergence: difference(delta) in utility between subsequent iterations
        ql.run()
        time = ql.time
        maxV = np.amax(ql.V)
        #run_stats.append(self._build_run_stat(i=self.iter, s=None, a=None, r=_np.max(policy_V),p=policy_next, v=policy_V, error=error))
        Q_stats = ql.run_stats
        #print(ql.run_stats)
        Vmax_vs_epsilon.append(maxV)
        Time_vs_epsilon.append(time)
        myplots.plot_run_stats(exp + ' epsilon='+str(epsilon)+' ', mdp_name, Q_stats, alg = 'QL',figsize=(4, 4))
    myplots.plot_x_y(exp, mdp_name, epsilons, Vmax_vs_epsilon, 'Epsilon', 'Reward')
    myplots.plot_x_y(exp, mdp_name, epsilons, Time_vs_epsilon, 'Epsilon', 'Time')



    #epsilon_decay_rates
    print("epsilon decay rate")
    Vmax_vs_epsilon_decay_rate = []
    Time_vs_epsilon_decay_rate = []

    for epsilon_decay in epsilon_decay_rates:
        print(epsilon_decay)
        ql = QLearning(P, R, gamma=gamma0, alpha = alpha0, epsilon = epsilon0, epsilon_decay = epsilon_decay, n_iter=2000000)
        #ql.setVerbose() #Convergence: difference(delta) in utility between subsequent iterations
        ql.run()
        time = ql.time
        maxV = np.amax(ql.V)
        #run_stats.append(self._build_run_stat(i=self.iter, s=None, a=None, r=_np.max(policy_V),p=policy_next, v=policy_V, error=error))
        Q_stats = ql.run_stats
        #print(ql.run_stats)
        Vmax_vs_epsilon_decay_rate.append(maxV)
        Time_vs_epsilon_decay_rate.append(time)
        myplots.plot_run_stats(exp + ' epsilon_decay=' + str(epsilon_decay) + ' ', mdp_name, Q_stats, alg = 'QL',figsize=(4, 4))
    myplots.plot_x_y(exp, mdp_name, epsilon_decay_rates, Vmax_vs_epsilon_decay_rate, 'Epsilon decay', 'Reward')
    myplots.plot_x_y(exp, mdp_name, epsilon_decay_rates, Time_vs_epsilon_decay_rate, 'Epsilon decay', 'Time')



    # , alpha_decay = 0.99, alpha_min = 0.001,
    # epsilon = 1.0, epsilon_min = 0.1, epsilon_decay = 0.99

if __name__ == '__main__':

    # forest_size = 10
    # P, R = forest(S=forest_size)
    # viz_optimal_policy_forest_QL(P, R, forest_size, gamma=0.99, epsilon=0.9)
    #
    # forest_size = 10000
    # P, R = forest(S=forest_size)
    # viz_optimal_policy_forest_QL(P, R, forest_size, gamma=0.99, epsilon=0.9)


    # create the environment
    size = 8
    p = 0.8
    np.random.seed(9)
    rand.seed(9)
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

    viz_optimal_policy_lake_QL(P_lake,R_lake, size,gamma = 0.5, alpha = 0.01, epsilon = 0.99, epsilon_decay=.999)

    # size = 25
    # p = 0.8
    # random_map = generate_random_map(size, p)
    # env = gym.make("FrozenLake-v0", desc=random_map)
    # env.reset()
    # env.render()
    # num_states = env.observation_space.n
    # num_actions = env.action_space.n
    # P_lake = np.zeros((num_actions, num_states, num_states))
    # R_lake = np.zeros((num_states, num_actions))
    # for S in env.env.P:
    #     for A in env.env.P[S]:
    #         for val in env.env.P[S][A]:
    #             P_lake[A][S][val[1]] += val[0]
    #             R_lake[S][A] += val[2]
    #
    # viz_optimal_policy_lake_QL(P_lake,R_lake, size, gamma = 0.5, alpha = 0.01, epsilon = 0.999, epsilon_decay=.999)
    #
    #
    # #alphas, epsilons, epsilon_decay_rates
    # gammas = np.arange(0.1, 0.99, 0.1)
    # #print(gammas)
    # alphas = np.arange(0.1, 0.99, 0.1)
    # #print(alphas)
    # epsilons = np.arange(0.1, 0.99, 0.1)
    # #print(epsilons)
    # epsilon_decay_rates = np.arange(0.1, 0.99, 0.1)
    # #print(epsilon_decay_rates)
    # #run_QL('q','lake', P_lake, R_lake, gammas, alphas, epsilons, epsilon_decay_rates, size, num_games=100)
    #


