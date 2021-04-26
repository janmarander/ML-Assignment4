# Assignment 4
#
# 1.	Come up with two interesting MDPs. Explain why they are interesting. They don't need to be overly complicated
# or directly grounded in a real situation, but it will be worthwhile if your MDPs are inspired by some process you are
# interested in or are familiar with. It's ok to keep it somewhat simple. For the purposes of this assignment, though,
# make sure one has a "small" number of states, and the other has a "large" number of states. I'm not going to go into
# detail about what large is, but 200 is not large. Furthermore, because I like variety no more than one of the MDPs
# you choose should be a so-called grid world problem.
# 2.	Solve each MDP using value iteration as well as policy iteration. How many iterations does it take to converge?
# Which one converges faster? Why? How did you choose to define convergence? Do they converge to the same answer? How
# did the number of states affect things, if at all?
# 3.	Now pick your favorite reinforcement learning algorithm and use it to solve the two MDPs. How does it perform,
# especially in comparison to the cases above where you knew the model, rewards, and so on? What exploration strategies
# did you choose? Did some work better than others?

import math
import random as rand
import gym
import mdptoolbox
import sys
import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import make_plots as myplots
import plotly
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from datetime import datetime
from collections import deque
from gym.envs.toy_text.frozen_lake import generate_random_map
from hiive.mdptoolbox.example import forest

import VI_PI as vipi
import QL as ql
# import QL_lake as ql_ls
# import QL_Lake2 as ql_ll

def make_lake(size = 4, p = 0.8):
    random_map = generate_random_map(size, p)

    env = gym.make("FrozenLake-v0", desc=random_map, is_slippery = True)
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

    return P_lake, R_lake

if __name__ == '__main__':
    np.random.seed(42)
    rand.seed(42)

    print('Assignment 4')
    stdoutOrigin = sys.stdout
    print(stdoutOrigin)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    sys.stdout = open("log" + timestr + ".txt", "w")
    print(timestr)

    print('Assignment 4')
    # 2.	Solve each MDP using value iteration as well as policy iteration.
    # How many iterations does it take to converge?

    sys.stdout.close()
    sys.stdout = stdoutOrigin
    # stdoutOrigin = sys.stdout
    # sys.stdout = open("Forest" + timestr + ".txt", "w")

    plt.close('all')
    gammas = np.arange(0.1, 0.99, 0.1)

    # FOREST
    mdp_name = "Forest"
    print(mdp_name)
    forest_size = 10
    P, R = forest(S=forest_size)
    vipi.run_VI_PI('gamma', mdp_name, P, R, gammas, size=forest_size, eps=0.0001, num_games=100)

    eps = np.arange(0.001, 0.05, 0.004)
    print(eps)
    vipi.run_VI_PI('epsilon', mdp_name, P, R, gammas, size=forest_size, eps=eps, num_games=100)
    vipi.viz_optimal_policy_forest(P, R, forest_size,gamma = 0.99,epsilon = 0.000001)

    forest_size = 10000
    P, R = forest(S=forest_size)
    vipi.run_VI_PI('gamma', mdp_name, P, R, gammas, size=forest_size, eps=0.0001, num_games=100)

    eps = np.arange(0.001, 0.05, 0.004)
    print(eps)
    vipi.run_VI_PI('epsilon', mdp_name, P, R, gammas, size=forest_size, eps=eps, num_games=100)
    vipi.viz_optimal_policy_forest(P, R, forest_size,gamma = 0.99,epsilon = 0.000001)


    gammas = np.arange(0.1, 0.99, 0.1)
    # print(gammas)
    alphas = np.arange(0.1, 0.99, 0.1)
    # print(alphas)
    epsilons = np.arange(0.1, 0.99, 0.1)

    print("Sizes")
    forest_sizes = [10, 100, 1000, 10000]

    VI_data = []
    PI_data = []
    for forest_size in forest_sizes:
        print(forest_size)
        Pf, Rf = forest(S=forest_size)
        VIres, PIres = vipi.run_VI_PI('size', mdp_name, Pf, Rf, gammas, forest_size, eps=epsilons, num_games=100)
        print(VIres[0])
        VI_data.append(VIres[0])
        PI_data.append(PIres[0])

    df_VI = pd.DataFrame(VI_data, columns=['Size', 'MaxV', 'Gamma', 'Epsilon', 'Time', 'Iterations'])
    df_PI = pd.DataFrame(PI_data, columns=['Size', 'MaxV', 'Gamma', 'Epsilon', 'Time', 'Iterations'])

    myplots.plot_vi_pi('size', mdp_name, df_VI, df_PI)


    sys.stdout.close()
    sys.stdout = stdoutOrigin
    stdoutOrigin = sys.stdout
    sys.stdout = open("Lake"+timestr+".txt", "w")

    plt.close('all')

    # LAKE
    print("LAKE")
    mdp_name = "Frozen Lake"

    lake_size = 8

    P_lake, R_lake = make_lake(size=lake_size, p=0.8)
    gammas = np.arange(0.1, .99, 0.03)
    print(gammas)
    vipi.run_VI_PI('gamma', mdp_name, P_lake, R_lake, gammas, lake_size, eps=0.0001, num_games=100)

    eps = np.arange(0.001,0.05, 0.004)
    print(eps)
    vipi.run_VI_PI('epsilon', mdp_name, P_lake, R_lake, gammas, lake_size, eps=eps, num_games=100)


    lake_size = 25

    P_lake, R_lake = make_lake(size=lake_size, p=0.8)
    gammas = np.arange(0.1, .99, 0.03)
    print(gammas)
    vipi.run_VI_PI('gamma', mdp_name, P_lake, R_lake, gammas, lake_size, eps=0.0001, num_games=100)

    eps = np.arange(0.001,0.05, 0.004)
    print(eps)
    vipi.run_VI_PI('epsilon', mdp_name, P_lake, R_lake, gammas, lake_size, eps=eps, num_games=100)

    # **Which one converges faster? Why? How did you choose to define convergence? Do they converge to the same answer?**
    # HOW DID YOU CHOOSE TO DEFINE CONVERGENCE???!?!?!??


    # **How did the number of states affect things, if at all?**
    # *** EXPERIMENT WITH DIFFERENT SIZES *** 4, 8, 16, 32??? 5, 25?
    print("Sizes")
    lake_sizes = [4,8,16,25]

    VI_data = []
    PI_data = []
    for lake_size in lake_sizes:
        print(lake_size)
        Pl, Rl = make_lake(size=lake_size, p=0.8)
        VIres, PIres = vipi.run_VI_PI('size', mdp_name, Pl, Rl, gammas, lake_size, eps=eps, num_games=100)
        print(VIres[0])
        VI_data.append(VIres[0])
        PI_data.append(PIres[0])

    df_VI = pd.DataFrame(VI_data, columns=['Size', 'MaxV', 'Gamma', 'Epsilon', 'Time', 'Iterations'])
    df_PI = pd.DataFrame(PI_data, columns=['Size', 'MaxV', 'Gamma', 'Epsilon', 'Time', 'Iterations'])

    myplots.plot_vi_pi('size', mdp_name, df_VI, df_PI)




# # -------------------------------------------------------------------------------------------------------------------
#     # 3.	Now pick your favorite reinforcement learning algorithm and use it to solve the two MDPs.
#     # How does it perform, especially in comparison to the cases above where you knew the model, rewards, and so on?
#
#     # What exploration strategies did you choose? Did some work better than others?
#
    # alphas, epsilons, epsilon_decay_rates
    gammas = np.arange(0.19, 1.0, 0.2)
    # print(gammas)
    alphas = np.arange(0.19, 1.0, 0.2)
    # print(alphas)
    epsilons = [0.2, 0.5, 0.8, 0.9, 0.99]
    # print(epsilons)
    epsilon_decay_rates = np.arange(0.19, 1.0, 0.2)
    # print(epsilon_decay_rates)

    # FOREST
    sys.stdout.close()
    sys.stdout = stdoutOrigin
    stdoutOrigin = sys.stdout
    sys.stdout = open("Q_Forest"+timestr+".txt", "w")
    plt.close('all')


    mdp_name = "Small Forest"
    print(mdp_name)
    forest_size = 10
    P, R = forest(S=forest_size)
    ql.run_QL('q', mdp_name, P, R, gammas, alphas, epsilons, epsilon_decay_rates, forest_size,num_games=100)

    ql.viz_optimal_policy_forest_QL(P, R, forest_size, gamma=0.99, alpha=0.1, epsilon_decay=.99, epsilon=0.99)

    mdp_name = "Large Forest"
    print(mdp_name)
    forest_size = 1000
    P, R = forest(S=forest_size)
    ql.run_QL('q', mdp_name, P, R, gammas, alphas, epsilons, epsilon_decay_rates, forest_size, num_games=100)

    ql.viz_optimal_policy_forest_QL(P, R, forest_size, gamma=0.99, alpha=0.1, epsilon_decay=.99, epsilon=0.99)

    # FROZEN LAKE
    sys.stdout.close()
    sys.stdout = stdoutOrigin
    stdoutOrigin = sys.stdout
    sys.stdout = open("Q_Lake"+timestr+".txt", "w")

    plt.close('all')

    # ql_ls
    #
    #
    # ql_ll

    # class Foo(object):
    #     def iter_callback(self, s, a, s_new):
    #         trans_prob = self.P[0][s, s_new] + self.P[1][s, s_new] + self.P[2][s, s_new] + self.P[3][s, s_new]
    #         if trans_prob > 0: return False
    #         else: return True

    #ql.viz_optimal_policy_forest_QL(P_lake, R_lake, forest_size, gamma=0.99, alpha=0.5, epsilon_decay=.99, epsilon=0.001)

    lake_size = 8
    P_lake, R_lake = make_lake(size=lake_size, p=0.8)
    ql.run_QL('q', 'Small Lake', P_lake, R_lake, gammas, alphas, epsilons, epsilon_decay_rates, lake_size, num_games=100)

    ql.viz_optimal_policy_lake_QL(P_lake, R_lake, lake_size, gamma=0.99, alpha=0.1, epsilon_decay=.999, epsilon=0.99)

    lake_size = 25
    P_lake, R_lake = make_lake(size=lake_size, p=0.8)
    ql.run_QL('q', 'Large Lake', P_lake, R_lake, gammas, alphas, epsilons, epsilon_decay_rates, lake_size, num_games=100)

    ql.viz_optimal_policy_lake_QL(P_lake, R_lake, lake_size, gamma=0.99, alpha=0.1, epsilon_decay=.999, epsilon=0.99)

    sys.stdout.close()
    sys.stdout = stdoutOrigin


