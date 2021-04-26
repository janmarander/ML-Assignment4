import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import gym

def test_policy(env, policy, num_games=100):
    '''
    Run some test games
    '''
    tot_rew = 0
    state = env.reset()

    for _ in range(num_games):
        actionsDictInv = {}
        actionsDictInv[" L "] = 0
        actionsDictInv[" D "] = 1
        actionsDictInv[" R "] = 2
        actionsDictInv[" U "] = 3

        done = False
        while not done:
            #print(actionsDictInv[policy[state]])
            action = actionsDictInv[policy[state]]
            next_state, reward, done, _ = env.step(action)

            state = next_state
            tot_rew += reward
            if done:
                state = env.reset()

    print('Won %i of %i games!' % (tot_rew, num_games))

def action_epsilon_greedy(q, s, epsilon=0.05):
    if np.random.rand() > epsilon:
        return np.argmax(q[s])
    return np.random.randint(4)

def greedy_policy(q, s):
    return np.argmax(q[s])

def average_performance(env, policy_fct, q):
    acc_returns = 0.
    n = 500
    for i in range(n):
        done = False
        s = env.reset()
        while not done:
            a = policy_fct(q, s)
            s, reward, done, info = env.step(a)
            acc_returns += reward
    return acc_returns / n

def run_QL_small_lake(is_slippery):
    env = gym.make('FrozenLake-v0', is_slippery=is_slippery)
    print("Action space = ", env.action_space)
    print("Observation space = ", env.observation_space)

    actionsDict = {}
    actionsDict[0] = " L "
    actionsDict[1] = " D "
    actionsDict[2] = " R "
    actionsDict[3] = " U "

    actionsDictInv = {}
    actionsDictInv[" L "] = 0
    actionsDictInv[" D "] = 1
    actionsDictInv[" R "] = 2
    actionsDictInv[" U "] = 3

    # actionsDict = {}
    # actionsDict[0] = "<"
    # actionsDict[1] = "v"
    # actionsDict[2] = ">"
    # actionsDict[3] = "^"
    #
    # actionsDictInv = {}
    # actionsDictInv["<"] = 0
    # actionsDictInv["v"] = 1
    # actionsDictInv[">"] = 2
    # actionsDictInv["^"] = 3

    env.reset()
    env.render()
    #print(env.desc==b'H')

    optimalPolicy = ["R/D", " R ", " D ", " L ",
                     " D ", " - ", " D ", " - ",
                     " R ", "R/D", " D ", " - ",
                     " - ", " R ", " R ", " ! ", ]

    print("Optimal policy:")
    idxs = [0, 4, 8, 12]
    for idx in idxs:
        print(optimalPolicy[idx + 0], optimalPolicy[idx + 1],
              optimalPolicy[idx + 2], optimalPolicy[idx + 3])


    q = np.ones((16, 4))
    # Set q(terminal,*) equal to 0
    q[5, :] = 0.0
    q[7, :] = 0.0
    q[11, :] = 0.0
    q[12, :] = 0.0
    q[15, :] = 0.0

    #print(q)
    i = 0
    disp = np.ones([4,4])
    row = 0
    while row < 4:
        col = 0
        while col < 4:
            print(row,col)
            print(env.desc[col][row])
            #if q[env.desc==b'H'] = 0
            if env.desc[col][row] == b'H': disp[col][row] = 2
            if env.desc[col][row] == b'S': disp[col][row] = 4
            if env.desc[col][row] == b'G': disp[col][row] = 3
            col += 1
        row+=1


        # SFFF
        # FHFH
        # FFFH
        # HFFG

    import matplotlib.colors as c
    colors = {"white": 1, "blue": 2, "green": 3,  "orange": 4}
    l_colors = sorted(colors, key=colors.get)
    cMap = c.ListedColormap(l_colors)
    fig, ax = plt.subplots()
    ax.pcolor(disp[::-1], cmap=cMap, vmin=1, vmax=len(colors), edgecolor='black')
    ax.tick_params(axis='both',bottom=False,left=False,labelbottom=False, labelleft=False)

    # plt.axis('off') # if you don't want the axis
    plt.show()

    nb_episodes = 40000
    STEPS = 2000
    alpha = 0.02
    gamma = 0.9
    epsilon_expl = 0.2

    q_performance = np.ndarray(nb_episodes // STEPS)

    # Q-Learning: Off-policy TD control algorithm
    for i in range(nb_episodes):

        done = False
        s = env.reset()
        while not done:
            a = action_epsilon_greedy(q, s, epsilon=epsilon_expl)  # behaviour policy
            new_s, reward, done, info = env.step(a)
            a_max = np.argmax(q[new_s])  # estimation policy
            q[s, a] = q[s, a] + alpha * (reward + gamma * q[new_s, a_max] - q[s, a])
            s = new_s

        # for plotting the performance
        if i % STEPS == 0:
            q_performance[i // STEPS] = average_performance(env, greedy_policy, q)

    plt.plot(STEPS * np.arange(nb_episodes // STEPS), q_performance)
    plt.xlabel("Epochs")
    plt.ylabel("Average reward of an epoch")
    plt.title("Learning progress for Q-Learning")
    plt.show()

    greedyPolicyAvgPerf = average_performance(env, greedy_policy, q=q)
    print("Greedy policy Q-learning performance =", greedyPolicyAvgPerf)

    q = np.round(q, 3)
    print("(A,S) Value function =", q.shape)
    print("First row")
    print(q[0:4, :])
    print("Second row")
    print(q[4:8, :])
    print("Third row")
    print(q[8:12, :])
    print("Fourth row")
    print(q[12:16, :])

    res_policy = []
    i=0
    while i < 16:
        row = i%4
        col = int((i-row)/4)
        #print(actionsDict[np.argmax(q[i, :])])
        if env.desc[col][row] == b'G': res_policy.append("!")
        elif env.desc[col][row] == b'H': res_policy.append("-")
        else: res_policy.append(actionsDict[np.argmax(q[i, :])])
        i+=1

    # print('pol',res_policy)
    # print(np.reshape(res_policy, (4,4)))

    policyFound = [actionsDict[np.argmax(q[0, :])], actionsDict[np.argmax(q[1, :])], actionsDict[np.argmax(q[2, :])],
                   actionsDict[np.argmax(q[3, :])],
                   actionsDict[np.argmax(q[4, :])], " - ", actionsDict[np.argmax(q[6, :])], " - ",
                   actionsDict[np.argmax(q[8, :])], actionsDict[np.argmax(q[9, :])], actionsDict[np.argmax(q[10, :])],
                   " - ",
                   " - ", actionsDict[np.argmax(q[13, :])], actionsDict[np.argmax(q[14, :])], " ! "]

    print("Greedy policy found:")
    idxs = [0, 4, 8, 12]
    idxs = np.arange(0, 16, 4)
    for idx in idxs:
        print(policyFound[idx + 0], policyFound[idx + 1],
              policyFound[idx + 2], policyFound[idx + 3])

    print(" ")

    print("Optimal policy:")
    idxs = [0, 4, 8, 12]
    for idx in idxs:
        print(optimalPolicy[idx + 0], optimalPolicy[idx + 1],
              optimalPolicy[idx + 2], optimalPolicy[idx + 3])

    test_policy(env, policyFound, num_games=100)

    # Optimal policy (pi)
    # LEFT = 0, DOWN = 1, RIGHT = 2, UP = 3
    plt.figure(figsize=(5, 16))
    sns.heatmap(q,  cmap="YlGnBu", annot=True, cbar=False, square=True)
    plt.show()


if __name__ == '__main__':
    #run_QL_small_lake(is_slippery=False)
    run_QL_small_lake(is_slippery=True)
