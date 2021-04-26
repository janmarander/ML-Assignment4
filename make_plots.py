# These are the plots believed to be necessary so far.
    #
    # # reward vs. iteration
    # # delta convergence (not sure if different from above)
    # # some measure of examining the effect of problem size (n_convergence vs problem size, maybe)
    #
    # RL Model Optimization (small and large environment):
    #       Note: MaxV == Reward
    #   -Reward (Utility) vs iterations (STILL NEED FOR Q) x
    #   -Reward (Utility) vs discount/gamma (PI, VI) x
    #   -Reward (Utility) vs epsilon (Q learning)
    #   -Reward (Utility) vs epsilon decay (Q learning)
    #   -Reward (Utility) vs alpha (Q learning)
    #   -Time vs iterations (STILL NEED FOR Q) x
    #   -Time vs discount/gamma (PI, VI) x
    #   -Time vs epsilon (Q learning)
    #   -Time vs epsilon decay (Q learning)
    #   -Time vs alpha (Q learning)
    #   -****Delta convergence (NEED FOR EVERYTHING)****


    #
    # Environment Analysis:
    #   -Original Environment view
    #   -Environment view with optimal policy found
    #   X?-Relative reward vs environment complexity (PI ,VI, Q learning)
    #   X?-Time to solution vs environment complexity (PI, VI, Q learning)
    #
    # Policy:
    #   -Output
    #   -Discussion:
    #       *Difference between sizes
    #       *Difference between problems
    #
    #
    #
    # https://omscs-study.slack.com/archives/C08LK14DV/p1618438327126500
    # John Miller
    # This is my best take on these. The third one I'm still trying to understand myself:
    #
    # **Original Environment View: We need to visualize the environment somehow. For gridworld problems this is easy,
    # just show the grid space somehow. For your non-gridworld problem its a bit harder but you need to show what your
    # problem "looks" like and how transitions work
    #
    # **Environment view w/ optimal policy: Show how your policy looks in the state space, for example, arrows showing the
    # optimal direction to take for gridworld
    #
    # **Relative reward vs environment complexity: This one is the one I'm the fuzziest on, I think this means the
    # expected reward of your policy as a function of how big your state space is. So you would need to train your MDP
    # on a specific problem size, simulate the problem to get the average reward, then repeat the process with larger
    # and larger problems? The "Relative" in relative reward is important because maybe as your state space grows, the
    # possible reward gets bigger. So you need to scale your reward as a percentage of the max possible reward to show
    # how that percentage changes across problem sizes.
    #
    # **Time to Solution vs environment complexity: This one is much easier, show how long it takes to converge as a
    # function of the problem size
    #
    # Taneem
    # @John Miller My understanding of the original environment view is the env.render() thing in openAI gym for
    # the grid-world


import matplotlib.pyplot as plt
import numpy as np
import random as rand
import seaborn as sns
import time
import csv

def plot_x_y(exp, name, x, y, xname, yname, figsize=(4,4)):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    f, (ax1) = plt.subplots(nrows=1, ncols=1,figsize=figsize)
    sns.lineplot(x=x, y=y, linestyle='-', label=exp, color='green')
    #sns.lineplot(x=df_PI['Size'], y=df_PI['Iterations'], linestyle='-', label="Policy Iteration",color='blue')
    ax1.set_title(name+' '+xname+' vs '+ yname)
    ax1.minorticks_on()
    ax1.grid(b=True, which='major', linestyle='-', alpha=0.2)
    ax1.grid(b=True, which='minor', linestyle='-', alpha=0.1)
    ax1.set_ylabel(yname)
    ax1.set_xlabel(xname)
    f.tight_layout()
    plt.savefig(exp + ' ' + name + ' ' + yname+' vs '+ xname + timestr + '.png')
    #plt.show()
    plt.close()

    with open(exp + ' ' + name + ' ' + yname+' vs '+ xname + timestr + '.csv', mode='w') as save_file:
        file_writer = csv.writer(save_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        file_writer.writerow(x)
        file_writer.writerow(y)

def plot_run_stats_double(exp, mdp_name, VI_stats, PI_stats, alg1='VI', alg2='PI', figsize=(4, 4)):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    print(np.shape(VI_stats))
    if np.shape(VI_stats):
        # print("HAS LENGTH")
        num_iters = np.shape(VI_stats)[0]
        print(num_iters)
        # collect data
        vTime = []
        vReward = []
        vError = []
        vAlpha = []
        vEpsilon = []
        vGamma = []
        vMax_V = []
        vMean_V = []
        vIteration = []

        for i in np.arange(num_iters):
            iter_data = VI_stats[i]
            vTime.append(iter_data['Time'])
            vReward.append(iter_data['Reward'])
            vError.append(iter_data['Error'])
            # Alpha.append(iter_data['Alpha'])
            # Epsilon.append(iter_data['Epsilon'])
            # Gamma.append(iter_data['Gamma'])
            vMax_V.append(iter_data['Max V'])
            vMean_V.append(iter_data['Mean V'])
            vIteration.append(iter_data['Iteration'])

        if np.shape(PI_stats):
            # print("HAS LENGTH")
            num_iters = np.shape(PI_stats)[0]
            print(num_iters)
            # collect data
            pTime = []
            pReward = []
            pError = []
            pAlpha = []
            pEpsilon = []
            pGamma = []
            pMax_V = []
            pMean_V = []
            pIteration = []

            for i in np.arange(num_iters):
                iter_data = PI_stats[i]
                pTime.append(iter_data['Time'])
                pReward.append(iter_data['Reward'])
                pError.append(iter_data['Error'])
                # Alpha.append(iter_data['Alpha'])
                # Epsilon.append(iter_data['Epsilon'])
                # Gamma.append(iter_data['Gamma'])
                pMax_V.append(iter_data['Max V'])
                pMean_V.append(iter_data['Mean V'])
                pIteration.append(iter_data['Iteration'])


        #   -Reward (Utility) vs iterations
        f, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        sns.lineplot(x=vIteration, y=np.cumsum(vReward), linestyle='-', label='VI', color='green')
        sns.lineplot(x=pIteration, y=np.cumsum(pReward), linestyle='-', label='PI', color='blue')

        ax1.set_title(mdp_name + ' Cumulative Reward vs iteration')
        ax1.minorticks_on()
        ax1.grid(b=True, which='major', linestyle='-', alpha=0.2)
        ax1.grid(b=True, which='minor', linestyle='-', alpha=0.1)
        ax1.set_ylabel('Cumulative Reward')
        ax1.set_xlabel('iteration')
        f.tight_layout()
        plt.savefig('PIVI ' + exp + ' ' + mdp_name + ' Reward vs iteration' + timestr + '.png')
        # plt.show()
        plt.close()

        #   Time vs iterations
        f, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        sns.lineplot(x=vIteration, y=vTime, linestyle='-', label="VI", color='green')
        sns.lineplot(x=pIteration, y=pTime, linestyle='-', label="PI", color='blue')
        ax1.set_title(mdp_name + ' Time vs iterations')
        ax1.minorticks_on()
        ax1.grid(b=True, which='major', linestyle='-', alpha=0.2)
        ax1.grid(b=True, which='minor', linestyle='-', alpha=0.1)
        ax1.set_ylabel('Time')
        ax1.set_xlabel('iterations')
        f.tight_layout()
        plt.savefig('PIVI' + exp + ' ' + mdp_name + ' Time vs iterations' + timestr + '.png')
        # plt.show()
        plt.close()

        #  Delta convergence
        f, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        sns.lineplot(x=vIteration, y=vError, linestyle='-', label="VI", color='green')
        sns.lineplot(x=pIteration, y=pError, linestyle='-', label="PI", color='blue')
        ax1.set_title(mdp_name + ' Error vs iterations')
        ax1.minorticks_on()
        ax1.grid(b=True, which='major', linestyle='-', alpha=0.2)
        ax1.grid(b=True, which='minor', linestyle='-', alpha=0.1)
        ax1.set_ylabel('Error')
        ax1.set_xlabel('iterations')
        f.tight_layout()
        plt.savefig('PIVI' + exp + ' ' + mdp_name + ' Error vs iterations' + timestr + '.png')
        # plt.show()
        plt.close()

    return


def plot_run_stats_final_reward(exp, name, run_stats, alg='nd', figsize=(4, 4)):
    if alg == "VI": color = 'Green'
    if alg == "PI": color = 'Blue'
    if alg == "QL": color = 'Orange'

    timestr = time.strftime("%Y%m%d-%H%M%S")
    print(np.shape(run_stats))
    if np.shape(run_stats):
        # print("HAS LENGTH")
        num_iters = np.shape(run_stats)[0]
        print(num_iters)
        # collect data
        Time = []
        Reward = []
        Error = []
        Alpha = []
        Epsilon = []
        Gamma = []
        Max_V = []
        Mean_V = []
        Iteration = []

        for i in np.arange(num_iters):
            iter_data = run_stats[i]
            Time.append(iter_data['Time'])
            Reward.append(iter_data['Reward'])
            Error.append(iter_data['Error'])
            # Alpha.append(iter_data['Alpha'])
            # Epsilon.append(iter_data['Epsilon'])
            # Gamma.append(iter_data['Gamma'])
            Max_V.append(iter_data['Max V'])
            Mean_V.append(iter_data['Mean V'])
            Iteration.append(iter_data['Iteration'])

        final_reward = np.sum(Reward)





def plot_run_stats(exp, name, run_stats, alg = 'nd', figsize=(4,4)):
    if alg == "VI": color = 'Green'
    if alg == "PI": color = 'Blue'
    if alg == "QL": color = 'Orange'

    timestr = time.strftime("%Y%m%d-%H%M%S")
    print(np.shape(run_stats))
    if np.shape(run_stats):
        #print("HAS LENGTH")
        num_iters = np.shape(run_stats)[0]
        print(num_iters)
        #collect data
        Time = []
        Reward = []
        Error = []
        Alpha = []
        Epsilon = []
        Gamma = []
        Max_V = []
        Mean_V = []
        Iteration = []

        for i in np.arange(num_iters):
            iter_data = run_stats[i]
            Time.append(iter_data['Time'])
            Reward.append(iter_data['Reward'])
            Error.append(iter_data['Error'])
            # Alpha.append(iter_data['Alpha'])
            # Epsilon.append(iter_data['Epsilon'])
            # Gamma.append(iter_data['Gamma'])
            Max_V.append(iter_data['Max V'])
            Mean_V.append(iter_data['Mean V'])
            Iteration.append(iter_data['Iteration'])

        final_reward = np.sum(Reward)

        #print(Reward)

        #   -Reward (Utility) vs iterations
        f, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        sns.lineplot(x=Iteration, y=np.cumsum(Reward), linestyle='-', label=exp, color='green')
        ax1.set_title(name + ' Cumulative Reward vs iteration')
        ax1.minorticks_on()
        ax1.grid(b=True, which='major', linestyle='-', alpha=0.2)
        ax1.grid(b=True, which='minor', linestyle='-', alpha=0.1)
        ax1.set_ylabel('Cumulative Reward')
        ax1.set_xlabel('iteration')
        f.tight_layout()
        plt.savefig(alg + ' ' + exp + ' ' + name + ' Reward vs iteration' + timestr + '.png')
        #plt.show()
        plt.close()

        with open(alg + ' ' + exp + ' ' + name + ' Reward vs iterations' + timestr + '.csv', mode='w') as save_file:
            file_writer = csv.writer(save_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            file_writer.writerow(Iteration)
            file_writer.writerow(Reward)

        #   Time vs iterations
        f, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        sns.lineplot(x=Iteration, y=Time, linestyle='-', label=exp, color='green')
        ax1.set_title(name + ' Time vs iterations')
        ax1.minorticks_on()
        ax1.grid(b=True, which='major', linestyle='-', alpha=0.2)
        ax1.grid(b=True, which='minor', linestyle='-', alpha=0.1)
        ax1.set_ylabel('Time')
        ax1.set_xlabel('iterations')
        f.tight_layout()
        plt.savefig(alg + ' ' + exp + ' ' + name + ' Time vs iterations' + timestr + '.png')
        #plt.show()
        plt.close()

        with open(alg + ' ' + exp + ' ' + name + ' Time vs iterations' + timestr + '.csv', mode='w') as save_file:
            file_writer = csv.writer(save_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            file_writer.writerow(Iteration)
            file_writer.writerow(Time)

        #  Delta convergence
        f, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        sns.lineplot(x=Iteration, y=Error, linestyle='-', label=exp, color='green')
        ax1.set_title(name + ' Error vs iterations')
        ax1.minorticks_on()
        ax1.grid(b=True, which='major', linestyle='-', alpha=0.2)
        ax1.grid(b=True, which='minor', linestyle='-', alpha=0.1)
        ax1.set_ylabel('Error')
        ax1.set_xlabel('iterations')
        f.tight_layout()
        plt.savefig(alg + ' ' + exp + ' ' + name + ' Error vs iterations' + timestr + '.png')
        #plt.show()
        plt.close()

        with open(alg + ' ' + exp + ' ' + name + ' Error vs iterations' + timestr + '.csv', mode='w') as save_file:
            file_writer = csv.writer(save_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            file_writer.writerow(Iteration)
            file_writer.writerow(Error)


def plot_vi_pi(exp, name, df_VI, df_PI, figsize=(4,4)):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    if exp == 'size':
        #print("SIZE PLOTS")
        f, (ax1) = plt.subplots(nrows=1, ncols=1,figsize=figsize)
        sns.lineplot(x=df_VI['Size'], y=df_VI['Iterations'], linestyle='-', label="Value Iteration", color='green')
        sns.lineplot(x=df_PI['Size'], y=df_PI['Iterations'], linestyle='-', label="Policy Iteration",color='blue')
        ax1.set_title(name+' Iterations vs Size')
        ax1.minorticks_on()
        ax1.grid(b=True, which='major', linestyle='-', alpha=0.2)
        ax1.grid(b=True, which='minor', linestyle='-', alpha=0.1)
        if name=='Forest': ax1.set_xscale('log')
        f.tight_layout()
        plt.savefig(exp + ' ' + name + ' Iterations vs Size' + timestr + '.png')
        #plt.show()
        plt.close()

        with open(exp + ' ' + name + ' Iterations vs Size' + timestr + '.csv', mode='w') as save_file:
            file_writer = csv.writer(save_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            file_writer.writerow(df_VI['Iterations'])
            file_writer.writerow(df_VI['Size'])
            file_writer.writerow(df_PI['Iterations'])
            file_writer.writerow(df_PI['Size'])

        # -Time vs discount/gamma (PI, VI)
        f, (ax2) = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        sns.lineplot(x=df_VI['Size'], y=df_VI['Time'], label="Value Iteration", ax=ax2, color='green')
        sns.lineplot(x=df_PI['Size'], y=df_PI['Time'], label="Policy Iteration", ax=ax2, color='blue')
        ax2.set_title(name+' Time vs Size')
        ax2.minorticks_on()
        ax2.grid(b=True, which='major', linestyle='-', alpha=0.2)
        ax2.grid(b=True, which='minor', linestyle='-', alpha=0.1)
        if name == 'Forest': ax2.set_xscale('log')
        f.tight_layout()
        plt.savefig(exp + ' ' + name + ' Time vs Size' + timestr + '.png')
        #plt.show()

        # -Reward (Utility) vs discount/gamma (PI, VI)
        f, (ax3) = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        sns.lineplot(x=df_VI['Size'], y=df_VI['MaxV'], label="Value Iteration", ax=ax3,color='green')
        sns.lineplot(x=df_PI['Size'], y=df_PI['MaxV'], label="Policy Iteration", ax=ax3,color='blue') #linestyle='--',
        ax3.set_title(name+' Reward (MaxV) vs Size')
        ax3.minorticks_on()
        ax3.grid(b=True, which='major', linestyle='-', alpha=0.2)
        ax3.grid(b=True, which='minor', linestyle='-', alpha=0.1)
        if name == 'Forest': ax3.set_xscale('log')
        f.tight_layout()
        plt.savefig(exp + ' ' + name + ' Reward (MaxV) vs Size' + timestr + '.png')
        #plt.show()

        return


    if exp == 'gamma':
        f, (ax1) = plt.subplots(nrows=1, ncols=1,figsize=figsize)
        sns.lineplot(x=df_VI['Gamma'], y=df_VI['Iterations'], linestyle='-', label="Value Iteration", color='green')
        sns.lineplot(x=df_PI['Gamma'], y=df_PI['Iterations'], linestyle='-', label="Policy Iteration",color='blue')
        ax1.set_title(name+' Iterations vs Gamma')
        ax1.minorticks_on()
        ax1.grid(b=True, which='major', linestyle='-', alpha=0.2)
        ax1.grid(b=True, which='minor', linestyle='-', alpha=0.1)
        f.tight_layout()
        #plt.show()

        # -Time vs discount/gamma (PI, VI)
        f, (ax2) = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        sns.lineplot(x=df_VI['Gamma'], y=df_VI['Time'], label="Value Iteration", ax=ax2, color='green')
        sns.lineplot(x=df_PI['Gamma'], y=df_PI['Time'], label="Policy Iteration", ax=ax2, color='blue')
        ax2.set_title(name+' Time vs Gamma')
        ax2.minorticks_on()
        ax2.grid(b=True, which='major', linestyle='-', alpha=0.2)
        ax2.grid(b=True, which='minor', linestyle='-', alpha=0.1)
        f.tight_layout()
        plt.savefig(exp + ' ' + name + ' Time vs Gamma' + timestr + '.png')
        #plt.show()

        # -Reward (Utility) vs discount/gamma (PI, VI)
        f, (ax3) = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        sns.lineplot(x=df_VI['Gamma'], y=df_VI['MaxV'], label="Value Iteration", ax=ax3,color='green')
        sns.lineplot(x=df_PI['Gamma'], y=df_PI['MaxV'], label="Policy Iteration", ax=ax3,color='blue') #linestyle='--',
        ax3.set_title(name+' Reward (MaxV) vs Gamma')
        ax3.minorticks_on()
        ax3.grid(b=True, which='major', linestyle='-', alpha=0.2)
        ax3.grid(b=True, which='minor', linestyle='-', alpha=0.1)
        f.tight_layout()
        plt.savefig(exp + ' ' + name + ' Reward (MaxV) vs Gamma' + timestr + '.png')
        #plt.show()

    if exp == 'epsilon':
        f, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        sns.lineplot(x=df_VI['Epsilon'], y=df_VI['Iterations'], linestyle='-', label="Value Iteration", color='green')
        sns.lineplot(x=df_PI['Epsilon'], y=df_PI['Iterations'], linestyle='-', label="Policy Iteration", color='blue')
        ax1.set_title(name + ' Iterations vs Epsilon')
        ax1.minorticks_on()
        ax1.grid(b=True, which='major', linestyle='-', alpha=0.2)
        ax1.grid(b=True, which='minor', linestyle='-', alpha=0.1)
        f.tight_layout()
        plt.savefig(exp + ' ' + name + ' Iterations vs Epsilon' + timestr + '.png')
        #plt.show()

        # -Time vs discount/gamma (PI, VI)
        f, (ax2) = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        sns.lineplot(x=df_VI['Epsilon'], y=df_VI['Time'], label="Value Iteration", ax=ax2, color='green')
        sns.lineplot(x=df_PI['Epsilon'], y=df_PI['Time'], label="Policy Iteration", ax=ax2, color='blue')
        ax2.set_title(name + ' Time vs Epsilon')
        ax2.minorticks_on()
        ax2.grid(b=True, which='major', linestyle='-', alpha=0.2)
        ax2.grid(b=True, which='minor', linestyle='-', alpha=0.1)
        f.tight_layout()
        plt.savefig(exp + ' ' + name + ' Time vs Epsilon' + timestr + '.png')
        #plt.show()

        # -Reward (Utility) vs discount/gamma (PI, VI)
        f, (ax3) = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        sns.lineplot(x=df_VI['Epsilon'], y=df_VI['MaxV'], label="Value Iteration", ax=ax3, color='green')
        sns.lineplot(x=df_PI['Epsilon'], y=df_PI['MaxV'], label="Policy Iteration", ax=ax3,
                     color='blue')  # linestyle='--',
        ax3.set_title(name + ' Reward (MaxV) vs Epsilon')
        ax3.minorticks_on()
        ax3.grid(b=True, which='major', linestyle='-', alpha=0.2)
        ax3.grid(b=True, which='minor', linestyle='-', alpha=0.1)
        f.tight_layout()
        plt.savefig(exp + ' ' + name + ' Reward (MaxV) vs Epsilon' + timestr + '.png')
        #plt.show()

    # -Reward vs iterations
    f, (ax4) = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    sns.lineplot(x=df_VI['Iterations'], y=df_VI['MaxV'], label="Value Iteration", ax=ax4,color='green')
    ax4.set_title(name+' VI Reward (MaxV) vs Iterations')
    ax4.minorticks_on()
    ax4.grid(b=True, which='major', linestyle='-', alpha=0.2)
    ax4.grid(b=True, which='minor', linestyle='-', alpha=0.1)
    f.tight_layout()
    plt.savefig(exp + ' ' + name + ' VI Reward (MaxV) vs Iterations' + timestr + '.png')
    #plt.show()

    f, (ax5) = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    sns.lineplot(x=df_PI['Iterations'], y=df_PI['MaxV'], label="Policy Iteration", linestyle='-', ax=ax5, color='blue')  # linestyle='--',
    ax5.set_title(name + ' PI Reward (MaxV) vs Iterations')
    ax5.minorticks_on()
    ax5.grid(b=True, which='major', linestyle='-', alpha=0.2)
    ax5.grid(b=True, which='minor', linestyle='-', alpha=0.1)
    f.tight_layout()
    plt.savefig(exp + ' ' + name + ' PI Reward (MaxV) vs Iterations' + timestr + '.png')
    #plt.show()

    # -Time vs iterations
    f, (ax4) = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    sns.lineplot(x=df_VI['Iterations'], y=df_VI['Time'], label="Value Iteration", ax=ax4,color='green')
    ax4.set_title(name+' VI Time vs Iterations')
    ax4.minorticks_on()
    ax4.grid(b=True, which='major', linestyle='-', alpha=0.2)
    ax4.grid(b=True, which='minor', linestyle='-', alpha=0.1)
    f.tight_layout()
    plt.savefig(exp + ' ' + name + ' VI Time vs Iterations' + timestr + '.png')
    #plt.show()

    f, (ax5) = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    sns.lineplot(x=df_PI['Iterations'], y=df_PI['Time'], label="Policy Iteration", linestyle='-', ax=ax5, color='blue')  # linestyle='--',
    ax5.set_title(name + ' PI Time vs Iterations')
    ax5.minorticks_on()
    ax5.grid(b=True, which='major', linestyle='-', alpha=0.2)
    ax5.grid(b=True, which='minor', linestyle='-', alpha=0.1)
    f.tight_layout()
    plt.savefig(exp + ' ' + name + ' PI Time vs Iterations' + timestr + '.png')
    #plt.show()

    # -combined time vs iterations
    f, (ax3) = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    sns.lineplot(x=df_VI['Iterations'], y=df_VI['Time'], label="Value Iteration", ax=ax3, color='green')
    sns.lineplot(x=df_PI['Iterations'], y=df_PI['Time'], label="Policy Iteration", ax=ax3,
                 color='blue')  # linestyle='--',
    ax3.set_title(name + ' Time vs Iterations')
    ax3.minorticks_on()
    ax3.grid(b=True, which='major', linestyle='-', alpha=0.2)
    ax3.grid(b=True, which='minor', linestyle='-', alpha=0.1)
    f.tight_layout()
    plt.savefig(exp + ' ' + name + ' Combined Time vs Iterations VI PI' + timestr + '.png')
    # plt.show()



def create_plot(x, y, x_lab, y_lab, linelab, title, savename, vline=0, vlineval=195, vlinelab='test'):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    plt.figure()
    print(np.ndim(y))
    print(y)
    if np.ndim(y) > 1 and np.ndim(y) < 10:
        for idx in np.arange(np.ndim(y)):
            plt.plot(x, y[idx], linestyle='-', label=linelab[idx])  # , marker='o')
    else:
        plt.plot(x, y, linestyle='-', label=linelab)  # , marker='o')
    if vline:
        plt.hlines(vlineval, 0, np.max(x), linestyles='dashed', label=vlinelab)
    plt.ylabel(y_lab, fontsize=10)
    plt.xlabel(x_lab, fontsize=10)
    plt.title(title, fontsize=14)
    # plt.xticks(fontsize=tick_size)
    # plt.yticks(fontsize=tick_size)
    plt.legend()
    plt.savefig(savename)
    #plt.show()