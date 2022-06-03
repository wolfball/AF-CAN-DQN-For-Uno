import seaborn as sns; sns.set()
import pandas as pd
import matplotlib.pyplot as plt
import os
explist = os.listdir('res')
explist = ['baseline0', 'aedqn_mse_1', 'aedqn_mse_05', 'aedqn_mse01',
           'aedqn_bce_1', 'aedqn_bce_05', 'aedqn_bce_01'
           ]
win_size = 20

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111)
for csvpath in explist:
    performance = pd.read_csv(f"res/{csvpath}/performance.csv")
    mean_perf = [performance[i:i+win_size]['reward'].mean() for i in range(len(performance)-win_size+1)]
    ax.plot(mean_perf)
    # df_reward = pd.DataFrame(performance)
    # print(len(df_reward))
    # if len(df_reward)>200:
    #     continue
    # sns.lineplot(x='timestep', y='reward', data=df_reward)
ax.set_title(f"Results")
ax.set_ylabel("Reward")
ax.set_xlabel("Timestep")
ax.legend(explist, loc="lower right")
plt.tight_layout()
plt.savefig("vis.png")

# print(f'Cool agent\'s test reward is (average over {runs} runs):', np.mean(test_recorded_episode_reward_log))
