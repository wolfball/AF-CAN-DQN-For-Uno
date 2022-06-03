# AF-CAN-DQN-For-Uno

SJTU 2021-2022 AI3617 Game Theory and Multi-Agent Learning Final Project

Members: Siyuan Li, Han Yan, Ziyuan Li, Zhesheng Xu


### Summary

Based on RL and Game theory, we train an agent that can play Uno. (In game theory and multi-agent reinforcement learning)

We propose two method to improve the performance of DQN. (It is not simple algorithm repetition and experimental comparison)

Each member's contribution are listed at last in the Minipaper.

The code is avaiable.

### Running commands

```bash
# baseline
python main.py --cuda 0 --algorithm dqn --log_dir path/to/save/baseline/

# AFDQN
python main.py --cuda 0 --algorithm afdqn --log_dir path/to/save/afdqn/

# CANDQN
python main.py --cuda 0 --algorithm candqn --log_dir path/to/save/candqn/
```



### Controllable Parameters

```bash
main.py
--algorithm dqn  # the algorithm choosed, 'dqn' or 'afdqn' or 'candqn'
--cuda 0  # the cuda id
--seed 42  # random seed
--num_episodes 100000  # training episodes
--num_eval_games 1000  # number of games when evaluation
--evaluate_every 2000  # when to evaluate
--color_dim 256  # the dimension of colorfeature when using candqn
--num_dim 128  # the dimension of numberfeature when using candqn
--lamb 1.5  # the coeffecient of loss function when using afdqn
--log_dir experiments/uno_dqn_result/  # path to save the logging and model

# evaluate.py and vis.py are used to analyse the experimental results.
```


