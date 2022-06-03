''' An example of training a reinforcement learning agent on the environments in RLCard
'''
import os
import argparse

import torch

import rlcard
from rlcard.agents import RandomAgent
from rlcard.utils import (
    get_device,
    set_seed,
    tournament,
    reorganize,
    Logger,
    plot_curve,
)
import time


def train(args):
    # Check whether gpu is available
    device = get_device()
    # Seed numpy, torch, random
    set_seed(args.seed)
    # Make the environment with seed
    env = rlcard.make(args.env, config={'seed': args.seed, })

    from ach import ACHAgent
    agent = ACHAgent(num_actions=env.num_actions,
                     state_shape=env.state_shape[0],
                     mlp_layers=[512, 512],
                     device=device)

    agents = [agent]
    for _ in range(1, env.num_players):
        agents.append(RandomAgent(num_actions=env.num_actions))
    env.set_agents(agents)
    t = time.time()
    # Start training
    with Logger(args.log_dir) as logger:
        for episode in range(args.num_episodes):
            # Generate data from the environment
            trajectories, payoffs = env.run(is_training=True)
            trajectories = reorganize(trajectories, payoffs)

            agent.feed(trajectories[0])

            if episode % args.evaluate_every == 0:
                logger.log_performance(
                    env.timestep,
                    tournament(
                        env,
                        args.num_eval_games,
                    )[0]
                )

        # Get the paths
        csv_path, fig_path = logger.csv_path, logger.fig_path

    # Plot the learning curve
    plot_curve(csv_path, fig_path, args.algorithm)
    print(f"Time used: {(time.time() - t) / 3600}")
    # Save model
    save_path = os.path.join(args.log_dir, 'model.pth')
    torch.save(agent, save_path)
    print('Model saved in', save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("DQN/NFSP example in RLCard")
    parser.add_argument('--env', type=str, default='uno', )
    # parser.add_argument('--algorithm', type=str, default='dqn', )
    parser.add_argument('--cuda', type=str, default='0', )
    parser.add_argument('--seed', type=int, default=42, )
    parser.add_argument('--num_episodes', type=int, default=50000, )
    parser.add_argument('--num_eval_games', type=int, default=1000, )
    parser.add_argument('--evaluate_every', type=int, default=2000, )
    parser.add_argument('--log_dir', type=str, default='experiments/uno_dqn_result/', )

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    train(args)