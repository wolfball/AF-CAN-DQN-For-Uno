import os
import argparse
import torch
import rlcard
from rlcard.agents import RandomAgent
from uno_human import UNORuleAgentV1
from rlcard.utils import (
    get_device,
    set_seed,
    tournament,
    reorganize,
    Logger,
    plot_curve,
)
def train(args):
    device = get_device()
    set_seed(args.seed)
    env = rlcard.make(args.env, config={'seed': args.seed,})

    # Initialize the agent and use random agents as opponents
    if args.algorithm == 'dqn':
        if len(args.pt_path) == 0:
            return
        agent = torch.load(os.path.join(args.pt_path, 'model.pth'))
        print(args.pt_path, " is loaded.")

    elif args.algorithm == 'nfsp':
        from rlcard.agents import NFSPAgent
        agent = NFSPAgent(
            num_actions=env.num_actions,
            state_shape=env.state_shape[0],
            hidden_layers_sizes=[512, 512],
            q_mlp_layers=[512, 512],
            device=device,
        )
    elif args.algorithm == 'human':
        agent = UNORuleAgentV1()
    elif args.algorithm == 'random':
        agent = RandomAgent(num_actions=env.num_actions)
    agents = [agent]
    for _ in range(1, env.num_players):
        agents.append(RandomAgent(num_actions=env.num_actions))
    env.set_agents(agents)

    # Start training
    if args.algorithm == 'nfsp':
        agents[0].sample_episode_policy()
        # Evaluate the performance. Play with random agents.
    print(tournament(env, args.num_eval_games,)[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser("DQN/NFSP example in RLCard")
    parser.add_argument('--env', type=str, default='uno',)
    parser.add_argument('--algorithm', type=str, default='dqn',
                        choices=['dqn', 'nfsp', 'human', 'random'],)
    parser.add_argument('--cuda', type=str, default='0',)
    parser.add_argument('--seed', type=int, default=42,)
    parser.add_argument('--num_eval_games', type=int, default=1000,)
    parser.add_argument('--pt_path', type=str, default='')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    train(args)