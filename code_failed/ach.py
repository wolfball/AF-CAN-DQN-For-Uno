import random
import numpy as np
import torch
import torch.nn as nn
from collections import namedtuple
from copy import deepcopy

from rlcard.utils.utils import remove_illegal

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'legal_actions', 'done'])

def softmax(x, axis=0):
    x -= np.max(x, axis=axis, keepdims=True)
    x = np.exp(x) / np.sum(x, axis=axis, keepdims=True)
    return x

class ACHAgent(object):
    def __init__(self,
                 replay_memory_size=20000,
                 replay_memory_init_size=100,
                 update_target_estimator_every=1000,
                 discount_factor=0.95,
                 epsilon_start=1.0,
                 epsilon_end=0.1,
                 epsilon_decay_steps=20000,
                 batch_size=32,
                 num_actions=2,
                 state_shape=None,
                 train_every=1,
                 mlp_layers=None,
                 learning_rate=0.00005,
                 device=None,
                 ratio_clip=0.5,
                 lth=2.0,
                 hedge_coef=1e-4,
                 entropy_coef=1e-2,
                 valueloss_coef=2.0):
        '''
        Args:
            replay_memory_size (int): Size of the replay memory
            replay_memory_init_size (int): Number of random experiences to sample when initializing
              the reply memory.
            update_target_estimator_every (int): Copy parameters from the Q estimator to the
              target estimator every N steps
            discount_factor (float): Gamma discount factor
            epsilon_start (float): Chance to sample a random action when taking an action.
              Epsilon is decayed over time and this is the start value
            epsilon_end (float): The final minimum value of epsilon after decaying is done
            epsilon_decay_steps (int): Number of steps to decay epsilon over
            batch_size (int): Size of batches to sample from the replay memory
            evaluate_every (int): Evaluate every N steps
            num_actions (int): The number of the actions
            state_space (list): The space of the state vector
            train_every (int): Train the network every X steps.
            mlp_layers (list): The layer number and the dimension of each layer in MLP
            learning_rate (float): The learning rate of the DQN agent.
            device (torch.device): whether to use the cpu or gpu
        '''
        self.use_raw = False
        self.replay_memory_init_size = replay_memory_init_size
        self.update_target_estimator_every = update_target_estimator_every
        self.discount_factor = discount_factor
        self.epsilon_decay_steps = epsilon_decay_steps
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.train_every = train_every
        self.ratio_clip = ratio_clip
        self.lth = lth
        self.hedge_coef = hedge_coef
        self.entropy_coef = entropy_coef
        self.valueloss_coef = valueloss_coef
        # Torch device
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # Total timesteps
        self.total_t = 0

        # Total training step
        self.train_t = 0

        # The epsilon decay scheduler
        self.epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

        # Create estimators
        self.policy_estimator = Estimator(num_actions=num_actions, learning_rate=learning_rate, state_shape=state_shape, \
                                     mlp_layers=mlp_layers, device=self.device)
        self.value_estimator = Estimator(num_actions=1, learning_rate=learning_rate, state_shape=state_shape, \
                                          mlp_layers=mlp_layers, device=self.device)

        # Create replay memory
        self.memory = Memory(replay_memory_size, batch_size)

    def feed(self, ts):
        if len(ts) == 1:
            return
        self.feed_memory(ts)
        # (state, action, reward, next_state, done) = tuple(ts)
        # self.feed_memory(state['obs'], action, reward, next_state['obs'],
        #                  list(state['legal_actions'].keys()), done)
        self.total_t += 1
        tmp = self.total_t - self.replay_memory_init_size
        if tmp >= 0 and tmp % self.train_every == 0:
            self.train()

    def step(self, state):
        ''' Predict the action for genrating training data but
            have the predictions disconnected from the computation graph
        Args:
            state (numpy.array): current state
        Returns:
            action (int): an action id
        '''
        q_values = self.predict(state)
        epsilon = self.epsilons[min(self.total_t, self.epsilon_decay_steps - 1)]
        legal_actions = list(state['legal_actions'].keys())
        probs = np.ones(len(legal_actions), dtype=float) * epsilon / len(legal_actions)
        best_action_idx = legal_actions.index(np.argmax(q_values))
        probs[best_action_idx] += (1.0 - epsilon)
        action_idx = np.random.choice(np.arange(len(probs)), p=probs)

        return legal_actions[action_idx]

    def eval_step(self, state):
        ''' Predict the action for evaluation purpose.
        Args:
            state (numpy.array): current state
        Returns:
            action (int): an action id
            info (dict): A dictionary containing information
        '''
        q_values = self.predict(state)
        best_action = np.argmax(q_values)

        info = {}
        info['values'] = {state['raw_legal_actions'][i]: float(q_values[list(state['legal_actions'].keys())[i]]) for i
                          in range(len(state['legal_actions']))}

        return best_action, info

    def predict(self, state):
        ''' Predict the masked Q-values
        Args:
            state (numpy.array): current state
        Returns:
            q_values (numpy.array): a 1-d array where each entry represents a Q value
        '''

        q_values = self.policy_estimator.predict_nograd(np.expand_dims(state['obs'], 0))[0]
        masked_q_values = -np.inf * np.ones(self.num_actions, dtype=float)
        legal_actions = list(state['legal_actions'].keys())
        masked_q_values[legal_actions] = q_values[legal_actions]

        return masked_q_values

    def train(self):
        # state_batch, action_batch, reward_batch, next_state_batch, legal_actions_batch, done_batch = self.memory.sample()
        # Calculate best next actions using Q-network (Double DQN)
        # state_batch, action_batch, reward_batch, next_state_batch, done_batch, \
        #     legal_action_batch, next_legal_action_batch \
        #         = self.memory.sample()  # h_1, h_2, ... (h=(s, a, r, ns, d, la, nla))
        trajs = self.memory.sample()
        self.value_estimator.qnet.train()
        self.policy_estimator.qnet.train()
        # print(traj[0][-1])
        policy_loss_total = torch.tensor([0.]).float().to(self.device)
        value_loss_total = torch.tensor([0.]).float().to(self.device)
        dist_loss_total = torch.tensor([0.]).float().to(self.device)
        for s, a, r, ns, d, la, nla, pi_old in trajs:
            a = torch.from_numpy(a)
            d = torch.from_numpy(d).float().to(self.device)
            r = torch.from_numpy(r).to(self.device)
            pi_old = torch.from_numpy(pi_old).gather(1, a).numpy()
            adv = r + (1-d) * (self.discount_factor * torch.from_numpy(self.value_estimator.predict_nograd(ns)).to(self.device) \
                               -self.value_estimator.qnet(torch.from_numpy(s).float().to(self.device)))
            G = torch.tensor(r[-1] * [[self.discount_factor**(len(r)-i-1)] for i in range(len(r))]).to(self.device)
            # adv = G - self.value_estimator.qnet(torch.from_numpy(s).float().to(self.device))
            advan = adv.detach().cpu().numpy()
            y = self.policy_estimator.predict_nograd(s)
            pi = torch.from_numpy(softmax(y, axis=1)).gather(1, a).numpy()
            y_avg = np.mean(y, axis=-1, keepdims=True)
            ind1 = pi / pi_old
            ind2 = torch.from_numpy(y - y_avg).gather(1, a).numpy()
            c = np.zeros_like(advan)

            c[advan >= 0] = (ind1<1+self.ratio_clip)[advan>=0] * (ind2<self.lth)[advan>=0]
            c[advan < 0] = (ind1>1-self.ratio_clip)[advan<0] * (ind2>-self.lth)[advan<0]
            y_a = self.policy_estimator.qnet(torch.from_numpy(s).float().to(self.device))
            policy_loss = torch.from_numpy(c * self.hedge_coef * advan / pi_old).to(self.device) \
                            * y_a.gather(1, a.to(self.device))
            value_loss = 0.5 * self.valueloss_coef * (G - self.value_estimator.qnet(torch.from_numpy(s).float().to(self.device)))**2
            # value_loss = 0.5 * self.valueloss_coef * adv**2
            pi_a = torch.softmax(y_a, 1)
            dist_loss = self.entropy_coef * torch.sum(pi_a*torch.log(pi_a), 1, keepdim=True)
            policy_loss_total += policy_loss.sum()
            value_loss_total += value_loss.sum()
            dist_loss_total += dist_loss.sum()
        policy_loss_total /= self.batch_size
        value_loss_total /= self.batch_size
        dist_loss_total /= self.batch_size
        self.value_estimator.optimizer.zero_grad()
        self.policy_estimator.optimizer.zero_grad()
        (policy_loss_total + value_loss_total + dist_loss_total).backward()
        self.value_estimator.optimizer.step()
        self.policy_estimator.optimizer.step()
        # batch_loss = batch_loss.item()

        # q_values_next = self.policy_estimator.predict_nograd(next_state_batch)
        # legal_actions = []
        # for b in range(self.batch_size):
        #     legal_actions.extend([i + b * self.num_actions for i in legal_actions_batch[b]])
        # masked_q_values = -np.inf * np.ones(self.num_actions * self.batch_size, dtype=float)
        # masked_q_values[legal_actions] = q_values_next.flatten()[legal_actions]
        # masked_q_values = masked_q_values.reshape((self.batch_size, self.num_actions))
        # best_actions = np.argmax(masked_q_values, axis=1)
        #
        # # Evaluate best next actions using Target-network (Double DQN)
        # q_values_next_target = self.value_estimator.predict_nograd(next_state_batch)
        # target_batch = reward_batch + np.invert(done_batch).astype(np.float32) * \
        #                self.discount_factor * q_values_next_target[np.arange(self.batch_size), best_actions]
        #
        # # Perform gradient descent update
        # state_batch = np.array(state_batch)
        #
        # loss = self.policy_estimator.update(state_batch, action_batch, target_batch)
        print('\rINFO - Step {}, rl-loss: {}, ({}, {}, {})'.format(
            self.total_t, (policy_loss_total + value_loss_total + dist_loss_total).item(),
            policy_loss_total.item(), value_loss_total.item(), dist_loss_total.item()), end='')

        # Update the target estimator
        # if self.train_t % self.update_target_estimator_every == 0:
        #     self.target_estimator = deepcopy(self.policy_estimator)
        #     print("\nINFO - Copied model parameters to target network.")

        # self.train_t += 1
        self.value_estimator.qnet.eval()
        self.policy_estimator.qnet.eval()

    def feed_memory(self, traj):
        ''' Feed transition to memory
        Args:
            state (numpy.array): the current state
            action (int): the performed action ID
            reward (float): the reward received
            next_state (numpy.array): the next state after performing the action
            legal_actions (list): the legal actions of the next state
            done (boolean): whether the episode is finished
        '''
        self.memory.save(traj, self.policy_estimator)

    def set_device(self, device):
        self.device = device
        self.value_estimator.device = device
        self.policy_estimator.device = device


class Estimator(object):
    '''
    Approximate clone of rlcard.agents.dqn_agent.Estimator that
    uses PyTorch instead of Tensorflow.  All methods input/output np.ndarray.
    Q-Value Estimator neural network.
    This network is used for both the Q-Network and the Target Network.
    '''

    def __init__(self, num_actions=2, learning_rate=0.001, state_shape=None, mlp_layers=None, device=None):
        ''' Initilalize an Estimator object.
        Args:
            num_actions (int): the number output actions
            state_shape (list): the shape of the state space
            mlp_layers (list): size of outputs of mlp layers
            device (torch.device): whether to use cpu or gpu
        '''
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.state_shape = state_shape
        self.mlp_layers = mlp_layers
        self.device = device

        # set up Q model and place it in eval mode
        qnet = EstimatorNetwork(num_actions, state_shape, mlp_layers)
        qnet = qnet.to(self.device)
        self.qnet = qnet
        self.qnet.eval()

        # initialize the weights using Xavier init
        for p in self.qnet.parameters():
            if len(p.data.shape) > 1:
                nn.init.xavier_uniform_(p.data)

        # set up loss function
        self.mse_loss = nn.MSELoss(reduction='mean')

        # set up optimizer
        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=self.learning_rate)

    def predict_nograd(self, s):
        ''' Predicts action values, but prediction is not included
            in the computation graph.  It is used to predict optimal next
            actions in the Double-DQN algorithm.
        Args:
          s (np.ndarray): (batch, state_len)
        Returns:
          np.ndarray of shape (batch_size, NUM_VALID_ACTIONS) containing the estimated
          action values.
        '''
        self.qnet.eval()
        with torch.no_grad():
            s = torch.from_numpy(s).float().to(self.device)
            q_as = self.qnet(s).cpu().numpy()
        self.qnet.train()
        return q_as

    def update(self, s, a, y):
        ''' Updates the estimator towards the given targets.
            In this case y is the target-network estimated
            value of the Q-network optimal actions, which
            is labeled y in Algorithm 1 of Minh et al. (2015)
        Args:
          s (np.ndarray): (batch, state_shape) state representation
          a (np.ndarray): (batch,) integer sampled actions
          y (np.ndarray): (batch,) value of optimal actions according to Q-target
        Returns:
          The calculated loss on the batch.
        '''
        self.optimizer.zero_grad()

        self.qnet.train()

        s = torch.from_numpy(s).float().to(self.device)
        a = torch.from_numpy(a).long().to(self.device)
        y = torch.from_numpy(y).float().to(self.device)

        # (batch, state_shape) -> (batch, num_actions)
        q_as = self.qnet(s)

        # (batch, num_actions) -> (batch, )
        Q = torch.gather(q_as, dim=-1, index=a.unsqueeze(-1)).squeeze(-1)

        # update model
        batch_loss = self.mse_loss(Q, y)
        batch_loss.backward()
        self.optimizer.step()
        batch_loss = batch_loss.item()

        self.qnet.eval()

        return batch_loss


class EstimatorNetwork(nn.Module):
    ''' The function approximation network for Estimator
        It is just a series of tanh layers. All in/out are torch.tensor
    '''

    def __init__(self, num_actions=2, state_shape=None, mlp_layers=None):
        ''' Initialize the Q network
        Args:
            num_actions (int): number of legal actions
            state_shape (list): shape of state tensor
            mlp_layers (list): output size of each fc layer
        '''
        super(EstimatorNetwork, self).__init__()

        self.num_actions = num_actions
        self.state_shape = state_shape
        self.mlp_layers = mlp_layers

        # build the Q network
        layer_dims = [np.prod(self.state_shape)] + self.mlp_layers
        fc = [nn.Flatten()]
        fc.append(nn.BatchNorm1d(layer_dims[0]))
        for i in range(len(layer_dims) - 1):
            fc.append(nn.Linear(layer_dims[i], layer_dims[i + 1], bias=True))
            fc.append(nn.Tanh())
        fc.append(nn.Linear(layer_dims[-1], self.num_actions, bias=True))
        self.fc_layers = nn.Sequential(*fc)

    def forward(self, s):
        ''' Predict action values
        Args:
            s  (Tensor): (batch, state_shape)
        '''
        return self.fc_layers(s)


class Memory(object):
    ''' Memory for saving transitions
    '''

    def __init__(self, memory_size, batch_size):
        ''' Initialize
        Args:
            memory_size (int): the size of the memroy buffer
        '''
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = []

    def save(self, traj, func):
        ''' Save transition into memory
        Args:
            state (numpy.array): the current state
            action (int): the performed action ID
            reward (float): the reward received
            next_state (numpy.array): the next state after performing the action
            legal_actions (list): the legal actions of the next state
            done (boolean): whether the episode is finished
        '''
        if len(self.memory) == self.memory_size:
            self.memory.pop(0)
        # transition = Transition(state, action, reward, next_state, legal_actions, done)
        state = np.array([h[0]['obs'] for h in traj])
        legal_action = [list(h[0]['legal_actions'].keys()) for h in traj]
        next_state = np.array([h[3]['obs'] for h in traj])
        next_legal_action = [list(h[3]['legal_actions'].keys()) for h in traj]
        action = np.array([[h[1]] for h in traj])
        reward = np.array([[h[2]] for h in traj])
        done = np.array([[h[4]] for h in traj])
        pi_old = softmax(func.predict_nograd(state))
        self.memory.append([state, action, reward, next_state, done, legal_action, next_legal_action, pi_old])

    def sample(self):
        ''' Sample a minibatch from the replay memory
        Returns:
            state_batch (list): a batch of states
            action_batch (list): a batch of actions
            reward_batch (list): a batch of rewards
            next_state_batch (list): a batch of states
            done_batch (list): a batch of dones
        '''
        samples = random.sample(self.memory, self.batch_size)
        return samples  # map(np.array, zip(*samples))