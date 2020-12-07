"""
Implementation of PPO
ref: Schulman, John, et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).
ref: https://github.com/Jiankai-Sun/Proximal-Policy-Optimization-in-Pytorch/blob/master/ppo.py
ref: https://github.com/openai/baselines/tree/master/baselines/ppo2

NOTICE:
    `Tensor2` means 2D-Tensor (num_samples, num_dims) 
"""

import gym
import torch
import torch.nn as nn
import torch.optim as opt
from torch import Tensor
from torch.autograd import Variable
from collections import namedtuple
from itertools import count
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from os.path import join as joindir
from os import makedirs as mkdir
import pandas as pd
import numpy as np
import seaborn as sns
import argparse
import datetime
import math


Transition = namedtuple('Transition', ('state', 'value', 'action', 'logproba', 'mask', 'next_state', 'reward'))
EPS = 1e-10
RESULT_DIR = '/home/wangxu/PycharmProjects/torchprojects/result/'
mkdir(RESULT_DIR, exist_ok=True)


class args(object):
    env_name = 'BipedalWalker-v3'
    seed = 1234
    num_episode = 2000  # 2000
    batch_size = 2048    # 2048
    max_step_per_round = 2000    # 2000
    gamma = 0.995
    lamda = 0.97
    log_num_episode = 1
    num_epoch = 10
    minibatch_size = 256
    clip = 0.2
    loss_coeff_value = 0.5
    loss_coeff_entropy = -0.01   # notice
    lr = 3e-4
    num_parallel_run = 5
    # tricks
    schedule_adam = 'linear'
    schedule_clip = 'linear'
    layer_norm = True
    state_norm = True
    advantage_norm = True
    lossvalue_norm = True


class RunningStat(object):
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape)   # shape = (state_dim, )
        self._S = np.zeros(shape)

    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)   # sn = (a0-mean)**2 + ... + (an-mean)**2

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape


class ZFilter:
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """

    def __init__(self, shape, demean=True, destd=True, clip=10.0):
        self.demean = demean
        self.destd = destd
        self.clip = clip

        self.rs = RunningStat(shape)

    def __call__(self, x, update=True):
        if update: self.rs.push(x)
        if self.demean:
            x = x - self.rs.mean
        if self.destd:
            x = x / (self.rs.std + 1e-8)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x

    def output_shape(self, input_space):
        return input_space.shape


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, layer_norm=True):
        super(ActorCritic, self).__init__()
        
        self.actor_fc1 = nn.Linear(num_inputs, 64)
        self.actor_fc2 = nn.Linear(64, 64)
        self.actor_fc3 = nn.Linear(64, num_outputs)
        self.actor_logstd = nn.Parameter(torch.zeros(1, num_outputs))

        self.critic_fc1 = nn.Linear(num_inputs, 64)
        self.critic_fc2 = nn.Linear(64, 64)
        self.critic_fc3 = nn.Linear(64, 1)

        if layer_norm:
            self.layer_norm(self.actor_fc1, std=1.0)
            self.layer_norm(self.actor_fc2, std=1.0)
            self.layer_norm(self.actor_fc3, std=0.01)

            self.layer_norm(self.critic_fc1, std=1.0)
            self.layer_norm(self.critic_fc2, std=1.0)
            self.layer_norm(self.critic_fc3, std=1.0)

    @staticmethod
    def layer_norm(layer, std=1.0, bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)

    def forward(self, states):
        """
        run policy network (actor) as well as value network (critic)
        :param states: a Tensor2 represents states
        :return: 3 Tensor2
        """
        action_mean, action_logstd = self._forward_actor(states)
        critic_value = self._forward_critic(states)
        return action_mean, action_logstd, critic_value

    def _forward_actor(self, states):
        x = torch.tanh(self.actor_fc1(states))
        x = torch.tanh(self.actor_fc2(x))
        action_mean = self.actor_fc3(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        return action_mean, action_logstd

    def _forward_critic(self, states):
        x = torch.tanh(self.critic_fc1(states))
        x = torch.tanh(self.critic_fc2(x))
        critic_value = self.critic_fc3(x)
        return critic_value

    def select_action(self, action_mean, action_logstd, return_logproba=True):
        """
        given mean and std, sample an action from normal(mean, std)
        also returns probability of the given chosen
        """
        action_std = torch.exp(action_logstd)
        action = torch.normal(action_mean, action_std)
        if return_logproba:
            logproba = self._normal_logproba(action, action_mean, action_logstd, action_std)
        return action, logproba

    @staticmethod
    def _normal_logproba(x, mean, logstd, std=None):
        if std is None:
            std = torch.exp(logstd)

        std_sq = std.pow(2)
        logproba = - 0.5 * math.log(2 * math.pi) - logstd - (x - mean).pow(2) / (2 * std_sq)
        return logproba.sum(1)

    def get_logproba(self, states, actions):
        """
        return probability of chosen the given actions under corresponding states of current network
        :param states: Tensor
        :param actions: Tensor
        """
        action_mean, action_logstd = self._forward_actor(states)
        logproba = self._normal_logproba(actions, action_mean, action_logstd)
        return logproba

    
class Memory(object):
    def __init__(self):
        self.memory = []       # list

    def push(self, *args):
        self.memory.append(Transition(*args))   # namedtuple_i into list[i]

    def sample(self):
        return Transition(*zip(*self.memory))   # many namedtuple list into a large list namedtuple

    def __len__(self):
        return len(self.memory)


def ppo(args):
    env = gym.make(args.env_name)
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]

    env.seed(args.seed)
    torch.manual_seed(args.seed)

    network = ActorCritic(num_states, num_actions, layer_norm=args.layer_norm)
    optimizer = opt.Adam(network.parameters(), lr=args.lr)
    network.train()

    normalized_state = ZFilter((num_states,), clip=5.0)
    
    # record average 1-round cumulative reward in every episode
    reward_record = []
    global_steps = 0

    lr_now = args.lr
    clip_now = args.clip

    for i_episode in range(args.num_episode):
        # step1: perform current policy to collect trajectories
        # this is an on-policy method!
        memory = Memory()
        num_steps = 0
        reward_list = []
        len_list = []
        while num_steps < args.batch_size:
            state = env.reset()
            if args.state_norm:
                state = normalized_state(state)
            reward_sum = 0  # for the whole trajectory reward
            for t in range(args.max_step_per_round):
                action_mean, action_logstd, value = network(Tensor(state).unsqueeze(0))
                action, logproba = network.select_action(action_mean, action_logstd)
                action = action.data.numpy()[0]
                logproba = logproba.data.numpy()[0]
                next_state, reward, done, _ = env.step(action)
                reward_sum += reward
                if args.state_norm:
                    next_state = normalized_state(next_state)
                mask = 0 if done else 1

                memory.push(state, value, action, logproba, mask, next_state, reward)
                # state.shape: (17, )     value.shape: torch.size([1,1])
                if done:
                    break
                    
                state = next_state

            # do every trajectory over
            num_steps += (t + 1)
            global_steps += (t + 1)
            reward_list.append(reward_sum)
            len_list.append(t + 1)
        reward_record.append({
            'episode': i_episode, 
            'steps': global_steps, 
            'mean_ep_reward': np.mean(reward_list),
            'mean_ep_len': np.mean(len_list),
            'trajectory_number': len(len_list)})

        batch = memory.sample()     # namedtuple     batch._fields     batch[1] = batch.value

        batch_size = len(memory)
        
        # step2: extract variables from trajectories
        rewards = Tensor(batch.reward)    # .shape: torch.Size([batch_size])     tuple to tensor
        values = Tensor(batch.value)      # .shape: torch.Size([batch_size])     batch.value[i]: value of step i
        masks = Tensor(batch.mask)        # .shape: torch.Size([batch_size])
        actions = Tensor(batch.action)    # .shape: torch.Size([batch_size, action_dim])
        states = Tensor(batch.state)      # .shape: torch.Size([batch_size, state_dim])
        oldlogproba = Tensor(batch.logproba)    # .shape: torch.Size([batch_size])

        returns = Tensor(batch_size)
        deltas = Tensor(batch_size)
        advantages = Tensor(batch_size)

        prev_return = 0   # define the last reward of trajectory is zero
        prev_value = 0
        prev_advantage = 0
        for i in reversed(range(batch_size)):
            returns[i] = rewards[i] + args.gamma * prev_return * masks[i]
            deltas[i] = rewards[i] + args.gamma * prev_value * masks[i] - values[i]
            # ref: https://arxiv.org/pdf/1506.02438.pdf (generalization advantage estimate)
            advantages[i] = deltas[i] + args.gamma * args.lamda * prev_advantage * masks[i]

            prev_return = returns[i]
            prev_value = values[i]
            prev_advantage = advantages[i]
        if args.advantage_norm:
            advantages = (advantages - advantages.mean()) / (advantages.std())

        for i_epoch in range(int(batch_size / args.minibatch_size)):
            # sample from current batch to training
            minibatch_ind = np.random.choice(batch_size, args.minibatch_size, replace=False)
            minibatch_states = states[minibatch_ind]
            minibatch_actions = actions[minibatch_ind]
            minibatch_oldlogproba = oldlogproba[minibatch_ind]
            #minibatch_newlogproba = network.get_logproba(minibatch_states, minibatch_actions)
            minibatch_advantages = advantages[minibatch_ind]
            minibatch_returns = returns[minibatch_ind]
            #minibatch_newvalues = network._forward_critic(minibatch_states).flatten()

            for _ in range(args.num_epoch):
                minibatch_newlogproba = network.get_logproba(minibatch_states, minibatch_actions)
                minibatch_newvalues = network._forward_critic(minibatch_states).flatten()
                ratio =  torch.exp(minibatch_newlogproba - minibatch_oldlogproba)
                surr1 = ratio * minibatch_advantages
                surr2 = ratio.clamp(1 - clip_now, 1 + clip_now) * minibatch_advantages
                loss_surr = - torch.mean(torch.min(surr1, surr2))

            # not sure the value loss should be clipped as well 
            # clip example: https://github.com/Jiankai-Sun/Proximal-Policy-Optimization-in-Pytorch/blob/master/ppo.py
            # however, it does not make sense to clip score-like value by a dimensionless clipping parameter
            # moreover, original paper does not mention clipped value 
                if args.lossvalue_norm:
                    minibatch_return_6std = 6 * minibatch_returns.std()
                    loss_value = torch.mean((minibatch_newvalues - minibatch_returns).pow(2)) / minibatch_return_6std
                else:
                    loss_value = torch.mean((minibatch_newvalues - minibatch_returns).pow(2))

                loss_entropy = torch.mean(torch.exp(minibatch_newlogproba) * minibatch_newlogproba)

                optimizer.zero_grad()
                total_loss = loss_surr + args.loss_coeff_value * loss_value + args.loss_coeff_entropy * loss_entropy

                total_loss.backward()
                optimizer.step()

        if args.schedule_clip == 'linear':
            ep_ratio = 1 - 0.5*(i_episode / args.num_episode)
            clip_now = args.clip * ep_ratio

        if args.schedule_adam == 'linear':
            ep_ratio = 1 - 0.5*(i_episode / args.num_episode)
            lr_now = args.lr * ep_ratio
            # set learning rate
            # ref: https://stackoverflow.com/questions/48324152/
            for g in optimizer.param_groups:
                g['lr'] = lr_now

        if i_episode % args.log_num_episode == 0:
            print('Finished episode: {} Reward: {:.4f} total_loss = {:.4f} = {:.4f} + {} * {:.4f} + {} * {:.4f}' \
                .format(i_episode, reward_record[-1]['mean_ep_reward'], total_loss.data, loss_surr.data, args.loss_coeff_value,
                loss_value.data, args.loss_coeff_entropy, loss_entropy.data))
            print('-----------------')

    return network, normalized_state, reward_record


def test_record(args):
    record_dfs = []
    for i in range(args.num_parallel_run):
        args.seed += 1
        network, normalized_state, reward_record = ppo(args)
        record = pd.DataFrame(reward_record)
        record['parallel_run'] = i
        record_dfs.append(record)
    record_dfs = pd.concat(record_dfs, axis=0)
    record_dfs.to_csv(joindir(RESULT_DIR, 'ppo-record-{}.csv'.format(args.env_name)))
    torch.save({
                'model_state_dict': network.state_dict(),
                'normalized_state': normalized_state
                }, joindir(RESULT_DIR, 'ppo-modelpara-{}.pth'.format(args.env_name)))

def test_run(args):
    network = ppo(args)[0]
    env = gym.make(args.env_name)
    for t in range(10):
        state = env.reset()
        done = False
        i = 0
        while (not done):
            env.render()
            action_mean, action_logstd, value = network(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0))
            action, logproba = network.select_action(action_mean, action_logstd)
            action = action.data.numpy()[0]
            state, _, done, _ = env.step(action)
            i += 1
            if done or i==2000:
                print('test {:.0f} time,  stop at {:.0f} step.'.format(t+1, i))
                env.close()
                break

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    env = 'Walker2d-v2'
    args.env_name = env
    is_record = False
    if is_record:
        test_record(args)
    else:
        test_run(args)

#  BipedalWalker-v3   Hopper-v2   Walker2d-v2
# 'Walker2d-v2', 'Swimmer-v2', 'Hopper-v2', 'Humanoid-v2', 'HalfCheetah-v2', 'Reacher-v2'
