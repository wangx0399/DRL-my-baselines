'''
from Spinningup PPO code
'''

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
import gym
from gym.spaces import Box, Discrete
import scipy.signal
import time
from os.path import join as joindir


def mlp(sizes, activation, out_activation):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else out_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, out_activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation, out_activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)  ###probs=probs

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)  # Categorical().log_prob(act)   probs to logits


class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, out_activation):
        super().__init__()
        log_std = 0.5 * np.ones(act_dim, dtype=np.float32)
        # log_std as update parameter
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std), requires_grad=True)
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation, out_activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution


class MLPCritic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, activation, out_activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation, out_activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.


class MLPActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, pi_hidden_sizes, v_hidden_sizes,
                 activation, pi_out_activation, v_out_activation):
        super().__init__()

        obs_dim = observation_space.shape[0]
        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], pi_hidden_sizes, activation, pi_out_activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, pi_hidden_sizes, activation, pi_out_activation)
        # build value function
        self.v = MLPCritic(obs_dim, v_hidden_sizes, activation, v_out_activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]


class PPOBuffer:
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros([size, obs_dim[0]], dtype=np.float32)
        if act_dim == ():
            self.act_buf = np.zeros(size, dtype=np.float32)
        else:
            self.act_buf = np.zeros([size, act_dim[0]], dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_episode(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        # the next two lines implement GAE-Lambda advantage calculation for the policy function
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = scipy.signal.lfilter([1], [1,float(-self.gamma * self.lam)],
                                                        deltas[::-1], axis=0)[::-1]
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = scipy.signal.lfilter([1],[1, float(-self.gamma)],
                                                        rews[::-1], axis=0)[::-1][:-1]
        # update the pointer
        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next three lines implement the GAE_advantage normalization trick
        adv_mean = np.mean(np.array(self.adv_buf, dtype=np.float32))
        adv_std = np.std(np.array(self.adv_buf, dtype=np.float32))
        self.adv_buf = (self.adv_buf - adv_mean) / (adv_std + 1e-8)
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)  # define dictionary
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}  #array to tensor

def ppo(env_fn, actor_critic=MLPActorCritic, ac_kwargs=dict(), seed=0, minibatch=4000,
        epochs=50, gamma=0.99, epsilon=0.2, pi_lr=3e-4,v_lr=1e-3, train_pi_iters=80, train_v_iters=80,
        lam=0.97, max_episode=1000, target_kl=0.01):

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make(env_fn)
    obs = env.observation_space
    act = env.action_space

    actic = actor_critic(obs, act, **ac_kwargs)
    print(actic)
    buffer = PPOBuffer(obs.shape, act.shape, minibatch, gamma, lam)
    pi_optimizer = Adam(actic.pi.parameters(), lr=pi_lr)
    v_optimizer = Adam(actic.v.parameters(), lr=v_lr)

    def compute_loss_pi(data):
        obs, act, adv, logp_old = data['obs'].to(device), data['act'].to(device), data['adv'].to(device),\
                                  data['logp'].to(device)
        pi, logp_new = actic.pi(obs, act)
        ratio = torch.exp(logp_new - logp_old)
        clip = torch.clamp(ratio, 1-epsilon, 1+epsilon) * adv
        loss_pi = -(torch.min(ratio * adv, clip)).mean()
        # extra info
        approx_kl = (logp_old - logp_new).mean().item()
        entropy = pi.entropy().mean().item()
        clipped = ratio.gt(1+epsilon) | ratio.lt(1-epsilon)   # >1.2 or <0.8: True
        clipfraction = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(KL=approx_kl, entropy=entropy, clipfrac=clipfraction)
        return loss_pi, pi_info

    def compute_loss_v(data):
        obs, ret = data['obs'].to(device), data['ret'].to(device)
        return ((actic.v(obs) - ret)**2).mean()

    start_time = time.time()
    reward_record = []

    for epoch in range(epochs):
        obs = env.reset()
        episode_r_list = []
        episode_r = 0
        episode_len = 0
        for t in range(minibatch):
            act, v, logp = actic.step(torch.as_tensor(obs, dtype=torch.float32))
            obs_next, reward, done, _ = env.step(act)
            episode_r += reward
            episode_len += 1
            buffer.store(obs, act, reward, v, logp)
            obs = obs_next

            timeout = episode_len == max_episode
            terminal = done or timeout
            minibath_end = t == minibatch-1
            if terminal or minibath_end:      # current trajectory should stop
                if minibath_end and (not terminal):
                    print('Warning: trajectory cut off by minibath_end at %d step.'
                          %episode_len, flush=True)
                if timeout or minibath_end:
                    _, v, _ = actic.step(torch.as_tensor(obs, dtype=torch.float32))
                else:
                    v = 0
                episode_r_list.append(episode_r)
                buffer.finish_episode(v)
                obs, episode_len, episode_r = env.reset(), 0, 0

        reward_record.append({'episode': epoch, 'mean_ep_reward': np.mean(episode_r_list)})

        print('epoch: ', epoch, '     mean r: ', np.mean(episode_r_list))
        # update
        data = buffer.get()
        actic.to(device)
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)

            kl = pi_info['KL']
            if kl > 1.5 * target_kl:
                print('Early stopping at train pi iterate %d due to reaching max KL.' %i)
                break

            loss_pi.backward()
            pi_optimizer.step()
            #print('epoch:', epoch, '  pi train:', i, '  loss:', loss_pi.item())

        for j in range(train_v_iters):
            v_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            v_optimizer.step()
            #print('epoch:', epoch, '  v train:', j, '  loss:', loss_v.item())
        actic.cpu()
    return actic, reward_record

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    game = 'Acrobot-v1'
    ac_hyper = dict(pi_hidden_sizes=[64,64], v_hidden_sizes=[64,64], activation=nn.ReLU,
                    pi_out_activation=nn.Identity, v_out_activation=nn.Identity)
    acnet , reward_record= ppo(env_fn=game,
                seed=30,
                ac_kwargs=ac_hyper,
                minibatch=10240,
                epochs=100,
                gamma=0.999,
                epsilon=0.2,
                pi_lr=0.0006,
                v_lr=0.018,
                train_pi_iters=80,
                train_v_iters=80,
                lam=0.999,
                max_episode=1000)
    reward_recode = pd.DataFrame(reward_record)
    RESULT_DIR = '/home/wangxu/PycharmProjects/torchprojects/result/'
    #reward_recode.to_csv(joindir(RESULT_DIR, 'ppo-simple-record-{}.csv'.format(game)))
    env = gym.make(game)
    for t in range(10):
        obs = env.reset()
        done = False
        i = 0
        while (not done):
            env.render()
            act = acnet.act(torch.as_tensor(obs, dtype=torch.float32))
            obs, _, done, _ = env.step(act)
            i += 1
            if done or i==1600:
                print('stop at %d step.', format(i))
                env.close()
                break
