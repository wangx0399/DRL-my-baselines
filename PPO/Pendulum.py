import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
import gym
from gym.spaces import Box, Discrete
import scipy.signal
import time
from os.path import join as joindir


class ActorGaussian(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden1=64, hidden2=32):
        super(ActorGaussian, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden1, hidden2)
        self.fc_mu = nn.Linear(hidden2, act_dim)
        self.fc_sigma = nn.Linear(hidden2, act_dim)
        #self.sigma = torch.nn.Parameter(torch.as_tensor(0.5 * np.ones(act_dim, dtype=np.float32)), requires_grad=True)

    def forward(self, obs, act=None):
        x = F.relu(self.fc1(obs))
        pi_mu = self.fc_mu(F.relu(self.fc2(x)))
        pi_sigma = torch.tanh(self.fc_sigma(F.relu(self.fc3(x))))
        pi_sigma = torch.exp(pi_sigma)    # important
        pi = Normal(pi_mu, pi_sigma)
        logp_a = None
        if act is not None:
            logp_a = pi.log_prob(act).sum(axis=-1)
        return pi, logp_a


class ActorCategorical(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden1=64, hidden2=32):
        super(ActorCategorical, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.prob_a = nn.Linear(hidden2, act_dim)

    def forward(self, obs, act=None):
        x = torch.sigmoid(self.prob_a(F.relu(self.fc2(F.relu(self.fc1(obs))))))
        pi = Categorical(probs=x)      # 'probs' need the x is > 0,  'logits' needn't torch.sigmoid()
        logp_a = None
        if act is not None:
            logp_a = pi.log_prob(act)
        return pi, logp_a


class Critic(nn.Module):
    def __init__(self, obs_dim, hidden1=64, hidden2=32):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.v = nn.Linear(hidden2, 1)

    def forward(self, obs):
        v = self.v(F.relu(self.fc2(F.relu(self.fc1(obs)))))
        return torch.squeeze(v, -1)        # important


class MLPActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, hidden1=64, hidden2=32):
        super().__init__()
        obs_dim = observation_space.shape[0]
        if isinstance(action_space, Box):
            self.pi = ActorGaussian(obs_dim, action_space.shape[0], hidden1, hidden2)
        elif isinstance(action_space, Discrete):
            self.pi = ActorCategorical(obs_dim, action_space.n, hidden1, hidden2)
        self.v = Critic(obs_dim, hidden1, hidden2)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi(obs)[0]
            a = pi.sample()
            logp_a = pi.log_prob(a).sum(axis=-1)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]


class PPOBuffer:
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        # size: buffer_size, gamma for r-t-g, gamma*lambda for GAE
        self.obs_buf = np.zeros([size, obs_dim[0]], dtype=np.float32)
        if act_dim == ():
            self.act_buf = np.zeros(size, dtype=np.float32)
        else:
            self.act_buf = np.zeros([size, act_dim[0]], dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.rtg_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.buf_len = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        assert self.ptr < self.buf_len  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew    # reward of every step
        self.val_buf[self.ptr] = val    # initial v function predict
        self.logp_buf[self.ptr] = logp   # log(action probability)
        self.ptr += 1

    def finish_episode(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.ptr)    # a trajectory length
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        # implement discount GAE-Lambda advantage calculation for the policy function
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = scipy.signal.lfilter([1], [1, float(-self.gamma * self.lam)],
                                                        deltas[::-1], axis=0)[::-1]
        # the next line computes discount rewards-to-go, to be targets for the value function
        self.rtg_buf[path_slice] = scipy.signal.lfilter([1], [1, float(-self.gamma)],
                                                        rews[::-1], axis=0)[::-1][:-1]
        # update the pointer
        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.buf_len  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next three lines implement the GAE_advantage normalization trick
        adv_mean = np.mean(np.array(self.adv_buf, dtype=np.float32))
        adv_std = np.std(np.array(self.adv_buf, dtype=np.float32))
        self.adv_buf = (self.adv_buf - adv_mean) / (adv_std + 1e-8)
        data = dict(obs=self.obs_buf, act=self.act_buf, rtg=self.rtg_buf,
                    adv=self.adv_buf, logp=self.logp_buf)  # define dictionary
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}  # array to tensor


def ppo(env_fn, actor_critic=MLPActorCritic, ac_kwargs=dict(), seed=0, minibatch=4000,
        epochs=50, gamma=0.99, epsilon=0.2, pi_lr=3e-4, v_lr=1e-3, train_pi_iters=80, train_v_iters=80,
        lam=0.97, max_episode=1000, target_kl=0.01):

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make(env_fn)#.unwrapped
    obs_sp = env.observation_space
    act_sp = env.action_space
    actic = actor_critic(obs_sp, act_sp, **ac_kwargs)
    buffer = PPOBuffer(obs_sp.shape, act_sp.shape, minibatch, gamma, lam)
    pi_optimizer = Adam(actic.pi.parameters(), lr=pi_lr)
    v_optimizer = Adam(actic.v.parameters(), lr=v_lr)

    start_time = time.time()
    reward_record = []

    for epoch in range(epochs):
        # collect buffer data, include many tarjectories
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
            minibath_end = t == minibatch - 1
            if terminal or minibath_end:  # current trajectory should stop
                # for endless game, limit max length of episode
                if minibath_end and (not terminal):
                    print('Warning: trajectory cut off by minibath_end at %d step.'
                          % episode_len, flush=True)
                if timeout or minibath_end:
                    _, v, _ = actic.step(torch.as_tensor(obs, dtype=torch.float32))
                else:
                    v = 0
                episode_r_list.append(episode_r)
                buffer.finish_episode(v)
                obs, episode_len, episode_r = env.reset(), 0, 0

        reward_record.append({'episode': epoch, 'mean_ep_reward': np.mean(episode_r_list)})
        print('epoch: {:.0f}          mean r: {:.4f}'.format(epoch, np.mean(episode_r_list)))
        # compute GAE--advantage and discount reward-to-go
        data = buffer.get()
        # train actic.pi and actic.v
        actic.to(device)
        obs, act, adv, logp_old, rtg = data['obs'].to(device), data['act'].to(device), data['adv'].to(device), \
                                            data['logp'].to(device), data['rtg'].to(device)
        print(data['obs'].mean(axis=0))
        for i in range(train_pi_iters):
            pi, logp_new = actic.pi(obs, act)
            ratio = torch.exp(logp_new - logp_old)
            clip = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * adv
            loss_pi = -(torch.min(ratio * adv, clip)).mean()
            # extra info
            approx_kl = (logp_old - logp_new).mean().item()
            entropy = pi.entropy().mean().item()
            kl = approx_kl
            if kl > 1.5 * target_kl:
                print('Early stopping at train pi iterate %d due to reaching max KL.' % i)
                break
            pi_optimizer.zero_grad()
            loss_pi.backward()
            pi_optimizer.step()

        for j in range(train_v_iters):
            v_optimizer.zero_grad()
            loss_v = ((actic.v(obs) - rtg) ** 2).mean()
            loss_v.backward()
            v_optimizer.step()
        actic.cpu()

    return actic, reward_record

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    game = 'Pendulum-v0'  # LunarLander-v2BipedalWalkerHardcore-v3
    acnet, reward_record = ppo(env_fn=game,
                seed=4321,
                ac_kwargs=dict(hidden1=64, hidden2=64),
                minibatch=20480,
                epochs=100,
                gamma=0.999,
                epsilon=0.2,
                pi_lr=0.0004,
                v_lr=0.008,
                train_pi_iters=80,
                train_v_iters=80,
                lam=0.999,
                max_episode=2000)
    reward_recode = pd.DataFrame(reward_record)
    RESULT_DIR = '/home/wangxu/PycharmProjects/torchprojects/result/'
    #reward_recode.to_csv(joindir(RESULT_DIR, 'pendulum-record-{}.csv'.format(game)))
    env = gym.make(game).unwrapped
    for t in range(10):
        obs = env.reset()
        done = False
        i = 0
        while (not done):
            env.render()
            act = acnet.act(torch.as_tensor(obs, dtype=torch.float32))
            obs, _, done, _ = env.step(act)
            i += 1
            if done or i==1000:
                print('test {:.0f} time,  stop at {:.0f} step.'.format(t+1, i))
                env.close()
                break

