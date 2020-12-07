import torch
from torch import Tensor
import numpy as np
import pandas as pd
from ppo import ActorCritic
from ppo import ZFilter
from ppo import RunningStat
import gym
from os.path import join as joindir

model_path = '/home/wangxu/PycharmProjects/torchprojects/result/'
game = 'Humanoid-v2'
env = gym.make(game)
num_states = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]
network = ActorCritic(num_states, num_actions)
network.eval()
checkpoint = torch.load(joindir(model_path, 'ppo-modelpara-{}.pth'.format(game)))
network.load_state_dict(checkpoint['model_state_dict'])
normalized_state = checkpoint['normalized_state']
#print(normalized_state.rs.mean)
#print(normalized_state.rs.std)

for t in range(3):
    state = env.reset()
    done = False
    i = 0
    while (not done):
        env.render()
        state = normalized_state(state)
        action_mean, action_logstd, value = network(Tensor(state).unsqueeze(0))
        action, logproba = network.select_action(action_mean, action_logstd)
        action = action.data.numpy()[0]
        state, _, done, _ = env.step(action)
        i += 1

        if done or i==4000:
            print('test {:.0f} time,  stop at {:.0f} step.'.format(t+1, i))
            env.close()
            break