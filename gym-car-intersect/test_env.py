import torch
import torch.nn as nn

t = torch.tensor([[0, 0], [1, 2], [2, 3]])
idx = torch.tensor([0, 1, 2, 0, 0, 1])
t[idx]

import gym
import gym_car_intersect
import numpy as np

env = gym.make('CarIntersect-v1')
env.reset()
for _ in range(500):
    env.render()
    a = np.random.choice(5)
    _, _, done, _ = env.step(a)
    if done:
        env.reset()
