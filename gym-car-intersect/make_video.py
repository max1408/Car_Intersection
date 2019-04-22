from __future__ import print_function, division
from IPython.core import display
import matplotlib.pyplot as plt
import numpy as np

import os
if type(os.environ.get("DISPLAY")) is not str or len(os.environ.get("DISPLAY")) == 0:
    os.system("bash ../xvfb start")

import gym
from rl_agents.agents import *
from rl_agents.wrappers import *
from rl_agents.train import *
import gym_car_intersect


def make_env():
    env = gym.make('CarIntersect-v0')

    env = MaxAndSkipEnv(env)
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, 4)

    return ScaledFloatFrame(env)

def make_disc_env():
    env = gym.make('CarIntersect-v1')

    env = MaxAndSkipEnv(env)
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, 4)

    return ScaledFloatFrame(env)

env = make_env()
disc_env = make_disc_env()

obs_shape = env.observation_space.shape
actions_shape = env.action_space.shape

import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
print("device: ", device)

n_parallel_games = 5
gamma = 0.99

# Discrete A2C training:
def video_disc_a2c():
    agent = A2CAgentDiscrete(disc_env, device).to(device)

    from gym import wrappers

    env_monitor = wrappers.Monitor(disc_env, directory="videos/disc_a2c", force=True)

    checkpoint = torch.load("checkpoint_disc_a2c.pt")
    agent.load_state_dict(checkpoint["agent"])

    rw = evaluate(agent, env_monitor, n_games=2,)
    env_monitor.close()
    print(rw)

# A2C training:
def video_a2c():
    agent = A2CAgent(env, device).to(device)

    from gym import wrappers

    env_monitor = wrappers.Monitor(env, directory="videos/a2c", force=True)

    checkpoint = torch.load("checkpoint_a2c.pt")
    agent.load_state_dict(checkpoint["agent"])

    rw = evaluate(agent, env_monitor, n_games=2,)
    env_monitor.close()
    print(rw)

# OC training:
def video_oc():
    agent = OptionCriticAgent(env, num_options = 4, device = device).to(device)

    from gym import wrappers

    env_monitor = wrappers.Monitor(env, directory="videos/oc", force=True)

    checkpoint = torch.load("checkpoint_oc.pt")
    agent.load_state_dict(checkpoint["agent"])

    rw = evaluate_oc(agent, env_monitor, n_games=2,)
    env_monitor.close()
    print(rw)

if __name__ == "__main__":
    import argparse
    video_disc_a2c()

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--show_oc", default=False, action="store_true", help="train oc or a2c")
    # args = parser.parse_args()
    #
    # if args.show_oc:
    #     print('oc')
    #     video_oc()
    # else:
    #     print("a2c")
    #     video_a2c()
