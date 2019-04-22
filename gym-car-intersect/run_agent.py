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

from gym_car_intersect.envs.my_env import CarRacing
def make_env_recording(epoch):
    env = CarRacing(agent = True, num_bots = 1, track_form = 'X',
                    write = False, data_path = 'car_racing_positions.csv',
                    start_file = True, training_epoch = epoch)

    env = MaxAndSkipEnv(env)
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, 4)

    return ScaledFloatFrame(env)

from gym_car_intersect.envs.my_env_discrete import CarRacingDiscrete
def make_disc_env_recording(epoch):
    env = CarRacingDiscrete(agent = True, num_bots = 1, track_form = 'X',
                            write = False, data_path = 'car_racing_positions.csv',
                            start_file = True, training_epoch = epoch)

    env = MaxAndSkipEnv(env)
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, 4)

    return ScaledFloatFrame(env)

env = make_env()
disc_env = make_disc_env()

# type(env.action_space)
# type(disc_env.action_space)
# isinstance(env.action_space, gym.spaces.box.Box)

# obs_shape = env.observation_space.shape
# actions_shape = env.action_space.shape
#
# print("Observation shape:", obs_shape)
# print("Num actions:", actions_shape)

# env.reset()
# for _ in range(100):
#     action = [10, -1, -1]
#     s, _, _, _ = env.step(action)
#
# # plt.title('Game image')
# # plt.imshow(env.render('rgb_array'))
# # plt.show()
#
# plt.title('Game image')
# plt.imshow(s[0,...])
# plt.show()

import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
print("device: ", device)

n_parallel_games = 5
gamma = 0.99

# A2C discrete training:
def train_discrete_a2c():
    obs_shape = disc_env.observation_space.shape
    actions_shape = disc_env.action_space.n

    print("Observation shape:", obs_shape)
    print("Num actions:", actions_shape)

    agent = A2CAgentDiscrete(disc_env, device).to(device)
    opt = torch.optim.Adam(agent.parameters(), lr=1e-4)

    # state = [disc_env.reset()]
    # logits, q = agent.step(state)
    # print("action logits:\n", logits)
    # print("state values:\n", q)
    #
    # from gym import wrappers
    #
    # env_monitor = wrappers.Monitor(disc_env, directory="videos", force=True)
    # rw = evaluate(agent, env_monitor, n_games=1,)
    # env_monitor.close()
    # print(rw)

    # for each of n_parallel_games, take 10 steps
    pool = EnvPool(agent, make_disc_env, n_parallel_games=5)
    # rollout_obs, rollout_actions, rollout_rewards, rollout_mask = pool.interact(10)
    #
    # print("Actions shape:", rollout_actions.shape)
    # print("Rewards shape:", rollout_rewards.shape)
    # print("Mask shape:", rollout_mask.shape)
    # print("Observations shape: ", rollout_obs.shape)
    #
    # rollout_obs, rollout_actions, rollout_rewards, rollout_mask = pool.interact(10)
    # loss = train_discrete_a2c_on_rollout(agent, opt, rollout_obs, rollout_actions,
    #                         rollout_rewards, rollout_mask, device = device)
    #
    # print("Loss: ", loss)

    from tqdm import trange
    from tensorboardX import SummaryWriter
    # with SummaryWriter(comment='RNN') as w:
    #     w.add_graph(agent, torch.zeros(1, *obs_shape).to(device), verbose=True)
    writer = SummaryWriter()
    for i in trange(10000):
        rollout_obs, rollout_actions, rollout_rewards, rollout_mask = pool.interact(10)
        loss, entropy_loss = train_discrete_a2c_on_rollout(agent, opt, rollout_obs, rollout_actions,
                                rollout_rewards, rollout_mask, device = device)
        writer.add_scalar("DiscA2C_train_loss: ", loss.item(), i)
        writer.add_scalar("DiscA2C_entropy_loss: ", entropy_loss.item(), i)
        if i % 10 == 0:
            eval_rwd = np.mean(evaluate(agent, make_disc_env_recording(i), n_games=1))
            writer.add_scalar("DiscA2C_rewards: ", eval_rwd, i//10)
        if i%100 == 0:
            torch.save({"agent": agent.state_dict(),
                        "opt" : opt.state_dict(),
                        }, "checkpoint_disc_a2c.pt")
    writer.close()

# A2C training:
def train_a2c():
    agent = A2CAgent(env, device).to(device)
    opt = torch.optim.Adam(agent.parameters(), lr=1e-4)

    # state = [env.reset()]
    # mu, var, q = agent.step(state)
    # print("action mu:\n", mu)
    # print("action var:\n", var)
    # print("state values:\n", q)
    #
    # from gym import wrappers
    #
    # env_monitor = wrappers.Monitor(env, directory="videos", force=True)
    # rw = evaluate(agent, env_monitor, n_games=1,)
    # env_monitor.close()
    # print(rw)

    # for each of n_parallel_games, take 10 steps
    pool = EnvPool(agent, make_env, n_parallel_games=5)
    # rollout_obs, rollout_actions, rollout_rewards, rollout_mask = pool.interact(10)
    #
    # print("Actions shape:", rollout_actions.shape)
    # print("Rewards shape:", rollout_rewards.shape)
    # print("Mask shape:", rollout_mask.shape)
    # print("Observations shape: ", rollout_obs.shape)

    # rollout_obs, rollout_actions, rollout_rewards, rollout_mask = pool.interact(10)
    # loss = train_a2c_on_rollout(agent, opt, rollout_obs, rollout_actions,
    #                         rollout_rewards, rollout_mask, device = device)
    #
    # print("Loss: ", loss)

    from tqdm import trange
    from tensorboardX import SummaryWriter
    # with SummaryWriter(comment='RNN') as w:
    #     w.add_graph(agent, torch.zeros(1, *obs_shape).to(device), verbose=True)
    writer = SummaryWriter()
    for i in trange(10000):
        rollout_obs, rollout_actions, rollout_rewards, rollout_mask = pool.interact(10)
        loss, entropy_loss = train_a2c_on_rollout(agent, opt, rollout_obs, rollout_actions,
                                rollout_rewards, rollout_mask, device = device)
        writer.add_scalar("A2C_train_loss: ", loss.item(), i)
        writer.add_scalar("A2C_entropy_loss: ", entropy_loss.item(), i)
        if i % 10 == 0:
            eval_rwd = np.mean(evaluate(agent, make_env_recording(i), n_games=1))
            writer.add_scalar("A2C_rewards: ", eval_rwd, i//10)
        if i%100 == 0:
            torch.save({"agent": agent.state_dict(),
                        "opt" : opt.state_dict(),
                        }, "checkpoint_a2c.pt")
    writer.close()

# OC training:
def train_oc():
    agent = OptionCriticAgent(env, num_options = 4, device = device).to(device)
    opt = torch.optim.Adam(agent.parameters(), lr=1e-4)

    target_network = OptionCriticAgent(env, num_options = 4, device = device).to(device)
    target_network.load_state_dict(agent.state_dict())

    # state = [env.reset()]
    # mu, var, q, beta = agent.step(state)
    # print("action mu:\n", mu)
    # print("action var:\n", var)
    # print("state values:\n", q)
    # print("terminal state:\n", beta)
    #

    # from gym import wrappers
    #
    # env_monitor = wrappers.Monitor(env, directory="videos", force=True)

    # checkpoint = torch.load("checkpoint_oc.pt")
    #
    # agent.load_state_dict(checkpoint["agent"])
    # opt.load_state_dict(checkpoint["opt"])

    # loss_history = checkpoint["loss_history"]
    # editdist_history = checkpoint["editdist_history"]
    # entropy_history = checkpoint["entropy_history"]

    # rw = evaluate_oc(agent, env_monitor, n_games=1,)
    # env_monitor.close()
    # print(rw)

    # for each of n_parallel_games, take 10 steps
    epsilon = (max(1-i, 0.1) for i in np.arange(0, 1e3, 1e-4))
    pool = EnvPoolOC(agent, make_env, n_parallel_games=5, epsilon=epsilon)
    # rollout_obs, rollout_actions, rollout_options, rollout_rewards, rollout_mask = pool.interact(10)
    #
    # print("Actions shape:", rollout_actions.shape)
    # print("Rewards shape:", rollout_rewards.shape)
    # print("Options shape:", rollout_options.shape)
    # print("Mask shape:", rollout_mask.shape)
    # print("Observations shape: ", rollout_obs.shape)

    # rollout_obs, rollout_actions, rollout_options, rollout_rewards, rollout_mask = pool.interact(10)
    # loss = train_oc_on_rollout(agent, target_network, opt, rollout_obs, rollout_actions,
    #                            rollout_options, rollout_rewards, rollout_mask,
    #                            device = device)
    # print("Loss: ", loss)

    from tqdm import trange
    from tensorboardX import SummaryWriter
    # with SummaryWriter(comment='RNN') as w:
    #     w.add_graph(agent, torch.zeros(1, *obs_shape).to(device), verbose=True)
    writer = SummaryWriter()
    for i in trange(10000):
        rollout_obs, rollout_actions, rollout_options, rollout_rewards, rollout_mask = pool.interact(10)
        loss, entropy_loss, beta_loss = train_oc_on_rollout(agent, target_network, opt, rollout_obs, rollout_actions,
                                   rollout_options, rollout_rewards, rollout_mask,
                                   device = device)
        writer.add_scalar("OC_train_loss: ", loss.item(), i)
        writer.add_scalar("OC_entropy_loss: ", entropy_loss.item(), i)
        writer.add_scalar("OC_beta_loss: ", beta_loss.item(), i)
        if i % 10 == 0:
            eval_rwd = np.mean(evaluate_oc(agent, make_env_recording(i), n_games=1))
            writer.add_scalar("OC_rewards: ", eval_rwd, i//10)
        if i % 50 == 0:
            target_network.load_state_dict(agent.state_dict())
        if i % 100 == 0:
            torch.save({"agent": agent.state_dict(),
                        "target": target_network.state_dict(),
                        "opt" : opt.state_dict(),
                        }, "checkpoint_oc.pt")
    writer.close()


if __name__ == "__main__":
    import argparse
    train_discrete_a2c()
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--train_oc", default=False, action="store_true", help="train oc or a2c")
    # args = parser.parse_args()
    #
    # if args.train_oc:
    #     print('oc')
    #     train_oc()
    # else:
    #     print("a2c")
    #     train_a2c()
