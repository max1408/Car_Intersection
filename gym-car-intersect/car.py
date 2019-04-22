import gym
import gym_car_intersect
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from a2c import wrappers

def make_env():
    env = gym.make('CarIntersect-v0')
    env = wrappers.MaxAndSkipEnv(env)
    env = wrappers.ProcessFrame84(env)
    env = wrappers.ImageToPyTorch(env)
    env = wrappers.BufferWrapper(env, 4)
    return wrappers.ScaledFloatFrame(env)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class SimpleRecurrentAgent(nn.Module):
    def __init__(self, obs_shape, actions_shape, reuse=False):
        """A simple actor-critic agent"""
        super(self.__class__, self).__init__()

        self.conv = nn.Sequential(nn.Conv2d(4, 32, kernel_size=(3, 3), stride=(2, 2)),
                                  nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2)),
                                  nn.ELU(),
                                  Flatten())

        size = np.prod(self.conv(torch.zeros(1, *obs_shape)).shape[1:])

        self.hid = nn.Linear(size, 128)
        self.rnn = nn.LSTMCell(128, 128)

        self.mu = nn.Linear(128, actions_shape[0])
        self.var = nn.Linear(128, actions_shape[0])
        self.state_value = nn.Linear(128, 1)

    def forward(self, prev_state, obs_t):
        """
        Takes agent's previous step and observation,
        returns next state and whatever it needs to learn (tf tensors)
        """

        # YOUR CODE: apply the whole neural net for one step here.
        # See docs on self.rnn(...)
        # the recurrent cell should take the last feedforward dense layer as input
        #<YOUR CODE >
        x = self.conv(obs_t)

        x = self.hid(x)
        x = F.relu(x)

        new_state = self.rnn(x, prev_state) #<YOUR CODE >
        mu_v = torch.tanh(self.mu(new_state[0])) #<YOUR CODE >
        var_v = F.softplus(self.var(new_state[0]))
        state_value = self.state_value(new_state[0]) #<YOUR CODE >

        return new_state, (mu_v, var_v, state_value)

    def get_initial_state(self, batch_size):
        """Return a list of agent memory states at game start. Each state is a np array of shape [batch_size, ...]"""
        return torch.zeros((batch_size, 128)), torch.zeros((batch_size, 128))

    def sample_actions(self, agent_outputs):
        """pick actions given numeric agent outputs (np arrays)"""
        mu_v, var_v, state_values = agent_outputs
        mu, var = mu_v.detach().cpu().numpy(), var_v.detach().cpu().numpy()
        actions = np.random.normal(mu, var)
        actions = np.clip(actions, env.action_space.low, env.action_space.high)
        return actions

    def step(self, prev_state, obs_t):
        """ like forward, but obs_t is a numpy array """
        obs_t = torch.tensor(np.asarray(obs_t), dtype=torch.float32)
        (h, c), (m, v, s) = self.forward(prev_state, obs_t)
        return (h.detach(), c.detach()), (m.detach(), v.detach(), s.detach())

def evaluate(agent, env, n_games=1):
    """Plays an entire game start to end, returns session rewards."""

    game_rewards = []
    for _ in range(n_games):
        # initial observation and memory
        observation = env.reset()
        prev_memories = agent.get_initial_state(1)

        total_reward = 0
        while True:
            new_memories, readouts = agent.step(
                prev_memories, observation[None, ...])
            action = agent.sample_actions(readouts)

            observation, reward, done, info = env.step(action[0])

            total_reward += reward
            prev_memories = new_memories
            if done:
                break

        game_rewards.append(total_reward)
    return game_rewards


if __name__ == '__main__':
    env = make_env()

    obs_shape, actions_shape = env.observation_space.shape, env.action_space.shape

    agent = SimpleRecurrentAgent(obs_shape, actions_shape)
    checkpoint = torch.load('a2c_0.pt')
    agent.load_state_dict(checkpoint['model_state_dict'])
    agent.eval()

    # Play video:
    from gym import wrappers

    env_monitor = wrappers.Monitor(env, directory="videos", force=True)
    rw = evaluate(agent, env_monitor, n_games=1,)
    env_monitor.close()
    print(rw)

    # state = [env.reset()]
    # memory = agent.get_initial_state(1)
    #
    # for i in range(2000):
    #     env.render()
    #     memory, agent_output = agent.step(memory, state)
    #     actions = agent.sample_actions(agent_output)
    #     state, reward, done, _ = env.step(actions[0]) # take a random action
    #     # print(state.shape)
    #     # plt.imshow(state)
    #     # plt.show()
    #
    #     state = [state]
    #
    #     if done:
    #         state = [env.reset()]
