from IPython.core import display
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class OptionCriticNet(nn.Module):
    def __init__(self, feature_size, num_actions, num_options):
        super().__init__()
        self.q = nn.Linear(feature_size, num_options)
        self.pi_mu = nn.Linear(feature_size, num_options*num_actions)
        self.pi_var = nn.Linear(feature_size, num_options*num_actions)
        self.beta = nn.Linear(feature_size, num_options)
        self.num_options = num_options
        self.num_actions = num_actions

    def forward(self, x):
        q = self.q(x)
        mu = torch.tanh(self.pi_mu(x))
        mu = mu.view(-1, self.num_options, self.num_actions)
        var = F.softplus(self.pi_var(x))
        var = var.view(-1, self.num_options, self.num_actions)
        beta = torch.sigmoid(self.beta(x))

        return mu, var, q, beta


class A2CAgentDiscrete(nn.Module):
    def __init__(self, env, device='cpu'):
        """A simple continuous actor-critic agent"""
        super(self.__class__, self).__init__()
        self.obs_shape = env.env.observation_space.shape
        self.action_shape = env.action_space.n
        self.env = env
        self.device = device

        # Preparing network for an agent:
        self.conv = nn.Sequential(nn.Conv2d(4, 32, kernel_size=(8, 8), stride=(4, 4)),
                                  nn.ELU(),
                                  nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2)),
                                  nn.ELU(),
                                  nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1)),
                                  nn.ELU(),
                                  Flatten())
        size = np.prod(self.conv(torch.zeros(1, *self.obs_shape)).shape[1:])

        self.fc = nn.Linear(size, 512)
        self.relu = nn.ReLU()

        self.logits = nn.Linear(512, self.action_shape)
        self.state_value = nn.Linear(512, 1)

    def forward(self, obs_t):
        """
        Takes agent's observation.
        Returns expectation, variance, next value state.
        """
        x = self.conv(obs_t)
        x = self.fc(x)
        x = self.relu(x)

        logits = self.logits(x)
        q = self.state_value(x)

        return logits, q

    def sample_actions(self, agent_outputs):
        """Return np arrays of actions given numeric agent outputs."""
        logits, q = agent_outputs
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1)[:, 0].cpu().data.numpy()

    def step(self, obs_t):
        """It's forward pass for numpy array of observation"""
        obs_t = torch.tensor(np.asarray(obs_t), dtype=torch.float32).to(self.device)
        logits, q = self.forward(obs_t)
        return logits.detach(), q.detach()


class A2CAgent(nn.Module):
    def __init__(self, env, device='cpu'):
        """A simple continuous actor-critic agent"""
        super(self.__class__, self).__init__()
        self.obs_shape = env.env.observation_space.shape
        self.action_shape = env.action_space.shape[0]
        self.env = env
        self.device = device

        # Preparing network for an agent:
        self.conv = nn.Sequential(nn.Conv2d(4, 32, kernel_size=(8, 8), stride=(2, 2)),
                                  nn.ELU(),
                                  nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2)),
                                  nn.ELU(),
                                  nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1)),
                                  nn.ELU(),
                                  Flatten())
        size = np.prod(self.conv(torch.zeros(1, *self.obs_shape)).shape[1:])

        self.fc = nn.Linear(size, 512)
        self.relu = nn.ReLU()

        self.mu = nn.Linear(512, self.action_shape)
        self.var = nn.Linear(512, self.action_shape)
        self.state_value = nn.Linear(512, 1)

    def forward(self, obs_t):
        """
        Takes agent's observation.
        Returns expectation, variance, next value state.
        """
        x = self.conv(obs_t)
        x = self.fc(x)
        x = self.relu(x)

        mu = torch.tanh(self.mu(x))
        var = F.softplus(self.var(x))
        q = self.state_value(x)

        return mu, var, q

    def sample_actions(self, agent_outputs):
        """Return np arrays of actions given numeric agent outputs."""
        mu, var, q = agent_outputs
        mu_np, sigma_np = mu.detach().cpu().numpy(), torch.sqrt(var).detach().cpu().numpy()
        actions = np.random.normal(mu_np, sigma_np)
        actions = np.clip(actions, self.env.action_space.low, self.env.action_space.high)
        return actions

    def step(self, obs_t):
        """It's forward pass for numpy array of observation"""
        obs_t = torch.tensor(np.asarray(obs_t), dtype=torch.float32).to(self.device)
        mu, var, q = self.forward(obs_t)
        return mu.detach(), var.detach(), q.detach()


class OptionCriticAgent(nn.Module):
    def __init__(self, env, num_options, device='cpu'):
        """A simple continuous actor-critic agent"""
        super(self.__class__, self).__init__()
        self.obs_shape = env.env.observation_space.shape
        self.action_shape = env.action_space.shape[0]
        self.env = env
        self.device = device
        self.num_options = num_options

        # Preparing network for an agent:
        self.conv = nn.Sequential(nn.Conv2d(4, 32, kernel_size=(8, 8), stride=(2, 2)),
                                  nn.ELU(),
                                  nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2)),
                                  nn.ELU(),
                                  nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1)),
                                  nn.ELU(),
                                  Flatten())
        size = np.prod(self.conv(torch.zeros(1, *self.obs_shape)).shape[1:])

        self.fc = nn.Linear(size, 512)
        self.relu = nn.ReLU()

        self.option_critic = OptionCriticNet(512, self.action_shape, self.num_options)

    def forward(self, obs_t):
        """
        Takes agent's observation.
        Returns expectation, variance, next value state.
        """
        x = self.conv(obs_t)
        x = self.fc(x)
        x = self.relu(x)

        mu, var, q, beta = self.option_critic(x)
        return mu, var, q, beta

    @staticmethod
    def epsilon_greedy(epsilon, q):
        q = q.data.cpu().numpy()
        idx = np.random.rand(q.shape[0]) < epsilon
        new_options = np.argmax(q, axis=-1)
        new_options[idx] = np.random.randint(0, q.shape[1], (idx.sum(),))
        # if np.random.rand() < epsilon:
        #     return torch.randint(0, q.shape[-1], (x.shape[0],))
        # else:
        #     return torch.argmax(q, dim=-1)
        return new_options

    def sample_actions(self, agent_outputs, options, epsilon=0.05):
        """Return np arrays of actions given numeric agent outputs."""
        mu, var, q, beta = agent_outputs
        mu, sigma = mu.detach().cpu().numpy(), torch.sqrt(var).detach().cpu().numpy()
        beta = beta.detach().cpu().numpy()
        # if option terminates in a state - choose new option:
        terminal_state = beta[range(beta.shape[0]), options] > 0.5
        options[terminal_state] = self.epsilon_greedy(epsilon, q)[terminal_state]
        # if options.shape[0] == 1:
        #     options = options.item()
        # else:
        #     options = options.detach().cpu().numpy()
        actions = np.random.normal(mu[range(mu.shape[0]),options],
                                   sigma[range(sigma.shape[0]),options])
        actions = np.clip(actions, self.env.action_space.low, self.env.action_space.high)
        return actions

    def step(self, obs_t):
        """It's forward pass for numpy array of observation"""
        obs_t = torch.tensor(np.asarray(obs_t), dtype=torch.float32).to(self.device)
        mu, var, q, beta = self.forward(obs_t)
        return mu.detach(), var.detach(), q.detach(), beta.detach()
