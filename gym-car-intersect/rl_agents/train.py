import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym

# TRAIN ACTOR-CRITIC:
class EnvPool(object):
    """
    A thin wrapper for openAI gym environments that maintains a set of parallel
    games and has a method to generate interaction sessions given agent one-step
    applier function.
    """
    def __init__(self, agent, make_env, n_parallel_games=1):
        """
        A special class that handles training on multiple parallel sessions
        and is capable of some auxilary actions like evaluating agent on one
        game session (See .evaluate()).

        :param agent: Agent which interacts with the environment.
        :param make_env: Factory that produces environments OR a name of the
                         gym environment.
        :param n_games: Number of parallel games. One game by default.
        :param epsilon: In case of OptionCritic we need eps-greedy strategy.
                        Pass generator.
        """
        # Create atari games.
        self.agent = agent
        self.make_env = make_env
        self.envs = [self.make_env() for _ in range(n_parallel_games)]

        # Initial observations.
        self.prev_observations = [env.reset() for env in self.envs]

        # Whether particular session has just been terminated and needs restarting.
        self.just_ended = [False] * len(self.envs)

    def interact(self, n_steps=100, verbose=False):
        """Generate interaction sessions with ataries
        (openAI gym atari environments)
        Sessions will have length n_steps. Each time one of games is finished,
        it is immediately getting reset and this time is recorded in
        is_alive_log (See returned values).

        :param n_steps: Length of an interaction.
        :returns: observation_seq, action_seq, reward_seq, is_alive_seq
        :rtype: a bunch of tensors [batch, tick, ...]
        """

        def env_step(i, action):
            if not self.just_ended[i]:
                new_observation, cur_reward, is_done, info = self.envs[i].step(action)
                if is_done:
                    # Game ends now, will finalize on next tick.
                    self.just_ended[i] = True
                # note: is_alive=True in any case because environment is still alive
                # (last tick alive) in our notation.
                return new_observation, cur_reward, True, info
            else:
                # Reset environment, get new observation to be used on next tick.
                new_observation = self.envs[i].reset()
                if verbose:
                    print("env %i reloaded" % i)
                self.just_ended[i] = False

                return new_observation, 0, False, {'end': True}

        history_log = []

        for i in range(n_steps - 1):
            readout = self.agent.step(self.prev_observations)
            actions = self.agent.sample_actions(readout)
            new_obs, cur_rwds, is_alive, infos = zip(*map(env_step, range(len(self.envs)), actions))
            # Append data tuple for this tick
            history_log.append((self.prev_observations, actions, cur_rwds, is_alive))

            self.prev_observations = new_obs

        #add last observation
        if isinstance(self.envs[0].action_space, gym.spaces.box.Box):
            dummy_actions = np.zeros((len(self.envs), self.envs[0].action_space.shape[0]))
        else:
            dummy_actions = [0] * len(self.envs)
        dummy_rewards = [0] * len(self.envs)
        dummy_mask = [1] * len(self.envs)
        history_log.append((self.prev_observations, dummy_actions, dummy_rewards, dummy_mask))
        # cast to numpy arrays, transpose from [time, batch, ...] to [batch, time, ...]
        history_log = [np.array(tensor).swapaxes(0, 1) for tensor in zip(*history_log)]
        observation_seq, action_seq, reward_seq, is_alive_seq = history_log

        return observation_seq, action_seq, reward_seq, is_alive_seq


def evaluate(agent, env, n_games=1):
    """Plays an entire game start to end, returns session rewards."""

    game_rewards = []
    for _ in range(n_games):
        # initial observation
        observation = env.reset()
        total_reward = 0

        while True:
            readouts = agent.step(observation[None, ...])
            action = agent.sample_actions(readouts)
            observation, reward, done, info = env.step(action[0])
            total_reward += reward
            if done:
                break

        game_rewards.append(total_reward)

    return game_rewards


def calc_logprob(mu, var, actions):
    p1 = - ((mu - actions) ** 2) / (2*var.clamp(min=1e-3))
    p2 = - torch.log(torch.sqrt(2 * np.pi * var))
    return (p1 + p2).sum(-1)


def train_a2c_on_rollout(agent, optimizer, states, actions, rewards, is_not_done, device = 'cpu', gamma=0.99):
    """
    Takes a sequence of states, actions and rewards produced by generate_session.
    Updates agent's weights by following the policy gradient above.
    Please use Adam optimizer with default parameters.
    """

    # shape: [batch_size, time, c, h, w]
    states = torch.tensor(np.asarray(states), dtype=torch.float32).to(device)
    # shape: [batch_size, time]
    actions = torch.tensor(np.array(actions), dtype=torch.float32).to(device)
    # shape: [batch_size, time]
    rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(device)
    # shape: [batch_size, time]
    is_not_done = torch.tensor(np.array(is_not_done), dtype=torch.float32).to(device)
    rollout_length = rewards.shape[1] - 1

    mu = []
    var = []
    state_values = []
    for t in range(rewards.shape[1]):
        obs_t = states[:, t]
        mu_v, var_v, q_v = agent(obs_t)

        mu.append(mu_v)
        var.append(var_v)
        state_values.append(q_v)

    mu = torch.stack(mu, dim=1)
    var = torch.stack(var, dim=1)
    state_values = torch.stack(state_values, dim=1)

    # select log-probabilities for chosen actions, log pi(a_i|s_i)
    logprobas_for_actions = calc_logprob(mu, var, actions)

    # Now let's compute two loss components:
    # 1) Policy gradient objective.
    # Notes: Please don't forget to call .detach() on advantage term. Also please use mean, not sum.
    # it's okay to use loops if you want
    J_hat = 0  # policy objective as in the formula for J_hat

    # 2) Temporal difference MSE for state values
    # Notes: Please don't forget to call on V(s') term. Also please use mean, not sum.
    # it's okay to use loops if you want
    value_loss = 0

    cumulative_returns = state_values[:, -1].detach()

    for t in reversed(range(rollout_length)):
        # current rewards
        r_t = rewards[:, t, None]
        # current state values
        V_t = state_values[:, t]
        # next state values
        V_next = state_values[:, t + 1].detach()
        # log-probability of a_t in s_t
        logpi_a_s_t = logprobas_for_actions[:, t, None]
        # update G_t = r_t + gamma * G_{t+1} as we did in week6 reinforce
        cumulative_returns = G_t = r_t + gamma * cumulative_returns
        # Compute temporal difference error (MSE for V(s))
        value_loss += (r_t + gamma*V_next - V_t).pow(2).mean()
        # compute advantage A(s_t, a_t) using cumulative returns and V(s_t) as baseline
        advantage = cumulative_returns - V_t
        advantage = advantage.detach()
        # compute policy pseudo-loss aka -J_hat.
        J_hat += (logpi_a_s_t*advantage).mean()
    # regularize with entropy
    entropy_reg = ((torch.log(2*np.pi*var) + 1)/2).sum(-1).mean()
    # add-up three loss components and average over time
    loss = (-J_hat / rollout_length) + (value_loss / rollout_length) + (-0.01 * entropy_reg)
    # Gradient descent step
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return loss.data.cpu().numpy(), entropy_reg.data.cpu().numpy()

# TRAIN ACTOR-CRITIC DISCRETE:
def to_one_hot(y, n_dims=None, device='cpu'):
    """ Take an integer tensor and convert it to 1-hot matrix. """
    y_tensor = y.view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).to(device)
    y_one_hot = y_one_hot.scatter_(1, y_tensor, 1)
    return y_one_hot

def train_discrete_a2c_on_rollout(agent, optimizer, states, actions, rewards, is_not_done, device = 'cpu', gamma=0.99):
    """
    Takes a sequence of states, actions and rewards produced by generate_session.
    Updates agent's weights by following the policy gradient above.
    Please use Adam optimizer with default parameters.
    """
    n_actions = agent.action_shape
    # shape: [batch_size, time, c, h, w]
    states = torch.tensor(np.asarray(states), dtype=torch.float32).to(device)
    # shape: [batch_size, time]
    actions = torch.tensor(np.array(actions), dtype=torch.int64).to(device)
    # shape: [batch_size, time]
    rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(device)
    # shape: [batch_size, time]
    is_not_done = torch.tensor(np.array(is_not_done), dtype=torch.float32).to(device)
    rollout_length = rewards.shape[1] - 1

    logits = []
    state_values = []
    for t in range(rewards.shape[1]):
        obs_t = states[:, t]
        logits_t, q_t = agent(obs_t)

        logits.append(logits_t)
        state_values.append(q_t)

    logits = torch.stack(logits, dim=1)
    state_values = torch.stack(state_values, dim=1)
    probas = F.softmax(logits, dim=2)
    logprobas = F.log_softmax(logits, dim=2)

    # select log-probabilities for chosen actions, log pi(a_i|s_i)
    actions_one_hot = to_one_hot(actions, n_actions, device).view(actions.shape[0], actions.shape[1], n_actions)
    logprobas_for_actions = torch.sum(logprobas * actions_one_hot, dim=-1)

    # Now let's compute two loss components:
    # 1) Policy gradient objective.
    # Notes: Please don't forget to call .detach() on advantage term. Also please use mean, not sum.
    # it's okay to use loops if you want
    J_hat = 0  # policy objective as in the formula for J_hat

    # 2) Temporal difference MSE for state values
    # Notes: Please don't forget to call on V(s') term. Also please use mean, not sum.
    # it's okay to use loops if you want
    value_loss = 0

    cumulative_returns = state_values[:, -1].detach()

    for t in reversed(range(rollout_length)):
        # current rewards
        r_t = rewards[:, t, None]
        # current state values
        V_t = state_values[:, t]
        # next state values
        V_next = state_values[:, t + 1].detach()
        # log-probability of a_t in s_t
        logpi_a_s_t = logprobas_for_actions[:, t, None]
        # update G_t = r_t + gamma * G_{t+1} as we did in week6 reinforce
        cumulative_returns = G_t = r_t + gamma * cumulative_returns
        # Compute temporal difference error (MSE for V(s))
        value_loss += (r_t + gamma*V_next - V_t).pow(2).mean()
        # compute advantage A(s_t, a_t) using cumulative returns and V(s_t) as baseline
        advantage = cumulative_returns - V_t
        advantage = advantage.detach()
        # compute policy pseudo-loss aka -J_hat.
        J_hat += (logpi_a_s_t*advantage).mean()
    # regularize with entropy
    entropy_reg = -((probas*logprobas).sum(-1)).mean()
    # add-up three loss components and average over time
    loss = (-J_hat / rollout_length) + (value_loss / rollout_length) + (-0.01 * entropy_reg)
    # Gradient descent step
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return loss.data.cpu().numpy(), entropy_reg.data.cpu().numpy()

# TRAIN OPTION-CRITIC:
class EnvPoolOC(object):
    """
    A thin wrapper for openAI gym environments that maintains a set of parallel
    games and has a method to generate interaction sessions given agent one-step
    applier function.
    """
    def __init__(self, agent, make_env, n_parallel_games=1, epsilon=0.05):
        """
        A special class that handles training on multiple parallel sessions
        and is capable of some auxilary actions like evaluating agent on one
        game session (See .evaluate()).

        :param agent: Agent which interacts with the environment.
        :param make_env: Factory that produces environments OR a name of the
                         gym environment.
        :param n_games: Number of parallel games. One game by default.
        :param epsilon: In case of OptionCritic we need eps-greedy strategy.
                        Pass generator.
        """
        # Create atari games.
        self.agent = agent
        self.make_env = make_env
        self.envs = [self.make_env() for _ in range(n_parallel_games)]
        self.epsilon = epsilon

        # Initial observations.
        self.prev_observations = [env.reset() for env in self.envs]

        # Whether particular session has just been terminated and needs restarting.
        self.just_ended = [False] * len(self.envs)

    def interact(self, n_steps=100, verbose=False):
        """Generate interaction sessions with ataries
        (openAI gym atari environments)
        Sessions will have length n_steps. Each time one of games is finished,
        it is immediately getting reset and this time is recorded in
        is_alive_log (See returned values).

        :param n_steps: Length of an interaction.
        :returns: observation_seq, action_seq, reward_seq, is_alive_seq
        :rtype: a bunch of tensors [batch, tick, ...]
        """
        epsilon = next(self.epsilon)
        readout = self.agent.step(self.prev_observations)
        _, _, q, beta = readout
        options = self.agent.epsilon_greedy(epsilon, q)

        def env_step(i, action):
            if not self.just_ended[i]:
                new_observation, cur_reward, is_done, info = self.envs[i].step(action)
                if is_done:
                    # Game ends now, will finalize on next tick.
                    self.just_ended[i] = True
                # note: is_alive=True in any case because environment is still alive
                # (last tick alive) in our notation.
                return new_observation, cur_reward, True, info
            else:
                # Reset environment, get new observation to be used on next tick.
                new_observation = self.envs[i].reset()
                if verbose:
                    print("env %i reloaded" % i)
                self.just_ended[i] = False

                return new_observation, 0, False, {'end': True}

        history_log = []

        for i in range(n_steps - 1):
            # for option-critic part
            actions = self.agent.sample_actions(readout, options, epsilon)
            new_obs, cur_rwds, is_alive, infos = zip(*map(env_step, range(len(self.envs)), actions))
            # Append data tuple for this tick.
            history_log.append((self.prev_observations, actions, options, cur_rwds, is_alive))

            self.prev_observations = new_obs
            readout = self.agent.step(self.prev_observations)

            _, _, q, beta = readout
            np_beta = beta.data.cpu().numpy()
            np_beta = np_beta[range(len(self.envs)), options]
            for i, b in enumerate(np_beta):
                if np.random.rand() < b:
                    options[i] = self.agent.epsilon_greedy(epsilon, q[None, i])

        # for option critic we add options as well
        dummy_actions = np.zeros((len(self.envs), self.envs[0].action_space.shape[0]))
        dummy_options = [0] * len(self.envs)
        dummy_rewards = [0] * len(self.envs)
        dummy_mask = [1] * len(self.envs)
        history_log.append((self.prev_observations, dummy_actions, dummy_options, dummy_rewards, dummy_mask))
        # cast to numpy arrays, transpose from [time, batch, ...] to [batch, time, ...]
        history_log = [np.array(tensor).swapaxes(0, 1) for tensor in zip(*history_log)]
        observation_seq, action_seq, option_seq, reward_seq, is_alive_seq = history_log

        return observation_seq, action_seq, option_seq, reward_seq, is_alive_seq


def evaluate_oc(agent, env, n_games=1):
    """Plays an entire game start to end, returns session rewards."""

    game_rewards = []
    for _ in range(n_games):
        # initial observation
        observation = env.reset()
        total_reward = 0
        readouts = agent.step(observation[None, ...])
        _, _, q, _ = readouts
        option = agent.epsilon_greedy(0.05, q)

        while True:
            action = agent.sample_actions(readouts, option)
            observation, reward, done, info = env.step(action[0])
            total_reward += reward
            _, _, q, beta = agent.step(observation[None, ...])
            if np.random.rand() < beta[0, option]:
                option = agent.epsilon_greedy(0.05, q)
            if done:
                break

        game_rewards.append(total_reward)

    return game_rewards


def train_oc_on_rollout(agent, target, optimizer, states, actions, options, rewards, is_not_done, device = 'cpu', gamma=0.99):
    """
    Takes a sequence of states, actions and rewards produced by generate_session.
    Updates agent's weights by following the policy gradient above.
    Please use Adam optimizer with default parameters.
    """
    # shape: [batch_size, time, c, h, w]
    states = torch.tensor(np.asarray(states), dtype=torch.float32).to(device)
    # shape: [batch_size, time, actions]
    actions = torch.tensor(np.array(actions), dtype=torch.float32).to(device)
    # shape: [batch_size, time]
    options = torch.tensor(np.array(options), dtype=torch.long).to(device)
    # shape: [batch_size, time]
    rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(device)
    # shape: [batch_size, time]
    is_not_done = torch.tensor(np.array(is_not_done), dtype=torch.float32).to(device)
    rollout_length = rewards.shape[1] - 1

    mu = []
    var = []
    state_values = []
    beta = []
    for t in range(rewards.shape[1]):
        obs_t = states[:, t]
        mu_v, var_v, q_v, beta_v = agent(obs_t)

        mu.append(mu_v)
        var.append(var_v)
        state_values.append(q_v)
        beta.append(beta_v)

    mu = torch.stack(mu, dim=1)
    var = torch.stack(var, dim=1)
    q = torch.stack(state_values, dim=1)
    beta = torch.stack(beta, dim=1)


    entropy_loss = 0
    pi_loss = 0
    q_loss = 0
    beta_loss = 0

    _, _, target_q_options, _ = target(states[:, -1])
    beta_last = beta[:,-1].gather(1, options[:,-1:])
    returns = (1 - beta_last) * target_q_options.gather(1, options[:,-1:]) + \
              beta_last * torch.max(target_q_options, dim=1, keepdim=True)[0]

    for t in reversed(range(rollout_length)):
        q_options = q[:, t]
        beta_t = beta[:, t]
        options_t = options[:, t, None]
        r_t = rewards[:, t, None]
        terminals = is_not_done[:, t, None]
        mu_t = mu[:, t]
        var_t = var[:, t]

        returns = r_t + gamma * returns * terminals

        q_omg = q_options.gather(1, options_t)
        mu_val = mu_t[range(mu_t.shape[0]), options_t.flatten()]
        var_val = var_t[range(var_t.shape[0]), options_t.flatten()]
        log_action_prob = calc_logprob(mu_val, var_val, actions[:,t])[:, None]
        entropy = (0.5*(torch.log(2*np.pi*var_t) + 1)).sum(-1).mean()

        pi_loss += -log_action_prob * (returns - q_omg.detach())
        entropy_loss += entropy
        q_loss += 0.5 * (q_omg - returns).pow(2)

        prev_options = options[:, t+1, None]
        advantage_omg = q_options.gather(1, prev_options) - \
                        torch.max(q_options, dim=1, keepdim=True)[0]
        beta_omg = beta_t.gather(1, prev_options) * (1 - terminals)
        beta_loss += advantage_omg * beta_omg

    # print(pi_loss, q_loss, entropy_loss, beta_loss)
    loss = (pi_loss.mean() / rollout_length) + (q_loss.mean() / rollout_length) + \
           (-0.01 * entropy_loss) + (beta_loss.mean() / rollout_length)
    # Gradient descent step
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return loss.data.cpu().numpy(), entropy_loss.data.cpu().numpy(), beta_loss.mean().data.cpu().numpy()
