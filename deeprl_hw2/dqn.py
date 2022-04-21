"""Main DQN agent."""
import os
import sys
import time

import torch
import numpy as np
from tensorboardX import SummaryWriter
import random

LOG_EVERY_N_STEPS = 1000
SAVE_MODEL_EVERY_N_STEPS = 50000


class DQNAgent:
    def __init__(self,
                 q_network,
                 memory,
                 gamma,
                 target_update_freq,
                 num_burn_in,
                 train_freq,
                 batch_size,
                 log_dir,
                 eps_max,
                 eps_min,
                 linear_frames,
                 is_double,
                 device):
        self.Q_ = q_network
        self.Q_target_ = q_network

        self.memory_ = memory
        self.gamma = gamma
        self.target_update_freq_ = target_update_freq
        self.num_burn_in_ = num_burn_in
        self.train_freq_ = train_freq
        self.batch_size_ = batch_size
        self.logger = SummaryWriter(log_dir=log_dir)
        self.epsilon_end = eps_min
        self.epsilon_decay_current_step = 0
        self.linear_decay_steps = linear_frames
        self.is_double_ = is_double
        self.device_ = device

        # exploration decay
        self.epsilon = eps_max
        self.epsilon_step = (eps_max - eps_min) / linear_frames

        # define network optimizer placeholder
        self.optimizer = None
        self.loss_func = None

        # env
        self.env_ = None
        self.input_shape_ = None
        self.num_action_ = -1

        # num episode
        self.num_episode = 0
        self.num_iteration = 0
        self.update_times = 0

        # others
        self.dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self.dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
        self.best_reward = 0

    def compile(self, optimizer, loss_func, learning_rate, alpha, eps, momentum):
        # self.optimizer = optimizer(self.Q_.parameters(), lr=learning_rate, alpha=alpha, eps=eps, momentum=momentum)
        self.optimizer = optimizer(self.Q_.parameters(), lr=learning_rate)
        self.loss_func = loss_func()

    def init_env(self, env):
        self.env_ = env
        self.num_action_ = env.action_space.n
        self.input_shape_ = list(env.observation_space.shape)

        self.Q_ = self.Q_(self.input_shape_, self.num_action_).to(self.device_)
        self.Q_target_ = self.Q_target_(self.input_shape_, self.num_action_).to(self.device_)

    def greedy_policy(self, state):
        state = torch.tensor(np.array([state]), dtype=torch.float).cuda(self.device_)
        q_s_a = self.Q_(state).datch()
        return torch.argmax(q_s_a)

    def fit(self, max_iteration):
        while True:
            self.num_episode += 1
            state = self.env_.reset()
            done = False

            while not done:
                self.num_iteration += 1

                # get sample
                action = self.__epsilon_greedy_policy(state)
                next_state, reward, done, info = self.env_.step(action)
                self.memory_.append(state, action, reward, next_state, done)
                state = next_state

                # if reply buffer has enough frames, start train
                if self.num_iteration > self.num_burn_in_:
                    # get batch
                    states, actions, rewards, next_states, dones = self.__batch_sample()

                    # get loss
                    if self.is_double_:
                        loss = self.__double_dqn(states, actions, rewards, next_states, dones)
                    else:
                        loss = self.__nature_dqn(states, actions, rewards, next_states, dones)
                    # print(loss)
                    # exit(0)
                    # backwards pass
                    self.optimizer.zero_grad()
                
                    loss.backward()

                    # update
                    self.optimizer.step()
                    self.update_times += 1

                    # update target Q network
                    if self.update_times % self.target_update_freq_ == 0:
                        self.Q_target_.load_state_dict(self.Q_.state_dict())

                if self.num_iteration % LOG_EVERY_N_STEPS == 0:
                    self.__stat_recorder()

                # save model
                if self.num_iteration % SAVE_MODEL_EVERY_N_STEPS == 0:
                    self.__save_model()

            if self.num_iteration > max_iteration:
                break
        return

    def __batch_sample(self):
        state, action, reward, next_state, done = self.memory_.sample()

        return self.__to_tensor(state), self.__to_tensor(action), self.__to_tensor(reward), self.__to_tensor(
            next_state), self.__to_tensor(done)

    def __to_tensor(self, data):
        return torch.tensor(data).to(self.device_)

    def __save_model(self):
        if not os.path.exists("models"):
            os.makedirs("models")
        add_str = self.Q_.__class__.__name__

        if self.is_double_:
            add_str += "-Double"

        model_save_path = "models/%s_%d_%s.model" % (
            add_str, self.num_iteration, str(time.ctime()).replace(' ', '_'))
        torch.save(self.Q_.state_dict(), model_save_path)

    def __stat_recorder(self):
        episode_rewards = list(self.env_.return_queue)
        mean_reward_100 = 0
        mean_reward_20 = 0
        std_reward_100 = -1
        last_reward = 0

        if len(episode_rewards) != 0:
            mean_reward_100 = np.mean(episode_rewards)
            mean_reward_20 = np.mean(episode_rewards[-20:])
            std_reward_100 = np.std(episode_rewards)
            last_reward = np.mean(episode_rewards[-1:])

            # update best
            self.best_reward = max(self.best_reward, mean_reward_20)

        print(
            "Episode: %d, "
            "Iteration: %d, "
            "Mean Reward (20): %f, "
            "Mean Reward (100): %f, "
            "Best Mean Reward: %f, "
            "Exploration: %f, "
            "Learning Rate: %f." % (
                self.num_episode,
                self.num_iteration,
                mean_reward_20,
                mean_reward_100,
                self.best_reward,
                self.epsilon,
                self.optimizer.param_groups[0]['lr']))

        # log info
        info = {
            'Episode': self.num_episode,
            'Iteration': self.num_iteration,
            'mean_reward_100': mean_reward_100,
            'mean_reward_20': mean_reward_20,
            'std_reward_100': std_reward_100,
            'best_mean_reward': self.best_reward,
            'last_episode_rewards': last_reward,
            'learning_started': (self.num_iteration > self.num_burn_in_),
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'exploration': self.epsilon,
        }
        for tag, value in info.items():
            self.logger.add_scalar(tag, value, self.num_iteration)

    def __epsilon_greedy_policy(self, state):
        if self.epsilon_decay_current_step < self.linear_decay_steps:
            self.epsilon -= self.epsilon_step
            self.epsilon_decay_current_step += 1

        if random.uniform(0, 1) < self.epsilon:
            if random.uniform(0, 1) < self.epsilon:
                return 1
            else:
                return np.random.randint(0, self.num_action_)
        else:
            state = torch.tensor(np.array([state]), dtype=torch.float).cuda(self.device_)
            q_s_a = self.Q_(state).detach()
            action = torch.argmax(q_s_a)
            return action
            
    def __nature_dqn(self, states, actions, rewards, next_states, dones):
        q_values = self.Q_(states)
        q_s_a = q_values.gather(1, actions.unsqueeze(1))
        q_s_a = q_s_a.squeeze()

        next_q_values = self.Q_target_(next_states).detach()
        next_q_s_a, _ = next_q_values.max(1)

        next_q_s_a[dones] = 0

        return self.loss_func(rewards + self.gamma * next_q_s_a, q_s_a).cuda(self.device_)

    def __double_dqn(self, states, actions, rewards, next_states, dones):
        q_values = self.Q_(states)
        q_s_a = q_values.gather(1, actions.unsqueeze(1)).squeeze()
        q_s_a = q_s_a.squeeze()

        next_q_values = self.Q_(next_states).detach()
        _, next_actions = next_q_values.max(1)

        next_q_target_values = self.Q_target_(next_states).detach()
        next_q_s_a = next_q_target_values.gather(1, next_actions.unsqueeze(1))
        next_q_s_a = next_q_s_a.squeeze()

        next_q_s_a[dones] = 0

        return self.loss_func(rewards + self.gamma * next_q_s_a, q_s_a).cuda(self.device_)