"""Main DQN agent."""
<<<<<<< Updated upstream

class DQNAgent:
    """Class implementing DQN.

    This is a basic outline of the functions/parameters you will need
    in order to implement the DQNAgnet. This is just to get you
    started. You may need to tweak the parameters, add new ones, etc.

    Feel free to change the functions and funciton parameters that the
    class provides.

    We have provided docstrings to go along with our suggested API.

    Parameters
    ----------
    q_network:
      Your Q-network model.
    preprocessor: deeprl_hw2.core.Preprocessor
      The preprocessor class. See the associated classes for more
      details.
    memory: deeprl_hw2.core.Memory
      Your replay memory.
    gamma: float
      Discount factor.
    target_update_freq: float
      Frequency to update the target network. You can either provide a
      number representing a soft target update (see utils.py) or a
      hard target update (see utils.py and Atari paper.)
    num_burn_in: int
      Before you begin updating the Q-network your replay memory has
      to be filled up with some number of samples. This number says
      how many.
    train_freq: int
      How often you actually update your Q-Network. Sometimes
      stability is improved if you collect a couple samples for your
      replay memory, for every Q-network update that you run.
    batch_size: int
      How many samples in each minibatch.
    """
=======
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
>>>>>>> Stashed changes
    def __init__(self,
                 q_network,
                 preprocessor,
                 memory,
                 policy,
                 gamma,
                 target_update_freq,
                 num_burn_in,
                 train_freq,
<<<<<<< Updated upstream
                 batch_size):
        pass

    def compile(self, optimizer, loss_func):
        """Setup all of the TF graph variables/ops.

        This is inspired by the compile method on the
        keras.models.Model class.

        This is a good place to create the target network, setup your
        loss function and any placeholders you might need.
        
        You should use the mean_huber_loss function as your
        loss_function. You can also experiment with MSE and other
        losses.

        The optimizer can be whatever class you want. We used the
        keras.optimizers.Optimizer class. Specifically the Adam
        optimizer.
        """
        pass

    def calc_q_values(self, state):
        """Given a state (or batch of states) calculate the Q-values.

        Basically run your network on these states.

        Return
        ------
        Q-values for the state(s)
        """
        pass

    def select_action(self, state, **kwargs):
        """Select the action based on the current state.

        You will probably want to vary your behavior here based on
        which stage of training your in. For example, if you're still
        collecting random samples you might want to use a
        UniformRandomPolicy.

        If you're testing, you might want to use a GreedyEpsilonPolicy
        with a low epsilon.

        If you're training, you might want to use the
        LinearDecayGreedyEpsilonPolicy.

        This would also be a good place to call
        process_state_for_network in your preprocessor.

        Returns
        --------
        selected action
        """
        pass

    def update_policy(self):
        """Update your policy.

        Behavior may differ based on what stage of training your
        in. If you're in training mode then you should check if you
        should update your network parameters based on the current
        step and the value you set for train_freq.

        Inside, you'll want to sample a minibatch, calculate the
        target values, update your network, and then update your
        target values.

        You might want to return the loss and other metrics as an
        output. They can help you monitor how training is going.
        """
        pass

    def fit(self, env, num_iterations, max_episode_length=None):
        """Fit your model to the provided environment.

        Its a good idea to print out things like loss, average reward,
        Q-values, etc to see if your agent is actually improving.

        You should probably also periodically save your network
        weights and any other useful info.

        This is where you should sample actions from your network,
        collect experience samples and add them to your replay memory,
        and update your network parameters.

        Parameters
        ----------
        env: gym.Env
          This is your Atari environment. You should wrap the
          environment using the wrap_atari_env function in the
          utils.py
        num_iterations: int
          How many samples/updates to perform.
        max_episode_length: int
          How long a single episode should last before the agent
          resets. Can help exploration.
        """
        pass

    def evaluate(self, env, num_episodes, max_episode_length=None):
        """Test your agent with a provided environment.
        
        You shouldn't update your network parameters here. Also if you
        have any layers that vary in behavior between train/test time
        (such as dropout or batch norm), you should set them to test.
=======
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
>>>>>>> Stashed changes

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

<<<<<<< Updated upstream
        You can also call the render function here if you want to
        visually inspect your policy.
        """
        pass
=======
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
>>>>>>> Stashed changes
