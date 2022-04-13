"""Main DQN agent."""
import os
import sys
import time

import torch
from torch.autograd import Variable
from deeprl_hw2.policy import UniformRandomPolicy, LinearDecayGreedyEpsilonPolicy, GreedyPolicy
# from deeprl_hw2.tensorboardlogger import TensorboardLogger
from deeprl_hw2.utils import *
import numpy as np

# from tensorboardlogger import Logger

LOG_EVERY_N_STEPS = 100
SAVE_MODEL_EVERY_N_STEPS = 2000


class DQNAgent:
    """Class implementing DQN.

    This is a basic outline of the functions/parameters you will need
    in order to implement the DQNAgnet. This is just to get you
    started. You may need to tweak the parameters, add new ones, etc.

    Feel free to change the functions and function parameters that the
    class provides.

    We have provided docstrings to go along with our suggested API.

    Parameters
    ----------
    q_network:
      Your Q-network model.
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

    def __init__(self,
                 q_network,
                 memory,
                 gamma,
                 target_update_freq,
                 num_burn_in,
                 train_freq,
                 batch_size,
                 learning_starts,
                 log_dir):
        self.Q_ = q_network
        self.Q_target_ = q_network

        # reply buffer
        self.memory_ = memory
        self.gamma = gamma
        self.target_update_freq_ = target_update_freq
        self.num_burn_in_ = num_burn_in
        self.train_freq_ = train_freq
        self.batch_size_ = batch_size
        self.learning_starts_ = learning_starts
        # self.logger = Logger(log_dir)

        # define network optimizer placeholder
        self.optimizer = None
        self.loss_func = None

        # Policy
        self.num_action_ = -1
        self.uniform_policy = None
        self.greedy_policy = None
        self.decay_policy = None

        # others
        self.dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self.dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
        self.best_reward = -1
        pass

    def compile(self, optimizer, loss_func, learning_rate):
        """Setup all the TF graph variables/ops.

        This is inspired by the compile method on the
        keras.models.Model class.

        This is a good place to create the target network, set up your
        loss function and any placeholders you might need.
        
        You should use the mean_huber_loss function as your
        loss_function. You can also experiment with MSE and other
        losses.

        The optimizer can be whatever class you want. We used the
        keras.optimizers.Optimizer class. Specifically the Adam
        optimizer.
        """
        self.optimizer = optimizer(self.Q_.parameters(), lr=learning_rate)
        self.loss_func = loss_func()

        pass

    def calc_q_values(self, state):
        """Given a state (or batch of states) calculate the Q-values.

        Basically run your network on these states.

        Return
        ------
        Q-values for the state(s)
        """

        return self.Q_.forward(state)

    def InitPolicy(self, num_actions, epsilon_start, epsilon_end_, num_step):
        self.num_action_ = num_actions
        self.uniform_policy = UniformRandomPolicy(num_actions)
        self.greedy_policy = GreedyPolicy()
        self.decay_policy = LinearDecayGreedyEpsilonPolicy(epsilon_start, epsilon_end_, num_step)

    def select_action(self, state, iteration, is_training=True):
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
        # Test stage, greedy policy
        if not is_training:
            q_s_a = self.Q_(Variable(state, volatile=True)).cpu()
            return self.greedy_policy.select_action(q_s_a)

        # Warm up for training, uniform policy
        if iteration < self.learning_starts_:
            return self.uniform_policy.select_action()

        # Training stage, decay policy
        q_s_a = self.Q_(Variable(state, volatile=True)).cpu()
        return self.decay_policy.select_action(q_s_a.detach().numpy())

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

        It's a good idea to print out things like loss, average reward,
        Q-values, etc. to see if your agent is actually improving.

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
        last_obs = env.reset()
        Q_update_times = 0
        for iteration in range(num_iterations):
            # 1. Check stopping criterion
            if iteration > 1000000:
                break

            # 2.  Step the env and store frame
            frame_idx = self.memory_.store_frame(last_obs)

            # 3. Get observations (state)
            observations = self.memory_.encode_recent_obs()

            # 4. select action
            obs_tensor = torch.from_numpy(observations).unsqueeze(0).type(self.dtype) / 255.0
            action = self.select_action(obs_tensor, iteration)

            next_obs, reward, done, info = env.step(action)
            env.render()

            # clipping the reward, noted in nature paper (Is this necessary? since env has been wrapped)
            # reward = np.clip(reward, -1.0, 1.0)

            # 5. store action and reward in replay buffer
            self.memory_.store_effect(frame_idx, action, reward, done)

            # if done, reset env
            if done:
                last_obs = env.reset()

            # 6. if reply buffer has enough frames, start train according to train frequency
            if iteration > self.num_burn_in_ and iteration % self.train_freq_ == 0 and self.memory_.can_sample(
                    self.batch_size_):
                # get batch
                states, actions, rewards, next_states, dones = self.memory_.sample(self.batch_size_)
                states = Variable(torch.from_numpy(states)).type(self.dtype) / 255.0
                actions = Variable(torch.from_numpy(actions)).type(self.dlongtype)
                rewards = Variable(torch.from_numpy(rewards)).type(self.dtype)
                next_states = Variable(torch.from_numpy(next_states)).type(self.dtype) / 255.0
                dones = Variable(torch.from_numpy(dones)).type(self.dtype)

                # feed states
                q_values = self.Q_(states)
                q_s_a = q_values.gather(1, actions.unsqueeze(1)).squeeze()

                # feed next states
                next_q_values = self.Q_target_(next_states).detach()
                next_q_value_prime, next_action_prime = next_q_values.max(1)
                next_q_value_prime = (1 - dones) * next_q_value_prime

                # get loss
                loss = self.loss_func(rewards + self.gamma * next_q_value_prime, q_s_a)

                # backwards pass
                self.optimizer.zero_grad()
                loss.backward()

                # update
                self.optimizer.step()
                Q_update_times += 1

                # update target Q network
                if Q_update_times % self.target_update_freq_ == 0:
                    self.Q_target_.load_state_dict(self.Q_.state_dict())

            # update best mean reward
            episode_rewards = get_wrapper_by_name(env, "Monitor").get_episode_rewards()
            mean_episode_reward = -1
            if len(episode_rewards) > 0:
                mean_episode_reward = np.mean(episode_rewards[-100:])
                self.best_reward = max(self.best_reward, mean_episode_reward)

            # summary
            if iteration % LOG_EVERY_N_STEPS == 0:
                self.summary(episode_rewards, iteration, mean_episode_reward)

            # save model
            if iteration % SAVE_MODEL_EVERY_N_STEPS == 0:
                self.save_model(iteration)

        return

    def summary(self, episode_rewards, iteration, mean_episode_reward):
        # log
        # for tag, value in self.Q_.named_parameters():
        #     tag = tag.replace('.', '/')
        #     self.logger.to_tf.log_histogram(tag, tensor_to_np(value), iteration + 1)
        #     self.logger.to_tf.log_histogram(tag + '/grad', tensor_to_np(value.grad), iteration + 1)

        print("---------------------------------")
        print("Timestep %d" % (iteration,))
        print("learning started? %d" % (iteration > self.learning_starts_))
        print("mean reward (100 episodes) %f" % mean_episode_reward)
        print("best mean reward %f" % self.best_reward)
        print("episodes %d" % len(episode_rewards))
        print("exploration %f" % self.decay_policy.epsilon)
        sys.stdout.flush()

        # ============ TensorBoard logging ============#
        # (1) Log the scalar values
        # info = {
        #     'learning_started': (iteration > self.learning_starts_),
        #     'num_episodes': len(episode_rewards),
        #     'exploration': self.decay_policy.cal_epsilon(),
        #     'learning_rate': self.optimizer.kwargs['lr'],
        # }
        #
        # for tag, value in info.items():
        #     self.logger.to_tf.log_scalar(tag, value, iteration + 1)
        #
        # if len(episode_rewards) > 0:
        #     info = {
        #         'last_episode_rewards': episode_rewards[-1],
        #     }
        #
        #     for tag, value in info.items():
        #         self.logger.to_tf.log_scalar(tag, value, iteration + 1)
        #
        # if self.best_reward != -float('inf'):
        #     info = {
        #         'mean_episode_reward_last_100': mean_episode_reward,
        #         'best_mean_episode_reward': self.best_reward
        #     }
        #
        #     for tag, value in info.items():
        #         self.logger.to_tf.log_scalar(tag, value, iteration + 1)

    def save_model(self, iteration):
        if not os.path.exists("models"):
            os.makedirs("models")
        add_str = ''
        model_save_path = "models/%s_%d_%s.model" % (
            add_str, iteration, str(time.ctime()).replace(' ', '_'))
        torch.save(self.Q_.state_dict(), model_save_path)

    def evaluate(self, env, num_episodes, max_episode_length=None):
        """Test your agent with a provided environment.

        You shouldn't update your network parameters here. Also if you
        have any layers that vary in behavior between train/test time
        (such as dropout or batch norm), you should set them to test.

        Basically run your policy on the environment and collect stats
        like cumulative reward, average episode length, etc.

        You can also call the render function here if you want to
        visually inspect your policy.
        """
        pass
