#!/usr/bin/env python
"""Run Atari Environment with DQN."""
import gym
import os
from deeprl_hw2.atari_wrappers_deepmind import wrap_deepmind
import warnings
from deeprl_hw2.dqn import DQNAgent
from deeprl_hw2.models import *
from deeprl_hw2.core import ReplayMemory
import torch
from gym import wrappers
import argparse
import time

warnings.filterwarnings("ignore")


def create_model(window, input_shape, num_actions,
                 model_name='q_network'):  # noqa: D103
    """Create the Q-network model.

    You can use any DL library you like, including Tensorflow, Keras or PyTorch.

    If you use Tensorflow or Keras, we highly recommend that you use tf.name_scope as discussed in
    class when creating the model and the layers. This will make it
    far easier to understand your network architecture if you are
    logging with tensorboard.

    Parameters
    ----------
    window: int
      Each input to the network is a sequence of frames. This value
      defines how many frames are in the sequence.
    input_shape: tuple(int, int)
      The expected input image size.
    num_actions: int
      Number of possible actions. Defined by the gym environment.
    model_name: str
      Useful when debugging. Makes the model show up nicer in tensorboard.

    Returns
    -------

      The Q-model.
    """
    pass


def get_output_folder(parent_dir, env_name, is_double):
    """Return save folder.

    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.

    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.
    env_name: str
      name of game in gym

    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    os.makedirs(parent_dir, exist_ok=True)

    if is_double:
      env_name += "-DoubleQ"

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + time.strftime("-%m-%d-%H-%M")
    return parent_dir


def main():  # noqa: D103
    parser = argparse.ArgumentParser(description='Run DQN on Atari Enduro')
    parser.add_argument('--env', default='Enduro-v0', help='Atari env name')
    parser.add_argument('--log', default='logs', help='Directory to save log to')
    parser.add_argument('--output', default='gym_monitor/', help='Directory to save video to')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
    parser.add_argument('--memory', type=int, default=1000000, help='replay_buffer_size')
    parser.add_argument('--gamma', default=0.99, type=float, help='discount factor')
    parser.add_argument('--num_burn_in', default=50000, type=int, help='num burn in')
    parser.add_argument('--freq', default=10000, type=int, help='Target network update frequency')
    parser.add_argument('--bs', default=32, type=int, help='batch size')
    parser.add_argument('--lr', default=0.00025, type=float, help='learning rate')
    parser.add_argument('--eps_max', default=1, type=float, help='epsilon start')
    parser.add_argument('--eps_min', default=0.1, type=float, help='epsilon end')
    parser.add_argument('--frames', default=1e6, type=int, help='linear_num_frames')
    parser.add_argument('--iters', default=5000001, type=int, help='iters')

    parser.add_argument('--model', default='DQN', type=str, help='LN, DQN')
    parser.add_argument('--is_double', default=False, type=bool, help='use double Q-netowkr if true')

    args = parser.parse_args()
    args.output = get_output_folder(args.output, args.model, args.is_double)
    args.log = get_output_folder(args.log, args.model, args.is_double)
    env = gym.make(args.env)
    env.reset()
    env = wrappers.Monitor(env, args.output, force=True)
    env = wrap_deepmind(env)

    # hyper parameters
    replay_buffer_size = args.memory
    window_size = 4
    gamma = args.gamma
    num_burn_in = args.num_burn_in
    target_update_freq = args.freq  # 10000
    train_freq = 4
    batch_size = args.bs
    learning_rate = args.lr
    num_actions = env.action_space.n
    img_h, img_w, img_c = env.observation_space.shape
    input_shape = [img_h, img_w, window_size * img_c]

    cuda_device = args.gpu

    with torch.cuda.device(cuda_device):
        if args.model == 'DQN':
            Q_model = DQN(input_shape[2], num_actions)
        elif args.model == 'LN':
            Q_model = LinearNetwork(input_shape, num_actions)
        else:
            raise NotImplementedError
        Q_model.to(cuda_device)
        replay_buffer = ReplayMemory(replay_buffer_size, window_size)

        agent = DQNAgent(Q_model, replay_buffer, gamma, target_update_freq, num_burn_in, train_freq, batch_size,
                         args.log, args.is_double)

        epsilon_start = args.eps_max
        epsilon_end = args.eps_min
        linear_num_frames = args.frames
        agent.InitPolicy(env.action_space.n, epsilon_start, epsilon_end, linear_num_frames)

        agent.compile(optimizer=torch.optim.RMSprop, loss_func=torch.nn.HuberLoss, learning_rate=learning_rate)

        agent.fit(env, args.iters)  # 500 0000


if __name__ == '__main__':
    main()
