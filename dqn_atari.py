#!/usr/bin/env python
"""Run Atari Environment with DQN."""
<<<<<<< Updated upstream
=======
import gym
import os
import warnings
import torch
>>>>>>> Stashed changes
import argparse
import os
import random

<<<<<<< Updated upstream
import numpy as np
# import tensorflow as tf
# from keras.layers import (Activation, Convolution2D, Dense, Flatten, Input,
#                           Permute)
# from keras.models import Model
# from keras.optimizers import Adam

import deeprl_hw2 as tfrl
from deeprl_hw2.dqn import DQNAgent
from deeprl_hw2.objectives import mean_huber_loss


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


def get_output_folder(parent_dir, env_name):
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

    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
=======
from deeprl_hw2.wrappers import Wrap
from deeprl_hw2.dqn import DQNAgent
from deeprl_hw2.models import *
from deeprl_hw2.replay_buffer import ReplayBuffer

warnings.filterwarnings("ignore")


def get_output_folder(parent_dir, name, double_DQN=False):
    if double_DQN:
        name += "double_DQN"

    parent_dir = os.path.join(parent_dir, name)
    parent_dir = parent_dir + time.strftime("-%m-%d-%H-%M")

    os.makedirs(parent_dir, exist_ok=True)
>>>>>>> Stashed changes
    return parent_dir


def main():  # noqa: D103
    parser = argparse.ArgumentParser(description='Run DQN on Atari Breakout')
    parser.add_argument('--env', default='Enduro-v0', help='Atari env name')
<<<<<<< Updated upstream
    parser.add_argument(
        '-o', '--output', default='atari-v0', help='Directory to save data to')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')

    args = parser.parse_args()
    args.input_shape = tuple(args.input_shape)

    args.output = get_output_folder(args.output, args.env)
=======
    parser.add_argument('--log', default='logs', help='Directory to save log to')
    parser.add_argument('--output', default='gym_monitor', help='Directory to save video to')
    parser.add_argument('--seed', default=1, type=int, help='Random seed')
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
    parser.add_argument('--memory', type=int, default=100000, help='replay_buffer_size')
    parser.add_argument('--gamma', default=0.99, type=float, help='discount factor')
    parser.add_argument('--num_burn_in', default=20000, type=int, help='learning starts')
    parser.add_argument('--freq', default=10000, type=int, help='Target network update frequency')
    parser.add_argument('--bs', default=32, type=int, help='batch size')
    parser.add_argument('--lr', default=0.00025, type=float, help='learning rate')
    parser.add_argument('--eps_max', default=1, type=float, help='epsilon start')
    parser.add_argument('--eps_min', default=0.1, type=float, help='epsilon end')
    parser.add_argument('--frames', default=5e5, type=int, help='linear_num_frames')
    parser.add_argument('--iters', default=2000000, type=int, help='iters')
    parser.add_argument('--alpha', default=0.99, type=float, help='opt param alpha')
    parser.add_argument('--eps', default=0.01, type=float, help='opt param eps')
    parser.add_argument('--momentum', default=0.95, type=float, help='opt param momentum')

    parser.add_argument('--model', default='DDQN', type=str, help='LN, DQN, DDQN')
    parser.add_argument('--double', default=0, type=int, help='is double')

    args = parser.parse_args()
    output_dir = get_output_folder(args.output, args.model, args.double)
    log_dir = get_output_folder(args.log, args.model, args.double)

    env = gym.make(args.env)
    env = Wrap(env, output_dir)

    # hyper parameters
    input_shape = list(env.observation_space.shape)

    cuda_device = args.gpu

    with torch.cuda.device(cuda_device):
        if args.model == 'DQN':
            Q_model = DQN
        elif args.model == 'LN':
            Q_model = LinearNetwork
        elif args.model == 'DDQN':
            Q_model = DDQN
        else:
            raise NotImplementedError

        replay_buffer = ReplayBuffer(args.memory, input_shape, args.bs)

        agent = DQNAgent(
            q_network=Q_model,
            memory=replay_buffer,
            gamma=args.gamma,
            target_update_freq=args.freq,
            num_burn_in=args.num_burn_in,
            train_freq=4,
            batch_size=args.bs,
            log_dir=log_dir,
            eps_max=args.eps_max,
            eps_min=args.eps_min,
            linear_frames=args.frames,
            is_double=args.double,
            device=cuda_device
        )
        agent.init_env(env)
        agent.compile(optimizer=torch.optim.RMSprop, loss_func=torch.nn.HuberLoss, learning_rate=args.lr,
                      alpha=args.alpha, eps=args.eps, momentum=args.momentum)
        agent.fit(args.iters)
>>>>>>> Stashed changes

    # here is where you should start up a session,
    # create your DQN agent, create your model, etc.
    # then you can run your fit method.

if __name__ == '__main__':
    main()
