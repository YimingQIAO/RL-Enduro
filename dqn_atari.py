#!/usr/bin/env python
"""Run Atari Environment with DQN."""
import imp
import gym
import os
from atari_wrappers_deepmind import wrap_deepmind
import warnings
from deeprl_hw2.dqn import DQNAgent
from deeprl_hw2.models import *
from deeprl_hw2.core import ReplayMemory
import torch
from gym import wrappers
import argparse
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
    return parent_dir


def main():  # noqa: D103
    parser = argparse.ArgumentParser(description='Run DQN on Atari Enduro')
    parser.add_argument('--env', default='Enduro-v0', help='Atari env name')
    parser.add_argument('--log', default='logs', help='Directory to save log to')
    parser.add_argument('--o','--output', default='gym_monitor/', help='Directory to save video to')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--gpu', type=int, default=0,help='GPU to use [default: GPU 0]')

    args = parser.parse_args()
    env = gym.make(args.env)
    env.reset()
    env = wrappers.Monitor(env, args.output, force=True)
    env = wrap_deepmind(env)

    # hyper parameters
    replay_buffer_size = 1000000
    window_size = 4
    gamma = 0.99
    num_burn_in = 64
    target_update_freq = 10000  # 10000
    train_freq = 4
    batch_size = 32 #32
    learning_rate = 0.00025
    learning_starts = 50000  # 50000
    num_actions = env.action_space.n
    img_h, img_w, img_c = env.observation_space.shape
    input_shape = [img_h, img_w, window_size * img_c]

    cuda_device = args.gpu

    args.output = get_output_folder(args.output, args.env)

    with torch.cuda.device(cuda_device):
      Q_model = DQN(input_shape[2], num_actions)
      Q_model.to(cuda_device)
      replay_buffer = ReplayMemory(replay_buffer_size, window_size)

      agent = DQNAgent(Q_model, replay_buffer, gamma, target_update_freq, num_burn_in, train_freq, batch_size,
                      learning_starts, args.log)

      epsilon_start = 1
      epsilon_end = 0.1
      linear_num_frames = 1e6
      agent.InitPolicy(env.action_space.n, epsilon_start, epsilon_end, linear_num_frames)

      agent.compile(optimizer=torch.optim.RMSprop, loss_func=torch.nn.HuberLoss, learning_rate=learning_rate)

      agent.fit(env, 5000000)#500 0000


if __name__ == '__main__':
    main()
