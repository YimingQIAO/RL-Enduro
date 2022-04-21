#!/usr/bin/env python
"""Run Atari Environment with DQN."""
import gym
import os
import warnings
import torch
import argparse
import time

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
    return parent_dir


def main():  # noqa: D103
    parser = argparse.ArgumentParser(description='Run DQN on Atari Enduro')
    parser.add_argument('--env', default='Enduro-v0', help='Atari env name')
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


if __name__ == '__main__':
    main()
