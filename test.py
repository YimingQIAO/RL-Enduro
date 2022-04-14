import gym
from atari_wrappers_berkeley import wrap_deepmind
import warnings
from deeprl_hw2.dqn import DQNAgent
from deeprl_hw2.models import *
from deeprl_hw2.core import ReplayMemory
import torch
from gym import wrappers

warnings.filterwarnings("ignore")

env = gym.make('Enduro-v0')
env.reset()

print("---" * 10 + "Before Wrap" + "---" * 10)
print("Action: {}".format(env.action_space))
print("Observation Space: {}".format(env.observation_space))

monitor_dir = "gym_monitor/"

env = wrappers.Monitor(env, monitor_dir, force=True)
print("---" * 10 + "Before Wrap" + "---" * 10)
env = wrap_deepmind(env)
print("Action: {}".format(env.action_space))
print("Observation Space: {}".format(env.observation_space))

# hyper parameters
replay_buffer_size = 1000000
window_size = 4
gamma = 0.9
target_update_freq = 1
num_burn_in = 64
train_freq = 4
batch_size = 32
learning_rate = 0.0001
learning_starts = 2000

num_actions = env.action_space.n

img_h, img_w, img_c = env.observation_space.shape
input_shape = [img_h, img_w, window_size * img_c]

Q_model = DQN(input_shape[2], num_actions)

replay_buffer = ReplayMemory(replay_buffer_size, window_size)

agent = DQNAgent(Q_model, replay_buffer, gamma, target_update_freq, num_burn_in, train_freq, batch_size,
                 learning_starts, './logs')

epsilon_start = 1
epsilon_end = 0.1
linear_num_frames = 4000
agent.InitPolicy(env.action_space.n, epsilon_start, epsilon_end, linear_num_frames)

agent.compile(optimizer=torch.optim.RMSprop, loss_func=torch.nn.HuberLoss, learning_rate=learning_rate)

agent.fit(env, 100000, 10000)
