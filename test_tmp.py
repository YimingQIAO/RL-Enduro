import matplotlib.pyplot as plt
import gym
from atari_wrappers import make_atari, wrap_deepmind, LazyFrames
import time
import matplotlib

matplotlib.rcParams['backend'] = 'TkAgg'

env = gym.make('Enduro-v0')
env.reset()
env = wrap_deepmind(env, scale=False, frame_stack=True)
n_actions = env.action_space.n
state_dim = env.observation_space.shape
test = env.reset()
for i in range(100):
    test = env.step(env.action_space.sample())[0]

# print(env._get_ob().frame(3))
# exit(0)
plt.imshow(env._get_ob().frame(3))
# plt.imshow(env.render("rgb_array"))
plt.show()
