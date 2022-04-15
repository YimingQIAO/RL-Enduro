import gym
import gnwrapper

import warnings
warnings.filterwarnings("ignore")

env = gnwrapper.Animation(gym.make('CartPole-v1'))

obs = env.reset()

for _ in range(1000):
    next_obs, reward, done, info = env.step(env.action_space.sample())
    env.render()

    obs = next_obs
    if done:
        obs = env.reset()

env.close()


#%%
