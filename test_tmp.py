import gym
from atari_wrappers import make_atari, wrap_deepmind,LazyFrames
import time
env = gym.make('Enduro-v0')
env.reset()
env = wrap_deepmind(env, scale = False, frame_stack=True )
n_actions = env.action_space.n
state_dim = env.observation_space.shape
print(state_dim)
exit(0)

for _ in range(1000):
    env.render()
    env.step(1)  # take a random action
    time.sleep(0.01)
env.close()
