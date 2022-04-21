from gym.wrappers import RecordVideo, RecordEpisodeStatistics, GrayScaleObservation, FrameStack, TransformReward, AtariPreprocessing
import gym
import cv2
from collections import deque
import numpy as np
import matplotlib.pyplot as plt 

def always_true(x):
    return True

class RepeatActionInFramesTakeMaxOfTwo(gym.Wrapper):
    def __init__(self, env, h=160, repeat=4):
        super().__init__(env)

        self.repeat = repeat
        self.h = h
        #self.shape = env.observation_space.low.shape
        self.shape = (h,env.observation_space.low.shape[1],env.observation_space.low.shape[2])
        self.frames = deque(maxlen=2)

        if repeat <= 0:
            raise ValueError('Repeat value needs to be 1 or higher')

    def step(self, action):

        total_reward = 0
        done = False
        info = {}

        for i in range(self.repeat):
            observation, reward, done, info = self.env.step(action)
            total_reward += reward
            self.frames.append(observation[:self.h,:,:])

            if done:
                break

        # Open queue into arguments for np.maximum
        maximum_of_frames = np.maximum(*self.frames)
        return maximum_of_frames, total_reward, done, info

    def reset(self):
        observation = self.env.reset()
        self.frames.clear()
        self.frames.append(observation[:self.h,:,:])
        return observation[:self.h,:,:]

class NormResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)

        # Create the new observation space for the env
        # Since we are converting to grayscale we set low of 0 and high of 1
        self.shape = shape

        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=self.shape, dtype=np.float32
        )

    def observation(self, observation):
        """Change from 255 grayscale to 0-1 scale
        """
        observation = cv2.resize(observation, self.shape, interpolation=cv2.INTER_AREA)
        return (observation / 255.0).reshape(self.shape)

class ClipRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def reward(self, reward):
        def clip(x):
            if x < 0:
                return -1
            if x > 0:
                return 1
            return 0
        vec_clip = np.vectorize(clip)
        return vec_clip(reward) 


def Wrap(env, video_dir):
    shape = (84, 84)
    env = RepeatActionInFramesTakeMaxOfTwo(env, repeat=4)
    env = ClipRewardWrapper(env)
    # env = TransformReward(env, np.sign)
    env = GrayScaleObservation(env)
    env = NormResizeObservation(env, shape)
    env = FrameStack(env, num_stack=4)
    # monitor
    env = RecordEpisodeStatistics(env, 100)
    env = RecordVideo(env, video_dir, episode_trigger=always_true)
    return env
