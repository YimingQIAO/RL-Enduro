import numpy as np
from gym.wrappers import FrameStack, TransformReward, AtariPreprocessing
from gym.wrappers import RecordVideo, RecordEpisodeStatistics


def PerHundred(x):
    return x % 100 == 0


def Wrap(env, video_dir):
    # preprocessor
    env.unwrapped._frameskip = True
    env = AtariPreprocessing(env)
    env = TransformReward(env, np.sign)
    env = FrameStack(env, 4, False)

    # monitor
    env = RecordEpisodeStatistics(env, 100)
    env = RecordEpisodeStatistics(env, 20)
    env = RecordVideo(env, video_dir, episode_trigger=PerHundred)

    return env
