"""Core classes."""
import random

import numpy as np


class ReplayMemory:
    """Interface for replay memories.

    We have found this to be a useful interface for the replay
    memory. Feel free to add, modify or delete methods/attributes to
    this class.

    It is expected that the replay memory has implemented the
    __iter__, __getitem__, and __len__ methods.

    If you are storing raw Sample objects in your memory, then you may
    not need the end_episode method, and you may want to tweak the
    append method. This will make the sample method easy to implement
    (just randomly draw samples saved in your memory).

    However, the above approach will waste a lot of memory (as states
    will be stored multiple times in s as next state and then s' as
    state, etc.). Depending on your machine resources you may want to
    implement a version that stores samples in a more memory efficient
    manner.
    """

    def __init__(self, max_size, window_length):
        """Setup memory.

        You should specify the maximum size o the memory. Once the
        memory fills up oldest values should be removed. You can try
        the collections.deque class as the underlying storage, but
        your sample method will be very slow.

        We recommend using a list as a ring buffer. Just track the
        index where the next sample should be inserted in the list.
        """
        self.max_size_ = max_size
        self.window_length_ = window_length

        self.next_idx = 0
        self.num_in_buffer = 0

        self.obs = None
        self.action = None
        self.reward = None
        self.done = None
        pass

    def store_frame(self, frame):
        """Store a single frame in the buffer at the next available index, overwriting
        old frames if necessary.
        Parameters
        ----------
        frame: np.array
            Array of shape (img_h, img_w, img_c) and dtype np.uint8
            the frame to be stored
        Returns
        -------
        idx: int
            Index at which the frame is stored. To be used for `store_effect` later.
        """
        # c, h, w
        frame = frame.transpose(2, 0, 1)

        if self.obs is None:
            self.obs = np.empty([self.max_size_] + list(frame.shape), dtype=np.uint8)
            self.action = np.empty([self.max_size_], dtype=np.int32)
            self.reward = np.empty([self.max_size_], dtype=np.float32)
            self.done = np.empty([self.max_size_], dtype=np.bool)
        self.obs[self.next_idx] = frame

        ret = self.next_idx
        self.next_idx = (self.next_idx + 1) % self.max_size_
        self.num_in_buffer = min(self.max_size_, self.num_in_buffer + 1)

        return ret

    def encode_recent_obs(self):
        assert self.num_in_buffer > 0
        return self._encode_observation((self.next_idx - 1) % self.max_size_)

    def store_effect(self, idx, action, reward, done):
        """Store effects of action taken after obeserving frame stored
        at index idx. The reason `store_frame` and `store_effect` is broken
        up into two functions is so that once can call `encode_recent_observation`
        in between.
        Paramters
        ---------
        idx: int
            Index in buffer of recently observed frame (returned by `store_frame`).
        action: int
            Action that was performed upon observing this frame.
        reward: float
            Reward that was received when the actions was performed.
        done: bool
            True if episode was finished after performing that action.
        """
        self.action[idx] = action
        self.reward[idx] = reward
        self.done[idx] = done

    def can_sample(self, batch_size):
        """Returns true if `batch_size` different transitions can be sampled from the buffer."""
        return batch_size + 1 <= self.num_in_buffer

    def sample(self, batch_size, indexes=None):
        assert self.can_sample(batch_size)
        if not indexes:
            indexes = self.sample_index(lambda: random.randint(0, self.num_in_buffer - 2), batch_size)

        return self._encode_sample(indexes)

    def _encode_sample(self, idxes):
        obs_batch = np.concatenate([self._encode_observation(idx)[None] for idx in idxes], 0)
        act_batch = self.action[idxes]
        rew_batch = self.reward[idxes]
        next_obs_batch = np.concatenate([self._encode_observation(idx + 1)[None] for idx in idxes], 0)
        done_mask = np.array([1.0 if self.done[idx] else 0.0 for idx in idxes], dtype=np.float32)

        return obs_batch, act_batch, rew_batch, next_obs_batch, done_mask

    def _encode_observation(self, idx):
        end_idx = idx + 1
        start_idx = end_idx - self.window_length_

        for i in range(start_idx, end_idx - 1):
            if self.done[i % self.max_size_]:
                start_idx = idx + 1

        missing_context = self.window_length_ - (end_idx - start_idx)
        if start_idx < 0 or missing_context > 0:
            frames = [np.zeros_like(self.obs[0]) for _ in range(missing_context)]
            for idx in range(start_idx, end_idx):
                frames.append(self.obs[idx % self.max_size_])

            return np.concatenate(frames, 0)
        else:
            img_h, img_w = self.obs.shape[2], self.obs.shape[3]
            return self.obs[start_idx: end_idx].reshape(-1, img_h, img_w)

    @staticmethod
    def sample_index(sampling_f, n):
        """Helper function. Given a function `sampling_f` that returns
        comparable objects, sample n such unique objects.
        """
        res = []
        while len(res) < n:
            candidate = sampling_f()
            if candidate not in res:
                res.append(candidate)
        return res
