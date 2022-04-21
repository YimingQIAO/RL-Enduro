import numpy as np


class ReplayBuffer:
    def __init__(self, max_size, state_shape, batch_size):
        self.max_size_ = max_size
        self.batch_size_ = batch_size
        self.is_full = False

        entry = [
            ('state', np.float32, state_shape), ('action', np.int64),
            ('reward', np.float32), ('next_state', np.float32, state_shape),
            ('done', bool)
        ]
        self.buffer = np.zeros(max_size, dtype=entry)
        self.idx = 0

    def append(self, state, action, reward, next_state, done):
        self.buffer[self.idx] = (state, action, reward, next_state, done)
        self.idx += 1
        if self.is_full or self.idx == self.max_size_:
            self.is_full = True

        self.idx %= self.max_size_

    def sample(self):
        max_range = self.max_size_ if self.is_full else self.idx
        indices = np.random.choice(max_range, self.batch_size_, replace=False)
        batch = self.buffer[indices]

        return (
            np.array(batch['state']),
            np.array(batch['action']),
            np.array(batch['reward']),
            np.array(batch['next_state']),
            np.array(batch['done'])
        )
