import numpy as np


class Memory(object):
    def __init__(self, size, state_dim, action_num):
        self.size = size
        self.store = np.zeros(shape=(self.size, state_dim * 2 + action_num + 1))

    def save(self, obs, action, reward, obs_, num):
        index = num % self.size
        temp = np.hstack((obs, action, reward, obs_))
        self.store[index, :] = temp

    def get_data(self, batch_size, num):
        if num < self.size:
            sample_index = np.random.choice(num, batch_size)
        else:
            sample_index = np.random.choice(self.size, batch_size)
        batch_memory = self.store[sample_index, :]
        return batch_memory

