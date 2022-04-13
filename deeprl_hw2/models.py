import torch
import torch.nn as nn
import numpy as np


class LinearNetwork(nn.Module):
    def __init__(self, in_channels, num_actions):
        super(LinearNetwork, self).__init__()

        num_features = 1
        for dim in in_channels:
            num_features *= dim

        self.Q_value = nn.Linear(in_features=num_features, out_features=num_actions)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.Q_value(x)


class DQN(nn.Module):
    def __init__(self, in_channels=4, num_actions=9):
        """
        Initialize a deep Q-learning network 
        Arguments:
            in_channels: number of channel of input.
                i.e The number of most recent frames stacked together as describe in the paper
            num_actions: number of action-value to output, one-to-one correspondence to action in game.
        """
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc5 = nn.Linear(512, num_actions)
        self.num_actions = num_actions

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.reshape(x.size(0), -1)))
        return self.fc5(x)

    def get_config(self):  # noqa: D102
        return {'num_actions': self.num_actions}
