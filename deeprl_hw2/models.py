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
