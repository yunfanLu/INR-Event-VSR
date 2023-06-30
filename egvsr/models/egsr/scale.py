import torch
import torch.nn as nn


class Weight(nn.Module):
    def __init__(self, init_value):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale
