import torch
from torch import nn
from torch.nn import functional as F


def to_tensor(x):
    return torch.as_tensor(x)


class LinearNet(nn.Module):
    def __init__(self, in_dim, out_dim, one_hot=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.one_hot = one_hot
        self.layer = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, state):
        state = to_tensor(state)
        if self.one_hot:
            state = F.one_hot(state, num_classes=self.in_dim).float()
        return self.layer(state)
