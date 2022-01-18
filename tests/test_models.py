import torch
from functionrl.models import LinearNet


def test_linear_net():
    net = LinearNet(2, 4)
    assert net(0).shape == (4,)
    assert net([0]).shape == (1, 4)

    net = LinearNet(2, 4, one_hot=False)
    assert net(torch.tensor([0, 1], dtype=torch.float)).shape == (4,)
    assert net(torch.tensor([[0, 1]], dtype=torch.float)).shape == (1, 4)
