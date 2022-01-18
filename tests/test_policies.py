import numpy as np
from functionrl.policies import make_random_policy, make_greedy_policy


def test_make_random_policy():
    policy = make_random_policy(4, seed=1)
    actions = [policy(None) for _ in range(8)]
    assert actions == [1, 2, 3, 3, 0, 0, 3, 3]


def test_make_greedy_policy():
    q = np.array([[0, 2, 1], [2, 1, 0]])
    policy = make_greedy_policy(q)
    assert policy(0) == 1
    assert policy(1) == 0
