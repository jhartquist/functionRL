from functionrl.policies import make_random_policy


def test_make_random_policy():
    policy = make_random_policy(4, seed=1)
    actions = [policy(None) for _ in range(8)]
    assert actions == [1, 2, 3, 3, 0, 0, 3, 3]
