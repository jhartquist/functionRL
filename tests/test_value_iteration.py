from functionrl.algorithms.value_iteration import value_iteration
from functionrl.envs import make_frozen_lake


def test_value_iteration():
    env = make_frozen_lake()
    pi, info = value_iteration(env)
    assert pi == [0, 3, 3, 3, 0, 0, 0, 0, 3, 1, 0, 0, 0, 2, 1, 0]
    assert info["steps"] == 571
