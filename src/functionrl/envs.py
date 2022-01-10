import gym


def make_frozen_lake():
    env = gym.make("FrozenLake-v1")
    env.seed(0)
    return env


def make_frozen_lake_not_slippery():
    env = gym.make("FrozenLake-v1", is_slippery=False)
    env.seed(0)
    return env
