import numpy as np
from .utils import decay_generator


def make_random_policy(n_actions, seed=None):
    rng = np.random.default_rng(seed=seed)

    def _policy(state):
        return rng.integers(n_actions)

    return _policy


def make_greedy_policy(q):
    def _policy(state):
        return q[state].argmax()

    return _policy


def make_epsilon_greedy_policy(q, epsilon_decay_fn, seed=None):
    rng = np.random.default_rng(seed=seed)

    n_actions = q.shape[1]
    greedy_policy = make_greedy_policy(q)
    random_policy = make_random_policy(n_actions, seed=seed)
    epsilon_generator = iter(decay_generator(epsilon_decay_fn))

    def _policy(state):
        epsilon = next(epsilon_generator)
        is_exploring = rng.random() < epsilon
        policy = random_policy if is_exploring else greedy_policy
        action = policy(state)
        return action, {"epsilon": epsilon}

    return _policy
