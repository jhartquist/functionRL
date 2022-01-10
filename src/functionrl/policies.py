import numpy as np
from .utils import linear_decay


def make_random_policy(n_actions, seed=None):
    rng = np.random.default_rng(seed=seed)

    def _policy(state, step):
        return rng.integers(n_actions)

    return _policy


def make_greedy_action_selector(q):
    return lambda state: q[state].argmax()


def make_random_action_selector(n_actions):
    return lambda state: np.random.randint(n_actions)


def make_epsilon_greedy_policy(q, eps_start, eps_end, eps_decay_steps):
    n_states, n_actions = q.shape
    greedy_selector = make_greedy_action_selector(q)
    random_selector = make_random_action_selector(n_actions)
    eps_decay = linear_decay(eps_start, eps_end, eps_decay_steps)

    step = 0

    def _policy(state, _):
        nonlocal step
        epsilon = eps_decay(step)
        step += 1
        is_exploring = np.random.random() < epsilon
        selector = random_selector if is_exploring else greedy_selector
        action = selector(state)
        return action

    return _policy


def make_greedy_policy(q):
    greedy_selector = make_greedy_action_selector(q)

    def _policy(state, step):
        action = greedy_selector(state)
        return action

    return _policy
