import numpy as np
import torch
from functionrl.experiences import calc_episode_return, gen_episodes
from .utils import decay_generator
from torch.distributions import Categorical


def make_random_policy(n_actions, seed=None):
    rng = np.random.default_rng(seed=seed)

    def _policy(state):
        return rng.integers(n_actions)

    return _policy


def make_greedy_policy(q):
    def _policy(state):
        return q[state].argmax()

    return _policy


def make_greedy_policy_from_model(model, n_states):
    with torch.no_grad():
        q = model(torch.arange(n_states)).numpy()
    return make_greedy_policy(q)


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
        return action, {"epsilon": epsilon, "is_exploring": is_exploring}

    return _policy


def make_categorical_policy_from_model(model):
    def _policy(state):
        logits = model(state)
        prob_dist = Categorical(logits=logits)
        action = prob_dist.sample()
        log_prob = prob_dist.log_prob(action)
        return action.item(), {"log_prob": log_prob}

    return _policy


def evaluate_policy(make_env, policy, n_episodes):
    env = make_env()
    episodes = gen_episodes(env, policy, n=n_episodes)
    returns = [calc_episode_return(episode) for episode in episodes]
    mean_return = np.mean(returns)
    return mean_return
