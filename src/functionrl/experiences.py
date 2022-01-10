from itertools import count
from collections import namedtuple
from .utils import limitable

Experience = namedtuple(
    "Experience",
    ["state", "action", "reward", "next_state", "is_done", "policy_info"],
)


def generate_episode(env, policy):
    next_state = env.reset()
    while True:
        state = next_state
        action = policy(state)
        policy_info = None
        if isinstance(action, tuple):
            action, policy_info = action
        next_state, reward, is_done, env_info = env.step(action)
        yield Experience(state, action, reward, next_state, is_done, policy_info)
        if is_done:
            break


@limitable
def generate_episodes(env, policy):
    while True:
        yield list(generate_episode(env, policy))


@limitable
def generate_experiences(env, policy):
    while True:
        yield from generate_episode(env, policy)
