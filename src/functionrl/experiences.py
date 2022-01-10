from itertools import count
from collections import namedtuple
from .utils import limitable

Experience = namedtuple("Experience", "state action reward next_state is_done")


def generate_episode(env, policy):
    next_state = env.reset()
    for step in count():
        state = next_state
        action = policy(state, step)
        next_state, reward, is_done, _ = env.step(action)
        yield Experience(state, action, reward, next_state, is_done)
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
