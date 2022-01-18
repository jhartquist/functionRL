from itertools import count
from collections import namedtuple
from .utils import limitable

Experience = namedtuple(
    "Experience",
    ["state", "action", "reward", "next_state", "is_done", "policy_info"],
)


def gen_episode(env, policy):
    next_state = env.reset()
    while True:
        state = next_state
        action = policy(state)
        policy_info = None
        if isinstance(action, tuple):
            action, policy_info = action
        next_state, reward, is_done, _ = env.step(action)
        yield Experience(state, action, reward, next_state, is_done, policy_info)
        if is_done:
            break


@limitable
def gen_episodes(env, policy):
    while True:
        yield list(gen_episode(env, policy))


@limitable
def gen_experiences(env, policy):
    while True:
        yield from gen_episode(env, policy)


def calc_episode_return(episode):
    return sum(experience.reward for experience in episode)
