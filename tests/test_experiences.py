import pytest

from functionrl.envs import make_frozen_lake
from functionrl.experiences import gen_episode, gen_episodes, gen_experiences
from functionrl.policies import make_random_policy


@pytest.fixture
def env():
    return make_frozen_lake()


@pytest.fixture
def policy():
    return make_random_policy(4, seed=1)


def test_generate_episode(env, policy):
    episode = list(gen_episode(env, policy))
    assert len(episode) == 5
    assert episode[-1].is_done


def test_episode_gen(env, policy):
    episodes = list(gen_episodes(env, policy, n=4))
    assert [len(episode) for episode in episodes] == [5, 2, 2, 3]


def test_generate_experiences(env, policy):
    experiences = list(gen_experiences(env, policy, n=10))
    assert len(experiences) == 10
    assert len([e for e in experiences if e.is_done]) == 3
