from functionrl.algorithms.reinforce import reinforce
from functionrl.envs import make_frozen_lake


def test_reinforce():
    policy = reinforce(
        make_frozen_lake,
        gamma=0.99,
        learning_rate=0.01,
        n_episodes=2,
        log_interval=2,
        eval_episodes=2,
        seed=1,
    )

    assert policy(0)[0] == 3
    assert policy(1)[0] == 3
    assert policy(2)[0] == 1
    assert policy(3)[0] == 3


def test_reinforce_no_seed():
    reinforce(
        make_frozen_lake,
        gamma=0.99,
        learning_rate=0.01,
        n_episodes=2,
        log_interval=1,
        eval_episodes=2,
    )
