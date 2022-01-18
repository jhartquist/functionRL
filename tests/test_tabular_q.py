from functionrl.algorithms.tabular_q import tabular_q
from functionrl.envs import make_frozen_lake


def test_tabular_q():
    q = tabular_q(
        make_frozen_lake,
        gamma=1,
        alpha_max=1e-1,
        alpha_min=1e-3,
        alpha_decay_steps=100_000,
        epsilon_max=1.0,
        epsilon_min=0.1,
        epsilon_decay_steps=100_000,
        n_steps=10,
        log_interval=5,
        seed=0,
    )
