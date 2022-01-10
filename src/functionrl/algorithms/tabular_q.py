import numpy as np
from ..utils import linear_decay
from ..policies import make_epsilon_greedy_policy, make_greedy_policy
from ..experiences import generate_experiences, generate_episodes
from ..envs import make_frozen_lake
from ..display import print_pi, print_v


def tabular_q(
    make_env,
    gamma: float = 1.0,
    alpha_max: float = 1e-2,
    alpha_min: float = 1e-4,
    alpha_decay_steps: int = 5000,
    epsilon_max: float = 1.0,
    epsilon_min: float = 0.1,
    epsilon_decay_steps: int = 5000,
    n_steps: int = 5000,
    log_interval: int = 1000,
    eval_episodes: int = 1000,
):
    env_train = make_env()
    env_eval = make_env()

    n_states = env_train.observation_space.n
    n_actions = env_train.action_space.n

    alpha_decay = linear_decay(alpha_max, alpha_min, alpha_decay_steps)
    epsilon_decay = linear_decay(epsilon_max, epsilon_min, epsilon_decay_steps)

    q = np.zeros((n_states, n_actions))

    # TODO: pass decay into make_eps
    policy_train = make_epsilon_greedy_policy(
        q, epsilon_max, epsilon_min, epsilon_decay_steps
    )
    policy_eval = make_greedy_policy(q)

    for step, exp in enumerate(
        generate_experiences(env_train, policy_train, n=n_steps)
    ):

        td_target = (
            exp.reward + gamma * float(not exp.is_done) * q[exp.next_state].max()
        )
        td_error = td_target - q[exp.state, exp.action]

        alpha = alpha_decay(step)
        q[exp.state, exp.action] += alpha * td_error

        if (step + 1) % log_interval == 0:
            episodes = list(generate_episodes(env_eval, policy_eval, n=eval_episodes))
            returns = [sum(e.reward for e in episode) for episode in episodes]
            mean_return = np.mean(returns)
            print(
                f"{step+1:5d}: {mean_return:.3f}, eps: {epsilon_decay(step):.3f}, alpha: {alpha:.6f}"
            )

    return q


if __name__ == "__main__":
    q = tabular_q(
        make_frozen_lake,
        gamma=1,
        alpha_max=1e-1,
        alpha_min=1e-3,
        alpha_decay_steps=100_000,
        epsilon_max=1.0,
        epsilon_min=0.1,
        epsilon_decay_steps=100_000,
        n_steps=100_000,
        log_interval=10_000,
    )
    pi = np.argmax(q, axis=1)
    print_pi(pi)
    v = np.max(q, axis=1)
    print_v(v)
