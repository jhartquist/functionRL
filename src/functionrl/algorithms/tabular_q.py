from itertools import count
from typing import Optional
import numpy as np
from ..utils import linear_decay
from ..policies import evaluate_policy, make_epsilon_greedy_policy, make_greedy_policy
from ..experiences import gen_experiences
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
    seed: Optional[int] = None,
):
    env = make_env()

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    q = np.zeros((n_states, n_actions))

    alpha_decay = linear_decay(alpha_max, alpha_min, alpha_decay_steps)
    epsilon_decay = linear_decay(epsilon_max, epsilon_min, epsilon_decay_steps)

    policy_train = make_epsilon_greedy_policy(q, epsilon_decay, seed=seed)
    policy_eval = make_greedy_policy(q)

    for i, exp in enumerate(gen_experiences(env, policy_train, n=n_steps), start=1):
        state, action, reward, next_state, is_done, policy_info = exp
        td_target = reward + gamma * float(not is_done) * q[next_state].max()
        td_error = td_target - q[state, action]

        alpha = alpha_decay(i)
        q[state, action] += alpha * td_error

        if i % log_interval == 0:
            epsilon = policy_info["epsilon"]
            mean_return = evaluate_policy(make_env, policy_eval, eval_episodes)
            print(f"{i:5d}: {mean_return:.3f}, eps: {epsilon:.3f}, alpha: {alpha:.6f}")
            pi = np.argmax(q, axis=1)
            print_pi(pi)

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
        seed=0,
    )
    v = np.max(q, axis=1)
    print_v(v)
