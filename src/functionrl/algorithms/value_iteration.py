import gym
from itertools import count
from ..display import print_pi, print_v


def argmax(arr):
    return max(range(len(arr)), key=lambda i: arr[i])


def value_iteration(env, gamma=0.99, theta=1e-10):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    transitions = env.P
    v = [0.0 for _ in range(n_states)]
    for step in count(1):  # pragma: no branch
        q = [
            [
                sum(
                    prob * (reward + (0 if is_done else gamma * v[next_state]))
                    for prob, next_state, reward, is_done in transitions[state][action]
                )
                for action in range(n_actions)
            ]
            for state in range(n_states)
        ]
        v_old = v.copy()
        v = list(map(max, q))
        max_delta = max(abs(a - b) for a, b in zip(v, v_old))
        if max_delta < theta:
            break

    pi = list(map(argmax, q))
    pi = [argmax(action_values) for action_values in q]
    return pi, {"steps": step, "q": q, "v": v}


if __name__ == "__main__":  # pragma: no cover
    env = gym.make("FrozenLake-v1")
    pi, info = value_iteration(env)
    print_pi(pi)
    print(info["steps"])
    print_v(info["v"])
