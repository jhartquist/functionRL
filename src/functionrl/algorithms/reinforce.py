from typing import Optional

import numpy as np
import torch
from functionrl.models import LinearNet
from functionrl.policies import (
    evaluate_policy,
    make_categorical_policy_from_model,
    make_greedy_policy_from_model,
)
from torch import optim

from ..envs import make_frozen_lake
from ..experiences import gen_episodes


def reinforce(
    make_env,
    gamma: float = 1.0,
    learning_rate: float = 1e-3,
    n_episodes: int = 10000,
    log_interval: int = 100,
    eval_episodes: int = 1000,
    seed: Optional[int] = None,
):
    if seed is not None:
        torch.manual_seed(seed)

    env = make_env()
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    pi = LinearNet(n_states, n_actions)
    print(pi)

    optimizer = optim.Adam(pi.parameters(), lr=learning_rate)
    policy = make_categorical_policy_from_model(pi)

    losses = []
    for i, episode in enumerate(gen_episodes(env, policy, n=n_episodes), start=1):
        T = len(episode)
        rewards = [exp.reward for exp in episode]
        log_probs = [exp.policy_info["log_prob"] for exp in episode]
        rets = np.empty(T, dtype=np.float32)
        future_ret = 0.0
        for t in reversed(range(T)):
            future_ret = rewards[t] + gamma * future_ret
            rets[t] = future_ret
        rets = torch.tensor(rets)
        # rets.sub_(rets.mean())
        log_probs = torch.stack(log_probs)
        loss = (-log_probs * rets).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if i % log_interval == 0:
            policy = make_greedy_policy_from_model(pi, n_states)
            mean_return = evaluate_policy(make_env, policy, eval_episodes)
            mean_loss = np.array(losses[-log_interval:]).mean()
            print(f"{i:5d}  mean_return: {mean_return:.3f} - loss: {mean_loss:8.4f}")


if __name__ == "__main__":
    reinforce(
        make_frozen_lake,
        gamma=0.99,
        learning_rate=0.01,
        n_episodes=10000,
        seed=1,
        eval_episodes=1000,
    )
