import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.distributions import Categorical
from ..utils import linear_decay
from ..policies import make_epsilon_greedy_policy, make_greedy_policy
from ..experiences import generate_episodes
from ..envs import make_frozen_lake
from ..display import print_pi, print_v


class Pi(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.model = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )

    def act(self, state):
        state_t = torch.tensor([state])
        state_oh = F.one_hot(state_t, num_classes=self.in_dim).float()
        logits = self.forward(state_oh)
        pd = Categorical(logits=logits)
        action = pd.sample()
        log_prob = pd.log_prob(action)
        return action.item(), log_prob  # .squeeze()

    def forward(self, x):
        return self.model(x)


def reinforce(
    make_env,
    gamma: float = 1.0,
    learning_rate: float = 1e-3,
    n_episodes: int = 10000,
    log_interval: int = 100,
    eval_episodes: int = 100,
):
    env_train = make_env()
    env_eval = make_env()

    n_states = env_train.observation_space.n
    n_actions = env_train.action_space.n

    pi = Pi(n_states, n_actions)
    optimizer = optim.Adam(pi.parameters(), lr=learning_rate)

    def policy(state):
        action, log_prob = pi.act(state)
        return action, {"log_prob": log_prob}

    for i, episode in enumerate(
        generate_episodes(env_train, policy, n=n_episodes), start=1
    ):
        T = len(episode)
        rewards = [exp.reward for exp in episode]
        log_probs = [exp.policy_info["log_prob"] for exp in episode]
        rets = np.empty(T, dtype=np.float32)
        future_ret = 0.0
        for t in reversed(range(T)):
            future_ret = rewards[t] + gamma * future_ret
            rets[t] = future_ret
        rets = torch.tensor(rets)
        log_probs = torch.stack(log_probs)
        loss = (-log_probs * rets).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % log_interval == 0:
            with torch.no_grad():
                episodes = list(generate_episodes(env_eval, policy, n=eval_episodes))
            returns = [sum(e.reward for e in episode) for episode in episodes]
            mean_return = np.mean(returns)
            print(f"{i:5d}: {mean_return:.3f}")


if __name__ == "__main__":
    reinforce(
        make_frozen_lake,
        gamma=0.99,
        n_episodes=100000,
    )
