# import gym
# import numpy as np
# import torch
# from torch import nn
# from algorithms.value_iteration import value_iteration
# from display import print_v, print_pi

# np.set_printoptions(suppress=True, precision=4)

# torch.manual_seed(0)

# NUM_EPOCHS = 10000
# LEARNING_RATE = 0.1

# # HIDDEN_DIM = 8
# # HIDDEN_DIM = 8
# HIDDEN_DIM = 16


# class QNetwork(nn.Module):
#     def __init__(self, in_dim, out_dim):
#         super().__init__()
#         self.layers = nn.Sequential(
#             nn.Linear(in_dim, out_dim, bias=False),
#             # nn.Linear(in_dim, HIDDEN_DIM),
#             # nn.ReLU(),
#             # nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
#             # nn.ReLU(),
#             # nn.Linear(HIDDEN_DIM, out_dim),
#         )

#     def forward(self, x):
#         return self.layers(x)


# if __name__ == "__main__":
#     env = gym.make("FrozenLake-v1")

#     pi, info = value_iteration(env, gamma=1)
#     q = np.array(info["q"])
#     n_states, n_actions = q.shape
#     print(q)

#     x = nn.functional.one_hot(torch.arange(n_states)).float()
#     y = torch.Tensor(q)
#     ds = torch.utils.data.TensorDataset(x, y)
#     dl = torch.utils.data.DataLoader(ds, batch_size=n_states)

#     net = QNetwork(16, 4)

#     print(f"{sum(p.numel() for p in net.parameters())} params")

#     opt = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE)
#     loss_fn = nn.MSELoss()

#     for epoch in range(1, NUM_EPOCHS + 1):
#         for batch in dl:
#             xb, yb = batch
#             opt.zero_grad()
#             yh = net(x)
#             loss = loss_fn(yh, y)
#             loss.backward()
#             opt.step()

#         if epoch % 100 == 0:
#             print(f"{epoch:4d}: {loss.item():.8f}")

#         if loss.item() < 1e-7:
#             print(f"Stopping early after {epoch} epochs")
#             break

#     # yh = net(x)
#     # q = yh.detach().numpy()
#     # v = np.max(q, axis=1).reshape(4, 4)
#     # print(v)

#     print_v(info["v"])
#     print_pi(pi)
