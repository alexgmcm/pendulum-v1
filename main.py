import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

env = gym.make("Pendulum-v1")
# obs, info = env.reset(seed=42)


def collect_random_data(env, episodes=10):
    # get data from random policy
    dataset = {"s": [], "a": [], "r": [], "s_next": [], "done": []}
    for i in range(episodes):
        s, _ = env.reset()
        done = False
        while not done:
            a = env.action_space.sample()
            s_next, r, terminated, truncated, info = env.step(a)
            done = terminated or truncated
            # record = DotMap({"s":s, "a":a, "r":r, "s_next":s_next,"done":done})
            dataset["s"].append(s)
            dataset["a"].append(a)
            dataset["r"].append(r)
            dataset["s_next"].append(s_next)
            dataset["done"].append(done)
            s = s_next

    return dataset


class Standardizer:
    def __init__(self, eps=1e-8):
        self.eps = eps
        self.mean = None
        self.std = None

    def fit(self, x):
        self.mean, self.std = torch.std_mean(x, dim=0, keepdim=True)
        self.std = np.maximum(self.std, self.eps)
        return self

    def transform(self, x):
        return (x - self.mean) / self.std

    def inverse_transform(self, x):
        return x * self.std + self.mean


class DynamicsMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


# TODO: implement training function
def train_mlp_model(dataset):
    # concat numpy arrays
    a = np.vstack(dataset["a"])
    s = np.vstack(dataset["s"])
    s_next = np.vstack(dataset["s_next"])
    delta_s = s_next - s
    print(a.shape)
    print(s.shape)
    print(delta_s.shape)

    Y = delta_s
    X = np.concatenate([s, a], axis=-1)  # add actions as column


# TODO: train simple neural net to predict state transitons (s,a) -> (s_next - s)

dataset = collect_random_data(env, 1)
train_mlp_model(dataset)

input("Press Enter to close...")
env.close()
