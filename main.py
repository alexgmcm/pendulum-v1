import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import argparse
import json

env = gym.make("Pendulum-v1")
# obs, info = env.reset(seed=42)


def collect_random_data(env, episodes=10, write_to_file=False):
    # get data from random policy
    dataset = {"s": [], "a": [], "r": [], "s_next": [], "done": []}
    for i in range(episodes):
        s, _ = env.reset()
        done = False
        while not done:
            a = env.action_space.sample()
            s_next, r, terminated, truncated, info = env.step(a)
            done = terminated or truncated
            dataset["s"].append(s)
            dataset["a"].append(a)
            dataset["r"].append(r)
            dataset["s_next"].append(s_next)
            dataset["done"].append(done)
            s = s_next

    if write_to_file:
        for k, v in dataset.items():
            print(f"{k}: {type(dataset[k][0])}")
            if isinstance(dataset[k][0], np.ndarray):
                dataset[k] = [x.tolist() for x in dataset[k]]
        with open("output.txt", "w") as f:
            json.dump(dataset, f, indent=4)

    return dataset


class Standardizer:
    def __init__(self, eps=1e-8):
        self.eps = eps
        self.mean = None
        self.std = None

    def fit(self, x):
        self.mean = x.mean(axis=0, keepdims=True)
        self.std = x.std(axis=0, keepdims=True)
        self.std[self.std < 1e-8] = 1.0
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


def train_dynamics_model(
    states,
    actions,
    next_states,
    hidden_dim=128,
    batch_size=256,
    lr=1e-3,
    weight_decay=1e-5,
    max_epochs=200,
    patience=20,
    device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build supervised dataset
    X = np.concatenate([states, actions], axis=1).astype(np.float32)
    Y = (next_states - states).astype(np.float32)

    # Train/val split
    N = len(X)
    perm = np.random.permutation(N)
    train_size = int(0.9 * N)
    train_idx = perm[:train_size]
    val_idx = perm[train_size:]

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]

    # Fit normalizers on train only
    x_scaler = Standardizer().fit(X_train)
    y_scaler = Standardizer().fit(Y_train)

    X_train_n = x_scaler.transform(X_train).astype(np.float32)
    Y_train_n = y_scaler.transform(Y_train).astype(np.float32)
    X_val_n = x_scaler.transform(X_val).astype(np.float32)
    Y_val_n = y_scaler.transform(Y_val).astype(np.float32)

    # PyTorch datasets
    train_ds = TensorDataset(
        torch.from_numpy(X_train_n),
        torch.from_numpy(Y_train_n),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val_n),
        torch.from_numpy(Y_val_n),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Model
    model = DynamicsMLP(
        input_dim=X.shape[1],
        output_dim=Y.shape[1],
        hidden_dim=hidden_dim,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_state_dict = None
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        # ---- Training ----
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb)
            loss = loss_fn(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * xb.size(0)
            train_count += xb.size(0)

        train_loss = train_loss_sum / train_count

        # ---- Validation ----
        model.eval()
        val_loss_sum = 0.0
        val_count = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                pred = model(xb)
                loss = loss_fn(pred, yb)

                val_loss_sum += loss.item() * xb.size(0)
                val_count += xb.size(0)

        val_loss = val_loss_sum / val_count

        print(
            f"Epoch {epoch+1:3d} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f}"
        )

        # ---- Early stopping ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return model, x_scaler, y_scaler


def save_model(path, model, x_scaler, y_scaler):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "x_mean": x_scaler.mean,
            "x_std": x_scaler.std,
            "y_mean": y_scaler.mean,
            "y_std": y_scaler.std,
        },
        path,
    )


def load_model(path, input_dim=4, output_dim=3, hidden_dim=128, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model = DynamicsMLP(input_dim, output_dim, hidden_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    x_scaler = Standardizer()
    x_scaler.mean = checkpoint["x_mean"]
    x_scaler.std = checkpoint["x_std"]

    y_scaler = Standardizer()
    y_scaler.mean = checkpoint["y_mean"]
    y_scaler.std = checkpoint["y_std"]

    return model, x_scaler, y_scaler


@torch.no_grad()
def predict_next_state(model, x_scaler, y_scaler, state, action, device=None):
    if device is None:
        device = next(model.parameters()).device

    single = state.ndim == 1
    if single:
        state = state[None, :]
        action = action[None, :]

    x = np.concatenate([state, action], axis=1).astype(np.float32)
    x_n = x_scaler.transform(x).astype(np.float32)

    x_tensor = torch.from_numpy(x_n).to(device)
    delta_n = model(x_tensor).cpu().numpy()
    delta = y_scaler.inverse_transform(delta_n)

    next_state_pred = state + delta
    return next_state_pred[0] if single else next_state_pred


@torch.no_grad()
def evaluate_one_step(
    model, x_scaler, y_scaler, states, actions, next_states, device=None
):
    pred_next = predict_next_state(
        model, x_scaler, y_scaler, states, actions, device=device
    )
    mse = np.mean((pred_next - next_states) ** 2)
    return mse


# for this we need to create 2nd validation set and test trajectories
# TODO: Fix model so the predicted trajectories don't explode
@torch.no_grad()
def rollout_model(model, x_scaler, y_scaler, init_state, actions, device=None):
    preds = [init_state.copy()]
    s = init_state.copy()

    for a in actions:
        s = predict_next_state(model, x_scaler, y_scaler, s, a, device=device)
        preds.append(s.copy())

    return np.array(preds)


def pendulum_reward(state, action):
    cos_theta, sin_theta, theta_dot = state
    theta = np.arctan2(sin_theta, cos_theta)
    u = action[0]

    cost = theta**2 + 0.1 * theta_dot**2 + 0.001 * (u**2)
    return -cost


parser = argparse.ArgumentParser()
parser.add_argument("--train", action="store_true")
parser.add_argument("--store_trajectories", action="store_true")
parser.add_argument("--simulate_trajectories", action="store_true")
args = parser.parse_args()

if args.store_trajectories:
    collect_random_data(env, 5, write_to_file=True)
elif args.train:
    dataset = collect_random_data(env, 50)
    a = np.vstack(dataset["a"])
    s = np.vstack(dataset["s"])
    s_next = np.vstack(dataset["s_next"])

    model, x_scaler, y_scaler = train_dynamics_model(
        states=a, actions=s, next_states=s_next
    )
    save_model("dynamics_checkpoint.pt", model, x_scaler, y_scaler)
else:
    model, x_scaler, y_scaler = load_model(
        "dynamics_checkpoint.pt",
    )


if args.simulate_trajectories:
    dataset = collect_random_data(env, 1)
    endpoints = [i for i, done in enumerate(dataset["done"]) if done]
    rmse_list = []
    for i, endpoint in enumerate(endpoints):
        min_range = endpoints[i - 1] if i != 0 else 0
        states = dataset["s"][min_range : (endpoint + 1)]
        init_state = states[0]
        actions = dataset["a"][min_range:(endpoint)]

        preds = rollout_model(
            model, x_scaler, y_scaler, init_state, actions, device=None
        )

        preds_json = [x.tolist() for x in preds]
        with open("preds.txt", "w") as f:
            json.dump(preds_json, f, indent=4)

        print(states[:5])
        print(preds[:5])
        rmse = np.sqrt(np.mean((preds - states) ** 2))
        print(np.max(np.abs(preds)))
        print(np.max(np.abs(states)))
        print(np.max(np.abs(preds - states)))
        rmse_list.append(rmse)
    print(f"avg rmse: {np.mean(rmse_list)}")


# TODO: Implement random-shooting MPC

input("Press Enter to close...")
env.close()
