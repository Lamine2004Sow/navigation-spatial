import argparse
import time

import numpy as np
import torch

from models import PolicyValueNet, RunningNorm, safe_torch_load
from space_env import EnvConfig, SpaceNavigationEnv


def load_policy(ckpt_path, obs_dim, act_dim, device):
    data = safe_torch_load(ckpt_path, device)
    model = PolicyValueNet(obs_dim, act_dim).to(device)
    model.load_state_dict(data["model"])
    model.eval()

    norm = RunningNorm(obs_dim)
    if "norm" in data:
        norm.load_state_dict(data["norm"])
    return model, norm


def run_episode(env, model, norm, device, deterministic=False, max_steps=600):
    obs, _ = env.reset()
    total = 0.0
    for _ in range(max_steps):
        obs_n = norm.normalize(obs)
        obs_t = torch.tensor(obs_n, dtype=torch.float32, device=device)
        with torch.no_grad():
            mean, log_std, _ = model(obs_t)
            if deterministic:
                action = mean
            else:
                std = torch.exp(log_std)
                action = torch.distributions.Normal(mean, std).sample()
        action_np = action.cpu().numpy()

        obs, reward, terminated, truncated, info = env.step(action_np)
        total += reward
        env.render()
        if terminated or truncated:
            break
        time.sleep(0.01)
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["dock", "orbit"], default="dock")
    parser.add_argument("--ckpt", type=str, default="artifacts/ppo_dock.pt")
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    env = SpaceNavigationEnv(EnvConfig(task=args.task, render_mode="human"))
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, norm = load_policy(args.ckpt, obs_dim, act_dim, device)

    total = run_episode(env, model, norm, device, deterministic=args.deterministic)
    print(f"episode return: {total:.2f}")


if __name__ == "__main__":
    main()
