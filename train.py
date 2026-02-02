import argparse
import csv
import os
import time

import numpy as np
import torch

from models import PPOConfig, PolicyValueNet, RunningNorm
from space_env import EnvConfig, SpaceNavigationEnv


def make_env(task: str, render_mode=None):
    config = EnvConfig(task=task, render_mode=render_mode)
    return SpaceNavigationEnv(config)


def compute_gae(rewards, values, dones, last_value, gamma, lam):
    adv = np.zeros_like(rewards)
    last_adv = 0.0
    for t in reversed(range(len(rewards))):
        next_nonterminal = 1.0 - dones[t]
        next_value = last_value if t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        adv[t] = last_adv = delta + gamma * lam * next_nonterminal * last_adv
    returns = adv + values
    return adv, returns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["dock", "orbit"], default="dock")
    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument("--horizon", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--minibatch", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="artifacts")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render-every", type=int, default=2)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    render_mode = "human" if args.render else None
    env = make_env(args.task, render_mode=render_mode)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PolicyValueNet(obs_dim, act_dim).to(device)
    norm = RunningNorm(obs_dim)
    cfg = PPOConfig()

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, "reward.csv")
    ckpt_path = os.path.join(args.out, f"ppo_{args.task}.pt")

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["update", "steps", "avg_return", "avg_ep_len"])

    obs, _ = env.reset(seed=args.seed)
    episode_returns = []
    episode_lens = []
    ep_return = 0.0
    ep_len = 0

    total_steps = 0
    update = 0
    start_time = time.time()

    while total_steps < args.total_steps:
        obs_buf = np.zeros((args.horizon, obs_dim), dtype=np.float32)
        act_buf = np.zeros((args.horizon, act_dim), dtype=np.float32)
        logp_buf = np.zeros((args.horizon,), dtype=np.float32)
        rew_buf = np.zeros((args.horizon,), dtype=np.float32)
        val_buf = np.zeros((args.horizon,), dtype=np.float32)
        done_buf = np.zeros((args.horizon,), dtype=np.float32)

        for t in range(args.horizon):
            norm.update(obs)
            obs_n = norm.normalize(obs)
            obs_buf[t] = obs_n

            obs_t = torch.tensor(obs_n, dtype=torch.float32, device=device)
            with torch.no_grad():
                action_t, logp_t, value_t = model.act(obs_t)

            action = action_t.cpu().numpy()
            logp = logp_t.cpu().numpy()
            value = value_t.cpu().numpy()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = float(terminated or truncated)

            act_buf[t] = action
            logp_buf[t] = logp
            rew_buf[t] = reward
            val_buf[t] = value
            done_buf[t] = done

            ep_return += reward
            ep_len += 1
            total_steps += 1

            if args.render and args.render_every > 0 and total_steps % args.render_every == 0:
                env.render()

            if done:
                episode_returns.append(ep_return)
                episode_lens.append(ep_len)
                ep_return = 0.0
                ep_len = 0
                next_obs, _ = env.reset()

            obs = next_obs

        with torch.no_grad():
            obs_n = norm.normalize(obs)
            last_value = model(
                torch.tensor(obs_n, dtype=torch.float32, device=device)
            )[2].cpu().numpy()

        adv_buf, ret_buf = compute_gae(
            rew_buf, val_buf, done_buf, last_value, cfg.gamma, cfg.gae_lambda
        )
        adv_buf = (adv_buf - adv_buf.mean()) / (adv_buf.std() + 1e-8)

        obs_t = torch.tensor(obs_buf, dtype=torch.float32, device=device)
        act_t = torch.tensor(act_buf, dtype=torch.float32, device=device)
        logp_old_t = torch.tensor(logp_buf, dtype=torch.float32, device=device)
        adv_t = torch.tensor(adv_buf, dtype=torch.float32, device=device)
        ret_t = torch.tensor(ret_buf, dtype=torch.float32, device=device)

        for _ in range(args.epochs):
            idx = np.random.permutation(args.horizon)
            for start in range(0, args.horizon, args.minibatch):
                mb = idx[start : start + args.minibatch]

                logp, entropy, value = model.evaluate_actions(obs_t[mb], act_t[mb])
                ratio = torch.exp(logp - logp_old_t[mb])
                surr1 = ratio * adv_t[mb]
                surr2 = torch.clamp(ratio, 1 - cfg.clip_ratio, 1 + cfg.clip_ratio) * adv_t[mb]
                actor_loss = -torch.min(surr1, surr2).mean()

                value_loss = ((value - ret_t[mb]) ** 2).mean()
                entropy_loss = -entropy.mean()

                loss = actor_loss + cfg.value_coef * value_loss + cfg.entropy_coef * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()

        update += 1
        avg_return = float(np.mean(episode_returns[-10:])) if episode_returns else 0.0
        avg_len = float(np.mean(episode_lens[-10:])) if episode_lens else 0.0

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([update, total_steps, f"{avg_return:.3f}", f"{avg_len:.1f}"])

        if update % 5 == 0 or total_steps >= args.total_steps:
            torch.save(
                {"model": model.state_dict(), "norm": norm.state_dict(), "cfg": vars(cfg)},
                ckpt_path,
            )

        if update % 10 == 0:
            elapsed = time.time() - start_time
            print(
                f"update {update} steps {total_steps} avg_return {avg_return:.2f} avg_len {avg_len:.1f} time {elapsed:.1f}s"
            )

    env.close()
    torch.save({"model": model.state_dict(), "norm": norm.state_dict()}, ckpt_path)
    print(f"saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
