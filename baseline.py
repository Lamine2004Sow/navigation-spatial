import argparse
import math

import numpy as np

from space_env import EnvConfig, SpaceNavigationEnv


def clamp_action(action):
    return np.clip(action, -1.0, 1.0)


def dock_policy(state, station):
    pos = state["pos"]
    vel = state["vel"]
    vec = station - pos
    thrust = 0.8 * vec - 0.4 * vel
    return clamp_action(thrust)


def orbit_policy(state, mu, target_radius):
    pos = state["pos"]
    vel = state["vel"]
    r = np.linalg.norm(pos) + 1e-6
    radial = pos / r
    tangential = np.array([-radial[1], radial[0]], dtype=np.float32)
    v_circ = math.sqrt(mu / r)
    desired_vel = tangential * v_circ

    radial_error = target_radius - r
    thrust = 0.6 * radial_error * radial + 0.6 * (desired_vel - vel)
    return clamp_action(thrust)


def run_episode(env, render=False):
    obs, info = env.reset()
    total = 0.0
    for _ in range(env.config.max_steps):
        state = info["state"]
        if env.config.task == "dock":
            action = dock_policy(state, env.station)
        else:
            action = orbit_policy(state, env.config.mu, env.config.target_orbit_radius)

        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        if render:
            env.render()
        if terminated or truncated:
            break
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["dock", "orbit"], default="dock")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    env = SpaceNavigationEnv(EnvConfig(task=args.task, render_mode="human" if args.render else None))

    returns = []
    for _ in range(args.episodes):
        returns.append(run_episode(env, render=args.render))

    avg_return = sum(returns) / len(returns)
    print(f"avg return over {args.episodes} episodes: {avg_return:.2f}")


if __name__ == "__main__":
    main()
