# navigation-spatial

A compact, self-contained project that mixes simplified space physics, deep learning, and reinforcement learning.
The goal is to train an autonomous agent to dock with a station or maintain a stable orbit using partial, noisy observations.

## Concept

- 2D continuous space with inertia, directional thrust, gravity, and limited fuel.
- Partial observations: noisy position and velocity plus coarse directional sensors.
- RL formulation: a POMDP (noisy, partial sensing). We still train a memoryless policy as a practical approximation.
- Agent: perception MLP + actor-critic (PPO) with input normalization.

## Files

- `space_env.py` : Gym-compatible environment with docking and orbit tasks.
- `models.py` : perception network, actor-critic, running normalization, PPO config.
- `train.py` : PPO training loop with logging and checkpointing.
- `evaluate.py` : run a trained policy with live rendering.
- `baseline.py` : naive controller for comparison.
- `plot_rewards.py` : training curve visualization.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy torch gymnasium matplotlib

# Train a docking policy
python train.py --task dock --total-steps 200000

# Train with live visualization (slower)
python train.py --task dock --total-steps 200000 --render --render-every 2

# Visualize a trained policy
python evaluate.py --task dock --ckpt artifacts/ppo_dock.pt --deterministic

# Compare with a naive controller
python baseline.py --task dock --episodes 10

# Plot rewards
python plot_rewards.py --log artifacts/reward.csv

# Generate LaTeX report assets (figures + results tables)
python latex/generate_assets.py --episodes 30

# Compile the LaTeX report manually (requires pdflatex)
cd latex
pdflatex report.tex

# Makefile shortcuts
make venv deps
make train TASK=dock STEPS=200000
make train TASK=dock STEPS=200000 RENDER=1 RENDER_EVERY=2
make eval TASK=dock CKPT=artifacts/ppo_dock.pt DETERMINISTIC=1
make plot LOG=artifacts/reward.csv
make report REPORT_EPISODES=30
```

## Environment details

- State (hidden): position, velocity, fuel.
- Observations: noisy position + velocity, fuel fraction, directional sensors.
- Actions: continuous thrust in x/y (clipped to [-1, 1]).
- Dynamics: Euler integration, central gravity, small perturbation noise.

### Rewards

Docking task:
- positive reward for progress toward station
- penalty for speed and fuel usage
- large bonus for successful dock

Orbit task:
- penalty for radius error and radial velocity
- penalty for tangential speed error from circular orbit
- bonus for a sustained stable orbit

## Project phases

1) Implement physics and the Gym-compatible environment.
2) Train a simple baseline PPO agent.
3) Add a perception network with input normalization.
4) Improve via reward shaping and curriculum (optional).

## Experiment ideas

- Compare PPO vs naive policy in `baseline.py`.
- Analyze trajectories from `evaluate.py`.
- Modify sensor count, noise level, or fuel cost.
- Curriculum: start without gravity, then enable it.

## Notes on control and space navigation

The dock task is a simplified rendezvous problem with thrust limits and fuel cost.
The orbit task mirrors a stabilizing control problem: keep radius and tangential velocity near a target.
Reward shaping acts like a soft optimal control objective (trade off fuel vs tracking error).

## Troubleshooting

- If training is unstable, reduce `max_thrust` or increase `fuel_cost` in `space_env.py`.
- If rewards plateau, try lowering `sensor_noise` or increasing `total_steps`.
- If rendering is slow, use `--deterministic` to reduce jitter.
