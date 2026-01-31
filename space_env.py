import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # gym fallback
    import gym
    from gym import spaces


@dataclass
class EnvConfig:
    task: str = "dock"  # 'dock' or 'orbit'
    dt: float = 0.1
    max_steps: int = 600
    continuous: bool = True
    max_thrust: float = 1.2
    fuel_capacity: float = 1.0
    fuel_cost: float = 0.08
    mu: float = 1.0  # gravitational parameter
    planet_radius: float = 0.35
    max_radius: float = 4.0
    station_radius: float = 0.08
    dock_radius: float = 0.12
    dock_speed: float = 0.25
    target_orbit_radius: float = 1.6
    stable_steps: int = 50
    pos_noise: float = 0.02
    vel_noise: float = 0.02
    sensor_noise: float = 0.01
    sensor_range: float = 4.0
    n_sensors: int = 8
    render_mode: Optional[str] = None


class SpaceNavigationEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, config: Optional[EnvConfig] = None):
        super().__init__()
        self.config = config or EnvConfig()
        self.rng = np.random.default_rng()

        self._init_spaces()
        self._init_render()
        self.reset()

    def _init_spaces(self) -> None:
        if self.config.continuous:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(5)  # 0 idle, 1 up, 2 right, 3 down, 4 left

        obs_dim = 5 + self.config.n_sensors
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

    def _init_render(self) -> None:
        self._fig = None
        self._ax = None
        self._trail = []

    def seed(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.seed(seed)

        self.t = 0
        self.fuel = self.config.fuel_capacity
        self.stable_count = 0

        if self.config.task == "dock":
            self.station = np.array([2.0, 0.0], dtype=np.float32)
            angle = self.rng.uniform(0, 2 * math.pi)
            radius = self.rng.uniform(1.2, 2.4)
            self.pos = np.array([radius * math.cos(angle), radius * math.sin(angle)], dtype=np.float32)
            self.vel = self.rng.normal(0, 0.2, size=2).astype(np.float32)
            self.sensor_target = "station"
        else:
            self.station = np.array([2.2, 0.0], dtype=np.float32)
            angle = self.rng.uniform(0, 2 * math.pi)
            radius = self.rng.uniform(1.2, 2.0)
            self.pos = np.array([radius * math.cos(angle), radius * math.sin(angle)], dtype=np.float32)
            # near circular orbit
            tangential = np.array([-math.sin(angle), math.cos(angle)], dtype=np.float32)
            v_circ = math.sqrt(self.config.mu / max(radius, 1e-4))
            self.vel = tangential * v_circ + self.rng.normal(0, 0.05, size=2).astype(np.float32)
            self.sensor_target = "planet"

        self.prev_dist = self._distance_to_target()
        obs = self._get_obs()
        info = {"state": self._get_state()}
        return obs, info

    def _get_state(self) -> dict:
        return {
            "pos": self.pos.copy(),
            "vel": self.vel.copy(),
            "fuel": float(self.fuel),
            "station": self.station.copy(),
        }

    def _distance_to_target(self) -> float:
        if self.config.task == "dock":
            return float(np.linalg.norm(self.pos - self.station))
        return float(abs(np.linalg.norm(self.pos) - self.config.target_orbit_radius))

    def _get_sensors(self) -> np.ndarray:
        directions = []
        for k in range(self.config.n_sensors):
            angle = 2 * math.pi * k / self.config.n_sensors
            directions.append(np.array([math.cos(angle), math.sin(angle)], dtype=np.float32))

        if self.sensor_target == "station":
            rel = self.station - self.pos
        else:
            rel = -self.pos

        sensors = []
        for d in directions:
            proj = float(np.dot(rel, d))
            if proj <= 0:
                val = 1.0
            else:
                val = min(proj / self.config.sensor_range, 1.0)
            val += self.rng.normal(0.0, self.config.sensor_noise)
            sensors.append(np.clip(val, 0.0, 1.0))
        return np.array(sensors, dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        pos_n = self.pos + self.rng.normal(0.0, self.config.pos_noise, size=2)
        vel_n = self.vel + self.rng.normal(0.0, self.config.vel_noise, size=2)
        fuel_frac = np.array([self.fuel / self.config.fuel_capacity], dtype=np.float32)
        sensors = self._get_sensors()
        obs = np.concatenate([pos_n, vel_n, fuel_frac, sensors]).astype(np.float32)
        return obs

    def _gravity(self) -> np.ndarray:
        r = np.linalg.norm(self.pos) + 1e-6
        return -self.config.mu * self.pos / (r**3)

    def _map_action(self, action) -> np.ndarray:
        if self.config.continuous:
            act = np.clip(action, -1.0, 1.0).astype(np.float32)
            return act * self.config.max_thrust
        if int(action) == 1:
            return np.array([0.0, 1.0], dtype=np.float32) * self.config.max_thrust
        if int(action) == 2:
            return np.array([1.0, 0.0], dtype=np.float32) * self.config.max_thrust
        if int(action) == 3:
            return np.array([0.0, -1.0], dtype=np.float32) * self.config.max_thrust
        if int(action) == 4:
            return np.array([-1.0, 0.0], dtype=np.float32) * self.config.max_thrust
        return np.zeros(2, dtype=np.float32)

    def step(self, action):
        thrust = self._map_action(action)
        thrust_mag = float(np.linalg.norm(thrust))
        fuel_use = self.config.fuel_cost * thrust_mag * self.config.dt

        if self.fuel <= 0.0:
            thrust = np.zeros(2, dtype=np.float32)
            fuel_use = 0.0
        else:
            self.fuel = max(0.0, self.fuel - fuel_use)

        accel = thrust + self._gravity()
        accel += self.rng.normal(0.0, 0.01, size=2)

        self.vel = self.vel + accel * self.config.dt
        self.pos = self.pos + self.vel * self.config.dt

        self.t += 1

        terminated = False
        truncated = False

        reward = 0.0
        info = {"state": self._get_state()}

        if np.linalg.norm(self.pos) < self.config.planet_radius:
            terminated = True
            reward -= 5.0
            info["event"] = "crash"
        if np.linalg.norm(self.pos) > self.config.max_radius:
            terminated = True
            reward -= 2.0
            info["event"] = "drift"

        if self.config.task == "dock":
            dist = np.linalg.norm(self.pos - self.station)
            speed = np.linalg.norm(self.vel)
            progress = self.prev_dist - dist
            reward += 1.2 * progress
            reward -= 0.05 * speed
            reward -= 0.6 * fuel_use

            if dist < self.config.dock_radius and speed < self.config.dock_speed:
                terminated = True
                reward += 10.0
                info["event"] = "dock"

            self.prev_dist = dist
        else:
            r = np.linalg.norm(self.pos) + 1e-6
            radial = np.dot(self.pos, self.vel) / r
            v_circ = math.sqrt(self.config.mu / r)
            tangential = np.linalg.norm(self.vel - radial * (self.pos / r))

            radial_err = abs(r - self.config.target_orbit_radius)
            tangential_err = abs(tangential - v_circ)

            reward -= 1.0 * radial_err
            reward -= 0.2 * abs(radial)
            reward -= 0.3 * tangential_err
            reward -= 0.6 * fuel_use

            if radial_err < 0.08 and abs(radial) < 0.1 and tangential_err < 0.2:
                self.stable_count += 1
            else:
                self.stable_count = 0

            if self.stable_count >= self.config.stable_steps:
                terminated = True
                reward += 8.0
                info["event"] = "stable_orbit"

        if self.t >= self.config.max_steps:
            truncated = True

        obs = self._get_obs()
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.config.render_mode != "human":
            return

        import matplotlib.pyplot as plt

        if self._fig is None:
            self._fig, self._ax = plt.subplots(figsize=(6, 6))
            self._ax.set_aspect("equal")
            self._ax.set_xlim(-self.config.max_radius, self.config.max_radius)
            self._ax.set_ylim(-self.config.max_radius, self.config.max_radius)
            self._ax.set_facecolor("#0b0f1a")

        self._ax.clear()
        self._ax.set_aspect("equal")
        self._ax.set_xlim(-self.config.max_radius, self.config.max_radius)
        self._ax.set_ylim(-self.config.max_radius, self.config.max_radius)
        self._ax.set_facecolor("#0b0f1a")

        # planet
        planet = plt.Circle((0, 0), self.config.planet_radius, color="#3d5a80")
        self._ax.add_patch(planet)

        # station
        if self.config.task == "dock":
            station = plt.Circle(tuple(self.station), self.config.station_radius, color="#e0fbfc")
            self._ax.add_patch(station)

        # trail
        self._trail.append(self.pos.copy())
        if len(self._trail) > 200:
            self._trail.pop(0)
        trail = np.array(self._trail)
        self._ax.plot(trail[:, 0], trail[:, 1], color="#98c1d9", linewidth=1)

        # ship
        self._ax.scatter([self.pos[0]], [self.pos[1]], color="#ee6c4d", s=40)

        self._ax.set_title(f"t={self.t} fuel={self.fuel:.2f}")
        plt.pause(0.001)

    def close(self):
        if self._fig is not None:
            import matplotlib.pyplot as plt

            plt.close(self._fig)
            self._fig = None
            self._ax = None
            self._trail = []
