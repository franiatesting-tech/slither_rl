from gymnasium import spaces
import numpy as np

from .base_env import BaseSlitherEnv


class SlitherSimEnv(BaseSlitherEnv):
    """Entorno simulado mínimo para Slither.io (MVP del Hito 1).

    La dinámica se detallará en la fase de diseño; por ahora es un esqueleto
    que devuelve observaciones y recompensas neutras.
    """

    N_SECTORS = 8

    def __init__(
        self,
        obs_dim: int = 8,
        world_size: float = 1.0,
        max_steps: int = 200,
        n_food: int = 10,
        speed: float = 0.05,
        turn_speed: float = float(np.pi / 16.0),
        food_radius: float = 0.05,
        food_reward: float = 2.0,
        death_penalty: float = 30.0,
        step_penalty: float = 0.02,
        obs_type: str = "features",
    ) -> None:
        super().__init__()
        # Reward config B (tuned against random baseline):
        # - small living cost
        # - big death penalty
        # - moderate food reward
        self.obs_dim = obs_dim
        self._obs_type = obs_type
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)

        self.world_size = float(world_size)
        self.max_steps = int(max_steps)
        self.n_food = int(n_food)
        self.speed = float(speed)
        self.turn_speed = float(turn_speed)
        self.food_radius = float(food_radius)
        self.food_reward = float(food_reward)
        self.death_penalty = float(death_penalty)
        self.step_penalty = float(step_penalty)

        self._rng = np.random.default_rng()
        self._step_count = 0
        self._x = 0.0
        self._y = 0.0
        self._theta = 0.0
        self._length = 1.0
        self._food_positions = np.zeros((self.n_food, 2), dtype=np.float32)

    def _reset_state(self) -> None:
        self._step_count = 0
        self._x = 0.0
        self._y = 0.0
        self._theta = 0.0
        self._length = 1.0
        self._food_positions = self._rng.uniform(
            low=-self.world_size,
            high=self.world_size,
            size=(self.n_food, 2),
        ).astype(np.float32)

    def _build_obs(self) -> np.ndarray:
        if self._obs_type == "features":
            return self._build_obs_features()
        if self._obs_type == "radar":
            return self._build_obs_radar()
        raise ValueError(f"Unknown obs_type={self._obs_type}")

    def _build_obs_features(self) -> np.ndarray:
        pos_x = float(np.clip(self._x / self.world_size, -1.0, 1.0))
        pos_y = float(np.clip(self._y / self.world_size, -1.0, 1.0))
        cos_t = float(np.cos(self._theta))
        sin_t = float(np.sin(self._theta))
        length_norm = float(self._length / (1.0 + self._length))

        if self.n_food > 0:
            diffs = self._food_positions - np.array([self._x, self._y], dtype=np.float32)
            dists = np.linalg.norm(diffs, axis=1)
            idx = int(np.argmin(dists))
            dist = float(dists[idx])
            dx = float(diffs[idx, 0])
            dy = float(diffs[idx, 1])
        else:
            dist = 2.0 * self.world_size
            dx = 0.0
            dy = 0.0

        dist_food_norm = float(np.clip(dist / (2.0 * self.world_size), 0.0, 1.0))
        angle = float(np.arctan2(dy, dx) - self._theta)
        angle = float((angle + np.pi) % (2.0 * np.pi) - np.pi)
        angle_food_norm = angle / np.pi

        cos_t_val = np.cos(self._theta)
        sin_t_val = np.sin(self._theta)
        eps = 1e-6
        candidates = []
        if abs(cos_t_val) > eps:
            if cos_t_val > 0.0:
                dist_x = (self.world_size - self._x) / cos_t_val
            else:
                dist_x = (-self.world_size - self._x) / cos_t_val
            if dist_x > 0.0:
                candidates.append(dist_x)
        if abs(sin_t_val) > eps:
            if sin_t_val > 0.0:
                dist_y = (self.world_size - self._y) / sin_t_val
            else:
                dist_y = (-self.world_size - self._y) / sin_t_val
            if dist_y > 0.0:
                candidates.append(dist_y)
        if candidates:
            dist_border = float(min(candidates))
        else:
            dist_border = 2.0 * self.world_size
        dist_border_norm = float(np.clip(dist_border / (2.0 * self.world_size), 0.0, 1.0))

        features = [
            pos_x,
            pos_y,
            cos_t,
            sin_t,
            length_norm,
            dist_food_norm,
            angle_food_norm,
            dist_border_norm,
        ]
        if len(features) < self.obs_dim:
            features.extend([0.0] * (self.obs_dim - len(features)))
        elif len(features) > self.obs_dim:
            features = features[: self.obs_dim]
        return np.asarray(features, dtype=np.float32)

    def _build_obs_radar(self) -> np.ndarray:
        cos_t = float(np.cos(self._theta))
        sin_t = float(np.sin(self._theta))
        length_norm = float(self._length / (1.0 + self._length))

        max_dist = float(self.world_size * np.sqrt(2.0))

        food_dists = np.ones(self.N_SECTORS, dtype=np.float32)
        border_dists = np.zeros(self.N_SECTORS, dtype=np.float32)

        sector_width = 2.0 * np.pi / float(self.N_SECTORS)

        for k in range(self.N_SECTORS):
            phi_k = self._theta + sector_width * float(k)
            dx = float(np.cos(phi_k))
            dy = float(np.sin(phi_k))
            eps = 1e-6
            candidates = []
            if abs(dx) > eps:
                if dx > 0.0:
                    t_x = (self.world_size - self._x) / dx
                else:
                    t_x = (-self.world_size - self._x) / dx
                if t_x > 0.0:
                    candidates.append(t_x)
            if abs(dy) > eps:
                if dy > 0.0:
                    t_y = (self.world_size - self._y) / dy
                else:
                    t_y = (-self.world_size - self._y) / dy
                if t_y > 0.0:
                    candidates.append(t_y)
            if candidates:
                dist_border = float(min(candidates))
            else:
                dist_border = 2.0 * self.world_size
            border_dists[k] = float(np.clip(dist_border / max_dist, 0.0, 1.0))

        if self.n_food > 0:
            diffs = self._food_positions - np.array([self._x, self._y], dtype=np.float32)
            dists = np.linalg.norm(diffs, axis=1)
            angles = np.arctan2(diffs[:, 1], diffs[:, 0]) - self._theta
            angles = (angles + np.pi) % (2.0 * np.pi) - np.pi

            for dist, ang in zip(dists, angles):
                dist_norm = float(np.clip(dist / max_dist, 0.0, 1.0))
                idx = int(np.floor((ang + np.pi) / sector_width))
                if idx < 0:
                    idx = 0
                elif idx >= self.N_SECTORS:
                    idx = self.N_SECTORS - 1
                if dist_norm < float(food_dists[idx]):
                    food_dists[idx] = dist_norm

        features = [cos_t, sin_t, length_norm]
        features.extend(food_dists.tolist())
        features.extend(border_dists.tolist())

        if len(features) < self.obs_dim:
            features.extend([0.0] * (self.obs_dim - len(features)))
        elif len(features) > self.obs_dim:
            features = features[: self.obs_dim]
        return np.asarray(features, dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):  # type: ignore[override]
        super().reset(seed=seed)
        if hasattr(self, "np_random"):
            self._rng = self.np_random
        self._reset_state()
        obs = self._build_obs()
        info: dict = {}
        return obs, info

    def step(self, action):  # type: ignore[override]
        self._step_count += 1
        action_int = int(action)
        speed = self.speed
        if action_int == 0:
            self._theta -= self.turn_speed
        elif action_int == 2:
            self._theta += self.turn_speed
        elif action_int == 3:
            speed = self.speed * 1.5
        self._theta = float((self._theta + np.pi) % (2.0 * np.pi) - np.pi)

        self._x += speed * float(np.cos(self._theta))
        self._y += speed * float(np.sin(self._theta))

        terminated = bool(
            abs(self._x) > self.world_size or abs(self._y) > self.world_size
        )
        reward = -self.step_penalty

        if not terminated and self.n_food > 0:
            diffs = self._food_positions - np.array([self._x, self._y], dtype=np.float32)
            dists = np.linalg.norm(diffs, axis=1)
            eat_indices = np.where(dists <= self.food_radius)[0]
            if eat_indices.size > 0:
                eaten = int(eat_indices.size)
                reward += self.food_reward * float(eaten)
                self._length += 0.1 * float(eaten)
                for idx in eat_indices:
                    self._food_positions[idx] = self._rng.uniform(
                        low=-self.world_size,
                        high=self.world_size,
                        size=(2,),
                    ).astype(np.float32)

        if terminated:
            reward -= self.death_penalty

        truncated = bool(not terminated and self._step_count >= self.max_steps)
        obs = self._build_obs()
        info: dict = {}
        return obs, float(reward), terminated, truncated, info
