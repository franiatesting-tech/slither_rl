import gymnasium as gym
from abc import ABC, abstractmethod


class BaseSlitherEnv(gym.Env, ABC):
    """Interfaz base para entornos tipo Slither.io."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 15}

    @abstractmethod
    def reset(self, *, seed: int | None = None, options: dict | None = None):  # type: ignore[override]
        """Reinicia el entorno y devuelve la primera observación."""
        super().reset(seed=seed)

    @abstractmethod
    def step(self, action):  # type: ignore[override]
        """Avanza un paso en el entorno."""
        raise NotImplementedError
