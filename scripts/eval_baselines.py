# scripts/eval_baselines.py

from __future__ import annotations

from typing import Tuple

import numpy as np
import gymnasium as gym

import envs  # registra SlitherSim-v0


def run_random(num_episodes: int = 50) -> Tuple[float, float]:
    """Evalúa una política random en SlitherSim-v0."""
    env = gym.make("SlitherSim-v0")
    returns: list[float] = []

    for _ in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        returns.append(total_reward)

    return float(np.mean(returns)), float(np.std(returns))


def run_greedy_food(num_episodes: int = 50) -> Tuple[float, float]:
    """Baseline heurístico que gira hacia la comida según el ángulo relativo."""
    env = gym.make("SlitherSim-v0")
    returns: list[float] = []

    for _ in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            food_rel_angle_norm = float(obs[6])  # angle_food_norm en [-1, 1]

            if food_rel_angle_norm > 0.05:
                action = 2  # RIGHT
            elif food_rel_angle_norm < -0.05:
                action = 0  # LEFT
            else:
                action = 1  # STRAIGHT

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        returns.append(total_reward)

    return float(np.mean(returns)), float(np.std(returns))


def main() -> None:
    mean_r, std_r = run_random()
    print(f"Random policy: mean_reward={mean_r:.2f} +/- {std_r:.2f}")

    # Baseline heurístico: gira hacia la comida más cercana usando el ángulo relativo.
    # Suponemos que la observación tiene la forma:
    # [pos_x, pos_y, cos_theta, sin_theta, length_norm, dist_food_norm, angle_food_norm, dist_border_norm]
    # donde angle_food_norm = ángulo relativo comida-cabeza / pi en [-1, 1].
    greedy_mean, greedy_std = run_greedy_food()
    print(f"Greedy food: mean_reward={greedy_mean:.2f} +/- {greedy_std:.2f}")


if __name__ == "__main__":
    main()