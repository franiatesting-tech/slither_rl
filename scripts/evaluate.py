# scripts/evaluate.py

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

import envs  # registra SlitherSim-v0


def main() -> None:
    base_env = gym.make("SlitherSim-v0")
    env = Monitor(base_env)

    model = PPO.load("models/ppo_slither_latest")

    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=50,
        deterministic=True,
        warn=False,  # ya sabemos que está envuelto
    )

    print(f"Eval PPO (Monitor): mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")


if __name__ == "__main__":
    main()