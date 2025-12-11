import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

import envs  # noqa: F401  # registra SlitherSim-v0 y SlitherSimRadar-v0


def eval_model(env_id: str, model_path: str, label: str) -> None:
    base_env = gym.make(env_id)
    env = Monitor(base_env)
    model = PPO.load(model_path)
    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=50,
        deterministic=True,
        warn=False,
    )
    print(f"{label}: mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")


def main() -> None:
    # 3 seeds para features
    for seed in [0, 1, 2]:
        eval_model(
            "SlitherSim-v0",
            f"models/ppo_slither_seed{seed}",
            f"PPO_features_seed{seed}",
        )

    # 3 seeds para radar
    for seed in [0, 1, 2]:
        eval_model(
            "SlitherSimRadar-v0",
            f"models/ppo_slither_radar_seed{seed}",
            f"PPO_radar_seed{seed}",
        )


if __name__ == "__main__":
    main()
