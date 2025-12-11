import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import envs  # noqa: F401  # registra SlitherSim-v0 y SlitherSimRadar-v0


def train_one(env_id: str, logdir: str, model_prefix: str, seed: int) -> None:
    env = make_vec_env(env_id, n_envs=8, seed=seed)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=logdir,
        seed=seed,
    )
    model.learn(total_timesteps=300_000)
    model.save(f"models/{model_prefix}_seed{seed}")


def main() -> None:
    # 3 seeds para features
    for seed in [0, 1, 2]:
        train_one("SlitherSim-v0", "logs/tb_slither", "ppo_slither", seed)

    # 3 seeds para radar
    for seed in [0, 1, 2]:
        train_one("SlitherSimRadar-v0", "logs/tb_slither_radar", "ppo_slither_radar", seed)


if __name__ == "__main__":
    main()
