import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import envs  # noqa: F401  (registra SlitherSim-v0)


def main() -> None:

    # 8 entornos en paralelo
    env = make_vec_env("SlitherSim-v0", n_envs=8)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="logs/tb_slither",
    )
    model.learn(total_timesteps=200_000)
    model.save("models/ppo_slither_latest")


if __name__ == "__main__":
    main()