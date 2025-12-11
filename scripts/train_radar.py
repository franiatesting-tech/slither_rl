import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import envs  # registra SlitherSimRadar-v0


def main() -> None:
    env = make_vec_env("SlitherSimRadar-v0", n_envs=8)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="logs/tb_slither_radar",
    )
    model.learn(total_timesteps=200_000)
    model.save("models/ppo_slither_radar_latest")


if __name__ == "__main__":
    main()
