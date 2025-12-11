from stable_baselines3.common.env_checker import check_env

from envs.slither_sim_env import SlitherSimEnv


def main() -> None:
    env = SlitherSimEnv()
    check_env(env, warn=True)


if __name__ == "__main__":
    main()
