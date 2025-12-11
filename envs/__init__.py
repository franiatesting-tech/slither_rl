from gymnasium.envs.registration import register

from .slither_sim_env import SlitherSimEnv  # noqa: F401  (para linters)


register(
    id="SlitherSim-v0",
    entry_point="envs.slither_sim_env:SlitherSimEnv",
)

register(
    id="SlitherSimRadar-v0",
    entry_point="envs.slither_sim_env:SlitherSimEnv",
    kwargs={"obs_dim": 19, "obs_type": "radar"},
)