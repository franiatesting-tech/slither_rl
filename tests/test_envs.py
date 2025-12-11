import numpy as np

from envs.slither_sim_env import SlitherSimEnv


def test_reset_and_step_shapes() -> None:
    env = SlitherSimEnv(obs_dim=8)
    obs, info = env.reset()
    assert obs.shape == (8,)

    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert obs.shape == (8,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_episode_terminates_on_border() -> None:
    env = SlitherSimEnv(obs_dim=8)
    obs, info = env.reset()
    env._x = env.world_size * 1.1
    env._y = 0.0
    obs, reward, terminated, truncated, info = env.step(1)
    assert terminated


def test_eating_food_increases_length_and_reward() -> None:
    env = SlitherSimEnv(obs_dim=8)
    obs, info = env.reset()
    env._x = 0.0
    env._y = 0.0
    env._theta = 0.0
    env._food_positions[0, 0] = 0.0
    env._food_positions[0, 1] = 0.0
    old_length = env._length
    obs, reward, terminated, truncated, info = env.step(1)
    assert env._length > old_length
    assert reward > 0.0


def test_reset_seed_reproducible() -> None:
    env1 = SlitherSimEnv(obs_dim=8)
    env2 = SlitherSimEnv(obs_dim=8)

    obs1, _ = env1.reset(seed=123)
    obs2, _ = env2.reset(seed=123)

    assert np.allclose(obs1, obs2)