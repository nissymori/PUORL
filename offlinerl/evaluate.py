from typing import Tuple

import d4rl
import gym
import jax
import jax.numpy as jnp
import numpy as np


def evaluate(
    policy_fn, env: gym.Env, num_episodes: int, obs_mean: float, obs_std: float
) -> float:
    episode_returns = []
    for _ in range(num_episodes):
        episode_return = 0
        observation, done = env.reset(), False
        while not done:
            observation = (observation - obs_mean) / (obs_std + 1e-5)
            action = policy_fn(observation)
            observation, reward, done, info = env.step(action)
            episode_return += reward
        episode_returns.append(episode_return)
    return env.get_normalized_score(np.mean(episode_returns)) * 100
