from typing import Callable, Dict, Tuple

import d4rl
import gym
import jax
import jax.numpy as jnp
import numpy as np
from einops import einops


def make_evaluation(
    env,
    config: Dict,
    obs_mean: float,
    obs_std: float,
    actor_fn: Callable,
    vectorized: bool = True,
) -> Callable:
    actor_vj = jax.jit(jax.vmap(actor_fn))
    actor_j = jax.jit(actor_fn)
    n_seeds = config.n_seeds

    def eval_d4rl_single(train_state):
        n_seeds = config.n_seeds
        returns = []
        for i in range(n_seeds):
            actor_params = jax.tree_util.tree_map(
                lambda x: x[i], train_state.actor.params
            )
            ts = train_state._replace(
                actor=train_state.actor.replace(params=actor_params)
            )
            episode_returns = []
            for _ in range(config.eval_episodes):
                obs = env.reset()
                done = False
                episode_return = 0
                while not done:
                    obs = (obs - obs_mean) / (obs_std)
                    action = actor_j(ts, obs)
                    obs, reward, done, _ = env.step(action)
                    episode_return += reward
                episode_returns.append(episode_return)
            returns.append(np.mean(episode_returns))
        return np.array(returns)

    def eval_d4rl_vectorized(train_state):
        num_envs = n_seeds * config.eval_episodes
        obs = env.reset()
        dones = jnp.zeros(num_envs)
        episode_returns = jnp.zeros((n_seeds, config.eval_episodes))
        while not jnp.all(dones):
            obs = (obs - obs_mean) / (obs_std)  # normalize states
            obs_reshaped = einops.rearrange(
                obs, "(n e) o -> n e o", n=n_seeds, e=config.eval_episodes
            )  # (n_seeds, eval_episodes, obs_dim)
            actions_reshaped = actor_vj(
                train_state, obs_reshaped
            )  # (n_seeds, eval_episodes, action_dim)
            actions = einops.rearrange(
                actions_reshaped,
                "n e a -> (n e) a",
                n=n_seeds,
                e=config.eval_episodes,
            )  # (n_seeds * eval_episodes, action_dim)
            actions = np.array(actions).astype(np.float64)
            obs, rewards, step_dones, _ = env.step(
                actions
            )  # (n_seeds * eval_episodes, obs_dim)
            rewards = jnp.where(dones, jnp.zeros_like(rewards), rewards)
            dones = jnp.logical_or(dones, step_dones)
            reward_reshaped = einops.rearrange(
                rewards, "(n e) -> n e", n=n_seeds, e=config.eval_episodes
            )
            episode_returns += reward_reshaped
        return jnp.mean(episode_returns, axis=1)  # (n_seeds,)

    return eval_d4rl_single if not vectorized else eval_d4rl_vectorized
