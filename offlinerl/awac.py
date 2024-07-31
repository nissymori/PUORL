# Copied and modifeid from JAX-CORL: https://github.com/nissymori/JAX-CORL/blob/main/algos/awac.py
# https://arxiv.org/abs/2006.09359
import os
import time
from functools import partial
from typing import Any, Callable, Dict, NamedTuple, Optional, Sequence, Tuple

import d4rl
import distrax
import flax
import flax.linen as nn
import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm
import wandb
from flax.training.train_state import TrainState
from omegaconf import OmegaConf
from pydantic import BaseModel

from .dataset import Transition

os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True "


def default_init(scale: Optional[float] = 1.0):
    return nn.initializers.variance_scaling(scale, "fan_avg", "uniform")


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: bool = False
    kernel_init: Callable[[Any, Sequence[int], Any], jnp.ndarray] = default_init()
    add_layer_norm: bool = False
    layer_norm_final: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, hidden_dims in enumerate(self.hidden_dims):
            x = nn.Dense(hidden_dims, kernel_init=self.kernel_init)(x)
            if self.add_layer_norm:  # Add layer norm after activation
                if self.layer_norm_final or i + 1 < len(self.hidden_dims):
                    x = nn.LayerNorm()(x)
            if (
                i + 1 < len(self.hidden_dims) or self.activate_final
            ):  # Add activation after layer norm
                x = self.activations(x)
        return x


class DoubleCritic(nn.Module):
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(
        self, observation: jnp.ndarray, action: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = jnp.concatenate([observation, action], axis=-1)
        q1 = MLP((*self.hidden_dims, 1), add_layer_norm=True)(x)
        q2 = MLP((*self.hidden_dims, 1), add_layer_norm=True)(x)
        return q1, q2


class GaussianPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    log_std_min: Optional[float] = -20.0
    log_std_max: Optional[float] = 2.0
    final_fc_init_scale: float = 1e-3

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, temperature: float = 1.0
    ) -> distrax.Distribution:
        outputs = MLP(
            self.hidden_dims,
            activate_final=True,
        )(observations)

        means = nn.Dense(
            self.action_dim, kernel_init=default_init(self.final_fc_init_scale)
        )(outputs)

        log_stds = self.param("log_stds", nn.initializers.zeros, (self.action_dim,))
        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(
            loc=means, scale_diag=jnp.exp(log_stds) * temperature
        )
        return distribution


def target_update(
    model: TrainState, target_model: TrainState, tau: float
) -> Tuple[TrainState, jnp.ndarray]:
    new_target_params = jax.tree_util.tree_map(
        lambda p, tp: p * tau + tp * (1 - tau), model.params, target_model.params
    )
    return target_model.replace(params=new_target_params)


def update_by_loss_grad(
    train_state: TrainState, loss_fn: Callable
) -> Tuple[float, Any]:
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(train_state.params)
    new_train_state = train_state.apply_gradients(grads=grad)
    return new_train_state, loss


class AWACTrainer(NamedTuple):
    rng: jax.random.PRNGKey
    critic: TrainState
    target_critic: TrainState
    actor: TrainState

    def update_actor(
        agent, batch: Transition, rng: jax.random.PRNGKey, config: Dict
    ) -> Tuple["AWACTrainer", jnp.ndarray]:
        def get_actor_loss(actor_params: flax.core.FrozenDict[str, Any]) -> jnp.ndarray:
            dist = agent.actor.apply_fn(actor_params, batch.observations)
            pi_actions = dist.sample(seed=rng)
            q_1, q_2 = agent.critic.apply_fn(
                agent.critic.params, batch.observations, pi_actions
            )
            v = jnp.minimum(q_1, q_2)

            lim = 1 - 1e-5
            actions = jnp.clip(batch.actions, -lim, lim)
            q_1, q_2 = agent.critic.apply_fn(
                agent.critic.params, batch.observations, actions
            )
            q = jnp.minimum(q_1, q_2)
            adv = q - v
            weights = jnp.exp(adv / config.beta)

            weights = jax.lax.stop_gradient(weights)

            log_prob = dist.log_prob(batch.actions)
            loss = -jnp.mean(log_prob * weights).mean()
            return loss

        new_actor, actor_loss = update_by_loss_grad(agent.actor, get_actor_loss)
        return agent._replace(actor=new_actor), actor_loss

    def update_critic(
        agent, batch: Transition, rng: jax.random.PRNGKey, config: Dict
    ) -> Tuple["AWACTrainer", jnp.ndarray]:
        def get_critic_loss(
            critic_params: flax.core.FrozenDict[str, Any]
        ) -> jnp.ndarray:
            dist = agent.actor.apply_fn(agent.actor.params, batch.observations)
            next_actions = dist.sample(seed=rng)
            n_q_1, n_q_2 = agent.target_critic.apply_fn(
                agent.target_critic.params, batch.next_observations, next_actions
            )
            next_q = jnp.minimum(n_q_1, n_q_2)
            q_target = batch.rewards + config.discount * (1 - batch.dones) * next_q
            q_target = jax.lax.stop_gradient(q_target)

            q_1, q_2 = agent.critic.apply_fn(
                critic_params, batch.observations, batch.actions
            )

            loss = jnp.mean((q_1 - q_target) ** 2 + (q_2 - q_target) ** 2)
            return loss

        new_critic, critic_loss = update_by_loss_grad(agent.critic, get_critic_loss)
        return agent._replace(critic=new_critic), critic_loss

    @partial(jax.jit, static_argnums=(3,))
    def update_n_times(
        agent,
        dataset: Transition,
        rng: jax.random.PRNGKey,
        config: Dict,
    ) -> Tuple["AWACTrainer", Dict]:
        for _ in range(config.n_jitted_updates):
            rng, batch_rng, critic_rng, actor_rng = jax.random.split(rng, 4)
            batch_indices = jax.random.randint(
                batch_rng, (config.batch_size,), 0, len(dataset.observations)
            )
            batch = jax.tree_util.tree_map(lambda x: x[batch_indices], dataset)

            agent, critic_loss = agent.update_critic(batch, critic_rng, config)
            new_target_critic = target_update(
                agent.critic,
                agent.target_critic,
                config.tau,
            )
            agent, actor_loss = agent.update_actor(batch, actor_rng, config)
        return agent._replace(target_critic=new_target_critic), {
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
        }

    @jax.jit
    def sample_actions(
        agent,
        observations: np.ndarray,
        seed: jax.random.PRNGKey,
        temperature: float = 1.0,
        max_action: float = 1.0,  # In D4RL envs, the action space is [-1, 1]
    ) -> jnp.ndarray:
        actions = agent.actor.apply_fn(
            agent.actor.params, observations, temperature=temperature
        ).sample(seed=seed)
        actions = jnp.clip(actions, -max_action, max_action)
        return actions


def create_awac_trainer(
    observations: jnp.ndarray, actions: jnp.ndarray, config: Dict
) -> AWACTrainer:
    rng = jax.random.PRNGKey(config.seed)
    rng, actor_rng, critic_rng, value_rng = jax.random.split(rng, 4)
    # initialize actor
    action_dim = actions.shape[-1]
    actor_model = GaussianPolicy(
        config.actor_hidden_dims,
        action_dim=action_dim,
    )
    actor = TrainState.create(
        apply_fn=actor_model.apply,
        params=actor_model.init(actor_rng, observations),
        tx=optax.adam(learning_rate=config.actor_lr),
    )
    # initialize critic
    critic_model = DoubleCritic(config.critic_hidden_dims)
    critic = TrainState.create(
        apply_fn=critic_model.apply,
        params=critic_model.init(critic_rng, observations, actions),
        tx=optax.adam(learning_rate=config.critic_lr),
    )
    # initialize target critic
    target_critic = TrainState.create(
        apply_fn=critic_model.apply,
        params=critic_model.init(critic_rng, observations, actions),
        tx=optax.adam(learning_rate=config.critic_lr),
    )
    return AWACTrainer(
        rng,
        critic=critic,
        target_critic=target_critic,
        actor=actor,
    )
