# source https://github.com/sfujim/TD3_BC
# https://arxiv.org/abs/2106.06860
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
from flax.training.train_state import TrainState
from omegaconf import OmegaConf
from pydantic import BaseModel

import wandb

os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True"


def default_init(scale: Optional[float] = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)


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


class TD3Actor(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    max_action: float = 1.0  # In D4RL, action is scaled to [-1, 1]

    @nn.compact
    def __call__(self, observation: jnp.ndarray) -> jnp.ndarray:
        action = MLP((*self.hidden_dims, self.action_dim))(observation)
        action = self.max_action * jnp.tanh(
            action
        )  # scale to [-max_action, max_action]
        return action


class Transition(NamedTuple):
    observations: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    next_observations: jnp.ndarray
    dones: jnp.ndarray


def target_update(
    model: TrainState, target_model: TrainState, tau: float
) -> TrainState:
    new_target_params = jax.tree_util.tree_map(
        lambda p, tp: p * tau + tp * (1 - tau), model.params, target_model.params
    )
    return target_model.replace(params=new_target_params)


def update_by_loss_grad(
    train_state: TrainState, loss_fn: Callable
) -> Tuple[TrainState, jnp.ndarray]:
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(train_state.params)
    new_train_state = train_state.apply_gradients(grads=grad)
    return new_train_state, loss


class TD3BCTrainState(NamedTuple):
    actor: TrainState
    critic: TrainState
    target_actor: TrainState
    target_critic: TrainState
    max_action: float = 1.0


class TD3BC(object):
    @classmethod
    def update_actor(
        self,
        train_state: TD3BCTrainState,
        batch: Transition,
        rng: jax.random.PRNGKey,
        config,
    ) -> Tuple["TD3BCTrainState", jnp.ndarray]:
        def actor_loss_fn(actor_params: flax.core.FrozenDict[str, Any]) -> jnp.ndarray:
            predicted_action = train_state.actor.apply_fn(
                actor_params, batch.observations
            )
            critic_params = jax.lax.stop_gradient(train_state.critic.params)
            q_value, _ = train_state.critic.apply_fn(
                critic_params, batch.observations, predicted_action
            )

            mean_abs_q = jax.lax.stop_gradient(jnp.abs(q_value).mean())
            loss_lambda = config.alpha / mean_abs_q

            bc_loss = jnp.square(predicted_action - batch.actions).mean()
            loss_actor = -1.0 * q_value.mean() * loss_lambda + bc_loss
            return loss_actor

        new_actor, actor_loss = update_by_loss_grad(train_state.actor, actor_loss_fn)
        return train_state._replace(actor=new_actor), actor_loss

    @classmethod
    def update_critic(
        self,
        train_state: TD3BCTrainState,
        batch: Transition,
        rng: jax.random.PRNGKey,
        config,
    ) -> Tuple["TD3BCTrainState", jnp.ndarray]:
        def critic_loss_fn(
            critic_params: flax.core.FrozenDict[str, Any]
        ) -> jnp.ndarray:
            q_pred_1, q_pred_2 = train_state.critic.apply_fn(
                critic_params, batch.observations, batch.actions
            )
            target_next_action = train_state.target_actor.apply_fn(
                train_state.target_actor.params, batch.next_observations
            )
            policy_noise = (
                config.policy_noise_std
                * train_state.max_action
                * jax.random.normal(rng, batch.actions.shape)
            )
            target_next_action = target_next_action + policy_noise.clip(
                -config.policy_noise_clip, config.policy_noise_clip
            )
            target_next_action = target_next_action.clip(
                -train_state.max_action, train_state.max_action
            )
            q_next_1, q_next_2 = train_state.target_critic.apply_fn(
                train_state.target_critic.params,
                batch.next_observations,
                target_next_action,
            )
            target = batch.rewards[..., None] + config.discount * jnp.minimum(
                q_next_1, q_next_2
            ) * (1 - batch.dones[..., None])
            target = jax.lax.stop_gradient(target)  # stop gradient for target
            value_loss_1 = jnp.square(q_pred_1 - target)
            value_loss_2 = jnp.square(q_pred_2 - target)
            value_loss = (value_loss_1 + value_loss_2).mean()
            return value_loss

        new_critic, critic_loss = update_by_loss_grad(
            train_state.critic, critic_loss_fn
        )
        return train_state._replace(critic=new_critic), critic_loss

    @classmethod
    def train_step(
        self,
        train_state: TD3BCTrainState,
        data,
        rng: jax.random.PRNGKey,
        config,
    ) -> TD3BCTrainState:
        for _ in range(config.policy_freq):
            rng, batch_rng = jax.random.split(rng)
            batch_idx = jax.random.randint(
                batch_rng, (config.batch_size,), 0, len(data.observations)
            )
            batch = jax.tree_util.tree_map(lambda x: x[batch_idx], data)

            rng, critic_rng = jax.random.split(rng)
            train_state, critic_loss = self.update_critic(
                train_state, batch, critic_rng, config
            )

        rng, actor_rng = jax.random.split(rng)
        train_state, actor_loss = self.update_actor(
            train_state, batch, actor_rng, config
        )

        # update target networks
        new_target_critic = target_update(
            train_state.critic, train_state.target_critic, config.tau
        )
        new_target_actor = target_update(
            train_state.actor, train_state.target_actor, config.tau
        )
        train_state = train_state._replace(
            target_critic=new_target_critic,
            target_actor=new_target_actor,
        )
        return train_state, critic_loss, actor_loss

    @classmethod
    def update_n_times(
        self,
        train_state: TD3BCTrainState,
        data: Transition,
        rng: jax.random.PRNGKey,
        config,
    ) -> Tuple["TD3BCTrainState", Dict]:
        def loop_fn(carry, _):
            train_state, rng = carry
            rng, key = jax.random.split(rng)
            train_state, critic_loss, actor_loss = self.train_step(
                train_state, data, key, config
            )
            return (train_state, rng), {
                "critic_loss": critic_loss,
                "actor_loss": actor_loss,
            }

        updates = int(config.n_jitted_updates // config.policy_freq)
        (train_state, _), losses = jax.lax.scan(
            loop_fn, (train_state, rng), None, length=updates
        )
        return train_state, losses

    @classmethod
    def get_action(
        self,
        train_state,
        obs: jnp.ndarray,
        max_action: float = 1.0,  # In D4RL, action is scaled to [-1, 1]
    ) -> jnp.ndarray:
        action = train_state.actor.apply_fn(train_state.actor.params, obs)
        action = action.clip(-max_action, max_action)
        return action


def create_td3bc_train_state(
    rng, observations: jnp.ndarray, actions: jnp.ndarray, config
):
    critic_model = DoubleCritic(
        hidden_dims=config.hidden_dims,
    )
    action_dim = actions.shape[-1]
    actor_model = TD3Actor(
        action_dim=action_dim,
        hidden_dims=config.hidden_dims,
    )
    rng, critic_rng, actor_rng = jax.random.split(rng, 3)
    # initialize critic
    critic_train_state: TrainState = TrainState.create(
        apply_fn=critic_model.apply,
        params=critic_model.init(critic_rng, observations, actions),
        tx=optax.adam(config.critic_lr),
    )
    target_critic_train_state: TrainState = TrainState.create(
        apply_fn=critic_model.apply,
        params=critic_model.init(critic_rng, observations, actions),
        tx=optax.adam(config.critic_lr),
    )
    # initialize actor
    actor_train_state: TrainState = TrainState.create(
        apply_fn=actor_model.apply,
        params=actor_model.init(actor_rng, observations),
        tx=optax.adam(config.actor_lr),
    )
    target_actor_train_state: TrainState = TrainState.create(
        apply_fn=actor_model.apply,
        params=actor_model.init(actor_rng, observations),
        tx=optax.adam(config.actor_lr),
    )
    return TD3BCTrainState(
        actor=actor_train_state,
        critic=critic_train_state,
        target_actor=target_actor_train_state,
        target_critic=target_critic_train_state,
    )
