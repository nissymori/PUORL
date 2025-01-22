# source https://github.com/ikostrikov/implicit_q_learning
# https://arxiv.org/abs/2110.06169
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
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, hidden_dims in enumerate(self.hidden_dims):
            x = nn.Dense(hidden_dims, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.layer_norm:  # Add layer norm after activation
                    x = nn.LayerNorm()(x)
                x = self.activations(x)
        return x


class Critic(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions], -1)
        critic = MLP((*self.hidden_dims, 1), activations=self.activations)(inputs)
        return jnp.squeeze(critic, -1)


def ensemblize(cls, num_qs, out_axes=0, **kwargs):
    split_rngs = kwargs.pop("split_rngs", {})
    return nn.vmap(
        cls,
        variable_axes={"params": 0},
        split_rngs={**split_rngs, "params": True},
        in_axes=None,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs,
    )


class ValueCritic(nn.Module):
    hidden_dims: Sequence[int]
    layer_norm: bool = False

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        critic = MLP((*self.hidden_dims, 1), layer_norm=self.layer_norm)(observations)
        return jnp.squeeze(critic, -1)


class GaussianPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    log_std_min: Optional[float] = -5.0
    log_std_max: Optional[float] = 2

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, temperature: float = 1.0
    ) -> distrax.Distribution:
        outputs = MLP(
            self.hidden_dims,
            activate_final=True,
        )(observations)

        means = nn.Dense(
            self.action_dim, kernel_init=default_init()
        )(outputs)
        log_stds = self.param("log_stds", nn.initializers.zeros, (self.action_dim,))
        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(
            loc=means, scale_diag=jnp.exp(log_stds) * temperature
        )
        return distribution


class Transition(NamedTuple):
    observations: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    next_observations: jnp.ndarray
    dones: jnp.ndarray
    dones_float: jnp.ndarray


def get_normalization(dataset: Transition) -> float:
    # into numpy.ndarray
    dataset = jax.tree_util.tree_map(lambda x: np.array(x), dataset)
    returns = []
    ret = 0
    for r, term in zip(dataset.rewards, dataset.dones_float):
        ret += r
        if term:
            returns.append(ret)
            ret = 0
    return (max(returns) - min(returns)) / 1000


def expectile_loss(diff, expectile=0.8) -> jnp.ndarray:
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)


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


class IQLTrainState(NamedTuple):
    rng: jax.random.PRNGKey
    critic: TrainState
    target_critic: TrainState
    value: TrainState
    actor: TrainState


class IQL(object):

    @classmethod
    def update_critic(
        self, train_state: IQLTrainState, batch: Transition, config: Dict
    ) -> Tuple["IQLTrainState", Dict]:
        next_v = train_state.value.apply_fn(
            train_state.value.params, batch.next_observations
        )
        target_q = batch.rewards + config.discount * (1 - batch.dones) * next_v
        
        def critic_loss_fn(
            critic_params: flax.core.FrozenDict[str, Any]
        ) -> jnp.ndarray:
            q1, q2 = train_state.critic.apply_fn(
                critic_params, batch.observations, batch.actions
            )
            critic_loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()
            return critic_loss

        new_critic, critic_loss = update_by_loss_grad(
            train_state.critic, critic_loss_fn
        )
        return train_state._replace(critic=new_critic), critic_loss

    @classmethod
    def update_value(
        self, train_state: IQLTrainState, batch: Transition, config: Dict
    ) -> Tuple["IQLTrainState", Dict]:
        q1, q2 = train_state.target_critic.apply_fn(
            train_state.target_critic.params, batch.observations, batch.actions
        )
        q = jax.lax.stop_gradient(jnp.minimum(q1, q2))
        def value_loss_fn(value_params: flax.core.FrozenDict[str, Any]) -> jnp.ndarray:
            v = train_state.value.apply_fn(value_params, batch.observations)
            value_loss = expectile_loss(q - v, config.expectile).mean()
            return value_loss

        new_value, value_loss = update_by_loss_grad(train_state.value, value_loss_fn)
        return train_state._replace(value=new_value), value_loss

    @classmethod
    def update_actor(
        self, train_state: IQLTrainState, batch: Transition, config: Dict
    ) -> Tuple["IQLTrainState", Dict]:
        v = train_state.value.apply_fn(train_state.value.params, batch.observations)
        q1, q2 = train_state.critic.apply_fn(
            train_state.target_critic.params, batch.observations, batch.actions
        )
        q = jnp.minimum(q1, q2)
        exp_a = jnp.exp((q - v) * config.beta)
        exp_a = jnp.minimum(exp_a, 100.0)
        def actor_loss_fn(actor_params: flax.core.FrozenDict[str, Any]) -> jnp.ndarray:
            dist = train_state.actor.apply_fn(actor_params, batch.observations)
            log_probs = dist.log_prob(batch.actions)
            actor_loss = -(exp_a * log_probs).mean()
            return actor_loss

        new_actor, actor_loss = update_by_loss_grad(train_state.actor, actor_loss_fn)
        return train_state._replace(actor=new_actor), actor_loss

    @classmethod
    def update_n_times(
        self,
        train_state: IQLTrainState,
        dataset: Transition,
        rng: jax.random.PRNGKey,
        config: Dict,
    ) -> Tuple["IQLTrainState", Dict]:
        for _ in range(config.n_jitted_updates):
            rng, subkey = jax.random.split(rng)
            batch_indices = jax.random.randint(
                subkey, (config.batch_size,), 0, len(dataset.observations)
            )
            batch = jax.tree_util.tree_map(lambda x: x[batch_indices], dataset)

            train_state, value_loss = self.update_value(train_state, batch, config)
            train_state, actor_loss = self.update_actor(train_state, batch, config)
            train_state, critic_loss = self.update_critic(train_state, batch, config)
            new_target_critic = target_update(
                train_state.critic, train_state.target_critic, config.tau
            )
            train_state = train_state._replace(target_critic=new_target_critic)
        return train_state, {
            "value_loss": value_loss,
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
        }

    @classmethod
    def get_action(
        self,
        train_state: IQLTrainState,
        observations: np.ndarray,
        seed: jax.random.PRNGKey,
        temperature: float = 1.0,
        max_action: float = 1.0,  # In D4RL, the action space is [-1, 1]
    ) -> jnp.ndarray:
        actions = train_state.actor.apply_fn(
            train_state.actor.params, observations, temperature=temperature
        ).sample(seed=seed)
        actions = jnp.clip(actions, -max_action, max_action)
        return actions


def create_iql_train_state(
    rng: jax.random.PRNGKey,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    config: Dict,
) -> IQLTrainState:
    rng, actor_rng, critic_rng, value_rng = jax.random.split(rng, 4)
    # initialize actor
    action_dim = actions.shape[-1]
    actor_model = GaussianPolicy(
        config.hidden_dims,
        action_dim=action_dim,
        log_std_min=-5.0,
    )
    if config.opt_decay_schedule:
        schedule_fn = optax.cosine_decay_schedule(-config.actor_lr, config.max_steps)
        actor_tx = optax.chain(optax.scale_by_adam(), optax.scale_by_schedule(schedule_fn))
    else:
        actor_tx = optax.adam(learning_rate=config.actor_lr)
    actor = TrainState.create(
        apply_fn=actor_model.apply,
        params=actor_model.init(actor_rng, observations),
        tx=actor_tx,
    )
    # initialize critic
    critic_model = ensemblize(Critic, num_qs=2)(config.hidden_dims)
    critic = TrainState.create(
        apply_fn=critic_model.apply,
        params=critic_model.init(critic_rng, observations, actions),
        tx=optax.adam(learning_rate=config.critic_lr),
    )
    target_critic = TrainState.create(
        apply_fn=critic_model.apply,
        params=critic_model.init(critic_rng, observations, actions),
        tx=optax.adam(learning_rate=config.critic_lr),
    )
    # initialize value
    value_model = ValueCritic(config.hidden_dims, layer_norm=config.layer_norm)
    value = TrainState.create(
        apply_fn=value_model.apply,
        params=value_model.init(value_rng, observations),
        tx=optax.adam(learning_rate=config.value_lr),
    )
    return IQLTrainState(
        rng,
        critic=critic,
        target_critic=target_critic,
        value=value,
        actor=actor,
    )
