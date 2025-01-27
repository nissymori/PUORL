import os
import pickle
from typing import List, Literal

import envpool
import gym
import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pyrallis
import torch
from tqdm import tqdm

import wandb
from offlinerl import make_agent, make_evaluation
from utils import (OfflineRLConfig, make_agent_params_path, make_classifier,
                   make_classifier_params_path, make_offline_rl_dataset,
                   make_shifted_dataset_path)

"""
Dataset
- Here, we train the RL agent with the dataset mixed with two differnt domain, between which there is a shift.
- We have three types of shift: body_mass, joint_noise, halfcheetah_vs_walker2d
- For each shift, we have two axis: shifted vs original, positive vs negative
- For each axis, we have four data quality: expert, replay, medium, random
- Such parameter is determined by config file at utils/config.py

Method
- We have 6 types of train: pu, pvu, oracle, sharing_all, uds, only_psitive
- pu: augment the observation with predicted label by pu classifier
- pvu: augment the observation with predicted label by pvu classifier (pvu means p vs u as negative)
- oracle: augment the observation with true label
- sharing_all: use all data
- only_p: use only positive data
- uds: use only unlabeled data

As an example, we consider the following setting:
- shift: body_mass
- positive_data_quality: expert
- negative_data_quality: random
- positive_ratio: 0.3
- labeled_ratio: 0.05
- method: pu
    
Then, the dataset is mixed with 900000 shifted data from expert and 100000 original data from random.
Then, 80% of the data is unlabeled and the rest is labeled.
We train the rl agent with the dataset augmented with predicted label by pu classifier.

To understand more about experimental setting, please refer to utils/config.py and experimental_setup_utils.py
"""


@pyrallis.wrap()
def main(config: OfflineRLConfig):
    train(config)


def train(config: OfflineRLConfig):
    print(
        f"Start RL training {config.env_name}, shift: {config.data.shift}, method: {config.method}, positive data quality: {config.data.positive_data_quality}, negative data quality: {config.data.negative_data_quality}, positive ratio: {config.data.positive_ratio}, labeled ratio: {config.data.labeled_ratio}"
    )
    # make positive (data) environment
    positive_data_env = gym.make(
        f"{config.env_name}-{config.data.positive_data_quality.replace('_', '-')}-v2"
    )
    # make eval env
    eval_env = envpool.make(
        config.eval_env_name,
        env_type="gym",
        num_envs=config.n_seeds * config.eval_episodes,
    )

    # load classifier if necessary
    sas_net_param_path, sa_net_param_path = make_classifier_params_path(config)
    print(sas_net_param_path)
    sas_net = make_classifier(
        config.classifier_hidden_dims, input_dim=positive_data_env.observation_space.shape[0] * 2 + positive_data_env.action_space.shape[0]
    )
    sa_net = make_classifier(
        config.classifier_hidden_dims, input_dim=positive_data_env.observation_space.shape[0] + positive_data_env.action_space.shape[0]
    )
    print(sas_net.state_dict().keys())

    if config.method == "pu" or config.method == "pvu" or config.method == "dara-pu" or config.method == "dara-pvu":
        # load classifier if method is pu
        sas_net_param = torch.load(sas_net_param_path)
        sas_net.load_state_dict(sas_net_param)            
        
        if config.method == "dara-pu" or config.method == "dara-pvu":
            sa_net_param = torch.load(sa_net_param_path)
            new_sa_net_param = OrderedDict()
            for key, value in sa_net_param.items():
                new_sa_net_param[key[7:]] = value
            sa_net.load_state_dict(new_sa_net_param, strict=False)

    # make agent
    algo, create_train_state, algo_config = make_agent(config)
    train_vj = jax.jit(
        jax.vmap(algo.update_n_times, in_axes=(0, None, 0, None)), static_argnums=(3,)
    )

    # make dataset
    shifted_dataset_path = make_shifted_dataset_path(config)
    dataset, obs_mean, obs_std = make_offline_rl_dataset(
        shifted_dataset_path,
        positive_data_env,
        config,
        sas_net,
        sa_net,
        algo_config.normalize_state,
        algo_config.normalize_reward,
    )

    # make evaluation function
    eval_fn = make_evaluation(
        eval_env,
        config,
        obs_mean,
        obs_std,
        algo.get_action,
        vectorized=True,
    )

    # init train state
    rng = jax.random.PRNGKey(config.seed)
    rng, subkey = jax.random.split(rng)
    rngs = jax.random.split(subkey, config.n_seeds)
    example_batch = jax.tree_util.tree_map(lambda x: x[0], dataset)
    train_state = jax.vmap(create_train_state, in_axes=(0, None, None, None))(
        rngs, example_batch.observations, example_batch.actions, algo_config
    )

    # train
    num_steps = int(config.max_steps // algo_config.n_jitted_updates)
    eval_interval = int(config.eval_interval // algo_config.n_jitted_updates)
    eval_returns = []
    for step in tqdm(range(num_steps)):
        rng, subkey = jax.random.split(rng)
        rngs = jax.random.split(subkey, config.n_seeds)
        train_state, loss = train_vj(train_state, dataset, rngs, algo_config)
        if step % eval_interval == 0:
            eval_return = eval_fn(train_state)
            normalized_eval_return = [round(float(positive_data_env.get_normalized_score(eval_return[i])) * 100, 2) for i in range(config.n_seeds)]
            eval_returns.append(normalized_eval_return)
            print(f"step: {step}, eval_return: {eval_return}, normalized_eval_return: {normalized_eval_return}")
    
    for seed in range(config.n_seeds):
        wandb.init(project=config.project, config=config, name=f"seed_{seed}", reinit=True)
        for step in range(len(eval_returns)):
            wandb.log({"eval_return": eval_returns[step][seed], "step": step * algo_config.n_jitted_updates * config.eval_interval})




if __name__ == "__main__":
    main()
