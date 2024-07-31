import os
import pickle
from typing import List, Literal

import gym
import h5py
import numpy as np
import pyrallis
import torch

import wandb
from utils import (
    make_classifier_params_path,
    make_agent_params_path,
    make_shifted_dataset_path,
    make_offline_rl_dataset,
    OfflineRLConfig,
    make_classifier,
)
from offlinerl import create_awac_trainer, create_td3bc_trainer, create_iql_trainer, evaluate
"""
Dataset
- Here, we train the RL agent with the dataset mixed with two differnt domain, between which there is a shift.
- We have three types of shift: body_mass, joint_noise, halfcheetah_vs_walker2d
- For each shift, we have two axis: shifted vs original, positive vs negative
- For each axis, we have four data quality: expert, replay, medium, random
- Such parameter is determined by config file at utils/config.py

Method
- We have three types of train: pu, pvu, oracle, sharing_all, uds, only_psitive
- pu: augment the observation with predicted label by pu classifier
- pvu: augment the observation with predicted label by pvu classifier (pvu means p vs u as negative)
- ground_true: augment the observation with true label


As an example, we consider the following setting:
- shift_type: body_mass
- positive_env: shifted
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


def train(config: OfflineRLConfig):
    # make positive and negative environment
    positive_env = make_pos_envs(config)

    # load classifier if necessary
    sas_net_param_path = make_classifier_params_path(config)
    print(sas_net_param_path, sa_net_param_path)
    sas_net = make_classifier(config.hidden_dims, input_dim=positive_env.observation_space.shape[0])
    # load classifier if train_type is pu
    sas_net = (
        sas_net.load_state_dict(torch.load(sas_net_param_path))
        if config.train_type == "pu"
        or config.train_type == "pvu"
        else None
    )

    # make dataset
    shifted_dataset_path = make_shifted_dataset_path(config)
    dataset = make_offline_rl_dataset(shifted_dataset_path, positive_env, config)

    # make agent

    if 

    
