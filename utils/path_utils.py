import os

import d4rl
import gym
import h5py

"""
Over all, the objective of experiment is to evaluate the performance of RL algorithm under the dynamics shift with PU structured data.

There are three types of shift:
1. body_mass: body mass of the agent is changed.
2. joint_noise: joint noise is added to the agent.
3. halfcheetah_vs_walker2d: halfcheetah is trained and walker2d is test.

For each shift, we can consider two types of correspondings between positive and negative data:
1. shifted=positive, original=negative
2. shifted=negative, original=positive

Finally, we consider the data quality for each domain, e.g. positive_data_quality = "expert", negative_data_quality = "random"
"""


def make_shifted_dataset_path(config):
    """
    To capture the configuration of dataset, we need to consider the following:
    """
    if config.data.shift == "halfcheetah_vs_walker2d":
        shifted_dataset_path = None
        return shifted_dataset_path
    shifted_dataset_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "dataset",
        f"{config.env_name}/{config.data.shift}/{config.data.negative_data_quality}.hdf5",
    )
    assert os.path.exists(shifted_dataset_path)
    return shifted_dataset_path


def make_agent_params_path(config):
    """
    To capture the configuration of RL algorithm, we need to consider the following:
    1. rl_algo: rl algorithm: [TD3BC, IQL]
    2. env_name: environment name: [Hopper, Halfcheetah, Walker2d]
    3. shift_type: shift type: [body_mass, joint_noise, halfcheetah_vs_walker2d]
    4. positive_data_quality: data quality of positive data
    5. negative_data_quality: data quality of negative data
    6. positive_ratio: positive data ratio
    7. labeled_ratio: labeled data ratio
    8. seed
    """
    data_quality = (
        f"{config.data.positive_data_quality}_vs_{config.data.negative_data_quality}"
    )
    rl_net_param_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "params/offlinerl",
        f"{config.method}/{config.rl_algo}/{config.env_name}/{config.data.shift}/{data_quality}/model_positive_ratio={config.data.positive_ratio}_labeled_ratio={config.data.labeled_ratio}_seed={config.seed}.pt",
    )
    if not os.path.exists(rl_net_param_path):
        os.makedirs(os.path.dirname(rl_net_param_path), exist_ok=True)
    return rl_net_param_path


def make_classifier_params_path(config):
    """
    To capture the configuration of classifier, we need to consider the following:
    1. method: train type: [pu, pvu]
    2. env_name: environment name: [Hopper, Halfcheetah, Walker2d]
    3. shift_type: shift type: [body_mass, joint_noise, halfcheetah_vs_walker2d]
    4. positive_data_quality: data quality of positive data
    5. negative_data_quality: data quality of negative data
    6. positive_ratio: positive data ratio
    7. labeled_ratio: labeled data ratio
    8. seed
    """
    data_quality = (
        f"{config.data.positive_data_quality}_vs_{config.data.negative_data_quality}"
    )
    param_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "params/classifier",
        f"{config.method}/{config.env_name}/{config.data.shift}/{data_quality}/{config.data.input_type}/model_positive_ratio={config.data.positive_ratio}_labeled_ratio={config.data.labeled_ratio}.pt",
    )
    return param_path
