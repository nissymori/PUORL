from typing import Dict, Tuple

import d4rl
import gym
import h5py
import numpy as np

# keys
KEYS = [
    "observations",
    "actions",
    "rewards",
    "next_observations",
    "terminals",
    "true_labels",
    "dones_float",
    "masks",
]


def shuffle_datadict(datadict: Dict) -> Dict:
    """
    shuffle datadict
    :datadict: datadict
    :return: shuffled datadict
    """
    indices = np.arange(len(datadict["observations"]))
    np.random.shuffle(indices)
    shuffled_datadict = {
        k: np.array(v)[indices] for k, v in datadict.items() if k in KEYS
    }
    return shuffled_datadict


def get_normalization(dataset: Dict, dones_float: np.ndarray) -> float:
    returns = []
    ret = 0
    for r, term in zip(dataset["rewards"], dones_float):
        ret += r
        if term:
            returns.append(ret)
            ret = 0
    return (max(returns) - min(returns)) / 1000


def normalize_dataset_reward(dataset: Dict):
    clip_to_eps = True
    eps = 1e-5

    if clip_to_eps:
        lim = 1 - eps
        dataset["actions"] = np.clip(dataset["actions"], -lim, lim)

    dones_float = np.zeros_like(dataset["rewards"])

    for i in range(len(dones_float) - 1):
        if (
            np.linalg.norm(
                dataset["observations"][i + 1] - dataset["next_observations"][i]
            )
            > 1e-6
            or dataset["terminals"][i] == 1.0
        ):
            dones_float[i] = 1
        else:
            dones_float[i] = 0
    dones_float[-1] = 1
    normalizing_factor = get_normalization(dataset, dones_float)
    dataset["rewards"] = np.array(dataset["rewards"]) / normalizing_factor
    return dataset


def make_pos_neg_datadict(
    shifted_dataset_path, positive_env: gym.Env, config, normalize_reward: bool = False
) -> Tuple[Dict, Dict]:
    """
    There are three types of shift:
    1. body_mass: body mass of the agent is changed.
    2. joint_noise: joint noise is added to the agent.
    3. halfcheetah_vs_walker2d: halfcheetah is trained and walker2d is test.

    For each shift, we can consider two types of correspondings between positive and negative data:
    1. shifted=positive, original=negative
    2. shifted=negative, original=positive

    Finally, we consider the data quality for each domain, e.g. positive_data_quality = "expert", negative_data_quality = "random"
    """
    if config.data.shift == "body_mass" or config.data.shift == "joint_noise":
        positive_datadict = d4rl.qlearning_dataset(env=positive_env)
        negative_datadict = dict(h5py.File(shifted_dataset_path, "r"))
        negative_datadict = {k: np.array(v) for k, v in negative_datadict.items()}
        if normalize_reward:
            positive_datadict = normalize_dataset_reward(positive_datadict)
            negative_datadict = normalize_dataset_reward(negative_datadict)
    elif config.data.shift == "mixture":
        assert len(shifted_dataset_path) == 2
        body_mass_path, joint_noise_path = shifted_dataset_path
        positive_datadict = d4rl.qlearning_dataset(env=positive_env)
        body_mass_datadict = dict(h5py.File(body_mass_path, "r"))
        joint_noise_datadict = dict(h5py.File(joint_noise_path, "r"))
        negative_datadict = {
            k: np.concatenate([body_mass_datadict[k], joint_noise_datadict[k]])
            for k in body_mass_datadict.keys()
        }
        if normalize_reward:
            positive_datadict = normalize_dataset_reward(positive_datadict)
            negative_datadict = normalize_dataset_reward(negative_datadict)
        negative_datadict = shuffle_datadict(negative_datadict)
        negative_datadict = {
            k: negative_datadict[k][: len(body_mass_datadict[k])]
            for k in negative_datadict.keys()
        }
    elif config.data.shift == "halfcheetah_vs_walker2d":
        positive_env = gym.make(
            f"halfcheetah-{config.data.positive_data_quality.replace('_', '-')}-v2"
        )
        negative_env = gym.make(
            f"walker2d-{config.data.negative_data_quality.replace('_', '-')}-v2"
        )
        positive_datadict = d4rl.qlearning_dataset(env=positive_env)
        negative_datadict = d4rl.qlearning_dataset(env=negative_env)
        if normalize_reward:
            positive_datadict = normalize_dataset_reward(positive_datadict)
            negative_datadict = normalize_dataset_reward(negative_datadict)
    else:
        raise NotImplementedError
    return positive_datadict, negative_datadict
