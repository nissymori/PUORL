from typing import Dict, List, NamedTuple, Tuple, Union

import d4rl
import gym
import h5py
import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data_utils import make_pos_neg_datadict, shuffle_datadict

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


class Transition(NamedTuple):
    observations: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    next_observations: jnp.ndarray
    dones: jnp.ndarray


def get_transitions(
    dataset,
    config,
    clip_to_eps: bool = True,
    eps: float = 1e-5,
    normalize_state: bool = False,
    normalize_reward: bool = False,
) -> Transition:

    if clip_to_eps:
        lim = 1 - eps
        dataset["actions"] = np.clip(dataset["actions"], -lim, lim)

    imputed_next_observations = np.roll(dataset["observations"], -1, axis=0)
    same_obs = np.all(
        np.isclose(imputed_next_observations, dataset["next_observations"], atol=1e-5),
        axis=-1,
    )
    dones = 1.0 - same_obs.astype(np.float32)
    dones[-1] = 1

    dataset = Transition(
        observations=jnp.array(dataset["observations"], dtype=jnp.float32),
        actions=jnp.array(dataset["actions"], dtype=jnp.float32),
        rewards=jnp.array(dataset["rewards"], dtype=jnp.float32),
        next_observations=jnp.array(dataset["next_observations"], dtype=jnp.float32),
        dones=jnp.array(dones, dtype=jnp.float32),
    )
    # shuffle data and select the first data_size samples
    data_size = min(config.data.size, len(dataset.observations))
    rng = jax.random.PRNGKey(config.seed)
    rng, rng_permute, rng_select = jax.random.split(rng, 3)
    perm = jax.random.permutation(rng_permute, len(dataset.observations))
    dataset = jax.tree_util.tree_map(lambda x: x[perm], dataset)
    assert len(dataset.observations) >= data_size
    dataset = jax.tree_util.tree_map(lambda x: x[:data_size], dataset)
    # normalize states
    obs_mean, obs_std = 0, 1
    if normalize_state:
        obs_mean = dataset.observations.mean(0)
        obs_std = dataset.observations.std(0)
        dataset = dataset._replace(
            observations=(dataset.observations - obs_mean) / (obs_std + 1e-5),
            next_observations=(dataset.next_observations - obs_mean) / (obs_std + 1e-5),
        )
    if normalize_reward:  # normalize rewards
        normalizing_factor = get_normalization(dataset)
        dataset = dataset._replace(rewards=dataset.rewards / normalizing_factor)
    return dataset, obs_mean, obs_std


def get_normalization(dataset: Transition) -> float:
    # into numpy.ndarray
    dataset = jax.tree_util.tree_map(lambda x: np.array(x), dataset)
    returns = []
    ret = 0
    for r, term in zip(dataset.rewards, dataset.dones):
        ret += r
        if term:
            returns.append(ret)
            ret = 0
    return (max(returns) - min(returns)) / 1000


def make_offline_rl_dataset(
    shifted_dataset_path: str,
    positive_env: gym.Env,
    config,
    sas_net: nn.Module = None,
    sa_net: nn.Module = None,
    normalize_state: bool = False,
    normalize_reward: bool = False,
) -> Transition:
    """
    make rl dataset
    :positive_datadict: positive data
    :negative_datadict: negative data
    :config: config
    :sas_net: sas_net
    :sa_net: for dara, currently not used
    :return: rl dataset (D4RL format)
    """
    positive_datadict, negative_datadict = make_pos_neg_datadict(
        shifted_dataset_path, positive_env, config
    )
    positive_datadict = shuffle_datadict(positive_datadict)
    negative_datadict = shuffle_datadict(negative_datadict)

    positive_num = int(config.data.size * config.data.positive_ratio)
    negative_num = int(config.data.size * (1 - config.data.positive_ratio))

    datadict = concatenate_datadict(
        positive_datadict,
        negative_datadict,
        positive_num,
        negative_num,
    )

    if (
        config.method == "pvu"
        or config.method == "pu"
        or config.method == "oracle"
        or config.method == "only_p"
    ):
        target = 0  # positive
        datadict = filtering_by_label(
            datadict, target, sas_net, config
        )  # filter positive

    dataset, obs_mean, obs_std = get_transitions(
        datadict,
        config,
        normalize_state=normalize_state,
        normalize_reward=normalize_reward,
    )
    return dataset, obs_mean, obs_std


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


def concat(
    p_data: np.ndarray, n_data: np.ndarray, positive_num: int, negative_num: int
):
    print(len(p_data), len(n_data), positive_num, negative_num)
    if positive_num == 0:
        data = n_data[:negative_num]
    elif negative_num == 0:
        data = p_data[:positive_num]
    elif positive_num > 0 and negative_num > 0:
        data = np.concatenate([p_data[:positive_num], n_data[:negative_num]], axis=0)
    else:
        raise ValueError("positive_num and negative_num must be positive")
    return data


def concatenate_datadict(
    positive_datadict: Dict,
    negative_datadict: Dict,
    positive_num: int,
    negative_num: int,
    add_true_labels: bool = True,
) -> Dict:
    """
    concatnate positive data and negative data
    :positive_datadict: positive data
    :negative_datadict: negative data
    :positive_num: number of positive data
    :negative_num: number of negative data
    """
    data = {
        k: concat(
            positive_datadict[k], negative_datadict[k], positive_num, negative_num
        )
        for k in positive_datadict.keys()
        if k in KEYS
    }
    if add_true_labels:
        true_labels = np.concatenate(
            [np.zeros(positive_num), np.ones(negative_num)], axis=0
        )  # 0: positive, 1: negative
        data["true_labels"] = true_labels
    # check the length of data
    for k, v in data.items():
        assert len(v) == positive_num + negative_num
    return data


def filtering_by_label(
    datadict: Dict, target_label: int, net: nn.Module, config
) -> Dict:
    """
    filtering by label
    :datadict: datadict
    :label: label
    :net: net
    :oracle: whether train with true label or not
    :return: filtered datadict
    """
    oracle = config.method == "oracle"
    if config.method == "only_p":
        labeled_positive_num = int(
            len(datadict["true_labels"]) * config.data.labeled_ratio
        )
        filtered_datadict = {
            k: v[datadict["true_labels"] == 0][:labeled_positive_num]
            for k, v in datadict.items()
        }
        return filtered_datadict
    if config.method == "oracle":
        filtered_datadict = {
            k: v[datadict["true_labels"] == target_label] for k, v in datadict.items()
        }
    elif config.method == "pu":
        net.eval()
        positive_indices = np.where(datadict["true_labels"] == 0)[0]
        negative_indices = np.where(datadict["true_labels"] == 1)[0]
        labeled_positive_num = int(
            (len(positive_indices) + len(negative_indices)) * config.data.labeled_ratio
        )
        # separate positive data into labeled and unlabeled
        positively_labeled_datadict = {
            k: v[datadict["true_labels"] == 0][:labeled_positive_num]
            for k, v in datadict.items()
        }
        rest_positive_datadict = {
            k: v[datadict["true_labels"] == 0][labeled_positive_num:]
            for k, v in datadict.items()
        }
        negative_datadict = {
            k: v[datadict["true_labels"] == 1] for k, v in datadict.items()
        }
        unlabeled_datadict = concatenate_datadict(
            rest_positive_datadict,
            negative_datadict,
            len(rest_positive_datadict["observations"]),
            len(negative_indices),
            add_true_labels=False,
        )  # filter from unlabeled data.
        _input = np.concatenate(
            [
                unlabeled_datadict["observations"],
                unlabeled_datadict["actions"],
                unlabeled_datadict["next_observations"],
            ],
            axis=1,
        )
        label = net(torch.from_numpy(_input).to(torch.float32)).cpu().detach().numpy()
        label = np.argmax(label, axis=1)
        print(
            "filtering acc",
            sum(label == unlabeled_datadict["true_labels"]) / len(label),
        )
        print(
            "filtering pos acc",
            sum(label[unlabeled_datadict["true_labels"] == 0] == 0)
            / len(label[unlabeled_datadict["true_labels"] == 0]),
        )
        print(
            "filtering neg acc",
            sum(label[unlabeled_datadict["true_labels"] == 1] == 1)
            / len(label[unlabeled_datadict["true_labels"] == 1]),
        )
        filtered_datadict = {
            k: v[label == target_label] for k, v in unlabeled_datadict.items()
        }
        # concat with labeled positive data
        filtered_datadict = concatenate_datadict(
            positively_labeled_datadict,
            filtered_datadict,
            positive_labeled_num,
            len(filtered_datadict["observations"]),
            add_true_labels=False,
        )
        filtered_datadict = shuffle_datadict(filtered_datadict)
    else:
        raise NotImplementedError
    return filtered_datadict


def augment_by_dara(datadict, sas_net, sa_net, config):
    """ """
    positive_indices = np.where(datadict["true_labels"] == 0)[0]
    negative_indices = np.where(datadict["true_labels"] == 1)[0]
    positive_labeled_num = int(
        (len(positive_indices) + len(negative_indices))
        * config.data.labeled_ratio
    )
    # separate positive data into labeled and unlabeled
    positively_labeled_datadict = {
        k: v[datadict["true_labels"] == 0][:positive_labeled_num]
        for k, v in datadict.items()
    }
    rest_positive_datadict = {
        k: v[datadict["true_labels"] == 0][positive_labeled_num:]
        for k, v in datadict.items()
    }
    negative_datadict = {
        k: v[datadict["true_labels"] == 1] for k, v in datadict.items()
    }
    unlabeled_data = concatenate_datadict(
        rest_positive_datadict,
        negative_datadict,
        len(rest_positive_datadict["observations"]),
        len(negative_indices),
        add_true_labels=False,
    )  # filter from unlabeled data.
    # apply dara with unlabeled data as source and positively labeled data as target
    modified_unlabeled_data = dara(
        unlabeled_data, sas_net, sa_net, source_idx=1, target_idx=0
    )
    augmented_datadict = concatenate_datadict(
        positively_labeled_datadict,
        modified_unlabeled_data,
        positive_labeled_num,
        len(modified_unlabeled_data["observations"]),
        add_true_labels=False,
    )
    # shuffle
    augmented_datadict = shuffle_datadict(augmented_datadict)
    assert len(augmented_datadict["observations"]) == len(datadict["observations"])
    return augmented_datadict


def dara(source_datadict, sas_net, sa_net, source_idx, target_idx, eta=0.1):
    """
    DARA augmentation
    we augment source data towards target data by DARA
    """
    sas_net.eval()
    sa_net.eval()

    states = np.array(source_datadict["observations"])
    actions = np.array(source_datadict["actions"])
    next_states = np.array(source_datadict["next_observations"])
    rewards = np.array(source_datadict["rewards"])

    source_sas_data = np.concatenate(
        (states, actions, next_states), axis=1
    )  # (B, data_dim)
    source_sas_data = torch.from_numpy(source_sas_data).float()  # (B, data_dim)

    source_sa_data = np.concatenate((states, actions), axis=1)  # (B, data_dim)
    source_sa_data = torch.from_numpy(source_sa_data).float()  # (B, data_dim)
    print(source_sas_data.shape, source_sa_data.shape)

    sas_logits = sas_net(source_sas_data)  # (B, 2)
    sas_probs = F.softmax(sas_logits, dim=-1)  # (B, 2)

    sa_logits = sa_net(source_sa_data)  # (B, 2)
    sa_probs = F.softmax(sa_logits, dim=-1)  # (B, 2)

    delta = torch.log(sas_probs[:, source_idx] / sas_probs[:, target_idx]) - torch.log(
        sa_probs[:, source_idx] / sa_probs[:, target_idx]
    ).to(
        torch.float32
    )  # (B,)

    assert delta.shape[0] == len(rewards)
    print(type(rewards.astype(np.float32)))
    rewards = rewards.reshape(rewards.shape[0], 1)  # (B, 1)  for medium-replay

    new_rewards = (
        rewards.astype(np.float32)[:, 0] - eta * delta.detach().cpu().numpy().squeeze()
    )  # (B,)
    source_datadict["rewards"] = new_rewards
    return source_datadict