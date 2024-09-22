from typing import Dict, Tuple

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils import data


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


def to_sas(data):
    state = data["observations"]
    action = data["actions"]
    next_state = data["next_observations"]

    return np.concatenate((state, action, next_state), axis=1)


def to_sa(data):
    state = data["observations"]
    action = data["actions"]

    return np.concatenate((state, action), axis=1)


class PN_data(torch.utils.data.Dataset):
    def __init__(self, positive_data, negative_data):
        self.p_data = positive_data
        self.n_data = negative_data
        print("n_data", len(negative_data))
        assert self.p_data.shape[1] == self.n_data.shape[1], "Check data dim"

    def __len__(self):
        return len(self.p_data) + len(self.n_data)


class PosData(torch.utils.data.Dataset):
    def __init__(
        self,
        data=None,
        index=None,
        data_type=None,
    ):
        self.data = data
        self.negatives = np.zeros(data.shape[0], dtype=np.int_)
        self.data_type = data_type
        self.index = index

    def __len__(self):
        return len(self.negatives)

    def __getitem__(self, idx):
        index, _inp, negative = self.index[idx], self.data[idx], self.negatives[idx]

        return index, _inp, negative


class UnlabelData(torch.utils.data.Dataset):
    def __init__(
        self,
        pos_data=None,
        neg_data=None,
        index=None,
        data_type=None,
    ):
        self.data = np.concatenate((pos_data, neg_data), axis=0)
        self.true_negatives = np.concatenate(
            (
                np.zeros(pos_data.shape[0], dtype=np.int_),
                np.ones(neg_data.shape[0], dtype=np.int_),
            ),
            axis=0,
        )
        self.negatives = np.ones_like(self.true_negatives, dtype=np.int_)

        self.data_type = data_type
        self.index = index

    def __len__(self):
        return len(self.negatives)

    def __getitem__(self, idx):
        index, _inp, negative, true_negative = (
            self.index[idx],
            self.data[idx],
            self.negatives[idx],
            self.true_negatives[idx],
        )
        return index, _inp, negative, true_negative


def get_PUDataSplits(data_obj, pos_size, alpha, beta, data_type=None):
    unlabel_size = int((1 - beta) * pos_size / beta)
    print(
        "total_size",
        pos_size + unlabel_size,
        "pos_size",
        pos_size,
        "unlabeled_size",
        unlabel_size,
        "data.p_data",
        len(data_obj.p_data),
        "data.n_data",
        len(data_obj.n_data),
        "alpha",
        alpha,
        "beta",
        beta,
    )
    assert (pos_size + int(unlabel_size * alpha)) <= len(
        data_obj.p_data
    ) + 1, "Check sizes again"
    assert (int(unlabel_size * (1 - alpha))) <= len(
        data_obj.n_data
    ) + 1, "Check sizes again"

    pos_data = data_obj.p_data[:pos_size]
    unlabel_pos_data = data_obj.p_data[pos_size : pos_size + int(unlabel_size * alpha)]
    unlabel_neg_data = data_obj.n_data[: int(unlabel_size * (1 - alpha))]

    return PosData(
        data=pos_data,
        index=np.array(range(pos_size)),
        data_type=data_type,
    ), UnlabelData(
        pos_data=unlabel_pos_data,
        neg_data=unlabel_neg_data,
        index=np.array(range(unlabel_size)),
        data_type=data_type,
    )


def make_classifier(hidden_dims: Tuple[int], input_dim=None):
    net = nn.Sequential()
    net.add_module("input", nn.Linear(input_dim, hidden_dims[0]))
    net.add_module("input_relu", nn.ReLU())
    for i in range(1, len(hidden_dims)):
        net.add_module(f"fc{i}", nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
        net.add_module(f"relu{i}", nn.ReLU())
    net.add_module("output", nn.Linear(hidden_dims[-1], 2))
    return net


def make_classification_dataset(
    shifted_dataset_path: str,
    device: str,
    alpha: float,
    beta: float,
    pos_size,
    config,
):
    import d4rl
    import gym
    import h5py

    if config.data.shift == "body_mass" or config.data.shift == "joint_noise":
        positive_env = gym.make(
            f"{config.env_name.lower()}-{config.data.positive_data_quality.replace('_', '-')}-v2"
        )
        positive_datadict = d4rl.qlearning_dataset(positive_env)
        negative_datadict = h5py.File(shifted_dataset_path, "r")
    elif config.data.shift == "halfcheetah_vs_walker2d":
        positive_env = gym.make(
            f"halfcheetah-{config.data.positive_data_quality.replace('_', '-')}-v2"
        )
        negative_env = gym.make(
            f"walker2d-{config.data.negative_data_quality.replace('_', '-')}-v2"
        )
        positive_datadict = d4rl.qlearning_dataset(positive_env)
        negative_datadict = d4rl.qlearning_dataset(negative_env)
    positive_datadict = shuffle_datadict(positive_datadict)
    negative_datadict = shuffle_datadict(negative_datadict)

    if config.data.input_type == "sas":
        positive_data: np.ndarray = to_sas(positive_datadict)
        negative_data: np.ndarray = to_sas(negative_datadict)
    elif config.data.input_type == "sa":
        positive_data: np.ndarray = to_sa(positive_datadict)
        negative_data: np.ndarray = to_sa(negative_datadict)
    else:
        raise ValueError("input_type should be sas or sa")

    train_positive_data, val_test_positive_data = train_test_split(
        positive_data, test_size=0.2, random_state=42
    )
    val_positive_data, test_souce_data = train_test_split(
        val_test_positive_data, test_size=0.5, random_state=42
    )

    train_negative_data, val_test_negative_data = train_test_split(
        negative_data, test_size=0.2, random_state=42
    )
    val_negative_data, test_test_negative_data = train_test_split(
        val_test_negative_data, test_size=0.5, random_state=43
    )

    pn_traindata = PN_data(train_positive_data, train_negative_data)
    pn_validdata = PN_data(val_positive_data, val_negative_data)
    pn_testdata = PN_data(test_souce_data, test_test_negative_data)

    p_traindata, u_traindata = get_PUDataSplits(
        rl_traindata, pos_size=int(pos_size * 0.8), alpha=alpha, beta=beta
    )
    p_validdata, u_validdata = get_PUDataSplits(
        rl_validdata, pos_size=int(pos_size * 0.1), alpha=alpha, beta=beta
    )
    p_testdata, u_testdata = get_PUDataSplits(
        rl_testdata, int(pos_size * 0.1), alpha=alpha, beta=beta
    )

    # Create train dataloader
    p_trainloader = torch.utils.data.DataLoader(
        p_traindata, batch_size=config.batch_size, shuffle=True, num_workers=0
    )
    u_trainloader = torch.utils.data.DataLoader(
        u_traindata, batch_size=config.batch_size, shuffle=True, num_workers=0
    )

    # Create validation dataloader
    p_validloader = torch.utils.data.DataLoader(
        p_validdata, batch_size=config.batch_size, shuffle=True, num_workers=0
    )
    u_validloader = torch.utils.data.DataLoader(
        u_validdata, batch_size=config.batch_size, shuffle=True, num_workers=0
    )

    # Create test dataloader
    p_testloader = torch.utils.data.DataLoader(
        p_testdata, batch_size=config.batch_size, shuffle=True, num_workers=0
    )
    u_testloader = torch.utils.data.DataLoader(
        u_testdata, batch_size=config.batch_size, shuffle=True, num_workers=0
    )
    return (
        p_trainloader,
        u_trainloader,
        p_validloader,
        u_validloader,
        p_testloader,
        u_testloader,
    )
