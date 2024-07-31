from dataclasses import dataclass, field
from typing import Sequence, Tuple


@dataclass
class DataConfig:
    env_name: str = "hopper"
    shift: str = "body_mass"  # (body_mass, joint_noise, h_v_w)
    positive_env: str = "original"  # (original shifted)
    positive_data_quality: str = "medium_expert"
    negative_data_quality: str = "random"
    input_type: str = "sas"  # (sas, sa)
    labeled_ratio: float = 0.05
    positive_ratio: float = 0.7
    size: int = 1000000
    train_ratio: float = 0.8

    def __hash__(self):
        return hash(self.__repr__())


@dataclass
class AWACConfig:
    # GENERAL
    batch_size: int = 256
    n_jitted_updates: int = 8
    normalize_state: bool = False
    # NETWORK
    actor_hidden_dims: Tuple[int, ...] = (256, 256, 256, 256)
    critic_hidden_dims: Tuple[int, ...] = (256, 256)
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    # AWAC SPECIFIC
    beta: float = 1.0
    tau: float = 0.003
    discount: float = 0.99

    def __hash__(self):
        return hash(self.__repr__())


@dataclass
class IQLConfig:
    # GENERAL
    batch_size: int = 256
    n_jitted_updates: int = 8
    normalize_state: bool = False
    # TRAINING
    hidden_dims: Tuple[int, int] = (256, 256)
    actor_lr: float = 3e-4
    value_lr: float = 3e-4
    critic_lr: float = 3e-4
    # IQL SPECIFIC
    expectile: float = 0.7  # FYI: for Hopper-me, 0.5 produce better result from CORL
    temperature: float = 3.0  # FYI: for Hopper-me, 6.0 produce better result from CORL
    tau: float = 0.005
    discount: float = 0.99


@dataclass
class TD3BCConfig:
    # GENERAL
    batch_size: int = 256
    n_jitted_updates: int = 8
    normalize_state: bool = True
    # NETWORK
    hidden_dims: Sequence[int] = (256, 256)
    critic_lr: float = 1e-3
    actor_lr: float = 1e-3
    # TD3-BC SPECIFIC
    policy_freq: int = 2  # update actor every policy_freq updates
    alpha: float = 2.5  # BC loss weight
    policy_noise_std: float = 0.2  # std of policy noise
    policy_noise_clip: float = 0.5  # clip policy noise
    tau: float = 0.005  # target network update rate
    discount: float = 0.99  # discount factor

    def __hash__(self):
        return hash(self.__repr__())


@dataclass
class OfflineRLConfig:
    # GENERAL
    project: str = "test-offlinerl"
    env: str = "hopper"
    seed: int = 0
    # DATA
    data: DataConfig = DataConfig()
    # ALGORITHM
    algorithm: str = "td3bc"
    awac: AWACConfig = AWACConfig()
    iql: IQLConfig = IQLConfig()
    td3bc: TD3BCConfig = TD3BCConfig()
    # TRAINING
    method: str = "pu"  # (oracle, pu, only_positive, sharing_all, uds, pvu)
    max_steps: int = 1000000
    eval_interval: int = 10000
    eval_episodes: int = 10
    log_interval: int = 1000

    def __hash__(self):
        return hash(self.__repr__())


@dataclass
class ClassifierConfig:
    # GENERAL
    project: str = "test-classifier"
    env_name: str = "hopper"
    seed: int = 0
    data: DataConfig = DataConfig()
    # NETWORK
    hidden_dims: Sequence[int] = (16, 16)
    lr: float = 1e-3
    wd: float = 5e-4
    # TRAINING
    warm_start_epochs: int = 3
    max_epochs: int = 100
    batch_size: int = 256
    # method
    method: str = "pu"  # (pu, pvu)

    def __hash__(self):
        return hash(self.__repr__())
