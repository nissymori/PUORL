from .evaluate import make_evaluation
from .iql import IQL, create_iql_train_state
from .td3bc import TD3BC, create_td3bc_train_state


def make_agent(config):
    if config.algorithm == "td3bc":
        return TD3BC, create_td3bc_train_state, config.td3bc
    elif config.algorithm == "iql":
        return IQL, create_iql_train_state, config.iql
    else:
        raise ValueError(f"Unknown algorithm: {config.algorithm}")
