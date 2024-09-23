import os
import time

import d4rl
import gym
import numpy
import pyrallis
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import wandb
from classifier import rank_inputs, test, train_PU_discard, train_PvU, validate, p_probs, u_probs, BBE_estimator
from utils import (ClassifierConfig, make_classification_dataset,
                   make_classifier, make_classifier_params_path,
                   make_shifted_dataset_path)


@pyrallis.wrap()
def main(config: ClassifierConfig):
    train(config)


def train(config):
    print(
        f"Start classifier training {config.env_name}, shift: {config.data.shift}, method: {config.method}, positive data quality: {config.data.positive_data_quality}, negative data quality: {config.data.negative_data_quality}"
    )
    wandb.init(project=f"train-" + config.project, config=config)
    positive_data_env = gym.make(
        f"{config.env_name}-{config.data.positive_data_quality.replace('_', '-')}-v2"
    )
    # paths
    shifted_dataset_path = make_shifted_dataset_path(config)

    sas_param_path, sa_params = make_classifier_params_path(config)

    if not os.path.exists(param_path):
        os.makedirs(os.path.dirname(param_path), exist_ok=True)
    if not os.path.exists(sa_params):
        os.makedirs(os.path.dirname(sa_params), exist_ok=True)
    param_path = sas_param_path if config.input_type == "sas" else sa_params

    positive_num = int(config.data.size * config.data.positive_ratio)
    negative_num = int(config.data.size * (1 - config.data.positive_ratio))
    unlabeled_num = int(config.data.size * (1 - config.data.labeled_ratio))

    alpha = (unlabeled_num - negative_num) / unlabeled_num  # positive data in unlabeled
    beta = (positive_num + negative_num - unlabeled_num) / (positive_num + negative_num)  # labeled data ratio
    device = "cuda" if torch.cuda.is_available() else "cpu"

    (
        p_trainloader,
        u_trainloader,
        p_validloader,
        u_validloader,
        p_testloader,
        u_testloader,
    ) = make_classification_dataset(
        shifted_dataset_path,
        positive_data_env,
        device,
        alpha,
        beta,
        pos_size=positive_num + negative_num - unlabeled_num,
        config=config,
    )

    input_dim = p_trainloader.dataset.data.shape[-1]
    net = make_classifier(config.hidden_dims, input_dim=input_dim).to(device)

    assert p_trainloader.dataset.__len__() <= u_trainloader.dataset.__len__()
    assert p_validloader.dataset.__len__() <= u_validloader.dataset.__len__()
    assert p_testloader.dataset.__len__() <= u_testloader.dataset.__len__()

    if device.startswith("cuda"):
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(net.parameters(), lr=config.lr, weight_decay=config.wd)

    if config.method == "pu":
        print("Warmup Start")
        for epoch in range(config.warm_start_epochs):
            sta = time.time()
            train_acc = train_PvU(
                epoch,
                net,
                p_trainloader,
                u_trainloader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
            )

            valid_acc = validate(
                epoch,
                net,
                u_validloader,
                criterion=criterion,
                device=device,
            )
            end = time.time()
            print(f"train time: {end - sta}")

            # Estimate alpha
            sta = time.time()
            pos_probs = p_probs(net, device, p_validloader)
            unlabeled_probs, unlabeled_negatives = u_probs(net, device, u_validloader)
            our_mpe_estimate, _, _ = BBE_estimator(
                pos_probs, unlabeled_probs, unlabeled_negatives
            )
            end = time.time()
            print(f"BBE_estimator time: {end - sta}")
            alpha_estimate = our_mpe_estimate

            log = {
                f"{config.env_name}/wamup/epoch": epoch,
                f"{config.env_name}/warmup/train_acc": train_acc,
                f"{config.env_name}/warmup/valid_acc": valid_acc,
                f"{config.env_name}/warmup/alpha_hat": alpha_estimate,
            }
            print(log)

        print("Algo Training")
        for epoch in range(config.epochs):
            alpha_used = alpha_estimate
            sta = time.time()
            keep_samples, neg_reject = rank_inputs(
                epoch,
                net,
                u_trainloader,
                device,
                alpha_used,
                u_size=u_trainloader.dataset.__len__(),
            )
            end = time.time()
            print(f"rank_inputs time: {end - sta}")

            sta = time.time()
            train_acc = train_PU_discard(
                epoch,
                config.epochs,
                net,
                p_trainloader,
                u_trainloader,
                optimizer,
                criterion,
                device,
                keep_sample=keep_samples,
            )
            end = time.time()
            print(f"train_PU_discard time: {end - sta}")
            valid_acc = validate(
                epoch, net, u_validloader, criterion=criterion, device=device
            )

            # Estimate alpha
            pos_probs = p_probs(net, device, p_validloader)
            unlabeled_probs, unlabeled_negatives = u_probs(net, device, u_validloader)
            our_mpe_estimate, _, _ = BBE_estimator(
                pos_probs, unlabeled_probs, unlabeled_negatives
            )
            alpha_estimate = our_mpe_estimate

            log = {
                f"{config.env_name}/epoch": epoch,
                f"{config.env_name}/train_acc": train_acc,
                f"{config.env_name}/valid_acc": valid_acc,
                f"{config.env_name}/alpha_hat": alpha_estimate,
            }
            print(log)
            wandb.log(log)
    elif config.method == "pvu":  # train classifier with policy and value
        for epoch in range(config.epochs):
            train_acc = train_PvU(
                epoch,
                net,
                p_trainloader,
                u_trainloader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
            )

            valid_acc = validate(
                epoch,
                net,
                u_validloader,
                criterion=criterion,
                device=device,
            )
            log = {
                f"{config.env_name}/epoch": epoch,
                f"{config.env_name}/train_acc": train_acc,
                f"{config.env_name}/valid_acc": valid_acc,
            }
            wandb.log(log)
            print(log)

    final_test_acc, final_pos_test_acc, final_neg_test_acc = test(
        epoch, net, u_testloader, criterion=criterion, device=device
    )

    log = {
        f"{config.env_name}/final_test_acc": final_test_acc,
        f"{config.env_name}/final_pos_test_acc": final_pos_test_acc,
        f"{config.env_name}/final_neg_test_acc": final_neg_test_acc,
    }
    torch.save(net.state_dict(), param_path)
    wandb.log(log)
    wandb.finish()


if __name__ == "__main__":
    main()
