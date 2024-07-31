"""
Algorithm implementation fof PU learning called (TED)^n: https://arxiv.org/pdf/2111.00980.pdf
The implimatation is modified version of https://github.com/acmi-lab/PU_learning
"""

import IPython
import numpy as np
import torch


def validate(
    epoch,
    net,
    u_validloader,
    criterion,
    device,
    separate=False,
):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    total_pos = 0
    correct_pos = 0

    total_neg = 0
    correct_neg = 0

    with torch.no_grad():
        for batch_idx, (_, inputs, _, true_targets) in enumerate(u_validloader):
            inputs, true_targets = inputs.to(device), true_targets.to(device)
            outputs = net(inputs.to(torch.float32))  # you need to specify float32

            predicted = torch.nn.functional.softmax(outputs, dim=-1).argmax(1)

            outputs = torch.nn.functional.softmax(outputs, dim=-1)
            # assert outputs.mean() * 2 == 1.0
            loss = criterion(outputs, true_targets)

            test_loss += loss.item()
            total += true_targets.size(0)

            correct_preds = predicted.eq(true_targets).cpu().numpy()
            correct += np.sum(correct_preds)

            if separate:
                pos_idx = (true_targets == 0).detach().cpu().numpy()
                neg_idx = (true_targets == 1).detach().cpu().numpy()

                total_pos += pos_idx.sum()
                correct_pos += np.sum(correct_preds[pos_idx])

                total_neg += neg_idx.sum()
                correct_neg += np.sum(correct_preds[neg_idx])

    if separate:
        return (
            100.0 * correct / total,
            100.0 * correct_pos / total_pos,
            100.0 * correct_neg / total_neg,
        )
    else:
        return 100.0 * correct / total


def test(
    epoch,
    net,
    u_testloader,
    criterion,
    device,
):
    net.eval()
    test_loss = 0
    correct = 0
    positive_correct = 0
    negative_correct = 0
    total = 0
    positive = 0
    negative = 0

    with torch.no_grad():
        for batch_idx, (_, inputs, _, true_targets) in enumerate(u_testloader):
            inputs, true_targets = inputs.to(device), true_targets.to(device)
            outputs = net(inputs.to(torch.float32))  # you need to specify float32

            predicted = torch.nn.functional.softmax(outputs, dim=-1).argmax(1)

            outputs = torch.nn.functional.softmax(outputs, dim=-1)

            loss = criterion(outputs, true_targets)

            test_loss += loss.item()
            total += true_targets.size(0)
            positive += (true_targets == 0).sum().item()
            negative += (true_targets == 1).sum().item()

            # count correct predictions
            correct_preds = predicted.eq(true_targets).cpu().numpy()
            corect_positive_preds = (
                predicted.eq(true_targets)[(true_targets == 0).detach().cpu().numpy()]
                .cpu()
                .numpy()
            )
            corect_negative_preds = (
                predicted.eq(true_targets)[(true_targets == 1).detach().cpu().numpy()]
                .cpu()
                .numpy()
            )

            correct += np.sum(correct_preds)
            positive_correct += np.sum(corect_positive_preds)
            negative_correct += np.sum(corect_negative_preds)

    total_acc = 100.0 * correct / total
    positive_acc = 100.0 * positive_correct / positive
    negative_acc = 100.0 * negative_correct / negative
    return total_acc, positive_acc, negative_acc


def train_PvU(
    epoch,
    net,
    p_trainloader,
    u_trainloader,
    optimizer,
    criterion,
    device,
    use_true_label=False,
):
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (p_data, u_data) in enumerate(zip(p_trainloader, u_trainloader)):
        optimizer.zero_grad()
        _, p_inputs, p_targets = p_data
        _, u_inputs, u_targets, u_true_targets = u_data

        p_targets = p_targets.to(device)
        if use_true_label:
            u_targets = u_true_targets.to(device)
        else:
            u_targets = u_targets.to(device)

        assert p_targets.sum() == 0  # confirm that p_targets are all 0

        inputs = torch.cat((p_inputs, u_inputs), dim=0)
        targets = torch.cat((p_targets, u_targets), dim=0)
        inputs = inputs.to(device).to(torch.float32)  # you need to specify float32
        outputs = net(inputs)

        p_outputs = outputs[: len(p_targets)]
        u_outputs = outputs[len(p_targets) :]

        p_loss = criterion(p_outputs, p_targets)
        u_loss = criterion(u_outputs, u_targets)
        loss = (p_loss + u_loss) / 2.0
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)

        correct_preds = predicted.eq(targets).cpu().numpy()
        correct += np.sum(correct_preds)

    return 100.0 * correct / total


def valid_PvU(
    epoch,
    net,
    p_validloader,
    u_validloader,
    criterion,
    device,
):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (p_data, u_data) in enumerate(zip(p_validloader, u_validloader)):
            _, p_inputs, p_targets = p_data
            _, u_inputs, u_targets, u_true_targets = u_data

            p_targets = p_targets.to(device)
            u_targets = u_targets.to(device)

            assert p_targets.sum() == 0  # confirm that p_targets are all 0

            inputs = torch.cat((p_inputs, u_inputs), dim=0)
            targets = torch.cat((p_targets, u_targets), dim=0)
            inputs = inputs.to(device).to(torch.float32)  # you need to specify float32

            outputs = net(inputs)

            p_outputs = outputs[: len(p_targets)]
            u_outputs = outputs[len(p_targets) :]

            p_loss = criterion(p_outputs, p_targets)
            u_loss = criterion(u_outputs, u_targets)
            loss = (p_loss + u_loss) / 2.0

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)

            correct_preds = predicted.eq(targets).cpu().numpy()
            correct += np.sum(correct_preds)

    return 100.0 * correct / total


def train_PU_discard(
    epoch,
    total_epochs,
    net,
    p_trainloader,
    u_trainloader,
    optimizer,
    criterion,
    device,
    keep_sample=None,
):
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (p_data, u_data) in enumerate(zip(p_trainloader, u_trainloader)):
        optimizer.zero_grad()

        _, p_inputs, p_targets = p_data
        u_index, u_inputs, u_targets, u_true_targets = u_data

        u_idx = np.where(keep_sample[u_index.numpy()] == 1)[0]

        if len(u_idx) < 1:
            continue

        u_targets = u_targets[u_idx]

        p_targets = p_targets.to(device)
        u_targets = u_targets.to(device)

        u_inputs = u_inputs[u_idx]
        inputs = torch.cat((p_inputs, u_inputs), dim=0)
        targets = torch.cat((p_targets, u_targets), dim=0)
        inputs = inputs.to(device).to(torch.float32)  # you need to specify float32
        outputs = net(inputs)

        p_outputs = outputs[: len(p_targets)]
        u_outputs = outputs[len(p_targets) :]

        p_loss = criterion(p_outputs, p_targets)
        u_loss = criterion(u_outputs, u_targets)

        loss = (p_loss + u_loss) / 2.0

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        _, predicted = outputs.max(1)
        total += targets.size(0)

        correct_preds = predicted.eq(targets).cpu().numpy()
        correct += np.sum(correct_preds)

    return 100.0 * correct / total


def rank_inputs(_, net, u_trainloader, device, alpha, u_size):
    net.eval()
    output_probs = np.zeros(u_size)
    keep_samples = np.ones_like(output_probs)
    true_targets_all = np.zeros(u_size)

    with torch.no_grad():
        for batch_num, (idx, inputs, _, true_targets) in enumerate(u_trainloader):
            idx = idx.numpy()

            inputs = inputs.to(device).to(torch.float32)  # you need to specify float32
            outputs = net(inputs)

            probs = torch.nn.functional.softmax(outputs, dim=-1)[:, 0]
            output_probs[idx] = probs.detach().cpu().numpy().squeeze()
            true_targets_all[idx] = true_targets.numpy().squeeze()

    sorted_idx = np.argsort(output_probs)

    keep_samples[sorted_idx[u_size - int(alpha * u_size) :]] = 0

    neg_reject = np.sum(
        true_targets_all[sorted_idx[u_size - int(alpha * u_size) :]] == 1.0
    )

    neg_reject = neg_reject / int(alpha * u_size)
    return keep_samples, neg_reject
