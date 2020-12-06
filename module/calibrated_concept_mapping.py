import random

import torch
import torch.nn as nn
import numpy as np

from sklearn.linear_model import LogisticRegression

def calibrate(logits, labels):
    """
    calibrate by minimizing negative log likelihood.
    :param logits: pytorch tensor with shape of [N, N_c]
    :param labels: pytorch tensor of labels
    :return: float
    """
    scale = nn.Parameter(torch.ones(
        1, 1, dtype=torch.float32), requires_grad=True)
    optim = torch.optim.LBFGS([scale])

    def loss():
        optim.zero_grad()
        lo = nn.CrossEntropyLoss()(logits * scale, labels)
        lo.backward()
        return lo

    state = optim.state[scale]
    for i in range(20):
        optim.step(loss)
        print(f'calibrating, {scale.item()}')
        if state['n_iter'] < optim.state_dict()['param_groups'][0]['max_iter']:
            break

    return scale.item()


def concept_mapping(train_logits, train_labels, validation_logits, validation_labels):
    """

    :param train_logits (ImageNet logits): [N, N_p], where N_p is the number of classes in pre-trained dataset
    :param train_labels:  [N], where 0 <= each number < N_t, and N_t is the number of target dataset
    :param validation_logits (ImageNet logits): [N, N_p]
    :param validation_labels:  [N]
    :return: [N_c, N_p] matrix representing the conditional probability p(pre-trained class | target_class)
     """

    def softmax_np(x):
        max_el = np.max(x, axis=1, keepdims=True)
        x = x - max_el
        x = np.exp(x)
        s = np.sum(x, axis=1, keepdims=True)
        return x / s

    # convert logits to probabilities
    train_probabilities = softmax_np(train_logits * 0.8840456604957581)
    validation_probabilities = softmax_np(
        validation_logits * 0.8840456604957581)
    # train_probabilities = softmax_np(train_logits * 0.921597957611084)
    # validation_probabilities = softmax_np(validation_logits * 0.921597957611084)

    Cs = []
    accs = []
    classifiers = []
    for C in [1e4, 3e3, 1e3, 3e2, 1e2, 3e1, 1e1, 3.0, 1.0, 3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4]:
        cls = LogisticRegression(
            multi_class='multinomial', C=C, fit_intercept=False)
        cls.fit(train_probabilities, train_labels)
        val_predict = cls.predict(validation_probabilities)
        val_acc = np.sum((val_predict == validation_labels).astype(
            np.float)) / len(validation_labels)
        Cs.append(C)
        accs.append(val_acc)
        classifiers.append(cls)

    accs = np.asarray(accs)
    ind = int(np.argmax(accs))
    cls = classifiers[ind]
    del classifiers

    validation_logits = np.matmul(validation_probabilities, cls.coef_.T)
    validation_logits = torch.from_numpy(validation_logits.astype(np.float32))
    validation_labels = torch.from_numpy(validation_labels)

    scale = calibrate(validation_logits, validation_labels)

    p_target_given_pretrain = softmax_np(
        cls.coef_.T * scale)  # shape of [N_p, N_c], conditional probability p(target_class | pre-trained class)

    pretrain_marginal = np.mean(all_probabilities, axis=0).reshape(
        (-1, 1))  # shape of [N_p, 1]
    p_joint_distribution = (p_target_given_pretrain * pretrain_marginal).T
    p_pretrain_given_target = p_joint_distribution / \
        np.sum(p_joint_distribution, axis=1, keepdims=True)

    return p_pretrain_given_target
