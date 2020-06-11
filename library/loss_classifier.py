import torch
from torch import nn

from torch.nn import functional as F
from Utils.flags import FLAGS

crossentropy = nn.CrossEntropyLoss()
softmax = nn.Softmax(1)
logsoftmax = nn.LogSoftmax(1)


def entropy(logits):
    prob = softmax(logits)
    logprob = logsoftmax(logits)
    return -torch.sum(prob * logprob, dim=1)


def loss_cross_entropy(netC, x_real, label):
    logits = netC(x_real)
    loss = crossentropy(logits, label)
    return loss.mean()


def loss_entropy_reg(netC, x_real):
    logits = netC(x_real)
    loss = entropy(logits)
    return loss.mean()


def loss_entropy_ssl(netC, x_l, l, x_u):
    loss_l = loss_cross_entropy(netC, x_l, l)
    loss_u = FLAGS.alpha_entropy * loss_entropy_reg(netC, x_u)
    return loss_l + loss_u


loss_dict = {
    "crossentropy": loss_cross_entropy,
    "entropyreg": loss_entropy_reg,
    "entropyssl": loss_entropy_ssl,
}
