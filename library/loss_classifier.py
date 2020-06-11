import torch
from torch import nn

from torch.nn import functional as F

crossentropy = nn.CrossEntropyLoss()


def loss_cross_entropy(netC, x_real, label):
    logits = netC(x_real)
    loss = crossentropy(logits, label)
    return loss.mean()


def loss_triple_gan(netC, x_real, label):
    logits = netC(x_real)
    loss = crossentropy(logits, label)
    return loss.mean()


loss_dict = {"crossentropy": loss_cross_entropy}
