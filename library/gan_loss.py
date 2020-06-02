import torch
from torch.nn import functional as F


def loss_dcgan_dis(dis, x_real, x_fake, label):
    d_real = dis(x_real, label)
    d_fake = dis(x_fake, label)
    loss_real = F.binary_cross_entropy_with_logits(d_real, 1)
    loss_fake = F.binary_cross_entropy_with_logits(d_fake, 0)
    return (loss_real + loss_fake).mean(), d_real.mean(), d_fake.mean()


def loss_dcgan_gen(dis, x_fake, label):
    d_fake = dis(x_fake, label)
    loss_fake = F.binary_cross_entropy_with_logits(d_fake, 1)
    return loss_fake.mean(), d_fake.mean()


def loss_hinge_dis(dis, x_real, x_fake, label):
    d_real = dis(x_real, label)
    d_fake = dis(x_fake, label)
    loss_real = torch.mean(torch.relu(1.0 - d_real))
    loss_fake = torch.mean(torch.relu(1.0 + d_fake))
    return (loss_real + loss_fake).mean(), d_real.mean(), d_fake.mean()


def loss_hinge_gen(dis, x_fake, label):
    d_fake = dis(x_fake, label)
    loss_fake = -torch.mean(d_fake)
    return loss_fake.mean(), d_fake.mean()


g_loss_dict = {"dcgan": loss_dcgan_gen, "hinge": loss_hinge_gen}
d_loss_dict = {"dcgan": loss_dcgan_dis, "hinge": loss_hinge_dis}
