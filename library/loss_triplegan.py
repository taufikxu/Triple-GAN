import torch

# from torch.nn import functional as F

from Utils.flags import FLAGS
from library.loss_cla import loss_cross_entropy

softmax = torch.nn.Softmax(1)



def loss_hinge_dis_elr(netD, netG, netC, x_l, z_rand, label, x_u):
    with torch.no_grad():
        x_fake = netG(z_rand, label).detach()
        logits_c = netC(x_u).detach()
        _, l_fake = torch.max(logits_c, 1)

    d_real = netD(x_l, label)
    d_fake_g = netD(x_fake, label)
    loss_real = torch.mean(torch.relu(1.0 - d_real))
    loss_fake_g = torch.mean(torch.relu(1.0 + d_fake_g))

    if FLAGS.gan_traind_c == "argmax":
        d_fake_c = netD(x_u, l_fake)
        loss_fake_c = torch.mean(torch.relu(1.0 + d_fake_c))
    elif FLAGS.gan_traind_c == "int":
        d_fake_c = netD(x_u)
        loss_fake_c = torch.mean(
            torch.sum(torch.relu(1.0 + d_fake_c) * softmax(logits_c), dim=1)
        )

    return (
        loss_real + 0.5 * loss_fake_g + 0.5 * loss_fake_c,
        d_real.mean(),
        d_fake_g.mean(),
        d_fake_c.mean(),
    )

def loss_hinge_dis(netD, netG, netC, netC_d, x_l, z_rand, label, x_u, x_u_d):
    with torch.no_grad():
        x_fake = netG(z_rand, label).detach()
        logits_c = netC(x_u).detach()
        _, l_fake = torch.max(logits_c, 1)
        _, l_d = torch.max(netC_d(x_u_d).detach(), 1)

    x_real_for_d = torch.cat([x_l[: FLAGS.bs_l_for_d], x_u_d[: FLAGS.bs_u_for_d]], 0)
    label_real_for_d = torch.cat(
        [label[: FLAGS.bs_l_for_d], l_d[: FLAGS.bs_u_for_d]], 0
    )
    d_real = netD(x_real_for_d, label_real_for_d)
    d_fake_g = netD(x_fake, label)
    loss_real = torch.mean(torch.relu(1.0 - d_real))
    loss_fake_g = torch.mean(torch.relu(1.0 + d_fake_g))

    if FLAGS.gan_traind_c == "argmax":
        d_fake_c = netD(x_u, l_fake)
        loss_fake_c = torch.mean(torch.relu(1.0 + d_fake_c))
    elif FLAGS.gan_traind_c == "int":
        d_fake_c = netD(x_u)
        loss_fake_c = torch.mean(
            torch.sum(torch.relu(1.0 + d_fake_c) * softmax(logits_c), dim=1)
        )

    return (
        loss_real + 0.5 * loss_fake_g + 0.5 * loss_fake_c,
        d_real.mean(),
        d_fake_g.mean(),
        d_fake_c.mean(),
    )


def loss_hinge_cla(netD, netC, x_u):
    probs = softmax(netC(x_u))
    d_fake = netD(x_u)
    loss_fake = -torch.mean(torch.sum(probs * d_fake, dim=1), dim=0)
    return loss_fake.mean(), d_fake.mean()


def pseudo_discriminative_loss(netC, netG, z_rand, label):
    with torch.no_grad():
        x_fake = netG(z_rand, label).detach()
    logit = netC(x_fake)
    return loss_cross_entropy(logit, label)


def loss_hinge_gen(netD, netG, z_rand, label):
    x_fake = netG(z_rand, label)
    d_fake = netD(x_fake, label)
    loss_fake = -torch.mean(d_fake)
    return loss_fake, d_fake.mean()


g_loss_dict = {
    "hinge": loss_hinge_gen,
}
c_loss_dict = {
    "hinge": loss_hinge_cla,
}
d_loss_dict = {
    "hinge": loss_hinge_dis,
}
