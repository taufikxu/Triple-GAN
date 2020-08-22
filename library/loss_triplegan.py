import torch

# from torch.nn import functional as F

from Utils.flags import FLAGS
from library.loss_cla import loss_cross_entropy

softmax = torch.nn.Softmax(1)
crossentropy_ele = torch.nn.CrossEntropyLoss(reduction="none")


def loss_cross_entropy_ele(logits, label):
    loss = crossentropy_ele(logits, label)
    return loss


def loss_hinge_dis_elr_icr(netD, netG, netC, x_l, z_rand, label, x_u, device):
    with torch.no_grad():
        z_rand_2 = torch.randn(FLAGS.bs_g, FLAGS.g_z_dim).to(device)
        z_rand_2 = z_rand + FLAGS.lam_sigma * z_rand_2
        x_fake_2 = netG(z_rand_2, label).detach()
        x_fake = netG(z_rand, label).detach()
        logits_c = netC(x_u).detach()
        _, l_fake = torch.max(logits_c, 1)

    d_real = netD(x=x_l, y=label, aug=True)
    d_real_2 = netD(x=x_l, y=label, aug=True)
    d_fake_g = netD(x=x_fake, y=label, aug=True)        
    d_fake_g_2 = netD(x=x_fake, y=label, aug=True)      
    d_fake_g_3 = netD(x=x_fake_2, y=label, aug=True)
    loss_real = torch.mean(torch.relu(1.0 - d_real))
    loss_fake_g = torch.mean(torch.relu(1.0 + d_fake_g))
    loss_bcr = FLAGS.lam_real * torch.mean((d_real_2 - d_real) ** 2, dim=[0, 1]) 
    loss_bcr = loss_bcr + FLAGS.lam_fake * torch.mean((d_fake_g_2 - d_fake_g) ** 2, dim=[0, 1]) 
    loss_zcr = FLAGS.lam_dis * torch.mean((0.5*d_fake_g+0.5*d_fake_g_2 -d_fake_g_3) ** 2, dim=[0, 1]) 

    if FLAGS.gan_traind_c == "argmax":
        d_fake_c = netD(x_u, l_fake)
        loss_fake_c = torch.mean(torch.relu(1.0 + d_fake_c))
    elif FLAGS.gan_traind_c == "int":
        d_fake_c = netD(x_u)
        loss_fake_c = torch.mean(
            torch.sum(torch.relu(1.0 + d_fake_c) * softmax(logits_c), dim=1)
        )

    return (
        loss_real + 0.5 * loss_fake_g + 0.5 * loss_fake_c + loss_bcr + loss_zcr,
        d_real.mean(),
        d_fake_g.mean(),
        d_fake_c.mean(),
    )

def loss_hinge_dis_elr_bcr(netD, netG, netC, x_l, z_rand, label, x_u):
    with torch.no_grad():
        x_fake = netG(z_rand, label).detach()
        logits_c = netC(x_u).detach()
        _, l_fake = torch.max(logits_c, 1)

    d_real = netD(x=x_l, y=label, aug=True)
    d_real_2 = netD(x=x_l, y=label, aug=True)
    d_fake_g = netD(x=x_fake, y=label, aug=True)        
    d_fake_g_2 = netD(x=x_fake, y=label, aug=True)
    loss_real = torch.mean(torch.relu(1.0 - d_real))
    loss_fake_g = torch.mean(torch.relu(1.0 + d_fake_g))
    loss_bcr = 10. * torch.mean((d_real_2 - d_real) ** 2, dim=[0, 1]) + 10. * torch.mean((d_fake_g_2 - d_fake_g) ** 2, dim=[0, 1]) 

    if FLAGS.gan_traind_c == "argmax":
        d_fake_c = netD(x_u, l_fake)
        loss_fake_c = torch.mean(torch.relu(1.0 + d_fake_c))
    elif FLAGS.gan_traind_c == "int":
        d_fake_c = netD(x_u)
        loss_fake_c = torch.mean(
            torch.sum(torch.relu(1.0 + d_fake_c) * softmax(logits_c), dim=1)
        )

    return (
        loss_real + 0.5 * loss_fake_g + 0.5 * loss_fake_c + loss_bcr,
        d_real.mean(),
        d_fake_g.mean(),
        d_fake_c.mean(),
    )

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


def loss_hinge_dis_bcr(netD, netG, netC, netC_d, x_l, z_rand, label, x_u, x_u_d):
    with torch.no_grad():
        x_fake = netG(z_rand, label).detach()
        logits_c = netC(x_u).detach()
        _, l_fake = torch.max(logits_c, 1)
        _, l_d = torch.max(netC_d(x_u_d).detach(), 1)

    x_real_for_d = torch.cat([x_l[: FLAGS.bs_l_for_d], x_u_d[: FLAGS.bs_u_for_d]], 0)
    label_real_for_d = torch.cat(
        [label[: FLAGS.bs_l_for_d], l_d[: FLAGS.bs_u_for_d]], 0
    )

    d_real = netD(x=x_real_for_d, y=label_real_for_d, aug=True)
    d_real_2 = netD(x=x_real_for_d, y=label_real_for_d, aug=True)
    d_fake_g = netD(x=x_fake, y=label, aug=True)        
    d_fake_g_2 = netD(x=x_fake, y=label, aug=True)
    loss_bcr = 10. * torch.mean((d_real_2 - d_real) ** 2, dim=[0, 1]) + 10. * torch.mean((d_fake_g_2 - d_fake_g) ** 2, dim=[0, 1]) 


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
        loss_real + 0.5 * loss_fake_g + 0.5 * loss_fake_c + loss_bcr,
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


def pseudo_discriminative_loss_MT(netC, netG, netC_T, z_rand, label):
    with torch.no_grad():
        x_fake = netG(z_rand, label).detach()
        x_fake_u = netG(torch.randn_like(z_rand), label).detach()

    logit_l = netC(x_fake)
    logit_u_1, logit_u_2 = netC(x_fake_u, double=True)
    logit_ut = netC_T(x_fake_u).detach()

    # only use data corrected predicted by the model
    if FLAGS.masked_pdl == True:
        max_logit, pred = torch.max(logit_l, dim=1)
        # mask = label == pred
        mask = (max_logit > torch.log(0.95))
    else:
        mask = 1.0

    loss_l = loss_cross_entropy_ele(logit_l, label)
    loss_l = torch.mean(mask * loss_l, dim=0)

    prob_u_2 = softmax(logit_u_2)
    prob_t = softmax(logit_ut)
    loss_u = FLAGS.max_consistency_cost * torch.mean(
        (prob_u_2 - prob_t) ** 2, dim=[0, 1]
    )
    loss_u = loss_u + FLAGS.alpha_mse * torch.mean(
        (logit_u_2 - logit_u_1) ** 2, dim=[0, 1]
    )
    return loss_l + loss_u


def pseudo_discriminative_loss_consist_MT(netC, netG, netC_T, z_rand, label):
    with torch.no_grad():
        x_fake_u = netG(torch.randn_like(z_rand), label).detach()

    logit_u_1, logit_u_2 = netC(x_fake_u, double=True)
    logit_ut = netC_T(x_fake_u).detach()

    prob_u_2 = softmax(logit_u_2)
    prob_t = softmax(logit_ut)
    loss_u = FLAGS.max_consistency_cost * torch.mean(
        (prob_u_2 - prob_t) ** 2, dim=[0, 1]
    )
    loss_u = loss_u + FLAGS.alpha_mse * torch.mean(
        (logit_u_2 - logit_u_1) ** 2, dim=[0, 1]
    )
    return loss_u


def loss_hinge_gen(netD, netG, z_rand, label):
    x_fake = netG(z_rand, label)
    d_fake = netD(x_fake, label)
    loss_fake = -torch.mean(d_fake)
    return loss_fake, d_fake.mean()

def loss_hinge_gen_icr(netD, netG, z_rand, label, device):
    x_fake = netG(z_rand, label)
    z_rand_2 = torch.randn(FLAGS.bs_g, FLAGS.g_z_dim).to(device)
    z_rand_2 = z_rand + FLAGS.lam_sigma * z_rand_2
    x_fake_2 = netG(z_rand, label)
    loss_zcr = FLAGS.lam_gen * torch.mean((x_fake - x_fake_2) ** 2) 
    d_fake = netD(x_fake, label)
    loss_fake = -torch.mean(d_fake) - loss_zcr
    return loss_fake, d_fake.mean()


g_loss_dict = {
    "hinge": loss_hinge_gen,
    "bcr": loss_hinge_gen,
}
c_loss_dict = {
    "hinge": loss_hinge_cla,
    "bcr": loss_hinge_cla,
}
d_loss_dict = {
    "hinge": loss_hinge_dis,
    "bcr": loss_hinge_dis_bcr,
}
