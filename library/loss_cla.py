import torch
from torch import nn

from torch.nn import functional as F
from Utils.flags import FLAGS
import numpy as np

crossentropy = nn.CrossEntropyLoss()
softmax = nn.Softmax(1)
logsoftmax = nn.LogSoftmax(1)


# some utils TODO
def update_average(model_tgt, model_src, beta=0.999):
    param_dict_src = dict(model_src.named_parameters())

    for p_name, p_tgt in model_tgt.named_parameters():
        p_src = param_dict_src[p_name]
        assert p_src is not p_tgt
        # p_tgt.copy_(beta * p_tgt + (1.0 - beta) * p_src)
        p_tgt.data.mul_(beta).add_((1 - beta) * p_src.data)


def sigmoid_rampup(global_step, rampup_length):
    global_step = np.clip(global_step, 0, rampup_length)
    phase = 1.0 - global_step / rampup_length
    return np.exp(-5.0 * phase * phase)


def sigmoid_rampdown(global_step, rampdown_length, training_length):
    if global_step >= training_length - rampdown_length:
        phase = 1.0 - (training_length - global_step) / rampdown_length
        return np.exp(-12.5 * phase * phase)
    else:
        return 1.0


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def cosine_rampdown(current, rampdown_length, training_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= training_length
    if current >= training_length - rampdown_length:
        return float(0.5 * (np.cos(np.pi * current / training_length) + 1))
    else:
        return 1.0


# losses
def entropy(logits):
    prob = softmax(logits)
    logprob = logsoftmax(logits)
    return -torch.sum(prob * logprob, dim=1)


def loss_cross_entropy(logits, label):
    loss = crossentropy(logits, label)
    return loss


def loss_supervised(netC, netC_T, it, iter_l, iter_u, device):
    data, label = iter_l.__next__()
    data, label = data.to(device), label.to(device)
    logit_l = netC(data)
    loss_l = loss_cross_entropy(logit_l, label)
    return loss_l, loss_l.detach(), torch.zeros_like(loss_l.detach())


def loss_entropy_ssl(netC, netC_T, it, iter_l, iter_u, device):
    data, label = iter_l.__next__()
    data, label = data.to(device), label.to(device)
    data_u, _ = iter_u.__next__()
    data_u = data_u.to(device)

    logit_l = netC(data)
    logit_u = netC(data_u)

    loss_l = loss_cross_entropy(logit_l, label)
    loss_u = FLAGS.alpha_entropy * entropy(logit_u)
    return loss_l + loss_u, loss_l.detach(), loss_u.detach()


def loss_res_MT_ssl(netC, netC_T, it, iter_l, iter_u, device):
    data, label = iter_l.__next__()
    data, label = data.to(device), label.to(device)
    data_u, _ = iter_u.__next__()
    data_u = data_u.to(device)

    sigmoid_rampup_value = sigmoid_rampup(it, FLAGS.rampup_length)
    cons_coefficient = sigmoid_rampup_value * FLAGS.max_consistency_cost
    logit_l = netC(data)
    logit_u = netC(data_u)
    logit_ut = netC_T(data_u).detach()

    loss_l = loss_cross_entropy(logit_l, label)
    prob = softmax(logit_u)
    prob_t = softmax(logit_ut)
    loss_u = cons_coefficient * torch.sum((prob - prob_t) ** 2, dim=1).mean(dim=0)
    return loss_l + loss_u, loss_l.detach(), loss_u.detach()


def loss_MT_ssl(netC, netC_T, it, iter_l, iter_u, device):
    data, label = iter_l.__next__()
    data, label = data.to(device), label.to(device)
    data_u, _ = iter_u.__next__()
    data_u = data_u.to(device)

    sigmoid_rampup_value = sigmoid_rampup(it, FLAGS.rampup_length)
    cons_coefficient = sigmoid_rampup_value * FLAGS.max_consistency_cost
    logit_l = netC(data)
    logit_u = netC(data_u)
    logit_ut = netC_T(data_u).detach()

    loss_l = loss_cross_entropy(logit_l, label)
    prob = softmax(logit_u)
    prob_t = softmax(logit_ut)
    loss_u = cons_coefficient * torch.sum((prob - prob_t) ** 2, dim=1).mean(dim=0)
    return loss_l + loss_u, loss_l.detach(), loss_u.detach()


def step_ramp(optim_c, netC, netC_T, it, tloss):
    sigmoid_rampup_value = sigmoid_rampup(it, FLAGS.rampup_length_lr)
    sigmoid_rampdown_value = sigmoid_rampdown(it, FLAGS.rampdown_length, FLAGS.n_iter)

    lr = FLAGS.c_lr * sigmoid_rampup_value * sigmoid_rampdown_value
    adam_beta_1 = (
        sigmoid_rampdown_value * FLAGS.adam_beta_1_before_rampdown
        + (1 - sigmoid_rampdown_value) * FLAGS.adam_beta_1_after_rampdown
    )
    if it < FLAGS.rampup_length:
        adam_beta_2 = FLAGS.adam_beta_2_during_rampup
        ema_decay = FLAGS.ema_decay_during_rampup
    else:
        adam_beta_2 = FLAGS.adam_beta_2_after_rampup
        ema_decay = FLAGS.ema_decay_after_rampup

    # update adam
    optim_c.param_groups[0]["betas"] = (adam_beta_1, adam_beta_2)
    optim_c.param_groups[0]["lr"] = lr

    # update student
    optim_c.zero_grad()
    tloss.backward()
    if FLAGS.clip_value > 0:
        torch.nn.utils.clip_grad_norm_(netC.parameters(), FLAGS.clip_value)
    optim_c.step()

    # update teacher
    update_average(netC_T, netC, ema_decay)


def step_ramp_linear(optim_c, netC, netC_T, it, tloss):

    ema_decay = FLAGS.ema_decay_after_rampup
    linear_rampup_value = linear_rampup(it, FLAGS.rampup_length_lr)
    cosine_rampdown_value = cosine_rampdown(it, FLAGS.rampdown_length, FLAGS.n_iter)
    lr = FLAGS.c_lr * linear_rampup_value * cosine_rampdown_value

    # update adam
    optim_c.param_groups[0]["lr"] = lr

    # update student
    optim_c.zero_grad()
    tloss.backward()
    if FLAGS.clip_value > 0:
        torch.nn.utils.clip_grad_norm_(netC.parameters(), FLAGS.clip_value)
    optim_c.step()

    # update teacher
    update_average(netC_T, netC, ema_decay)


def step_regular(optim_c, netC, netC_T, it, tloss):

    ema_decay = FLAGS.ema_decay_after_rampup
    if it > FLAGS.lr_anneal_num:
        times = (it - FLAGS.lr_anneal_num) // FLAGS.lr_anneal_interval
        lr = FLAGS.c_lr * (FLAGS.lr_anneal_coe ** times)
        optim_c.param_groups[0]["lr"] = lr

    # update student
    optim_c.zero_grad()
    tloss.backward()
    if FLAGS.clip_value > 0:
        torch.nn.utils.clip_grad_norm_(netC.parameters(), FLAGS.clip_value)
    optim_c.step()

    # update teacher
    update_average(netC_T, netC, ema_decay)


c_loss_dict = {
    "crossentropy": loss_supervised,
    "entropyssl": loss_entropy_ssl,
    "mtssl": loss_MT_ssl,
    "res_mtssl": loss_res_MT_ssl,
}

c_step_func = {
    "ramp": step_ramp,
    "ramp_linear": step_ramp_linear,
    "regular": step_regular,
}

