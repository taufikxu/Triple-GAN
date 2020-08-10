import torch
from torch import nn

from torch.nn import functional as F
from Utils.flags import FLAGS
import numpy as np
from torch.autograd import Variable


crossentropy = nn.CrossEntropyLoss()
softmax = nn.Softmax(1)
logsoftmax = nn.LogSoftmax(1)


# some utils TODO
def update_average(model_tgt, model_src, beta=0.999):
    param_dict_src = dict(model_src.named_parameters())

    for p_name, p_tgt in model_tgt.named_parameters():
        p_src = param_dict_src[p_name]
        assert p_src is not p_tgt
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


def cosine_rampdown_1(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return max(0.0, float(0.5 * (np.cos(np.pi * current / rampdown_length) + 1)))


# losses
def entropy(logits):
    prob = softmax(logits)
    logprob = logsoftmax(logits)
    return -torch.sum(prob * logprob, dim=1)


def loss_cross_entropy(logits, label):
    loss = crossentropy(logits, label)
    return loss


def loss_elr(netC, it, iter_l, device):
    data, label = iter_l.__next__()
    data, label = data.to(device), label.to(device)
    logit_l = netC(data)
    loss_l = loss_cross_entropy(logit_l, label)
    return loss_l


def loss_elr_wrap(netC, netC_T, it, iter_l, itr, device):
    data, label = iter_l.__next__()
    data, label = data.to(device), label.to(device)
    logit_l = netC(data)
    loss_l = loss_cross_entropy(logit_l, label)
    return loss_l, loss_l, loss_l


def loss_supervised(netC, netC_T, it, iter_l, iter_u, device):
    data, label = iter_l.__next__()
    data_u, _ = iter_u.__next__()
    data_u = data_u.to(device)
    logit_ut = netC_T(data_u).detach()

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
    logit_ut = netC_T(data_u).detach()

    loss_l = loss_cross_entropy(logit_l, label)
    loss_u = FLAGS.alpha_entropy * entropy(logit_u)
    return loss_l + loss_u, loss_l.detach(), loss_u.detach()


def loss_MT_double_ssl(netC, netC_T, it, iter_l, iter_u, device):
    if it == 0:
        print("Using Double MT SSL")
    data, label = iter_l.__next__()
    data, label = data.to(device), label.to(device)
    data_u, _ = iter_u.__next__()
    data_u = data_u.to(device)

    sigmoid_rampup_value = sigmoid_rampup(it, FLAGS.rampup_length)
    cons_coefficient = sigmoid_rampup_value * FLAGS.max_consistency_cost
    logit_l = netC(data)
    logit_u_1, logit_u_2 = netC(data_u, double=True)
    logit_ut = netC_T(data_u).detach()

    loss_l = loss_cross_entropy(logit_l, label)
    prob_u_2 = softmax(logit_u_2)
    prob_t = softmax(logit_ut)
    loss_u = cons_coefficient * torch.mean((prob_u_2 - prob_t) ** 2, dim=[0, 1])
    loss_u = loss_u + FLAGS.alpha_mse * torch.mean(
        (logit_u_2 - logit_u_1) ** 2, dim=[0, 1]
    )
    return loss_l + loss_u, loss_l.detach(), loss_u.detach()


def loss_MT_ssl(netC, netC_T, it, iter_l, iter_u, device):
    data, label = iter_l.__next__()
    data, label = data.to(device), label.to(device)
    data_u, _ = iter_u.__next__()
    data_u = data_u.to(device)

    sigmoid_rampup_value = sigmoid_rampup(it, FLAGS.rampup_length)
    cons_coefficient = sigmoid_rampup_value * FLAGS.max_consistency_cost

    lpi = FLAGS.num_label_per_batch
    batch_input = torch.cat([data[:lpi], data_u[lpi:]], dim=0)
    logit = netC(batch_input)
    logit_ut = netC_T(batch_input).detach()
    logit_l = logit[:lpi]
    logit_u = logit
    # logit_l = netC(data)
    # logit_u = netC(data_u)
    # logit_ut = netC_T(data_u).detach()

    loss_l = loss_cross_entropy(logit_l, label[:lpi])
    prob = softmax(logit_u)
    prob_t = softmax(logit_ut)
    loss_u = cons_coefficient * torch.mean((prob - prob_t) ** 2, dim=[0, 1])
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


def step_ramp_swa(optim_c, swa_optim, netC, netC_T, it, tloss):

    ema_decay = FLAGS.ema_decay_after_rampup
    linear_rampup_value = linear_rampup(it, FLAGS.rampup_length_lr)

    if it >= FLAGS.swa_start and (it - FLAGS.swa_start) % FLAGS.cycle_interval == 0:
        swa_optim.update(netC)

    if it < FLAGS.swa_start:
        # Cosine LR rampdown from https://arxiv.org/abs/1608.03983
        assert FLAGS.rampdown_length >= FLAGS.swa_start
        cosine_rampdown_value = cosine_rampdown_1(it, FLAGS.rampdown_length)
    elif it >= FLAGS.swa_start:
        cosine_rampdown_value = cosine_rampdown_1(
            (FLAGS.swa_start - FLAGS.cycle_interval)
            + ((it - FLAGS.swa_start) % FLAGS.cycle_interval),
            FLAGS.rampdown_length,
        )

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


def step_vat(optim_c, netC, netC_T, it, tloss):
    ema_decay = FLAGS.ema_decay_after_rampup
    if it > FLAGS.lr_anneal_num:
        decayed_lr = (
            (FLAGS.n_iter - it) * FLAGS.c_lr / (FLAGS.n_iter - FLAGS.lr_anneal_num)
        )
        optim_c.param_groups[0]["lr"] = decayed_lr

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


def kl_div_with_logit(q_logit, p_logit):
    q = F.softmax(q_logit, dim=1)
    logq = F.log_softmax(q_logit, dim=1)
    logp = F.log_softmax(p_logit, dim=1)

    qlogq = (q * logq).sum(dim=1).mean(dim=0)
    qlogp = (q * logp).sum(dim=1).mean(dim=0)

    return qlogq - qlogp


def vat_loss_ssl(netC, netC_T, it, iter_l, iter_u, device):
    data, label = iter_l.__next__()
    data, label = data.to(device), label.to(device)
    data_u, _ = iter_u.__next__()
    data_u = data_u.to(device)

    logit_l = netC(data)
    logit_u = netC(data_u)

    loss_u = vat_loss(
        netC, data_u, logit_u, FLAGS.vat_xi, FLAGS.vat_eps, FLAGS.vat_iters
    )
    loss_u = loss_u + torch.mean(entropy(logit_u), dim=0)

    loss_l = loss_cross_entropy(logit_l, label)

    return loss_l + loss_u, loss_l.detach(), loss_u.detach()


def _l2_normalize(d):
    d = d.numpy()
    d /= np.sqrt(np.sum(d ** 2, axis=(1, 2, 3))).reshape((-1, 1, 1, 1)) + 1e-16
    return torch.from_numpy(d)


def vat_loss(model, ul_x, ul_y, xi=1e-6, eps=2.5, num_iters=1):
    # find r_adv
    d = torch.Tensor(ul_x.size()).normal_()
    for i in range(num_iters):
        d = xi * _l2_normalize(d)
        d = Variable(d.cuda(), requires_grad=True)
        y_hat = model(ul_x + d)
        delta_kl = kl_div_with_logit(ul_y.detach(), y_hat)
        delta_kl.backward()

        d = d.grad.data.clone().cpu()
        model.zero_grad()

    d = _l2_normalize(d)
    d = Variable(d.cuda())
    r_adv = eps * d
    # compute lds
    y_hat = model(ul_x + r_adv.detach())
    delta_kl = kl_div_with_logit(ul_y.detach(), y_hat)
    return delta_kl


c_loss_dict = {
    "crossentropy": loss_supervised,
    "entropyssl": loss_entropy_ssl,
    "mtssl": loss_MT_ssl,
    "mtdoublessl": loss_MT_double_ssl,
    "vatssl": vat_loss_ssl,
    "loss_elr_wrap": loss_elr_wrap,
}

c_step_func = {
    "ramp": step_ramp,
    "ramp_linear": step_ramp_linear,
    "regular": step_regular,
    "step_vat": step_vat,
    "ramp_swa": step_ramp_swa,
}

