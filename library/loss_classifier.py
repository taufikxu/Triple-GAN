import torch
from torch import nn

from torch.nn import functional as F
from Utils.flags import FLAGS

crossentropy = nn.CrossEntropyLoss()
softmax = nn.Softmax(1)
logsoftmax = nn.LogSoftmax(1)


# some utils TODO
def update_average(model_tgt, model_src, beta=0.999):
    param_dict_src = dict(model_src.named_parameters())
    for p_name, p_tgt in model_tgt.named_parameters():
        p_src = param_dict_src[p_name]
        assert p_src is not p_tgt
        p_tgt.copy_(beta * p_tgt + (1.0 - beta) * p_src)

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

# losses
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


def loss_MT(netC, necC_T, i, data, label, data_u):
    sigmoid_rampup_value = sigmoid_rampup(i, FLAGS.rampup_length)
    cons_coefficient = sigmoid_rampup_value * FLAGS.max_consistency_cost
    loss_l = loss_cross_entropy(netC, data, label)
    logits = netC(data_u)
    prob = softmax(logits)
    logits_t = netC_T(data_u)
    prob_t = softmax(logits_t)
    loss_u = cons_coefficient * torch.sum((prob-prob_t)**2, dim=1).mean(dim=0)
    return loss_l + loss_u, loss_l.detach(), loss_u.detach()
        
def step_MT(optim_c, netC, netC_T, i, t_loss):
    sigmoid_rampup_value = sigmoid_rampup(i, FLAGS.rampup_length)
    sigmoid_rampdown_value = sigmoid_rampdown(i, FLAGS.rampup_length, max_iter)

    lr = FLAGS.lr * sigmoid_rampup_value * sigmoid_rampdown_value
    adam_beta_1 = sigmoid_rampdown_value * FLAGS.adam_beta_1_before_rampdown + (1 - sigmoid_rampdown_value) * FLAGS.adam_beta_1_after_rampdown
    if i < FLAGS.rampup_length:
        adam_beta_2 = FLAGS.adam_beta_2_during_rampup
        ema_decay = FLAGS.ema_decay_during_rampup
    else:
        adam_beta_2 = FLAGS.adam_beta_2_after_rampup
        ema_decay = FLAGS.ema_decay_after_rampup

    # update adam
    optim_c.param_groups[0]['betas'] = (adam_beta_1, adam_beta_2)
    optim_c.param_groups[0]['lr'] = lr
    
    # update student
    optim_c.zero_grad()
    tloss.backward()
    if FLAGS.clip_value > 0:
        torch.nn.utils.clip_grad_norm_(netC.parameters(), FLAGS.clip_value)
    optim_c.step()

    # update teacher
    update_average(netC_T, netC, ema_decay)



c_loss_dict = {
    "crossentropy": loss_cross_entropy,
    "entropyreg": loss_entropy_reg,
    "entropyssl": loss_entropy_ssl,
    "mtssl": loss_MT
}
