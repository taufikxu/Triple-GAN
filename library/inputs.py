import torch
from torch import nn


import library.data_iters as dataset_iters
from library.model_generators import generator_dict
from library.model_discriminators import discriminator_dict
from library.model_classifiers import classifier_dict


from Utils import flags

FLAGS = flags.FLAGS

hw_dict = {
    "cifar10": (32, 3, 10),
    "cifar100": (32, 3, 100),
    "stl10": (96, 3, 10),
    "svhn": (32, 3, 10),
    "mnist": (32, 1, 10),
    "fashionmnist": (32, 1, 10),
    "tinyimagenet": (64, 3, 10),
    "tinyimagenet32": (32, 3, 10),
}
actvn_dict = {
    "relu": nn.ReLU,
    "softplus": nn.Softplus,
    "lrelu": lambda: nn.LeakyReLU(0.2),
}


def get_optimizer(params, opt_name, lr, beta1, beta2):
    if opt_name.lower() == "adam":
        optim = torch.optim.Adam(params, lr, betas=(beta1, beta2))
    elif opt_name.lower() == "nesterov":
        optim = torch.optim.SGD(
            params, lr, momentum=beta1, weight_decay=FLAGS.c_weight_decay, nesterov=True
        )
    return optim


def get_data_iter(batch_size=None, train=True, infinity=True, subset=0):
    if batch_size is None:
        batch_size = FLAGS.batch_size
    return dataset_iters.inf_train_gen(batch_size, train, infinity, subset)


def get_data_iter_test(batch_size=None, infinity=False):
    if batch_size is None:
        batch_size = FLAGS.batch_size
    return dataset_iters.inf_train_gen(batch_size, train=False, infinity=infinity)


def get_generator_optimizer():
    module = generator_dict[FLAGS.g_model_name.lower()]
    hw, c, nlabel = hw_dict[FLAGS.dataset.lower()]
    actvn = actvn_dict[FLAGS.g_actvn]()
    G = module(
        z_dim=FLAGS.g_z_dim,
        n_label=nlabel,
        im_size=hw,
        im_chan=c,
        embed_size=FLAGS.g_embed_size,
        nfilter=FLAGS.g_nfilter,
        nfilter_max=FLAGS.g_nfilter_max,
        actvn=actvn,
    )
    optim = get_optimizer(
        G.parameters(), FLAGS.g_optim, FLAGS.g_lr, FLAGS.g_beta1, FLAGS.g_beta2
    )
    return G, optim


class discriminator_wrapper(nn.Module):
    def __init__(self, discriminator):
        super().__init__()
        self.dis = discriminator
        self.trans = dataset_iters.AugmentWrapper_DIS()

    def forward(self, x, y=None, aug=False):
        if aug:
            x = self.trans(x, self.training)
        logits = self.dis(x=x, y=y)
        return logits


def get_discriminator_optimizer():
    module = discriminator_dict[FLAGS.g_model_name.lower()]
    hw, c, nlabel = hw_dict[FLAGS.dataset.lower()]
    D = module(
        z_dim=FLAGS.d_z_dim,
        n_label=nlabel,
        im_size=hw,
        im_chan=c,
        embed_size=FLAGS.d_embed_size,
        nfilter=FLAGS.d_nfilter,
        nfilter_max=FLAGS.d_nfilter_max,
        actvn=actvn_dict[FLAGS.d_actvn](),
    )

    D = discriminator_wrapper(D)

    optim = get_optimizer(
        D.parameters(), FLAGS.d_optim, FLAGS.d_lr, FLAGS.d_beta1, FLAGS.d_beta2
    )

    return D, optim


class classifier_wrapper(nn.Module):
    def __init__(self, classifier):
        super().__init__()
        self.cla = classifier
        self.trans = dataset_iters.AugmentWrapper()

    def forward(self, dat, double=False, aug=True):
        if aug:
            dat = self.trans(dat, self.training)
        logits = self.cla(dat)
        if len(logits) == 2:
            logits1, logits2 = logits[0], logits[1]
        else:
            logits1, logits2 = logits
        if double is True:
            return logits1, logits2
        else:
            return logits1


def get_classifier_optimizer():
    module = classifier_dict[FLAGS.c_model_name]
    _, _, nlabel = hw_dict[FLAGS.dataset.lower()]
    C = module(num_classes=nlabel)
    C = classifier_wrapper(C)
    optim = get_optimizer(
        C.parameters(), FLAGS.c_optim, FLAGS.c_lr, FLAGS.c_beta1, FLAGS.c_beta2
    )
    return C, optim
