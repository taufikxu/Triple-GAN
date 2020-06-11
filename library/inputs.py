import torch
from torch import nn


import library.data_iters as dataset_iters
from library.generators import generator_dict
from library.discriminators import discriminator_dict
from library.classifiers import classifier_dict


from Utils import flags

FLAGS = flags.FLAGS

hw_dict = {
    "cifar10": (32, 3, 10),
    "svhn": (32, 3, 10),
    "mnist": (32, 1, 10),
    "fashionmnist": (32, 1, 10),
}
actvn_dict = {"relu": nn.ReLU, "softplus": nn.Softplus}


def get_optimizer(params, opt_name, lr, beta1, beta2):
    if opt_name.lower() == "adam":
        optim = torch.optim.Adam(params, lr, betas=(beta1, beta2))
    return optim


def get_data_iter(train=True, infinity=True, subset=0):
    return dataset_iters.inf_train_gen(FLAGS.batch_size, train, infinity, subset)


def get_data_iter_twice(train=True, infinity=True, subset=0):
    return dataset_iters.inf_train_gen_twice(FLAGS.batch_size, train, infinity, subset)


def get_data_iter_test(infinity=False):
    return dataset_iters.inf_train_gen(FLAGS.batch_size, train=False, infinity=infinity)


def get_generator_optimizer():
    module = generator_dict[FLAGS.g_model_name.lower()]
    hw, c, nlabel = hw_dict[FLAGS.dataset]
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


def get_discriminator_optimizer():
    module = discriminator_dict[FLAGS.g_model_name.lower()]
    hw, c, nlabel = hw_dict[FLAGS.dataset]
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

    optim = get_optimizer(
        D.parameters(), FLAGS.d_optim, FLAGS.d_lr, FLAGS.d_beta1, FLAGS.d_beta2
    )

    return D, optim


def get_classifier_optimizer():
    module = classifier_dict[FLAGS.c_model_name]
    _, _, nlabel = hw_dict[FLAGS.dataset]
    C = module(num_classes=nlabel)
    optim = get_optimizer(
        C.parameters(), FLAGS.c_optim, FLAGS.c_lr, FLAGS.c_beta1, FLAGS.c_beta2
    )
    return C, optim
