import os
import sys
import math

import numpy as np
from PIL import Image
import scipy.linalg

import chainer
import chainer.cuda
from chainer import Variable
from chainer import serializers
from chainer import cuda
import chainer.functions as F

# sys.path.append(os.path.dirname(__file__))
# sys.path.append("../")
from library.chainer_evaluation.inception_score import inception_score, Inception


def load_inception_model(path=None):
    path = "./allresults/chainer_inception"
    model = Inception()
    serializers.load_hdf5(path, model)
    model.to_gpu()
    return model


def calc_inception(ims, batchsize=100, n_ims=50000, splits=10):
    model = load_inception_model()
    mean, std = inception_score(model, ims, splits=splits)
    return mean, std


def get_mean_cov(model, ims, batch_size=100):
    n, c, w, h = ims.shape
    n_batches = int(math.ceil(float(n) / float(batch_size)))
    xp = model.xp
    print("Batch size:", batch_size)
    print("Total number of images:", n)
    print("Total number of batches:", n_batches)
    ys = xp.empty((n, 2048), dtype=xp.float32)
    for i in range(n_batches):
        # print("Running batch", i + 1, "/", n_batches, "...")
        batch_start = i * batch_size
        batch_end = min((i + 1) * batch_size, n)

        ims_batch = ims[batch_start:batch_end]
        ims_batch = xp.asarray(ims_batch)  # To GPU if using CuPy
        ims_batch = Variable(ims_batch)

        # Resize image to the shape expected by the inception module
        if (w, h) != (299, 299):
            ims_batch = F.resize_images(ims_batch, (299, 299))  # bilinear

        # Feed images to the inception module to get the features
        with chainer.using_config("train", False), chainer.using_config(
            "enable_backprop", False
        ):
            y = model(ims_batch, get_feature=True)
        ys[batch_start:batch_end] = y.data

    mean = xp.mean(ys, axis=0).get()
    # cov = F.cross_covariance(ys, ys, reduce="no").datasets.get()
    cov = np.cov(ys.get().T)
    return mean, cov


def FID(m0, c0, m1, c1):
    ret = 0
    ret += np.sum((m0 - m1) ** 2)
    ret += np.trace(c0 + c1 - 2.0 * scipy.linalg.sqrtm(np.dot(c0, c1)))
    return np.real(ret)


def calc_FID(ims, batchsize=100, stat_file=None):
    """Frechet Inception Distance proposed by https://arxiv.org/abs/1706.08500"""
    model = load_inception_model()
    with chainer.using_config("train", False), chainer.using_config(
        "enable_backprop", False
    ):
        mean, cov = get_mean_cov(model, ims)
    if stat_file is None or os.path.exists(stat_file) is False:
        if stat_file is None:
            stat_file = "./cifar10_total.npz"
        np.savez(stat_file, mean=mean, cov=cov)
        return None
    stat = np.load(stat_file)
    fid = FID(stat["mean"], stat["cov"], mean, cov)
    return fid
