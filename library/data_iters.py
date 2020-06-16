import torch
import numpy as np

import torch.nn.functional as F
from torchvision import datasets, transforms

from PIL import Image
from scipy import linalg
from Utils.flags import FLAGS


class ZCA(object):
    def __init__(self, regularization=1e-5, x=None):
        self.regularization = regularization
        if x is not None:
            self.fit(x)

    def fit(self, x):
        s = x.shape
        x = x.copy().reshape((s[0], np.prod(s[1:])))
        m = np.mean(x, axis=0)
        x -= m
        sigma = np.dot(x.T, x) / x.shape[0]
        U, S, V = linalg.svd(sigma)
        tmp = np.dot(U, np.diag(1.0 / np.sqrt(S + self.regularization)))
        tmp2 = np.dot(U, np.diag(np.sqrt(S + self.regularization)))
        self.ZCA_mat = np.dot(tmp, U.T)
        self.inv_ZCA_mat = np.dot(tmp2, U.T)
        self.mean = m

    def apply(self, x):
        s = x.shape
        return np.dot(
            x.reshape((s[0], np.prod(s[1:]))) - self.mean, self.ZCA_mat,
        ).reshape(s)

    def invert(self, x):
        s = x.shape
        return (
            np.dot(x.reshape((s[0], np.prod(s[1:]))), self.inv_ZCA_mat) + self.mean
        ).reshape(s)


class AugmentWrapper(object):
    def __init__(self):
        zca = FLAGS.zca

        if zca is True:
            self.zca = ZCA()
            dataset = get_dataset(train=True, subset=0)
            tloader = torch.utils.data.DataLoader(
                dataset, 100, drop_last=False, shuffle=True, num_workers=8,
            )
            tensor_list = []
            for img, _ in tloader:
                tensor_list.append(img)
            tensors = torch.cat(tensor_list, dim=0)
            tensors = tensors.data.cpu().numpy()
            self.zca.fit(tensors)
        else:
            self.zca = None

    def __call__(self, tensor):
        assert isinstance(tensor, torch.Tensor)
        assert len(tensor.shape) == 4

        if self.zca is not None:
            tensor = tensor.data.cpu().numpy()
            tensor = self.zca.apply(tensor)
        else:
            tensor = tensor.data.cpu().numpy()

        if FLAGS.translate > 0:
            bs, lenx, leny = tensor.shape[0], tensor.shape[2], tensor.shape[3]
            pad = FLAGS.translate
            tensor = np.pad(
                tensor,
                ([0, 0], [0, 0], [pad, pad], [pad, pad]),
                "constant",
                constant_values=0,
            )
            index = np.random.randint(0, pad * 2, size=[2, bs])
            indexx, indexy = index[0], index[1]

            new_tensor_list = []
            for i in range(bs):
                ten = tensor[
                    i : i + 1,
                    :,
                    indexx[i] : indexx[i] + lenx,
                    indexy[i] : indexy[i] + leny,
                ]
                if FLAGS.flip_horizontal is True and np.random.randint(0, 2) == 1:
                    ten = ten[:, :, ::-1, :]
                new_tensor_list.append(ten)
            tensor = np.concatenate(new_tensor_list, 0)

        return torch.from_numpy(tensor.astype(np.float32))


def get_dataset(train, subset):
    transf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    if FLAGS.dataset.lower() == "svhn":
        if train is True:
            split = "train"
        else:
            split = "test"

        sets = datasets.SVHN(
            "/home/LargeData/svhn", split=split, download=True, transform=transf
        )
    elif FLAGS.dataset.lower() == "cifar10":
        sets = datasets.CIFAR10(
            "/home/LargeData/cifar", train=train, download=True, transform=transf
        )

    if subset > 0:
        generator = np.random.default_rng(FLAGS.ssl_seed)
        indices = list(range(len(sets)))
        generator.shuffle(indices)
        sets = torch.utils.data.Subset(sets, indices[:subset])
        assert len(sets) == subset
    return sets


def inf_train_gen(batch_size, train=True, infinity=True, subset=0):

    loader = torch.utils.data.DataLoader(
        get_dataset(train, subset),
        batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=8,
    )
    if infinity is True:
        while True:
            for img, labels in loader:
                yield img, labels
    else:
        for img, labels in loader:
            yield img, labels
