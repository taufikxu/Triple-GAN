import torch
import numpy as np
from torchvision import transforms, datasets
import PIL.Image
import torch.nn.functional as F

totensor_normalized = [
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]


def inf_train_gen_cifar(batch_size, flip=True, train=True, infinity=True):
    if flip:
        transf = transforms.Compose(
            [transforms.RandomHorizontalFlip(p=0.5)] + totensor_normalized
        )
    else:
        transf = transforms.ToTensor()

    loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            "/home/LargeData/cifar/", train=train, download=True, transform=transf
        ),
        batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=8,
    )
    if infinity is True:
        while True:
            for img, labels in loader:
                # print(img.shape)
                yield img, labels
    else:
        for img, labels in loader:
            yield img, labels


def inf_train_gen_mnist(batch_size, train=True, infinity=True):

    transf = totensor_normalized

    loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "/home/LargeData/", train=train, download=True, transform=transf
        ),
        batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=8,
    )

    if infinity is True:
        while True:
            for img, labels in loader:
                img = F.pad(img, [2, 2, 2, 2])
                yield img, labels
    else:
        for img, labels in loader:
            img = F.pad(img, [2, 2, 2, 2])
            yield img, labels


def inf_train_gen_fashionmnist(batch_size, train=True, infinity=True):

    transf = totensor_normalized

    loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(
            "/home/LargeData/", train=train, download=True, transform=transf
        ),
        batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=8,
    )

    if infinity is True:
        while True:
            for img, labels in loader:
                img = F.pad(img, [2, 2, 2, 2])
                yield img, labels
    else:
        for img, labels in loader:
            img = F.pad(img, [2, 2, 2, 2])
            yield img, labels


def inf_train_gen_svhn(batch_size, split="train", infinity=True):

    transf = totensor_normalized

    loader = torch.utils.data.DataLoader(
        datasets.SVHN(
            "/home/LargeData/svhn", split=split, download=True, transform=transf
        ),
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
