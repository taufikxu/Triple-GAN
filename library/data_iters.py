import torch
import numpy as np

# import torch.nn.functional as F
from torchvision import datasets, transforms

# import PIL.Image

from Utils.flags import FLAGS


class TransformTwice(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        output = torch.cat([out1, out2], 0)
        return output


def get_augmentation(train):
    totensor_normalized = [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    if train is False:
        return transforms.Compose(totensor_normalized)
    trans = []
    if FLAGS.data_augment.data_flip is True:
        trans.append(transforms.RandomHorizontalFlip())
    trans += totensor_normalized
    return transforms.Compose(trans)


def get_augmentation_twice(train):
    assert train is True
    totensor_normalized = [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    trans = []
    if FLAGS.data_augment.twice_data_flip is True:
        trans.append(transforms.RandomHorizontalFlip())
    trans += totensor_normalized
    return TransformTwice(transforms.Compose(trans))


def get_dataset(train, subset, transf):
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
    transf = get_augmentation(train)
    loader = torch.utils.data.DataLoader(
        get_dataset(train, subset, transf),
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


def inf_train_gen_twice(batch_size, train=True, infinity=True, subset=0):
    transf = get_augmentation_twice(train)
    loader = torch.utils.data.DataLoader(
        get_dataset(train, subset, transf),
        batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=8,
    )
    if infinity is True:
        while True:
            for img, labels in loader:
                img1, img2 = torch.split(img, img.shape[1] // 2, 1)
                yield img1, img2, labels
    else:
        for img1, img2, labels in loader:
            img1, img2 = torch.split(img, img.shape[1] // 2, 1)
            yield img1, img2, labels
