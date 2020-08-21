import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import linalg
import torchvision
from torchvision import datasets, transforms

import Utils
from Utils.flags import FLAGS
from library.randaugment import RandomAugment


class NumpyDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, transf):
        super().__init__()
        assert X.shape[0] == Y.shape[0]
        self.X = X
        self.Y = Y
        self.transf = transf

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        image = Image.fromarray(self.X[i])
        image = self.transf(image)
        return image, self.Y[i]


class ZCA(object):
    def __init__(self, regularization=1e-3, x=None):
        self.regularization = regularization
        if x is not None:
            self.fit(x)

    def fit(self, x):
        s = x.shape
        x = x.copy().reshape((s[0], np.prod(s[1:])))
        m = np.mean(x, axis=0)
        x -= m
        sigma = np.dot(x.T, x) / x.shape[0]
        sigma += np.eye(sigma.shape[0], sigma.shape[1]) * 0.1
        U, S, V = linalg.svd(sigma)
        tmp = np.dot(U, np.diag(1.0 / np.sqrt(S + self.regularization)))
        tmp2 = np.dot(U, np.diag(np.sqrt(S + self.regularization)))
        self.ZCA_mat = np.dot(tmp, U.T)
        self.inv_ZCA_mat = np.dot(tmp2, U.T)
        self.mean = m

        self.ZCA_mat_t = torch.from_numpy(self.ZCA_mat).cuda()
        self.inv_ZCA_mat_t = torch.from_numpy(self.inv_ZCA_mat).cuda()
        self.mean_t = torch.from_numpy(self.mean).cuda()

    def apply(self, x):
        s = x.shape
        # return np.dot(
        #     x.reshape((s[0], np.prod(s[1:]))) - self.mean, self.ZCA_mat,
        # ).reshape(s)
        result = torch.mm(x.view(s[0], -1) - self.mean_t, self.ZCA_mat_t)
        return result.view(*s)

    def invert(self, x):
        s = x.shape
        return (
            np.dot(x.reshape((s[0], np.prod(s[1:]))), self.inv_ZCA_mat) + self.mean
        ).reshape(s)


class AugmentWrapper_DIS(object):
    def __init__(self):
        self.zca = None

    def __call__(self, tensor, training):
        assert isinstance(tensor, torch.Tensor)
        assert len(tensor.shape) == 4
        
        if training is False:
            exit('warning: bcr loss is true but evaluate dis. see AugmentWrapper_DIS in data_iters.py')
            return tensor

        if FLAGS.dataset == 'cifar10':
            translate = 4
            flip_horizontal = True
        if FLAGS.dataset == 'tinyimagenet32':
            translate = 4
            flip_horizontal = True
        if FLAGS.dataset == 'svhn' or FLAGS.dataset == 'SVHN':
            translate = 2
            flip_horizontal = False

        if translate > 0:
            bs, lenx, leny = tensor.shape[0], tensor.shape[2], tensor.shape[3]
            pad = translate
            tensor = F.pad(tensor, [pad, pad, pad, pad])
            index = np.random.randint(0, pad * 2, size=[2, bs])
            indexx, indexy = index[0], index[1]
            inv_idx = torch.arange(leny - 1, -1, -1).long().cuda()

            new_tensor_list = []
            for i in range(bs):
                ten = tensor[
                    i : i + 1,
                    :,
                    indexx[i] : indexx[i] + lenx,
                    indexy[i] : indexy[i] + leny,
                ]
                if flip_horizontal is True and np.random.randint(0, 2) == 1:
                    ten = ten.index_select(3, inv_idx)
                new_tensor_list.append(ten)
            tensor = torch.cat(new_tensor_list, 0)
        return tensor


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

    def __call__(self, tensor, training):
        assert isinstance(tensor, torch.Tensor)
        assert len(tensor.shape) == 4

        if FLAGS.norm is True:
            assert FLAGS.zca is False
            assert FLAGS.dataset == "cifar10"
            tensor = (tensor + 1) / 2
            channel_mean = (
                torch.from_numpy(np.array([0.4914, 0.4822, 0.4465]).astype(np.float32))
                .to(tensor.device)
                .view(1, 3, 1, 1)
            )
            channel_std = (
                torch.from_numpy(np.array([0.2470, 0.2435, 0.2616]).astype(np.float32))
                .to(tensor.device)
                .view(1, 3, 1, 1)
            )
            tensor = (tensor - channel_mean) / channel_std

        if self.zca is not None:
            tensor = self.zca.apply(tensor)
        if training is False:
            return tensor

        if FLAGS.translate > 0:
            bs, lenx, leny = tensor.shape[0], tensor.shape[2], tensor.shape[3]
            pad = FLAGS.translate
            tensor = F.pad(tensor, [pad, pad, pad, pad])
            index = np.random.randint(0, pad * 2, size=[2, bs])
            indexx, indexy = index[0], index[1]
            inv_idx = torch.arange(leny - 1, -1, -1).long().cuda()

            new_tensor_list = []
            for i in range(bs):
                ten = tensor[
                    i : i + 1,
                    :,
                    indexx[i] : indexx[i] + lenx,
                    indexy[i] : indexy[i] + leny,
                ]
                if FLAGS.flip_horizontal is True and np.random.randint(0, 2) == 1:
                    ten = ten.index_select(3, inv_idx)
                new_tensor_list.append(ten)
            tensor = torch.cat(new_tensor_list, 0)
        return tensor


def get_dataset(train, subset):

    transf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]
    )
    if FLAGS.dataset.lower() == "svhn":
        if train is True:
            split = "train"
        else:
            split = "test"

        sets = datasets.SVHN(
            "/home/LargeData/Regular/svhn", split=split, download=True, transform=transf
        )
    elif FLAGS.dataset.lower() == "cifar10":
        sets = datasets.CIFAR10(
            "/home/LargeData/Regular/cifar",
            train=train,
            download=True,
            transform=transf,
        )
    elif FLAGS.dataset.lower() == "cifar100":
        sets = datasets.CIFAR100(
            "/home/LargeData/Regular/cifar",
            train=train,
            download=True,
            transform=transf,
        )
    elif FLAGS.dataset.lower() == "stl10":
        if train is True:
            split = "train+unlabeled"
        else:
            split = "test"
        sets = datasets.STL10(
            "/home/LargeData/Regular/", split=split, download=True, transform=transf,
        )
    elif FLAGS.dataset.lower() in ["tinyimagenet", "tiny-imagenet"]:
        if train is True:
            train_data = np.load("/home/LargeData/Regular/tinyimagenet/train.npz")
            X = train_data["X"]
            Y = train_data["Y"].flatten()
        else:
            test_data = np.load("/home/LargeData/Regular/tinyimagenet/test.npz")
            X = test_data["X"]
            Y = test_data["Y"].flatten()
        select = [
            0,
            9,
            10,
            20,
            29,
            35,
            40,
            72,
            76,
            81,
            94,
            104,
            115,
            120,
            124,
            154,
            160,
            179,
            187,
            193,
            197,
        ]
        img_list, label_list = [], []
        for j in range(10):
            tempx = X[Y == select[j]]
            tempy = Y[Y == select[j]] * 0 + j
            img_list.append(tempx)
            label_list.append(tempy)
        img_list = np.concatenate(img_list, axis=0)
        label_list = np.concatenate(label_list, axis=0)
        sets = NumpyDataset(
            img_list.astype(np.uint8), label_list.astype(np.int), transf
        )
    elif FLAGS.dataset.lower() in ["tinyimagenet32", "tiny-imagenet32"]:
        if train is True:
            train_data = np.load("/home/LargeData/Regular/tinyimagenet/train.npz")
            X = train_data["X"]
            Y = train_data["Y"].flatten()
        else:
            test_data = np.load("/home/LargeData/Regular/tinyimagenet/test.npz")
            X = test_data["X"]
            Y = test_data["Y"].flatten()
        select = [
            0,
            9,
            10,
            20,
            29,
            35,
            40,
            72,
            76,
            81,
            94,
            104,
            115,
            120,
            124,
            154,
            160,
            179,
            187,
            193,
            197,
        ]
        img_list, label_list = [], []
        for j in range(10):
            tempx = X[Y == select[j]]
            tempy = Y[Y == select[j]] * 0 + j
            img_list.append(tempx)
            label_list.append(tempy)
        transf = transforms.Compose([transforms.Resize(32), transf])
        img_list = np.concatenate(img_list, axis=0)
        label_list = np.concatenate(label_list, axis=0)
        sets = NumpyDataset(
            img_list.astype(np.uint8), label_list.astype(np.int), transf
        )

    if subset > 0:
        generator = np.random.default_rng(FLAGS.ssl_seed)
        labels, indexs = [], []
        for i in range(len(sets)):
            _, lab = sets.__getitem__(i)
            labels.append(lab)
            indexs.append(i)
        labels = np.array(labels)
        indexs = np.array(indexs)
        num_labels = np.max(labels) + 1

        assert subset % num_labels == 0

        final_indices = []
        for i in range(num_labels):
            tind = list(indexs[labels == i])
            generator.shuffle(tind)
            final_indices.extend(tind[: (subset // num_labels)])

        sets = torch.utils.data.Subset(sets, final_indices)
        assert len(sets) == subset
    return sets


def get_dataset_randaug(train, subset):

    transf = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]
        )
    transf_strong = transforms.Compose([
            transforms.PadandRandomCrop(border=4, cropsize=(32, 32)),
            transforms.RandomHorizontalFlip(p=0.5),
            RandomAugment(2, 10),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.ToTensor(),
        ])

    if FLAGS.dataset.lower() == "cifar10":
        sets = datasets.CIFAR10(
            "/home/LargeData/Regular/cifar",
            train=train,
            download=True,
            transform=transf,
        )
    if train:
        sets_strong = datasets.CIFAR10(
            "/home/LargeData/Regular/cifar",
            train=train,
            download=True,
            transform=transf_strong,
        )

    if subset > 0:
        generator = np.random.default_rng(FLAGS.ssl_seed)
        labels, indexs = [], []
        for i in range(len(sets)):
            _, lab = sets.__getitem__(i)
            labels.append(lab)
            indexs.append(i)
        labels = np.array(labels)
        indexs = np.array(indexs)
        num_labels = np.max(labels) + 1

        assert subset % num_labels == 0

        final_indices = []
        for i in range(num_labels):
            tind = list(indexs[labels == i])
            generator.shuffle(tind)
            final_indices.extend(tind[: (subset // num_labels)])

        sets = torch.utils.data.Subset(sets, final_indices)
        if train:
            sets_strong = torch.utils.data.Subset(sets_strong, final_indices)
        assert len(sets) == subset
    if train:
        return sets, sets_strong
    else:
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


if __name__ == "__main__":
    Utils.config.load_config("./configs/classifier_cifar10_mt_aug.yaml")
    FLAGS.zca = True
    FLAGS.translate = 2

    wrapper = AugmentWrapper()
    dataset = get_dataset(True, 0)
    img_list = []
    for i in range(100):
        img, _ = dataset.__getitem__(i)
        img_list.append(img)
    img_list = torch.stack(img_list, 0).cuda()
    torchvision.utils.save_image((img_list + 1) / 2, "./tmp.png", nrow=10)

    img_list = wrapper(img_list)
    print(torch.max(img_list), torch.min(img_list))
    torchvision.utils.save_image((img_list + 1) / 2, "./tmp1.png", nrow=10)
