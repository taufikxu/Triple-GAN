import torch
from torch import nn
import numpy as np
from library.model_layers import ResnetBlock

import math
import torch.nn.functional as F
from torch.nn import init
from torch.nn import utils


class Discriminator(nn.Module):
    def __init__(
        self,
        z_dim=256,
        n_label=10,
        im_size=32,
        im_chan=3,
        embed_size=256,
        nfilter=64,
        nfilter_max=512,
        actvn=nn.ReLU(),
    ):
        super().__init__()
        self.actvn = actvn
        self.embed_size = embed_size
        self.nlabels = n_label
        s0 = self.s0 = 4
        nf = self.nf = nfilter
        nf_max = self.nf_max = nfilter_max

        self.im_size = im_size
        self.im_chan = im_chan

        # Submodules
        nlayers = int(np.log2(im_size / s0))
        self.nf0 = min(nf_max, nf * 2 ** nlayers)

        blocks = [ResnetBlock(nf, nf, actvn=self.actvn)]

        for i in range(nlayers):
            nf0 = min(nf * 2 ** i, nf_max)
            nf1 = min(nf * 2 ** (i + 1), nf_max)
            blocks += [
                nn.AvgPool2d(3, stride=2, padding=1),
                ResnetBlock(nf0, nf1, actvn=self.actvn),
            ]

        self.conv_img = nn.Conv2d(im_chan, 1 * nf, 3, padding=1)
        self.resnet = nn.Sequential(*blocks)
        self.fc = nn.Linear(self.nf0 * s0 * s0, n_label)

    def forward(self, x, y=None):
        batch_size = x.size(0)

        out = self.conv_img(x)
        out = self.resnet(out)
        out = out.view(batch_size, self.nf0 * self.s0 * self.s0)
        out = self.fc(self.actvn(out))

        if y is None:
            return out

        index = torch.LongTensor(range(out.size(0)))
        if x.is_cuda:
            index = index.cuda()
            y = y.cuda()
        out = out[index, y]
        return out


class Block(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        h_ch=None,
        ksize=3,
        pad=1,
        activation=F.relu,
        downsample=False,
    ):
        super(Block, self).__init__()

        self.activation = activation
        self.downsample = downsample

        self.learnable_sc = (in_ch != out_ch) or downsample
        if h_ch is None:
            h_ch = in_ch
        else:
            h_ch = out_ch

        self.c1 = utils.spectral_norm(nn.Conv2d(in_ch, h_ch, ksize, 1, pad))
        self.c2 = utils.spectral_norm(nn.Conv2d(h_ch, out_ch, ksize, 1, pad))
        if self.learnable_sc:
            self.c_sc = utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 1, 1, 0))

        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.c1.weight.data, math.sqrt(2))
        init.xavier_uniform_(self.c2.weight.data, math.sqrt(2))
        if self.learnable_sc:
            init.xavier_uniform_(self.c_sc.weight.data)

    def forward(self, x):
        return self.shortcut(x) + self.residual(x)

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
        if self.downsample:
            return F.avg_pool2d(x, 2)
        return x

    def residual(self, x):
        h = self.c1(self.activation(x))
        h = self.c2(self.activation(h))
        if self.downsample:
            h = F.avg_pool2d(h, 2)
        return h


class OptimizedBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ksize=3, pad=1, activation=F.relu):
        super(OptimizedBlock, self).__init__()
        self.activation = activation

        self.c1 = utils.spectral_norm(nn.Conv2d(in_ch, out_ch, ksize, 1, pad))
        self.c2 = utils.spectral_norm(nn.Conv2d(out_ch, out_ch, ksize, 1, pad))
        self.c_sc = utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 1, 1, 0))

        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.c1.weight.data, math.sqrt(2))
        init.xavier_uniform_(self.c2.weight.data, math.sqrt(2))
        init.xavier_uniform_(self.c_sc.weight.data)

    def forward(self, x):
        return self.shortcut(x) + self.residual(x)

    def shortcut(self, x):
        return self.c_sc(F.avg_pool2d(x, 2))

    def residual(self, x):
        h = self.activation(self.c1(x))
        return F.avg_pool2d(self.c2(h), 2)


class SNResNetProjectionDiscriminator(nn.Module):
    def __init__(
        self,
        z_dim=256,
        n_label=10,
        im_size=32,
        im_chan=3,
        embed_size=256,
        nfilter=64,
        nfilter_max=512,
        actvn=nn.ReLU(),
    ):
        super(SNResNetProjectionDiscriminator, self).__init__()
        self.num_features = num_features = nfilter
        self.num_classes = num_classes = n_label
        self.activation = activation = actvn

        width_coe = 8
        self.block1 = OptimizedBlock(3, num_features * width_coe)
        self.block2 = Block(
            num_features * width_coe,
            num_features * width_coe,
            activation=activation,
            downsample=True,
        )
        self.block3 = Block(
            num_features * width_coe,
            num_features * width_coe,
            activation=activation,
            downsample=True,
        )
        self.block4 = Block(
            num_features * width_coe,
            num_features * width_coe,
            activation=activation,
            downsample=True,
        )
        self.l7 = utils.spectral_norm(nn.Linear(num_features * width_coe, 1))
        if num_classes > 0:
            self.l_y = utils.spectral_norm(
                nn.Embedding(num_classes, num_features * width_coe)
            )

        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.l7.weight.data)
        optional_l_y = getattr(self, "l_y", None)
        if optional_l_y is not None:
            init.xavier_uniform_(optional_l_y.weight.data)

    def forward(self, x, y=None):
        bs = x.shape[0]
        h = x
        for i in range(1, 5):
            h = getattr(self, "block{}".format(i))(h)
        h = self.activation(h)
        # Global pooling
        h = torch.sum(h, dim=(2, 3))
        output = self.l7(h)
        if y is not None:
            output += torch.sum(self.l_y(y) * h, dim=1, keepdim=True)
        else:
            output_list = []
            for i in range(self.num_classes):
                ty = torch.ones([bs,], dtype=torch.long) * i
                toutput = output + torch.sum(
                    self.l_y(ty.to(x.device)) * h, dim=1, keepdim=True
                )
                output_list.append(toutput)
            output = torch.cat(output_list, dim=1)
        return output


class SNResNetUnconditionalDiscriminator(nn.Module):
    def __init__(
        self,
        z_dim=256,
        n_label=10,
        im_size=32,
        im_chan=3,
        embed_size=256,
        nfilter=64,
        nfilter_max=512,
        actvn=nn.ReLU(),
    ):
        super(SNResNetUnconditionalDiscriminator, self).__init__()
        self.num_features = num_features = nfilter
        self.num_classes = num_classes = 0
        self.activation = activation = actvn

        width_coe = 8
        self.block1 = OptimizedBlock(3, num_features * width_coe)
        self.block2 = Block(
            num_features * width_coe,
            num_features * width_coe,
            activation=activation,
            downsample=True,
        )
        self.block3 = Block(
            num_features * width_coe,
            num_features * width_coe,
            activation=activation,
            downsample=True,
        )
        self.block4 = Block(
            num_features * width_coe,
            num_features * width_coe,
            activation=activation,
            downsample=True,
        )
        self.l7 = utils.spectral_norm(nn.Linear(num_features * width_coe, 1))
        if num_classes > 0:
            self.l_y = utils.spectral_norm(
                nn.Embedding(num_classes, num_features * width_coe)
            )

        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.l7.weight.data)
        optional_l_y = getattr(self, "l_y", None)
        if optional_l_y is not None:
            init.xavier_uniform_(optional_l_y.weight.data)

    def forward(self, x, y=None):
        bs = x.shape[0]
        h = x
        for i in range(1, 5):
            h = getattr(self, "block{}".format(i))(h)
        h = self.activation(h)
        # Global pooling
        h = torch.sum(h, dim=(2, 3))
        output = self.l7(h)
        return output


class SNResNetProjectionDiscriminator96(nn.Module):
    def __init__(
        self,
        z_dim=256,
        n_label=10,
        im_size=32,
        im_chan=3,
        embed_size=256,
        nfilter=64,
        nfilter_max=512,
        actvn=nn.ReLU(),
    ):
        super(SNResNetProjectionDiscriminator96, self).__init__()
        self.num_features = num_features = nfilter
        self.num_classes = num_classes = n_label
        self.activation = activation = actvn

        width_coe = 1
        self.block1 = OptimizedBlock(3, num_features * width_coe)
        self.block2 = Block(
            num_features * width_coe * 1,
            num_features * width_coe * 2,
            activation=activation,
            downsample=True,
        )
        self.block3 = Block(
            num_features * width_coe * 2,
            num_features * width_coe * 4,
            activation=activation,
            downsample=True,
        )
        self.block4 = Block(
            num_features * width_coe * 4,
            num_features * width_coe * 8,
            activation=activation,
            downsample=True,
        )
        self.block5 = Block(
            num_features * width_coe * 8,
            num_features * width_coe * 16,
            activation=activation,
            downsample=True,
        )
        self.l7 = utils.spectral_norm(nn.Linear(num_features * width_coe * 16, 1))
        if num_classes > 0:
            self.l_y = utils.spectral_norm(
                nn.Embedding(num_classes, num_features * width_coe * 16)
            )

        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.l7.weight.data)
        optional_l_y = getattr(self, "l_y", None)
        if optional_l_y is not None:
            init.xavier_uniform_(optional_l_y.weight.data)

    def forward(self, x, y=None):
        bs = x.shape[0]
        h = x
        for i in range(1, 6):
            h = getattr(self, "block{}".format(i))(h)
        h = self.activation(h)
        # Global pooling
        h = torch.sum(h, dim=(2, 3))
        output = self.l7(h)
        if y is not None:
            output += torch.sum(self.l_y(y) * h, dim=1, keepdim=True)
        else:
            output_list = []
            for i in range(self.num_classes):
                ty = torch.ones([bs,], dtype=torch.long) * i
                toutput = output + torch.sum(
                    self.l_y(ty.to(x.device)) * h, dim=1, keepdim=True
                )
                output_list.append(toutput)
            output = torch.cat(output_list, dim=1)
        return output


class SNResNetProjectionDiscriminator64(nn.Module):
    def __init__(
        self,
        z_dim=256,
        n_label=10,
        im_size=32,
        im_chan=3,
        embed_size=256,
        nfilter=64,
        nfilter_max=512,
        actvn=nn.ReLU(),
    ):
        super(SNResNetProjectionDiscriminator64, self).__init__()
        self.num_features = num_features = nfilter
        self.num_classes = num_classes = n_label
        self.activation = activation = actvn

        width_coe = 8
        self.block1 = OptimizedBlock(3, num_features * width_coe)
        self.block2 = Block(
            num_features * width_coe,
            num_features * width_coe,
            activation=activation,
            downsample=True,
        )
        self.block3 = Block(
            num_features * width_coe,
            num_features * width_coe,
            activation=activation,
            downsample=True,
        )
        self.block4 = Block(
            num_features * width_coe,
            num_features * width_coe,
            activation=activation,
            downsample=True,
        )
        self.block5 = Block(
            num_features * width_coe,
            num_features * width_coe,
            activation=activation,
            downsample=True,
        )
        self.l7 = utils.spectral_norm(nn.Linear(num_features * width_coe, 1))
        if num_classes > 0:
            self.l_y = utils.spectral_norm(
                nn.Embedding(num_classes, num_features * width_coe)
            )

        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.l7.weight.data)
        optional_l_y = getattr(self, "l_y", None)
        if optional_l_y is not None:
            init.xavier_uniform_(optional_l_y.weight.data)

    def forward(self, x, y=None):
        bs = x.shape[0]
        h = x
        for i in range(1, 6):
            h = getattr(self, "block{}".format(i))(h)
        h = self.activation(h)
        # Global pooling
        h = torch.sum(h, dim=(2, 3))
        output = self.l7(h)
        if y is not None:
            output += torch.sum(self.l_y(y) * h, dim=1, keepdim=True)
        else:
            output_list = []
            for i in range(self.num_classes):
                ty = torch.ones([bs,], dtype=torch.long) * i
                toutput = output + torch.sum(
                    self.l_y(ty.to(x.device)) * h, dim=1, keepdim=True
                )
                output_list.append(toutput)
            output = torch.cat(output_list, dim=1)
        return output


discriminator_dict = {
    "resnet_reggan": Discriminator,
    "resnet_sngan": SNResNetProjectionDiscriminator,
    "resnet_sngan_un": SNResNetUnconditionalDiscriminator,
    "resnet_sngan96": SNResNetProjectionDiscriminator96,
    "resnet_sngan64": SNResNetProjectionDiscriminator64,
}

