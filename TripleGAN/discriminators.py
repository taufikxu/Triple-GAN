import torch
from torch import nn
import numpy as np
from TripleGAN.layers import ResnetBlock


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

    def forward(self, x, y):
        batch_size = x.size(0)

        out = self.conv_img(x)
        out = self.resnet(out)
        out = out.view(batch_size, self.nf0 * self.s0 * self.s0)
        out = self.fc(self.actvn(out))

        index = torch.LongTensor(range(out.size(0)))
        if x.is_cuda:
            index = index.cuda()
            y = y.cuda()
        out = out[index, y]

        return out


discriminator_dict = {"resnet_reggan": Discriminator}
