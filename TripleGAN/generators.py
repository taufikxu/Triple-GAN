import torch
from torch import nn
import numpy as np
from TripleGAN.layers import ResnetBlock


class Generator(nn.Module):
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
        s0 = self.s0 = 4
        nf = self.nf = nfilter
        nf_max = self.nf_max = nfilter_max

        self.z_dim = z_dim
        self.nlabels = n_label
        self.im_size = im_size
        self.im_chan = im_chan

        # Submodules
        nlayers = int(np.log2(im_size / s0))
        self.nf0 = min(nf_max, nf * 2 ** nlayers)

        self.embedding = nn.Embedding(n_label, embed_size)
        self.fc = nn.Linear(z_dim + embed_size, self.nf0 * s0 * s0)

        blocks = []
        for i in range(nlayers):
            nf0 = min(nf * 2 ** (nlayers - i), nf_max)
            nf1 = min(nf * 2 ** (nlayers - i - 1), nf_max)
            blocks += [
                ResnetBlock(nf0, nf1, actvn=self.actvn),
                nn.Upsample(scale_factor=2),
            ]

        blocks += [ResnetBlock(nf, nf, actvn=self.actvn)]

        self.resnet = nn.Sequential(*blocks)
        self.conv_img = nn.Conv2d(nf, im_chan, 3, padding=1)

    def forward(self, z, y):
        assert z.size(0) == y.size(0)
        batch_size = z.size(0)

        yembed = self.embedding(y)
        yembed = yembed / torch.norm(yembed, p=2, dim=1, keepdim=True)
        # print(z.shape, yembed.shape)

        yz = torch.cat([z, yembed], dim=1)
        out = self.fc(yz)
        out = out.view(batch_size, self.nf0, self.s0, self.s0)

        out = self.resnet(out)

        out = self.conv_img(self.actvn(out))
        out = torch.tanh(out)

        return out


generator_dict = {"resnet_reggan": Generator}
