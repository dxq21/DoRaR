# School: ISU
# Name: Dong Qin
# Creat Time: 1/16/2022 4:16 PM
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import Flatten, idxtobool, cuda
from pathlib import Path


class generative_model1_cifar10(nn.Module):

    def __init__(self, **kwargs):

        super(generative_model1_cifar10, self).__init__()

        self.args = kwargs['args']
        self.mode = self.args.mode

        self.device = torch.device("cuda" if self.args.cuda else "cpu")
        # self.tau = 0.5  # float, parameter for concrete random variable distribution
        self.K = self.args.K  # number of variables selected
        self.model_type = self.args.explainer_type  # 'nn', 'cnn', 'nn4', 'cnn4'
        self.chunk_size = self.args.chunk_size

        self.decode_cnn = nn.Sequential(
            Flatten(),
            nn.Linear(32 * 32 * 3, 32 * 32 * 3),
            nn.ReLU(True),
            nn.Linear(32 * 32 * 3, 32 * 32 * 3),
            nn.Sigmoid()

        )
        self.decode = self.decode_cnn


    def decoder(self, x, Z_hat, mu, sigma, num_sample=1):

        assert num_sample > 0

        eps = (torch.randn_like(mu) * sigma + mu)
        eps = eps.swapaxes(1, 2)
        eps = eps.swapaxes(0, 1)

        Z_hat0 = Z_hat.view(Z_hat.size(0), Z_hat.size(1),
                            int(np.sqrt(Z_hat.size(-1))),
                            int(np.sqrt(Z_hat.size(-1))))

        ## Upsampling
        if self.chunk_size > 1 and Z_hat0.size(2) != 32:
            Z_hat0 = F.interpolate(Z_hat0,
                                   scale_factor=(self.chunk_size, self.chunk_size),
                                   mode='nearest')

        ## feature selection
        newsize = [x.size(0), 3]
        newsize.extend(list(map(lambda x: x, x.size()[2:])))
        net = torch.mul(x.expand(torch.Size(newsize)), Z_hat0)
        if self.args.with_noise:
            net = net + (1 - Z_hat0) * eps
        ## decode
        newsize2 = [-1, 3]
        newsize2.extend(newsize[2:])
        net = net.view(torch.Size(newsize2))
        fake_image = self.decode(net)


        return fake_image, Z_hat0, net.view(self.args.batch_size, 3, 32, 32)

    def forward(self, x, Z_hat, mu, sigma, num_sample=1):
        fake_image, Z_hat0, noised_input = self.decoder(x, Z_hat, mu, sigma, num_sample)
        return Z_hat0, fake_image.view(self.args.batch_size, 3, 32, 32), noised_input


    def weight_init(self):
        for m in self._modules:
            xavier_init(self._modules[m])

def xavier_init(ms):
    for m in ms:
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            m.bias.data.zero_()