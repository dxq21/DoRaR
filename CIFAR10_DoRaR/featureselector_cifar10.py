# School: ISU
# Name: Dong Qin
# Creat Time: 1/16/2022 4:16 PM
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical
from torch.distributions.one_hot_categorical import OneHotCategorical
from utils import Flatten, idxtobool, cuda
from pathlib import Path


class featureselector_cifar10(nn.Module):

    def __init__(self, **kwargs):

        super(featureselector_cifar10, self).__init__()

        self.args = kwargs['args']
        self.mode = self.args.mode

        self.device = torch.device("cuda" if self.args.cuda else "cpu")
        self.tau = self.args.tau  # float, parameter for concrete random variable distribution
        self.K = self.args.K  # number of variables selected
        self.model_type = self.args.explainer_type  # 'nn', 'cnn', 'nn4', 'cnn4'
        self.chunk_size = self.args.chunk_size


        if self.chunk_size == 4:
            self.encode_cnn4 = nn.Sequential(
                nn.Conv2d(3, 8, kernel_size=(5, 5), padding=2),
                # nn.BatchNorm2d(8),
                nn.ReLU(True),
                nn.MaxPool2d(kernel_size=(2, 2)),
                nn.Conv2d(8, 16, kernel_size=(5, 5), padding=2),
                # nn.BatchNorm2d(16),
                nn.ReLU(True),
                nn.MaxPool2d(kernel_size=(2, 2)),
                nn.Conv2d(16, 1, kernel_size=(1, 1)),
                Flatten(),
                nn.LogSoftmax(1)

            )
        else:
            self.encode_cnn4 = nn.Sequential(
                nn.Conv2d(3, 8, kernel_size=(5, 5), padding=2),
                nn.BatchNorm2d(8),
                nn.ReLU(True),

                nn.Conv2d(8, 16, kernel_size=(5, 5), padding=2),
                nn.BatchNorm2d(16),
                nn.ReLU(True),

                nn.Conv2d(16, 1, kernel_size=(1, 1)),
                Flatten(),

                nn.LogSoftmax(1)
                )
        self.encode = self.encode_cnn4

    def encoder(self, x):
        p_i = self.encode(x)
        return p_i  # [batch_size, d]

    def reparameterize(self, p_i, tau, k, num_sample=1, x_check=1):
        p_i_ = p_i.view(p_i.size(0), 1, -1)
        p_i_ = p_i_.expand(p_i_.size(0), k, p_i_.size(-1))
        V = F.gumbel_softmax(p_i_, tau=tau, hard=False)
        V = V.sum(1)
        V = V.view(V.size(0), 1, -1)
        V = 1/(1 + torch.exp(-(V.add(-0.5)).mul(10)))

        ## without sampling
        V_fixed_size = p_i.unsqueeze(1).size()
        _, V_fixed_idx = p_i.unsqueeze(1).topk(k, dim=-1)  # batch * 1 * k
        V_fixed = idxtobool(V_fixed_idx, V_fixed_size, is_cuda=self.args.cuda)
        V_fixed = V_fixed.type(torch.float)

        return V, V_fixed

    def forward(self, x, num_sample=1):
        p_i = self.encoder(x)  # probability of each element to be selected [batch-size, d]

        Z_hat, Z_hat_fixed = self.reparameterize(p_i, tau=self.tau,
                                                 k=self.K,
                                                 num_sample=num_sample,
                                                 x_check=x,
                                                 )  # torch.Size([batch-size, num-samples, d])
        return p_i, Z_hat, Z_hat_fixed

    def weight_init(self):
        for m in self._modules:
            xavier_init(self._modules[m])

def xavier_init(ms):
    for m in ms:
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            m.bias.data.zero_()