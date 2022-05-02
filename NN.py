from torch import nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torch
import torch
from torch import nn


class Rnn(nn.Module):
    def __init__(self, INPUT_SIZE):
        super(Rnn, self).__init__()

        self.rnn = nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=30,
            num_layers=1,

        )

        self.out = nn.Linear(30, 2)

    def forward(self, x):
        r_out, h_state = self.rnn(x, None)

        out = self.out(r_out[:, -1, :])
        return out

class BPNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.linear1 = torch.nn.Linear(46, 64)
        # linear(x1,x2)的两个参数分别表示输入输出的维度（矩阵的列数）
        # 这里的9与数据集的列数一至
        self.linear2 = torch.nn.Linear(64, 64)
        self.linear3 = torch.nn.Linear(64, 2)
        self.sigmoid = torch.nn.Sigmoid()

        # self.out = nn.Linear(hidden_size,2)

    def forward(self,x):
        x1 = self.sigmoid(self.linear1(x))
        x1 = self.sigmoid(self.linear2(x1))
        x1 = self.sigmoid(self.linear3(x1))
        # x1 = self.linear3(x1)
        return x1