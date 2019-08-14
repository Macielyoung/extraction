# coding:utf-8

import torch
from torch import nn
import torch.nn.functional as F

class PositionwiseFeedForward(nn.Module):
    # "Positionwise feed-forward网络全连接层"
    # 计算公式： FFN(x) = max(0, x*W1+b1)*W2+b2
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))