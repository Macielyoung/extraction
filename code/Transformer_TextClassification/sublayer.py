# coding:utf-8

import torch
from torch import nn

class LayerNorm(nn.Module):
    # 层归一化模块
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    # 标准归一化
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerOutput(nn.Module):
    # 子层输出，每个子层后面加上一个残差
    def __init__(self, size, dropout):
        super(SublayerOutput, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, sublayer):
        # 层归一化公式输出： LayerNorm(x+Sublayer(x)) 包含所有子层
        return x + self.dropout(sublayer(self.norm(x)))