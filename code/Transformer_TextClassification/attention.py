# coding:utf-8

import torch
from torch import nn
import math
import torch.nn.functional as F
from utils import clones

def attention(qurey, key, value, mask=None, dropout=None):
    # "点乘注意力机制实现"
    # 计算公式： att(Q,K,V) = softmax(QK^T/sqrt(d_k))V
    d_k = qurey.size(-1)
    scores = torch.matmul(qurey, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask==0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        # "词向量长度和多头数目"
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        # "多头注意力机制实现"
        if mask is not None:
            mask = mask.unsqueeze(1)
        n_batches = query.size(0)

        # 1) 执行线性映射 d_model => h * d_k
        query, key, value = [l(x).view(n_batches, -1, self.h, self.d_k).transpose(1, 2) 
                            for l, x in zip(self.linears, (query, key, value))]
        # 2) 在每个映射层中执行attention操作
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # 3) 将多头结果拼接起来
        x = x.transpose(1, 2).contiguous().view(n_batches, -1, self.h*self.d_k)
        return self.linears[-1](x)