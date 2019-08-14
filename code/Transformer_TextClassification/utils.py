# coding:utf-8

import torch
from torch import nn
from torch.autograd import Variable
import copy
import math

def clones(module, N):
    # 生成N层encoder子层
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Embedding(nn.Module):
    # embedding层
    def __init__(self, d_model, pre_train_weight, embedding_size):
        super(Embedding, self).__init__()
        # 使用已经训练好的Glove 100维词向量模型
        self.lut = nn.Embedding.from_pretrained(pre_train_weight)
        self.lut.weight.requires_grad = True
        self.fc = nn.Linear(embedding_size, d_model)
        self.d_model = d_model
    
    def forward(self, x):
        return self.fc(self.lut(x)) * math.sqrt(self.d_model)                                                                                                                                                                     

class PositionalEncoding(nn.Module):
    # 使用正弦余弦公式来计算位置编码
    # PE(pos, 2i)   = sin(pos/10000^(2i/d_model))
    # PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        # 函数 : e ^ (2i * -log(10000) / d), i是维度
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(torch.as_tensor(position.numpy() * div_term.unsqueeze(0).numpy()))
        pe[:, 1::2] = torch.cos(torch.as_tensor(position.numpy() * div_term.unsqueeze(0).numpy()))
        pe = pe.unsqueeze(0)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(torch.as_tensor(position.numpy() * div_term.unsqueeze(0).numpy()))
        pe[:, 1::2] = torch.cos(torch.as_tensor(position.numpy() * div_term.unsqueeze(0).numpy()))#torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)