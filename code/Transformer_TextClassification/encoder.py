# coding:utf-8

from torch import nn
from utils import clones
from sublayer import LayerNorm, SublayerOutput

class Encoder(nn.Module):
    # 编码器，包含多个子层
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
    
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class EncoderLayer(nn.Module):
    # 单个编码层，包含自注意力层和全连接层
    # 每个子层都有残差和层归一化，通过SublayerOutput实现
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer_output = clones(SublayerOutput(size, dropout), 2)
        self.size = size

    def forward(self, x, mask=None):
        # 第一层自编码层输出，第二层全连接层输出
        x = self.sublayer_output[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer_output[1](x, self.feed_forward)