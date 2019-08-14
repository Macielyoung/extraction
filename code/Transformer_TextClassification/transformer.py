# coding:utf-8

import torch
from torch import nn
from copy import deepcopy
from utils import Embedding, PositionalEncoding
from attention import MultiHeadedAttention
from encoder import EncoderLayer, Encoder
from ffn import PositionwiseFeedForward

class Transformer(nn.Module):
    def __init__(self, config, pre_train_weight, embedding_size):
        super(Transformer, self).__init__()
        self.config = config
        self.pre_train_weight = pre_train_weight
        self.embedding_size = embedding_size
        
        # 超参数
        # h是多头数量， N是层数， dropout是比率
        h, N, dropout = self.config.h, self.config.N, self.config.dropout
        # 词向量维度，全连接维度
        d_model, d_ff = self.config.d_model, self.config.d_ff

        # 多头注意力层
        attn = MultiHeadedAttention(h, d_model)
        # 全连接层
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        # 位置向量
        position = PositionalEncoding(d_model, dropout)

        self.encoder = Encoder(EncoderLayer(config.d_model, deepcopy(attn), deepcopy(ff), dropout), N)
        self.src_embed = nn.Sequential(Embedding(self.config.d_model, self.pre_train_weight, self.embedding_size), deepcopy(position)) # embedding with position encoding
    
    def forward(self, x):
        embedded_sentences = self.src_embed(x) # shape = (batch_size, sen_len, d_model)
        encoded_sentences = self.encoder(embedded_sentences)
        # print(encoded_sentences.shape)

        # Convert input to (batch_size, d_model) for linear layer
        final_feature_map = encoded_sentences[:,-1,:]
        final_out = self.fc(final_feature_map)
        return final_out