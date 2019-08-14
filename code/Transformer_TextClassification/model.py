# coding:utf-8

import torch
from torch import nn
from transformer import Transformer
from copy import deepcopy
from constants import Const
import numpy as np
import torch.nn.functional as F
from crf import CRF

# Transformer + CRF model 
class Transformer_CRF(nn.Module):
    def __init__(self, config, src_vocab, nb_labels, ):
        super(Transformer_CRF, self).__init__()
        self.config = config
        self.src_vocab = src_vocab
        
        self.transformer = Transformer(self.config, self.src_vocab)
        self.crf = CRF(
            nb_labels, 
            Const.BOS_TAG_ID, 
            Const.EOS_TAG_ID, 
            pad_tag_id=Const.PAD_TAG_ID, 
            batch_first=True,
        )

    def forward(self, x, mask=None):
        emissions = self.transformer(x)
        score, path = self.crf.decode(emissions, mask=mask)
        return score, path