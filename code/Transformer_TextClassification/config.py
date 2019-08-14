# coding:utf-8

class Config(object):
    N = 2 #编码器中子层数量，transformer论文中使用6个
    d_model = 256 #模型维数，transformer中使用512
    d_ff = 512 
    h = 8
    dropout = 0.1
    tag_size = 7
    lr = 0.0003
    max_epochs = 1
    batch_size = 2
    max_sen_len = 15
    batches_num = 1
    START_TAG = "<START>"
    STOP_TAG = "<STOP>"