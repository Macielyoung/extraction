# coding:utf-8

from torch import nn
import sys
from config import Config
import torch.optim as optim
import torch
from model import Transformer_CRF
from data_loader import Dataset

if __name__ == "__main__":
    config = Config()
    embedding_size = 50
    train_file = "../../data/processed/train/train_data.txt"
    test_file = "../../data/processed/test/dg_test.txt"

    dataset = Dataset(config)
    # print(dataset.word_to_ix)
    # print(dataset.tag_to_ix)
    # 读取文件
    train_lines = dataset.read_lines(train_file)
    dataset.get_words_tags(train_lines)
    test_lines = dataset.read_lines(test_file)
    dataset.get_test_words(test_lines)

    # print(dataset.word_to_ix)
    # print(dataset.tag_to_ix)

    sen_tag_data = dataset.get_sentence_and_tag(train_lines)
    # print("sen_tag: ", sen_tag_data[:2])
    train_input_batches, train_label_batches = dataset.get_batches(sen_tag_data, config.batches_num, config.batch_size)
    # print(train_input_batches[0])
    # print(train_label_batches[0])