# coding:utf-8

import torch
import numpy as np
import random
from config import Config
from constants import Const
import random

class Dataset(object):
    def __init__(self, config):
        self.config = config
        self.word_to_ix = {
            Const.UNK_TOKEN: Const.UNK_ID,
            Const.PAD_TOKEN: Const.PAD_ID,
            Const.BOS_TOKEN: Const.BOS_ID,
            Const.EOS_TOKEN: Const.EOS_ID,
        }
        self.ix_to_word = {
            Const.UNK_ID: Const.UNK_TOKEN,
            Const.PAD_ID: Const.PAD_TOKEN,
            Const.BOS_ID: Const.BOS_TOKEN,
            Const.EOS_ID: Const.EOS_TOKEN,
        }
        self.tag_to_ix = {
            Const.PAD_TAG_TOKEN: Const.PAD_TAG_ID,
            Const.BOS_TAG_TOKEN: Const.BOS_TAG_ID,
            Const.EOS_TAG_TOKEN: Const.EOS_TAG_ID,
        }
        self.ix_to_tag = {
            Const.PAD_TAG_ID: Const.PAD_TAG_TOKEN,
            Const.BOS_TAG_ID: Const.BOS_TAG_TOKEN,
            Const.EOS_TAG_ID: Const.EOS_TAG_TOKEN,
        }

    def read_lines(self, datafile):
        with open(datafile, 'r') as f:
            lines = f.readlines()
        return lines
        
    def get_words_tags(self, lines):
        for rid, line in enumerate(lines):
            if rid % 2 == 0:
                # 奇数行是句子
                for word in line.strip().split(" "):
                    if word not in self.word_to_ix:
                        num = len(self.word_to_ix)
                        self.word_to_ix[word] = num
                        self.ix_to_word[num] = word
            else:
                # 偶数行是标注
                for tag in line.strip().split(" "):
                    if tag not in self.tag_to_ix:
                        num = len(self.tag_to_ix)
                        self.tag_to_ix[tag] = num
                        self.ix_to_tag[num] = tag

    def get_test_words(self, lines):
        for line in lines:
            for word in line.strip().split(" "):
                if word != "" and word not in self.word_to_ix:
                    num = len(self.word_to_ix)
                    self.word_to_ix[word] = num
                    self.ix_to_word[num] = word
    
    def get_sentence_and_tag(self, lines):
        sen_tag_data = []
        one_line = []
        for rid, line in enumerate(lines):
            if rid % 2 == 0:
                one_line.append(line.strip().split())
            else:
                one_line.append(line.strip().split())
                sen_tag_data.append(one_line)
                one_line = []
        return sen_tag_data

    def get_batches(self, sen_tag_data, batches_num, batch_size):
        input_seqs, label_seqs = [], []
        for _ in range(batches_num):
            input_seqs = random.sample(sen_tag_data, batch_size)
            trans_input_seqs = []
            trans_label_seqs = []
            for seq in input_seqs:
                # print(seq[0])
                trans_sen = [self.word_to_ix[word] for word in seq[0]]
                # print(trans_sen)
                trans_tag = [self.tag_to_ix[tag] for tag in seq[1]]
                trans_input_seqs.append(trans_sen)
                trans_label_seqs.append(trans_tag)
            print("input: ", trans_input_seqs)
            # print("label: ", trans_label_seqs)
            input_seqs.append(trans_input_seqs)
            label_seqs.append(trans_label_seqs)
        return input_seqs, label_seqs

if __name__ == "__main__":
    config = Config()
    train_file = "../../data/processed/train/processed_data.txt"
    dev_file = "../../data/processed/dev/sen_tag.txt"

    dataset = Dataset(config)
    train_lines = dataset.read_lines(train_file)