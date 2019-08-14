# coding:utf-8

import torch

# get the pre_trained word2vec weight (Glove 100d)
def get_weight(model, embedding_size, dataset):
    vocab_size = len(dataset.vocab)
    weight = torch.zeros(vocab_size, embedding_size)
    for i in range(len(model.index2word)):
        try:
            index = dataset.word_to_ix[model.index2word[i]]
        except:
            continue
        weight[index, :] = torch.from_numpy(model.get_vector(
            dataset.ix_to_word[dataset.word_to_ix[model.index2word[i]]]
        ))
    return weight