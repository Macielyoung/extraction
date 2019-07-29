# coding:utf-8

# @Created : Macielyoung
# @Time : 2019/7/26
# @Function : Get sentences and corresponding labels

root = "/Users/maciel/Documents/Competition/InformationExtraction/"
root2 = "/Users/maciel/Documents/Competition/extraction/data/processed/"

class Intergration():
    def __init__(self, read_file):
        self.read_file = read_file

    def intergrate(self):
        with open(root+self.read_file, 'r', encoding='utf-8') as f:
            sentences, labels = [], []
            sentence, label = [], []
            for line in f.readlines():
                if line == "\n":
                    sentences.append(sentence)
                    labels.append(label)
                    sentence, label = [], []
                else:
                    # print(id, line)
                    word, mark = line.strip("\n").split("\t")
                    sentence.append(word)
                    label.append(mark)
        return sentences, labels

    def write_txt(self, sen_tag_file, sen_file, tag_file, sentences, labels):
        with open(root2+sen_tag_file, 'w', encoding='utf-8') as f:
            for s, l in zip(sentences, labels):
                sen_str = " ".join(s)
                f.write(sen_str+"\n")
                lab_str = " ".join(l)
                f.write(lab_str+"\n")

        with open(root2+sen_file, 'w', encoding='utf-8') as f:
            for s in sentences:
                sen_str = " ".join(s)
                f.write(sen_str+"\n")
        with open(root2+tag_file, 'w', encoding='utf-8') as f:
            for l in labels:
                lab_str = " ".join(l)
                f.write(lab_str+"\n")

if __name__ == "__main__":
    read_file = "data/datagrand/dg_train.txt"
    train_sen_file = "train/sentences.txt"
    train_tag_file = "train/tags.txt"
    val_sen_file = "val/sentences.txt"
    val_tag_file = "val/tags.txt"
    train_sen_tag_file = "train/sen_tag.txt"
    val_sen_tag_file = "val/sen_tag.txt"
    inter = Intergration(read_file)
    sentences, labels = inter.intergrate()
    val_sentences, val_labels = sentences[:3000], labels[:3000]
    train_sentences, train_labels = sentences[3000:], labels[3000:]
    inter.write_txt(train_sen_tag_file, train_sen_file, train_tag_file, train_sentences, train_labels)
    inter.write_txt(val_sen_tag_file, val_sen_file, val_tag_file, val_sentences, val_labels)