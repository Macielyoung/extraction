# coding:utf-8

root = "/Users/maciel/Documents/Competition/InformationExtraction/"

class Intergration():
    def __init__(self, read_file, save_file):
        self.read_file = read_file
        self.save_file = save_file
        self.sentences = []
        self.labels = []

    def intergrate(self):
        with open(root+self.read_file, 'r', encoding='utf-8') as f:
            sentence, label = [], []
            for line in f.readlines():
                if line == "\n":
                    self.sentences.append(sentence)
                    self.labels.append(label)
                    sentence, label = [], []
                else:
                    # print(id, line)
                    word, mark = line.strip("\n").split("\t")
                    sentence.append(word)
                    label.append(mark)

    def write_txt(self):
        with open(root+self.save_file, 'w', encoding='utf-8') as f:
            for s, l in zip(self.sentences, self.labels):
                sen_str = " ".join(s)
                f.write(sen_str+"\n")
                lab_str = " ".join(l)
                f.write(lab_str+"\n")

if __name__ == "__main__":
    read_file = "data/datagrand/dg_train.txt"
    save_file = "data/datagrand/train_sen_label.txt"
    inter = Intergration(read_file, save_file)
    inter.intergrate()
    inter.write_txt()
    # print(inter.sentences[:2])
    # print(inter.labels[:2])