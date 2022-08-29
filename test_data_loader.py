import json
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import prettytable as pt
from gensim.models import KeyedVectors
from transformers import AutoTokenizer
import os
import utils
import requests
os.environ["TOKENIZERS_PARALLELISM"] = "false"

dis2idx = np.zeros((1000), dtype='int64')
dis2idx[1] = 1
dis2idx[2:] = 2
dis2idx[4:] = 3
dis2idx[8:] = 4
dis2idx[16:] = 5
dis2idx[32:] = 6
dis2idx[64:] = 7
dis2idx[128:] = 8
dis2idx[256:] = 9


class Vocabulary(object):
    PAD = '<pad>'
    UNK = '<unk>'
    SUC = '<suc>'

    def __init__(self):
        self.label2id = {self.PAD: 0, self.SUC: 1}
        self.id2label = {0: self.PAD, 1: self.SUC}

    def add_label(self, label):
        label = label.lower()
        if label not in self.label2id:
            self.label2id[label] = len(self.label2id)    #统计实体的类别
            self.id2label[self.label2id[label]] = label

        assert label == self.id2label[self.label2id[label]]

    def __len__(self):
        return len(self.token2id)

    def label_to_id(self, label):
        label = label.lower()
        return self.label2id[label]

    def id_to_label(self, i):
        return self.id2label[i]


def collate_fn(data):
    bert_inputs, grid_mask2d, pieces2word, dist_inputs, sent_length = map(list, zip(*data))

    max_tok = np.max(sent_length)   #以最长的句的长度子作为这个batchsize的长度
    sent_length = torch.LongTensor(sent_length)
    max_pie = np.max([x.shape[0] for x in bert_inputs])
    bert_inputs = pad_sequence(bert_inputs, True)    #padding
    batch_size = bert_inputs.size(0)

    def fill(data, new_data):
        for j, x in enumerate(data):
            new_data[j, :x.shape[0], :x.shape[1]] = x   #将数据转移到padding后的矩阵上
        return new_data

    dis_mat = torch.zeros((batch_size, max_tok, max_tok), dtype=torch.long)
    dist_inputs = fill(dist_inputs, dis_mat)
    labels_mat = torch.zeros((batch_size, max_tok, max_tok), dtype=torch.long)
    # grid_labels = fill(grid_labels, labels_mat)
    mask2d_mat = torch.zeros((batch_size, max_tok, max_tok), dtype=torch.bool)
    grid_mask2d = fill(grid_mask2d, mask2d_mat)
    sub_mat = torch.zeros((batch_size, max_tok, max_pie), dtype=torch.bool)
    pieces2word = fill(pieces2word, sub_mat)

    return bert_inputs, grid_mask2d, pieces2word, dist_inputs, sent_length

class RelationDataset(Dataset):
    def __init__(self, bert_inputs, pieces2word, grid_mask2d, dist_inputs, sent_length):
        self.bert_inputs = bert_inputs
        # self.grid_labels = grid_labels
        self.grid_mask2d = grid_mask2d
        self.pieces2word = pieces2word
        self.dist_inputs = dist_inputs
        self.sent_length = sent_length
        # self.entity_text = entity_text

    def __getitem__(self, item):
        return torch.LongTensor(self.bert_inputs[item]), \
               torch.LongTensor(self.grid_mask2d[item]), \
               torch.LongTensor(self.pieces2word[item]), \
               torch.LongTensor(self.dist_inputs[item]), \
               self.sent_length[item]
            #    self.entity_text[item]

    def __len__(self):
        return len(self.bert_inputs)


def process_bert(data, tokenizer, vocab):

    bert_inputs = []
    grid_labels = []
    grid_mask2d = []
    dist_inputs = []
    entity_text = []
    pieces2word = []
    sent_length = []

    for index, instance in enumerate(data):
        if len(instance['sentence']) == 0:
            continue

        tokens = [tokenizer.tokenize(word) for word in instance['sentence']]   #有些英文单词可以分成两个或者多个,tokens为完整的字
        pieces = [piece for pieces in tokens for piece in pieces]              #pieces为每个字切成的片段  一般中文pieces和tokens是一样的
        _bert_inputs = tokenizer.convert_tokens_to_ids(pieces)                  #将字转成id
        _bert_inputs = np.array([tokenizer.cls_token_id] + _bert_inputs + [tokenizer.sep_token_id])  #前后加上cls和sep

        length = len(instance['sentence'])                                     #句子的长度即字和标点的个数
        _grid_labels = np.zeros((length, length), dtype=np.int)               
        _pieces2word = np.zeros((length, len(_bert_inputs)), dtype=np.bool)
        _dist_inputs = np.zeros((length, length), dtype=np.int)
        _grid_mask2d = np.ones((length, length), dtype=np.bool)

        if tokenizer is not None:
            start = 0
            for i, pieces in enumerate(tokens):  #遍历tokenizer后的数据
                if len(pieces) == 0:
                    continue
                pieces = list(range(start, start + len(pieces)))  #每段pieces的开始和结束索引
                _pieces2word[i, pieces[0] + 1:pieces[-1] + 2] = 1  #pieces[-1] + 2是因为pieces[-1]为索引，正常取到最后一位是要加1（因为第一个cls），但是由于需要向后一位，因此是加2
                                                                   #在二维矩阵中用每个字对应的piece标记为1
                start += len(pieces)

        for k in range(length):   #字的个数
            _dist_inputs[k, :] += k
            _dist_inputs[:, k] -= k     #_dist_inputs即位置输入，第一行只有一个0其余为负，第二行1,0，其余为负，第三行2，1,0，其余为负

        for i in range(length):
            for j in range(length):
                if _dist_inputs[i, j] < 0:
                    _dist_inputs[i, j] = dis2idx[-_dist_inputs[i, j]] + 9
                else:
                    _dist_inputs[i, j] = dis2idx[_dist_inputs[i, j]]
        _dist_inputs[_dist_inputs == 0] = 19                            #定义位置为20个

        # for entity in instance["ner"]:
        #     index = entity["index"]
        #     for i in range(len(index)):
        #         if i + 1 >= len(index):
        #             break
        #         _grid_labels[index[i], index[i + 1]] = 1
        #     _grid_labels[index[-1], index[0]] = vocab.label_to_id(entity["type"])  #构建word-pair grid，右上部分为NNW，左下部分为 THW（用标签类型的id表示）

        # _entity_text = set([utils.convert_index_to_text(e["index"], vocab.label_to_id(e["type"])) #总共10个类别
        #                     for e in instance["ner"]])  #将实体的索引和实体的类别通过#连接起来，并放入到一个集合中

        sent_length.append(length)        #句子的长度即字和标点的个数
        bert_inputs.append(_bert_inputs)  #将字通过tokenizer切成piece并前后加上cls和sep并转成id
        # grid_labels.append(_grid_labels)  #右上部分为NNW（用1表示，即SUC），左下部分为 THW（用标签类型的id表示）
        grid_mask2d.append(_grid_mask2d)  #通过句子的长度设置
        dist_inputs.append(_dist_inputs)
        pieces2word.append(_pieces2word)  #二维矩阵中用每个字对应的一个或多个piece标记为1
        # entity_text.append(_entity_text)  # 将实体的索引和实体的类别通过#连接起来，并放入到一个集合中

    return bert_inputs, pieces2word, grid_mask2d, dist_inputs, sent_length

def fill_vocab(vocab, dataset):
    entity_num = 0
    for instance in dataset:
        for entity in instance["ner"]:
            vocab.add_label(entity["type"])
        entity_num += len(instance["ner"])
    return entity_num


class LoadTestDataBert:
    def __init__(self,config):

        with open('./data/{}/train.json'.format(config.dataset), 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        with open('./data/{}/dev.json'.format(config.dataset), 'r', encoding='utf-8') as f:
            dev_data = json.load(f)
        with open('./data/{}/test.json'.format(config.dataset), 'r', encoding='utf-8') as f:
            test_data = json.load(f)

        self.tokenizer = AutoTokenizer.from_pretrained(config.bert_name, cache_dir="./cache/")

        self.vocab = Vocabulary()
        train_ent_num = fill_vocab(self.vocab, train_data)  #返回数据集中的所有实体数量以及统计数据集中的label2id和id2label
        dev_ent_num = fill_vocab(self.vocab, dev_data)
        test_ent_num = fill_vocab(self.vocab, test_data)

    # table = pt.PrettyTable([config.dataset, 'sentences', 'entities'])
    # # table.add_row(['train', len(train_data), train_ent_num])
    # # table.add_row(['dev', len(dev_data), dev_ent_num])
    # table.add_row(['test', len(test_data), test_ent_num])
    # config.logger.info("\n{}".format(table))

        config.label_num = len(self.vocab.label2id)
        config.vocab = self.vocab

    # train_dataset = RelationDataset(*process_bert(train_data, tokenizer, vocab))  #RelationDataset将数据存入类中，并且转成LongTensor
    # dev_dataset = RelationDataset(*process_bert(dev_data, tokenizer, vocab))
    def load_test_data_bert(self,data):
        test_dataset = RelationDataset(*process_bert(data, self.tokenizer, self.vocab))
        return test_dataset, data
