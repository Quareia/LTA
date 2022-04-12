# -*- coding = 'utf-8' -*-

import torch
import os
import csv
import pickle
import gensim
import numpy as np
# from gensim.models import Word2Vec

from utils import ensure_dir


np.random.seed(1)

BASE_DIR = './data/ver1'
RSC_DIR = './data/resources'
SAVE_DIR = './data/ver1'


def load_data_from_pkl(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


class Vocab():
    def __init__(self, w2v_dim=300, w2v_file_name='no'):
        self.w2v_dim = w2v_dim

        self.word2id = {'<PAD>': 0, '<UNK>': 1}
        self.id2word = {0: '<PAD>', 1: '<UNK>'}
        self.word2vec = {'<PAD>': np.zeros(w2v_dim).tolist(),
                         '<UNK>': np.random.uniform(-0.25, 0.25, self.w2v_dim).round(6).tolist()
                         }
        self.word_f = {'<PAD>': 0, '<UNK>': 0}
        self.pretrain_vocab_size = 0

        if w2v_file_name != 'no' and w2v_file_name != 'self':
            # try:
            self._load_word_vectors(w2v_file_name)
            # except:
            #     raise Exception("Not Found Word2vec File: {}".format(w2v_file_name))

    def add_word(self, word, vector=None):
        id = len(self.word2id)
        self.word2id[word] = id
        self.id2word[id] = word
        if vector is not None:
            self.word2vec[word] = vector
        else:
            uniform = np.random.uniform(-0.25, 0.25, self.w2v_dim).round(6).tolist()
            self.word2vec[word] = uniform

    def set_w2v(self, word, vector):
        try:
            self.word2vec[word] = vector
        except:
            raise Exception("Not Found Word When Set Word2vec: {}".format(word))

    def _load_word_vectors(self, w2v_file_name):
        # load word2vec from pre-train file
        # 2: GoogleNews 300 SLIM

        w2v_path = os.path.join(RSC_DIR, w2v_file_name)

        # elif w2v_file_name == 'wiki.en.vec':
        #     with open(w2v_path, 'r') as f:
        #         _, self.w2v_dim = [int(_) for _ in f.readline().strip().split(' ')]
        #         lines = f.readlines()
        #         for line in lines:
        #             word, vector = line.strip().split(' ', 1)
        #             if word not in self.word2id:
        #                 self.add_word(word, [float(_) for _ in vector.split(' ')])
        if w2v_file_name == 'GoogleNews-vectors-negative300-SLIM.bin':
            model = gensim.models.KeyedVectors.load_word2vec_format(w2v_path, binary=True)
            words = model.index2word
            self.w2v_dim = model.vector_size
            for i, word in enumerate(words):
                vector = model[word]
                self.add_word(word, vector)
        else:
            # if w2v_file_name == 'sgns_merge_subsetSMP.txt':
            with open(w2v_path, 'r') as f:
                _, self.w2v_dim = [int(_) for _ in f.readline().strip().split(' ')]
                lines = f.readlines()
                for line in lines:
                    word, vector = line.strip().split(' ', 1)
                    if word not in self.word2id:
                        self.add_word(word, [float(_) for _ in vector.split(' ')])
            # raise Exception("Not Found Word2Vec File Path: {}".format(w2v_path))

        self.pretrain_vocab_size = len(self.word2id) - 2
        # print(self.pretrain_vocab_size)

    def config(self):
        config_dict = {
            'dataset': {
                'vocab': {
                    'pretrain vocab size': self.pretrain_vocab_size,
                    'vocab size': len(self.word2id),
                    'word2vec dim': self.w2v_dim
                }
            }
        }
        # print(self.print_dict)
        return config_dict


class Corpus():
    def __init__(self, data_name, lang, w2v, min_freq=1, max_len=60):
        """
        :param dataname: dataset name
        :param lang: language
        :param w2v: 'no' or 'self' or w2v filename
        :param save_path: save file path
        :param min_freq: min word frequency to cut
        :param max_len: max length of a sentence to cut
        """

        self.data_name = data_name
        self.lang = lang
        self.w2v = w2v
        self.save_path = os.path.join(SAVE_DIR, data_name, '{}_{}.pkl'.format(data_name, w2v[:4]))
        self.min_freq = min_freq
        self.max_len = max_len
        self.is_pretrain = True if w2v else False

        self.vocab = Vocab(w2v_file_name=self.w2v)
        self._preprocess()

    def _process_text(self, text, type='train'):
        words = text.split(' ')
        raw_ids = []
        for word in words:
            if type == 'test':
                raw_ids.append(self.vocab.word2id.get(word, 1))
                continue
            self.vocab.word_f[word] = self.vocab.word_f.get(word, 0) + 1
            if word not in self.vocab.word2id:
                self.vocab.add_word(word)
            raw_ids.append(self.vocab.word2id[word])

        length = len(raw_ids[:self.max_len])
        pad_ids = raw_ids[:self.max_len] + [0] * (self.max_len - length)

        sample = {
            'text': text,
            'raw_ids': raw_ids,
            'pad_ids': pad_ids,
            'length': length
        }
        return sample

    def _read_from_csv(self, file_path, type='train'):
        raws = []
        samples = []

        csv_file = csv.reader(open(file_path, encoding='utf-8'))
        next(csv_file)
        for row in csv_file:
            raw = row[0]
            sample = self._process_text(raw, type)
            sample['y'] = int(row[1])
            samples.append(sample)
            raws.append(raw)

        return samples, raws

    def _save2pkl(self, data):
        # ensure_dir(SAVE_DIR)

        with open(self.save_path, 'wb') as f:
            pickle.dump(data, f)

    def _train_and_set_word2vec(self, sentences):
        # gensim word2vec
        model = Word2Vec(sentences, size=self.vocab.w2v_dim, window=5, min_count=1, workers=4)

        # set word2vec
        for word, idx in self.vocab.word2id.items():
            if word in model.wv:
                self.vocab.set_w2v(word, model[word])
            else:
                uniform = np.random.uniform(-0.25, 0.25, self.vocab.w2v_dim).round(6).tolist()
                self.vocab.set_w2v(word, uniform)
        return model

    def _preprocess(self):
        # files = ['seen_class', 'val_unseen_class', 'unseen_class',
        #          'train_seen', 'val_seen', 'test_seen', 'val_unseen', 'test_unseen']
        files = ['seen_class', 'unseen_class',
                 'train_seen', 'test_seen', 'test_unseen']

        data = {}
        for file in files:
            file_path = os.path.join(BASE_DIR, '{}/{}.csv'.format(self.data_name, file))
            locals()[file], locals()['{}_raws'.format(file)] = self._read_from_csv(file_path)
            data[file] = locals()[file]

        # train gensim word2vec model
        if self.w2v == 'self':
            sentences = seen_class_raws + train_seen_raws
            self._train_and_set_word2vec(sentences)

        data['corpus'] = self

        self._save2pkl(data)

    def get_wordembedding(self):
        wv_list = [[0.0] * self.vocab.w2v_dim] * (len(self.vocab.word2id))
        for idx, word in self.vocab.id2word.items():
            wv_list[idx] = self.vocab.word2vec[word]
        wv_tensor = torch.Tensor(wv_list)
        return wv_tensor

    def config(self):
        self.config_dict = self.vocab.config()
        self.config_dict['dataset']['name'] = self.data_name
        self.config_dict['dataset']['language'] = self.lang
        self.config_dict['dataset']['pretrain'] = self.is_pretrain
        self.config_dict['dataset']['word2vec'] = self.w2v
        self.config_dict['dataset']['max_sent_len'] = self.max_len
        return self.config_dict


if __name__ == '__main__':
    # corpus = Corpus('SNIPS', 'EN', 'wiki.en.vec')
    # print(corpus.config())
    # corpus = Corpus('SMP', 'CH', 'sgns_merge_subsetSMP.txt')
    # print(corpus.config())
    # corpus = Corpus('ATIS', 'EN', 'wiki.en.vec')
    # print(corpus.config())
    corpus = Corpus('Clinc', 'EN', 'GoogleNews-vectors-negative300-SLIM.bin')
    print(corpus.config())
    # # corpus = Corpus('Quora', 'EN', 'wiki.en.vec')
    # corpus = Corpus('Quora', 'EN', 'GoogleNews-vectors-negative300-SLIM.bin')
    # print(corpus.config())
    #
