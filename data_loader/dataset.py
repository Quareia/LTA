from data_preprocess import *
import numpy as np
from collections import defaultdict
from copy import deepcopy


class Dataset():
    """Generalized Zero-Shot Learning Dataset, including all label."""

    def __init__(self, data_path):
        # Dict (class_name: (list) Sample)
        # self.all_class = defaultdict(list)

        # List (class_name)
        self.seen_class = None
        self.val_unseen_class = None
        self.unseen_class = None

        # List (samples)
        self.train_seen = None
        self.test_seen = None
        self.val_unseen = None
        self.test_unseen = None

        self._init_dataset(data_path)

    def _init_dataset(self, data_path):
        data = load_data_from_pkl(data_path)

        for k, v in data.items():
            setattr(self, k, v)

        # for sample in self.seen_class + self.unseen_class:
        # self.all_class[sample['y']].append(sample)

        self.n_seen_class = len(self.seen_class)
        self.n_val_unseen_class = 0
        self.n_unseen_class = len(self.unseen_class)
        self.all_class = self.seen_class + self.unseen_class
        self.samples = self.train_seen + self.test_seen + self.test_unseen
        # print(self.all_class)

    def reset_dataset_by_random(self, train_class_num=0, val_unseen_num=0, unseen_num=0, keep_prob=True):
        """Reset dataset randomly

        Args:
            train_class_num: (int) the number of training set class
            val_unseen_num: (int) the number of validation set class
            unseen_num: (int) the number of testing set class
            keep_prob: (bool) keep prob for each class or not
        """
        self.n_seen_class = train_class_num if train_class_num else self.n_seen_class
        self.n_val_unseen_class = val_unseen_num if val_unseen_num else self.n_val_unseen_class
        self.n_unseen_class = unseen_num if unseen_num else self.n_unseen_class
        assert self.n_seen_class + self.n_val_unseen_class + self.n_unseen_class <= len(
            self.all_class), 'Exceeded class total number.'

        all_class = deepcopy(self.all_class)
        np.random.shuffle(all_class)

        idx = {sample['y']: i for i, sample in enumerate(all_class)}
        for cls in all_class:
            cls['y'] = idx[cls['y']]

        self.seen_class = all_class[:self.n_seen_class]
        self.val_unseen_class = all_class[self.n_seen_class: self.n_seen_class + self.n_val_unseen_class]
        self.unseen_class = all_class[self.n_seen_class + self.n_val_unseen_class: \
                                      self.n_seen_class + self.n_val_unseen_class + self.n_unseen_class]

        samples = deepcopy(self.samples)
        for sample in samples:
            sample['y'] = idx[sample['y']]
            samples[sample['y']].append(sample)
        # for l in samples.values():
        #     np.random.shuffle(l)

        self.train_seen = []
        self.test_seen = []
        self.test_unseen = []

        if keep_prob:
            for i in range(len(samples)):
                l = samples[i]
                np.random.shuffle(l)
                if i < self.n_seen_class:
                    self.train_seen += l[:len(l) * 7 // 10]
                    self.test_seen += l[len(l) * 7 // 10:]
                elif self.n_seen_class <= i < self.n_seen_class + self.val_unseen_class:
                    self.val_unseen += l
                else:
                    self.test_unseen += l
        else:
            train = []
            for i in range(len(samples)):
                l = samples[i]
                np.random.shuffle(l)
                if i < self.n_seen_class:
                    train.append(l)
                elif self.n_seen_class <= i < self.n_seen_class + self.val_unseen_class:
                    self.val_unseen += l
                else:
                    self.test_unseen += l
            np.random.shuffle(train)
            self.train_seen += train[:len(train) * 7 // 10]
            self.test_seen += train[len(train) * 7 // 10:]


if __name__ == '__main__':
    # a = {1:2,2:3}
    # print(a.keys())
    # np.random.shuffle(a)
    dataset = Dataset('../data/Clinc/Clinc_Goog.pkl')
    print(dataset.train_seen)
