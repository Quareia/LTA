import numpy as np
import torch
from torch.utils.data import BatchSampler, RandomSampler


class CLSBatchSampler(object):
    """Traditional batch random sampler."""
    def __init__(self, labels, batch_size):
        super().__init__()
        self.labels = labels
        self.batch_size = batch_size
        self.sampler = BatchSampler(RandomSampler(labels), batch_size=self.batch_size, drop_last=False)

    def __len__(self):
        return len(self.sampler)

    def __iter__(self):
        for idxs in self.sampler:
            yield idxs


class ClassBaseBatchSampler(object):
    """Class-based batch sampler."""
    def __init__(self, labels):
        self.labels = labels
        self.process_dict_tensor()

    def process_dict_tensor(self):
        """Get a 2d dict tensor."""
        self.classes, idx, counts = torch.unique(torch.LongTensor(self.labels), return_inverse=True, return_counts=True)
        self.num_classes = len(self.classes)
        self.class2idx = {c: i for i, c in enumerate(self.classes)}

        # self.idxs = range(len(self.labels))
        self.indexes = torch.zeros(self.num_classes, max(counts)) * np.nan
        self.num_per_class = torch.zeros_like(self.classes, dtype=torch.long)

        for i, label in enumerate(self.labels):
            label_idx = idx[i]
            self.indexes[label_idx, np.where(np.isnan(self.indexes[label_idx]))[0][0]] = i
            self.num_per_class[label_idx] += 1


class MRDNFSBatchSampler(ClassBaseBatchSampler):
    def __init__(self, labels, iterations, seen_N, seen_K, unseen_N, unseen_K):
        """ Initialize the seen(random) and unseen class(few-shot) BatchSampler object.

        Args:
            labels: The training dataset labels list.
            iterations: Number of iterations (episodes) per epoch.
            seen_N: The number of seen class in an episode.
            seen_K: The number of samples for each seen class.
            unseen_N: The number of unseen class in an episode.
            unseen_K: The number of samples for each unseen class.
        """
        super().__init__(labels)
        self.iterations = iterations
        self.seen_N = seen_N
        self.seen_K = seen_K
        self.unseen_N = unseen_N
        self.unseen_K = unseen_K

    def __len__(self):
        """Returns the number of iterations (episodes) per epoch."""
        return self.iterations

    def __iter__(self):
        """Yield a batch of indexes."""

        for it in range(self.iterations):
            unseen_classes = []
            unseen_query = []
            seen_classes = []
            seen_query = []

            rand_classes = torch.randperm(self.num_classes)

            # classes for unseen
            class_idxs = rand_classes[:self.unseen_N]
            for i, c in enumerate(self.classes[class_idxs]):
                sample_idxs = torch.randperm(self.num_per_class[c])[:self.unseen_K]
                unseen_classes.append(int(c))
                unseen_query += self.indexes[c][sample_idxs].int().tolist()

            # classes for remain
            query_len = len(unseen_query)
            remain_class_idxs = rand_classes[self.unseen_N:]
            for i, c in enumerate(self.classes[remain_class_idxs]):
                sample_idxs = torch.randperm(self.num_per_class[c])
                seen_classes.append(int(c))
                seen_query += self.indexes[c][sample_idxs].int().tolist()

            # seen_query = seen_query[torch.randperm(len(seen_query))]
            np.random.shuffle(seen_query)

            # seen_support = seen_query[:-query_len]    Support set won't be taken into consideration.
            seen_query = seen_query[-query_len:]

            yield unseen_classes, unseen_query, seen_classes, seen_query


class MNFBatchSampler(ClassBaseBatchSampler):
    def __init__(self, labels, iterations, seen_N, seen_K, unseen_N, unseen_K):
        """ Initialize the seen and unseen class few-shot BatchSampler object. (In Our Paper)

        Args:
            labels: The training dataset labels list.
            iterations: Number of iterations (episodes) per epoch.
            seen_N: The number of seen class in an episode.
            seen_K: The number of samples for each seen class.
            unseen_N: The number of unseen class in an episode.
            unseen_K: The number of samples for each unseen class.
        """
        super().__init__(labels)
        self.iterations = iterations
        self.seen_N = seen_N
        self.seen_K = seen_K
        self.unseen_N = unseen_N
        self.unseen_K = unseen_K

    def __len__(self):
        """Returns the number of iterations (episodes) per epoch."""
        return self.iterations

    def __iter__(self):
        """Yield a batch of indexes."""

        for it in range(self.iterations):
            unseen_classes = []
            unseen_query = []
            seen_classes = []
            seen_query = []

            rand_classes = torch.randperm(self.num_classes)

            # classes for unseen
            class_idxs = rand_classes[:self.unseen_N]
            for i, c in enumerate(self.classes[class_idxs]):
                sample_idxs = torch.randperm(self.num_per_class[c])[:self.unseen_K]
                unseen_classes.append(int(c))
                unseen_query += self.indexes[c][sample_idxs].int().tolist()

            # classes for remain
            seen_class_idxs = rand_classes[self.seen_N:]
            for i, c in enumerate(self.classes[seen_class_idxs]):
                sample_idxs = torch.randperm(self.num_per_class[c])[:self.seen_K]
                seen_classes.append(int(c))
                seen_query += self.indexes[c][sample_idxs].int().tolist()

            yield unseen_classes, unseen_query, seen_classes, seen_query


if __name__ == '__main__':
    a = np.random.randint(0, 10, 200)

    print()
