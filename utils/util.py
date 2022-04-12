import json
import pandas as pd
import torch
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
import numpy as np


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    """wrapper function for endless data loader."""
    for loader in repeat(data_loader):
        yield from loader


class MetricTracker:
    """Metric Tracker Class for Training and validation."""
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()
        
    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]
    
    def result(self):
        return dict(self._data.average)


def euclidean_dist(x, y):
    """Compute euclidean distance between two tensors

    Args:
        x: (Tensor) N x D
        y: (Tensor) M x D

    Returns:
        Euclidean distance of x and y, a float
    """
    flag = False
    if y is None:
        y = x
        flag = True
    if len(list(x.size())) == len(list(y.size())) == 1:
        return torch.pow(x - y, 2).sum()
    x_norm = (x ** 2).sum(1).view(-1, 1)
    y_t = torch.transpose(y, 0, 1)
    y_norm = (y ** 2).sum(1).view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    if flag:
        dist = dist - torch.diag(dist.diag())
    return torch.clamp(dist, 0.0, np.inf)


def cos_sim(x, y=None, eps=1e-8):
    """Cosine Similarity calculate. Add eps for numerical stability"""
    y = x if y is None else y
    if len(list(x.size())) == len(list(y.size())) == 1:
        return torch.nn.CosineSimilarity(dim=-1, eps=eps)(x, y)
    x_n, y_n = x.norm(dim=1)[:, None], y.norm(dim=1)[:, None]
    x_norm = x / torch.max(x_n, eps * torch.ones_like(x_n))
    y_norm = y / torch.max(y_n, eps * torch.ones_like(y_n))
    sim_mt = torch.mm(x_norm, y_norm.transpose(0, 1))
    return sim_mt


def get_samples(samples, device, type):
    """

    Args:
        samples: (List) list of sample
        device: (str) CUDA device
        type: (str) bert or lstm. If encoder is bert, return raw text else ids.

    Returns:
        x, x_length, y which means text(raw/ids), word length, label
    """
    if type == 'bert':
        x = [sample['text'] for sample in samples]
    else:
        x = torch.LongTensor([sample['pad_ids'] for sample in samples]).to(device)
    len = torch.IntTensor([sample['length'] for sample in samples]).to(device)
    y = torch.LongTensor([sample['y'] for sample in samples]).to(device)
    return x, len, y


def log_dict(logger, log):
    """Print result dict in format."""
    for key, value in log.items():
        if isinstance(value, int):
            logger.info('    {:30s}: {}'.format(str(key), value))
        else:
            logger.info('    {:30s}: {:.6}'.format(str(key), value))
        if 'f1' in key:
            logger.info('    {:30s}: '.format('-----------'))


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """Kernel matrix."""
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)

    total0 = total.unsqueeze(0).expand(int(total.size(0)), \
                                       int(total.size(0)), \
                                       int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), \
                                       int(total.size(0)), \
                                       int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for \
                  bandwidth_temp in bandwidth_list]

    return sum(kernel_val)


def mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """Maximum Mean Discrepancy."""
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul,
                              kernel_num=kernel_num,
                              fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]  # Source<->Source
    YY = kernels[batch_size:, batch_size:]  # Target<->Target
    XY = kernels[:batch_size, batch_size:]  # Source<->Target
    YX = kernels[batch_size:, :batch_size]  # Target<->Source
    mmd = XX.mean() - XY.mean() - YX.mean() + YY.mean()
    return mmd


def cos_mmd(x, y):
    """Cosine similarity MMD(Maximum Mean Discrepancy in transfer learning)."""
    return cos_sim(x).mean() - 2 * cos_sim(x, y).mean() + cos_sim(y).mean()