import torch
import torch.nn
import torch.nn.functional as F
from utils import cos_sim, euclidean_dist


def EucSoftmax(protos, querys, querys_y, t=1.0):
    """Calculate cos distance loss.

    Args:
        protos: protos vector in now episode (class_size, hidden_size)
        querys: queres vector to classify (querys_len, hidden_size)
        querys_y: corresponding y for query samples (querys_len, )
        t: (FloatTensor) temperature

    Returns:
        return loss, neg dist (N*M)
    """
    eu_dists = euclidean_dist(querys, protos)
    neg_eu_dists = -eu_dists
    loss_sfm = F.cross_entropy(neg_eu_dists, querys_y)

    return loss_sfm, neg_eu_dists


def CosT(protos, querys, querys_y, t=1.0):
    """Calculate cos distance loss.

    Args:
        protos: protos vector in now episode (class_size, hidden_size)
        querys: queres vector to classify (querys_len, hidden_size)
        querys_y: corresponding y for query samples (querys_len, )
        t: (FloatTensor) temperature

    Returns:
        return loss, neg dist (N*M)
    """
    cos = cos_sim(querys, protos)
    t_cos = t * cos
    loss_cos = F.cross_entropy(t_cos, querys_y)
    return loss_cos, cos


def EucTriplet(protos, querys, querys_y, t=1.0):
    """Calculate cos distance loss.

    Args:
        protos: protos vector in now episode (class_size, hidden_size)
        querys: queres vector to classify (querys_len, hidden_size)
        querys_y: corresponding y for query samples (querys_len, )
        t: (FloatTensor) temperature

    Returns:
        return loss, neg dist (N*M)
    """
    device = protos.device
    querys_size = querys.size(0)
    a = torch.arange(querys_size).to(device)

    eu_dists = euclidean_dist(querys, protos)
    neg_eu_dists = -eu_dists

    masked_dists = eu_dists.clone()
    masked_dists[a, querys_y] = float('inf')

    neg_samples_dists = masked_dists.min(dim=-1)[0]
    pos_samples_dists = eu_dists[a, querys_y]

    loss_triplet = F.relu(pos_samples_dists - neg_samples_dists + 1.0).mean()
    return loss_triplet, neg_eu_dists
