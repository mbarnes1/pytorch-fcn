"""
Correlation Clustering methods
"""
import numpy as np
import random
import torch


def kwik_cluster(V, cost_function):
    """
    KwikCluster (Ailon2008) based on cosine similarity
    :param V:               N x D torch.FloatTensor (or cuda.FloatTensor) where N is number of pixels and D is the embedding dimension.
    :param cost_function:   Function that maps vector dot product (cosine similarity) to a cost for creating graph.
    :return labels:         N numpy array of pixel labels.
    """
    n, d = V.size()
    labels = V.new(n).long().fill_(-1)
    unlabeled_indices = np.array(xrange(0, n))
    counter = 0
    while unlabeled_indices.size > 0:  # until all samples are labeled
        pivot_index = random.sample(unlabeled_indices, 1)[0]
        pivot = V[pivot_index]
        cosine_similarities = torch.mv(V, pivot)  # N entries
        probability = cost_function(cosine_similarities)
        cluster = torch.ge(probability, probability.new(probability.size()).uniform_(0.0, 1.0))

        # If already in a cluster, do not reassign
        labeled_indices_mask = torch.ne(labels, -1)
        cluster[labeled_indices_mask] = 0

        labels[cluster] = counter
        counter += 1

        unlabeled_indices = torch.nonzero(torch.eq(labels, -1)).squeeze().cpu().numpy()
    return labels


def l1_cost(x):
    """
    :param x:  FloatTensor of vector dot products (i.e. 1 - cosine similarity)
    :return:   Cost of placing these two samples in different clusters. In [0, 1]
    """
    return x


def lp_cost(x, p=2):
    """
    :param x: FloatTensor of vector dot products (i.e. 1 - cosine similarity)
    :param p: Power to raise x
    :return:  Cost of placing these two samples in different clusters. In [0, 1]
    """
    num = torch.pow(x, p)
    den = num + torch.pow((1.0 - x), p)
    return torch.div(num, den)
