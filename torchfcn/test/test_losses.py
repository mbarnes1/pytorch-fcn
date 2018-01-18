import torch
from torch.autograd import Variable
from torchfcn.trainer import MSEAdjacencyLoss
import unittest


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self._n_sample = 10
        self.mse = MSEAdjacencyLoss(self._n_sample)

    def test_mse(self):
        N = 2  # batch size
        C = 3  # pixel embedding dimension
        H = 10  # image height
        W = 20  # image width

        # All one cluster vs. all different clusters has maximal mistake
        labels = range(0, H*W)  # all different clusters
        labels = Variable(torch.LongTensor(labels))
        labels = labels.view(H, W)
        labels = labels.unsqueeze(dim=0).expand(N, -1, -1).contiguous()

        pixel_embeddings = Variable(torch.FloatTensor(N, C, H, W).fill_(0.0))
        pixel_embeddings[:, 1, :, :] = 1.0
        loss = self.mse.forward(pixel_embeddings, labels)
        self.assertAlmostEqual(loss.data[0], 1.0 - 1.0 / self._n_sample)  # all edges are incorect, except for diagonal (self edges)

        # Same clusters should have zero loss
        labels = Variable(torch.LongTensor(N, H, W).fill_(2))  # all belong to cluster 2
        pixel_embeddings = Variable(torch.FloatTensor(N, C, H, W).fill_(0.0))
        pixel_embeddings[:, 1, :, :] = 1.0
        loss = self.mse.forward(pixel_embeddings, labels)
        self.assertAlmostEqual(loss.data[0], 0.0)