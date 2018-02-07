import torch
from torch.autograd import Variable
from torchfcn.trainer import MSEAdjacencyLoss, ConfusionMatrix
import unittest


class MyTestCase(unittest.TestCase):
    def setUp(self):
        pass

    def test_mse(self):
        n_sample = 10
        mse = MSEAdjacencyLoss(n_sample)

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
        loss = mse.forward(pixel_embeddings, labels)
        self.assertAlmostEqual(loss.data[0], 1.0 - 1.0 / n_sample)  # all edges are incorect, except for diagonal (self edges)

        # Same clusters should have zero loss
        labels = Variable(torch.LongTensor(N, H, W).fill_(2))  # all belong to cluster 2
        pixel_embeddings = Variable(torch.FloatTensor(N, C, H, W).fill_(0.0))
        pixel_embeddings[:, 1, :, :] = 1.0
        loss = mse.forward(pixel_embeddings, labels)
        self.assertAlmostEqual(loss.data[0], 0.0)

    def test_acc_prec_recall(self):
        N = 1  # batch size
        C = 3  # pixel embedding dimension
        H = 3  # image height
        W = 5  # image width
        confusion = ConfusionMatrix(H * W)

        labels = Variable(torch.LongTensor(H, W))
        labels = labels.view(H, W)
        labels[0, :] = 0  # top row is label 0
        labels[1, :] = 1  # top row is label 1
        labels[2, :] = 2  # top row is label 2
        labels = labels.unsqueeze(dim=0).expand(N, -1, -1).contiguous()

        # Predict all positive
        pixel_embeddings = Variable(torch.ByteTensor(N, C, H, W).fill_(0))
        pixel_embeddings[:, 1, :, :] = 1
        tp, fp, tn, fn = confusion(pixel_embeddings, labels)
        p = tp + fn
        n = tn + fp
        acc = (tp + tn) / (p + n)
        prec = tp / (tp + fp)
        recall = tp / p
        self.assertEqual(tp, float(H * W**2))
        self.assertEqual(fp, H * W * (2 * W))
        self.assertEqual(tn, 0)
        self.assertEqual(fn, 0)
        self.assertEqual(acc, float(H * W**2) / (H*W)**2)  # only get within-cluster edges correct
        self.assertEqual(prec, float(H * W ** 2) / (H * W) ** 2)  # predicted all positive, so same as accuracy
        self.assertEqual(recall, 1.0)  # predicted all positive, so same as accuracy

