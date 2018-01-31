import torch
from torchfcn.cluster.correlation import kwik_cluster, l1_cost, l2_cost
import unittest


class MyTestCase(unittest.TestCase):
    def setUp(self):
        pass

    def test_perfect_singular_vectors(self):
        """
        Clustering should be perfect.
        """
        singular_vectors = torch.FloatTensor([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]])
        labels = kwik_cluster(singular_vectors, l1_cost)
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[2], labels[3])
        self.assertEqual(labels[4], labels[5])
        self.assertNotEqual(labels[1], labels[2])
        self.assertNotEqual(labels[3], labels[4])
        self.assertNotEqual(labels[1], labels[4])

    def test_l2_cost(self):
        x = torch.FloatTensor([1.0, 0.9, 0.5, 0.0])
        costs = l2_cost(x)
        self.assertTrue(torch.equal(costs, torch.FloatTensor([1.0, 0.81/0.82, 0.5, 0])))