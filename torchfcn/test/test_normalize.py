import numpy as np
import torch
from torch.autograd import Variable
from torchfcn.utils import normalize_unit
import unittest


class MyTestCase(unittest.TestCase):
    def setUp(self):
        pass

    def test_normalize_unit(self):
        x = Variable(torch.rand(2, 5, 10))

        x_unit = normalize_unit(x, dim=2)
        np.testing.assert_array_almost_equal(torch.norm(x_unit, 2, 2, keepdim=True).data.numpy(), np.ones((2, 5, 1)))
        inner_product = torch.bmm(x_unit, x_unit.transpose(1, 2))
        for idx in range(0, x.size(0)):
            inner_product_diag = torch.diag(inner_product[idx, :, :].squeeze())
            np.testing.assert_array_almost_equal(inner_product_diag.data.numpy(), np.ones(5))

        x_unit = normalize_unit(x, dim=1)
        np.testing.assert_array_almost_equal(torch.norm(x_unit, 2, 1, keepdim=True).data.numpy(), np.ones((2, 1, 10)))
        inner_product = torch.bmm(x_unit.transpose(1, 2), x_unit)
        for idx in range(0, x.size(0)):
            inner_product_diag = torch.diag(inner_product[idx, :, :].squeeze())
            np.testing.assert_array_almost_equal(inner_product_diag.data.numpy(), np.ones(10))

    def test_normalize_unit_already_normalized(self):
        x_unit = Variable(torch.zeros(2, 5, 10).float())
        x_unit[:, :, 2] = 1.0
        x_renormalized = normalize_unit(x_unit, dim=2)
        np.testing.assert_array_almost_equal(x_unit.data.numpy(), x_renormalized.data.numpy())

    def test_normalize_assertions(self):
        x = Variable(torch.zeros(2, 5, 10).float())
        self.assertRaises(AssertionError, normalize_unit, x)