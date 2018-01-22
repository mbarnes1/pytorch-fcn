"""
Write some synthetic data to tensorboardX to test the visualizations
"""
import numpy as np
from tensorboardX import SummaryWriter
import torch


writer = SummaryWriter(log_dir='temp2')
d = 4
n = 20  # samples per cluster
cov = 0.05 * np.eye(d)
x_0 = np.random.multivariate_normal([1, 0, 0, 0], cov, n)
y_0 = np.zeros(n)
x_1 = np.random.multivariate_normal([0, 1, 0, 0], cov, n)
y_1 = np.ones(n)
x_2 = np.random.multivariate_normal([0, 0, 0, 1], cov, n)
y_2 = 2 * np.ones(n)
x = torch.FloatTensor(np.concatenate((x_0, x_1, x_2), axis=0))  # 3n x d
y = list(np.concatenate((y_0, y_1, y_2), axis=0))  # 3n

writer.add_embedding(x, metadata=y, global_step=2)