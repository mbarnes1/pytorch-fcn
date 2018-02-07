import datetime
import fcn
import math
import numpy as np
import os
import os.path as osp
import pytz
import random
import scipy.misc
import shutil
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchfcn
from torchfcn.utils import normalize_unit
from torch import Size

MAX_TENSORBOARD_EMBEDDINGS = 10000  # write at most this many pixels to tensorboard embedding


class RandomEdgeSampler(nn.Module):
    """
    Randomly sample nodes in graph and compute the edge adjacency matrix
    """
    def __init__(self, n_nodes):
        """
        :param n_nodes: Number of random nodes to sample from the graph.
        """
        super(RandomEdgeSampler, self).__init__()
        self._n_nodes = n_nodes

    def forward(self, input, target):
        """
        Compute an unbiased estimate of the MSE by randomly selecting nodes from the graph adjacency matrices.
        :param input: N x C x H x W Variable. Predicted eigenvectors, i.e. the embedding for every pixel.
        :param target: N x H x W Long Variable. Instance labels for every channel
        :return input_adjacency: N x n_nodes x n_nodes Variable. (same type as input)
        :return target_adjacency: N x n_nodes x n_nodes Byte Variable.
        """
        # Randomly sample nodes
        n, c, h, w = input.size()
        total_nodes_per_image = h * w
        if total_nodes_per_image >= self._n_nodes:
            n_nodes_this_iter = self._n_nodes
        else:
            n_nodes_this_iter = total_nodes_per_image
        random_indices = Variable(
            target.data.new(random.sample(xrange(total_nodes_per_image), n_nodes_this_iter * n)).view(n,
                                                                                                      n_nodes_this_iter))  # n x n_nodes

        # Compute adjacency matrices
        input_subsample = torch.gather(input.view(n, c, -1), 2,
                                       random_indices.unsqueeze(dim=1).expand(-1, c, -1))  # N x C x n_nodes
        target_subsample = torch.gather(target.view(n, -1), 1, random_indices)  # N x n_nodes

        input_adjacency = torch.bmm(input_subsample.transpose(1, 2), input_subsample)  # N x n_nodes x n_nodes
        target_adjacency = labels_to_adjacency(target_subsample.view(n, -1))
        return input_adjacency, target_adjacency


class ConfusionMatrix(nn.Module):
    """
    An unbiased estimator of the pairwise true positives, true negatives, false positives and false negatives.
    """
    def __init__(self, n_nodes):
        super(ConfusionMatrix, self).__init__()
        self._sampler = RandomEdgeSampler(n_nodes)

    def forward(self, input, target):
        """
        Compute an unbiased estimate of the MSE by randomly selecting nodes from the graph adjacency matrices.
        :param input: N x C x H x W Byte Variable. Hot one encoding.
        :param target: N x H x W Long Variable. Instance labels for every channel
        :return tp: True positives, float
        :return fp: False positives, float
        :return tn: True negatives, float
        :return fn: False negatives, float
        """
        # Randomly sample nodes
        input_adjacency, target_adjacency = self._sampler(input, target)
        input_adjacency, target_adjacency = input_adjacency.data, target_adjacency.data

        # Assert binary adjacency matrices
        assert isinstance(input_adjacency, torch.ByteTensor) or isinstance(input_adjacency, torch.cuda.ByteTensor)
        assert isinstance(target_adjacency, torch.ByteTensor) or isinstance(target_adjacency, torch.cuda.ByteTensor)

        # Compute metrics
        tp = float(torch.sum(input_adjacency[target_adjacency]))
        fp = float(torch.sum(input_adjacency[~target_adjacency]))
        tn = float(torch.sum(~input_adjacency[~target_adjacency]))
        fn = float(torch.sum(~input_adjacency[target_adjacency]))
        return tp, fp, tn, fn


class MSEAdjacencyLoss(nn.Module):
    """
    An unbiased estimator of the mean squared error between the true and predicted graph adjacency matrices.
    Loss is computed per edge.
    """
    def __init__(self, n_nodes):
        """
        :param n_nodes: Number of random nodes to sample from the graph.
        """
        super(MSEAdjacencyLoss, self).__init__()
        self._sampler = RandomEdgeSampler(n_nodes)
        self._mse = torch.nn.MSELoss()

    def forward(self, input, target):
        """
        Compute an unbiased estimate of the MSE by randomly selecting nodes from the graph adjacency matrices.
        :param input: N x C x H x W Float Variable. Predicted eigenvectors, i.e. the embedding for every pixel.
        :param target: N x H x W Long Variable. Instance labels for every channel
        :return loss: Float Variable
        """
        # Preprocess the input
        # input = F.softmax(input, dim=1)  # (optional) softmax
        input = normalize_unit(input, dim=1)  # (optional) normalize to unit vectors

        # Randomly sample nodes
        input_adjacency, target_adjacency = self._sampler(input, target)
        target_adjacency = target_adjacency.float()

        #off_diagonal_mask = ~torch.eye(self._n_nodes).byte().unsqueeze(dim=0).expand(n, -1, -1)
        #loss = self._mse(input_adjacency[off_diagonal_mask], target_adjacency[off_diagonal_mask])  # MSE per edge, excluding self edges
        #loss = torch.norm(input_adjacency - target_adjacency, p=2) / (self._n_nodes ** 2)  # Frobenius norm per edge
        loss = self._mse(input_adjacency, target_adjacency)  # MSE per edge
        return loss

        # TODO Other random sample strategies: Choose random edges


def labels_to_adjacency(labels):
    """
    :param labels: N x M LongTensor Variable, where N is the batch size and M is the number of nodes.
    :return adjacency: N x M x M ByteTensor Variable.
    """
    m = labels.size(1)
    labels = labels.unsqueeze(dim=1).expand(-1, m, -1)  # N x M x M matrix
    adjacency = (labels == labels.transpose(1, 2))
    return adjacency


def cross_entropy2d(input, target, weight=None, size_average=True):
    """
    Compute the cross-entropy loss on valid targets (i.e. entires >= 0, invalid labels have entry -1).
    :param input: 1 x C x H x W Variable with FloatTensor
    :param target: 1 x H x W Variable with LongTensor
    :param weight:
    :param size_average:
    :return:
    """
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    # log_p: (n, c, h, w)
    log_p = F.log_softmax(input, dim=1)
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= mask.data.sum()
    return loss


class Trainer(object):

    def __init__(self, cuda, model, optimizer,
                 train_loader, val_loader, out, max_iter,
                 size_average=False, interval_validate=None, tensorboard_writer=None, interval_train_loss=10,
                 n_class=21):
        """
        :param cuda:
        :param model:
        :param optimizer:
        :param train_loader:
        :param val_loader:
        :param out:
        :param max_iter:
        :param size_average:
        :param interval_validate: Validate, print and write to tensorboard every this many iterations.
        :param tensorboard_writer: TensorboardX SummaryWriter object
        :param interval_train_loss: Print train loss to Tensorboard every this many iterations.
        :param n_class: Embedding dimension, or number of semantic classes
        """
        self.cuda = cuda
        self.model = model
        self.out = out
        self.optim = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_iter = max_iter
        self.size_average = size_average
        if interval_validate is None:
            self.interval_validate = len(self.train_loader)
        else:
            self.interval_validate = interval_validate
        self._tensorboard_writer = tensorboard_writer
        self._interval_train_loss = interval_train_loss
        self._n_class = n_class

        self.timestamp_start = \
            datetime.datetime.now(pytz.timezone('Asia/Tokyo'))

        if not osp.exists(self.out):
            os.makedirs(self.out)

        self.log_headers = [
            'epoch',
            'iteration',
            'train/loss',
            'train/acc',
            'train/acc_cls',
            'train/mean_iu',
            'train/fwavacc',
            #'valid/loss_crossentropy',
            'valid/loss_mse',
            'valid/acc',
            'valid/acc_cls',
            'valid/mean_iu',
            'valid/fwavacc',
            'elapsed_time',
        ]
        if not osp.exists(osp.join(self.out, 'log.csv')):
            with open(osp.join(self.out, 'log.csv'), 'w') as f:
                f.write(','.join(self.log_headers) + '\n')

        self.epoch = 0
        self.iteration = 0
        self.best_mean_iu = 0

        self.mse_loss = MSEAdjacencyLoss(20000)

    def validate(self):
        """
        Writes a bunch of metrics to the log file and Tensorboard.
        """
        training = self.model.training
        self.model.eval()

        n_class = self._n_class  # len(self.val_loader.dataset.class_names)

        #val_loss_crossentropy = 0
        val_loss_mse = 0
        #visualizations = []
        label_trues, label_preds = [], []
        for batch_idx, (data, target) in enumerate(self.val_loader):
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            score = self.model(data)

            if np.isnan(score.data.cpu()).any():
                print score
                raise ValueError('Scores are NaN')
            if batch_idx == 0:
                n, d, h, w = score.size()
                assert n == 1
                first_image_scores = score.squeeze(dim=0).permute(1, 2, 0)  # H x W x D
                first_image_scores = first_image_scores.contiguous().view(-1, d)  # hw x d
                first_image_labels = target.squeeze(dim=0).contiguous().view(-1)  # hw
                if h*w > MAX_TENSORBOARD_EMBEDDINGS:
                    random_indices = Variable(first_image_labels.data.new(random.sample(xrange(h*w), MAX_TENSORBOARD_EMBEDDINGS)))
                    first_image_scores = first_image_scores[random_indices, :]
                    first_image_labels = first_image_labels[random_indices]
                self._tensorboard_writer.add_embedding(first_image_scores.data,
                                                       metadata=list(first_image_labels.data.cpu().numpy()),
                                                       global_step=self.iteration)

            #loss_crossentropy = cross_entropy2d(score, target, size_average=self.size_average)
            loss_mse = self.mse_loss(score, target)

            #if np.isnan(float(loss_crossentropy.data[0])):
            #    raise ValueError('Cross entropy loss is nan while validating')
            if np.isnan(float(loss_mse.data[0])):
                raise ValueError('MSE loss is nan while validating')

            #val_loss_crossentropy += float(loss_crossentropy.data[0])
            val_loss_mse += float(loss_mse.data[0])

            imgs = data.data.cpu()
            lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
            lbl_true = target.data.cpu()
            for img, lt, lp in zip(imgs, lbl_true, lbl_pred):
                img, lt = self.val_loader.dataset.untransform(img, lt)
                label_trues.append(lt)
                label_preds.append(lp)
                # if len(visualizations) < 9:
                #     viz = fcn.utils.visualize_segmentation(
                #         lbl_pred=lp, lbl_true=lt, img=img, n_class=n_class)
                #     visualizations.append(viz)
        metrics = torchfcn.utils.label_accuracy_score(
            label_trues, label_preds, n_class)

        # out = osp.join(self.out, 'visualization_viz')
        # if not osp.exists(out):
        #     os.makedirs(out)
        # out_file = osp.join(out, 'iter%012d.jpg' % self.iteration)
        # scipy.misc.imsave(out_file, fcn.utils.get_tile_image(visualizations))

        #val_loss_crossentropy /= len(self.val_loader)
        val_loss_mse /= len(self.val_loader)

        with open(osp.join(self.out, 'log.csv'), 'a') as f:
            elapsed_time = (
                datetime.datetime.now(pytz.timezone('Asia/Tokyo')) -
                self.timestamp_start).total_seconds()
            #log = [self.epoch, self.iteration] + [''] * 5 + \
            #      [val_loss_crossentropy] + [val_loss_mse] + list(metrics) + [elapsed_time]
            log = [self.epoch, self.iteration] + [''] * 5 + [val_loss_mse] + list(metrics) + [elapsed_time]
            log = map(str, log)
            f.write(','.join(log) + '\n')

        mean_iu = metrics[2]
        is_best = mean_iu > self.best_mean_iu
        if is_best:
            self.best_mean_iu = mean_iu
        torch.save({
            'epoch': self.epoch,
            'iteration': self.iteration,
            'arch': self.model.__class__.__name__,
            'optim_state_dict': self.optim.state_dict(),
            'model_state_dict': self.model.state_dict(),
            'best_mean_iu': self.best_mean_iu,
        }, osp.join(self.out, 'checkpoint.pth.tar'))
        if is_best:
            shutil.copy(osp.join(self.out, 'checkpoint.pth.tar'),
                        osp.join(self.out, 'model_best.pth.tar'))

        if training:
            self.model.train()
        val_acc = metrics[0]

        # Write outputs to Tensorboard
        if self._tensorboard_writer is not None:
            #self._tensorboard_writer.add_scalar('loss_crossentropy/validation', val_loss_crossentropy, self.iteration)
            self._tensorboard_writer.add_scalar('loss_mse/validation', val_loss_mse, self.iteration)
            self._tensorboard_writer.add_scalar('acc/validation', val_acc, self.iteration)
            self._tensorboard_writer.add_scalar('mean_iu/validation', mean_iu, self.iteration)

    def train_epoch(self):
        self.model.train()

        n_class = self._n_class  # len(self.train_loader.dataset.class_names)

        for batch_idx, (data, target) in enumerate(self.train_loader):
            iteration = batch_idx + self.epoch * len(self.train_loader)
            if self.iteration != 0 and (iteration - 1) != self.iteration:
                continue  # for resuming
            self.iteration = iteration

            if self.iteration % self.interval_validate == 0:
                print 'Epoch {}. Iteration {}. Validating...'.format(self.epoch, self.iteration)
                self.validate()

            assert self.model.training

            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            self.optim.zero_grad()
            score = self.model(data)
            if np.isnan(score.data.cpu()).any():
                print score
                raise ValueError('Scores are NaN')

            #loss_crossentropy = cross_entropy2d(score, target, size_average=self.size_average) / len(data)
            #if np.isnan(float(loss_crossentropy.data[0])):
            #    raise ValueError('Cross entropy loss is nan while training')

            loss_mse = self.mse_loss(score, target) / len(data)
            loss = loss_mse

            print 'Epoch {}. Iteration {}. Training loss {}'.format(self.epoch, self.iteration, loss.data[0])

            if np.isnan(float(loss_mse.data[0])):
                raise ValueError('MSE loss is nan while training')
            loss.backward()
            self.optim.step()

            metrics = []
            lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
            lbl_true = target.data.cpu().numpy()
            for lt, lp in zip(lbl_true, lbl_pred):
                acc, acc_cls, mean_iu, fwavacc = \
                    torchfcn.utils.label_accuracy_score(
                        [lt], [lp], n_class=n_class)
                metrics.append((acc, acc_cls, mean_iu, fwavacc))
            metrics = np.mean(metrics, axis=0)

            with open(osp.join(self.out, 'log.csv'), 'a') as f:
                elapsed_time = (
                    datetime.datetime.now(pytz.timezone('Asia/Tokyo')) -
                    self.timestamp_start).total_seconds()
                #log = [self.epoch, self.iteration] + [loss_crossentropy.data[0]] + \
                #      metrics.tolist() + [''] * 5 + [elapsed_time]  # [loss_mse.data[0]] + \ (in above line)
                log = [self.epoch, self.iteration] + metrics.tolist() + [''] * 5 + [elapsed_time]  # [loss_mse.data[0]] + \ (in above line)
                log = map(str, log)
                f.write(','.join(log) + '\n')

            # Write results to Tensorboard
            if self._tensorboard_writer is not None:
                if self.iteration % self._interval_train_loss == 0:
                    #self._tensorboard_writer.add_scalar('loss_crossentropy/train', loss_crossentropy.data[0], self.iteration)
                    self._tensorboard_writer.add_scalar('loss_mse/train', loss_mse.data[0], self.iteration)

                if self.iteration % self.interval_validate == 0:
                    # Network gradients and weights
                    for name, param in self.model.named_parameters():
                        self._tensorboard_writer.add_histogram(name + 'value', param.data.cpu().numpy(),
                                                               global_step=self.iteration)
                        self._tensorboard_writer.add_histogram(name + 'gradient', param.grad.data.cpu().numpy(),
                                                               global_step=self.iteration)

            if self.iteration >= self.max_iter:
                break

    def train(self):
        max_epoch = int(math.ceil(1. * self.max_iter / len(self.train_loader)))
        for epoch in xrange(self.epoch, max_epoch):
            self.epoch = epoch
            self.train_epoch()
            if self.iteration >= self.max_iter:
                break
