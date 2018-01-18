import numpy as np
import torch


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.

      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc


def normalize_unit(scores, dim=1):
    """ Normalize scores such that vectors along dimension have unit l2 norm.

    :param scores: Tensor
    :param dim: Dimension to normalize along
    :return normalized_scores: Tensor with same shape as scores, with all vectors along dimension having unit l2 norm.
    """
    l2_norms = torch.norm(scores, 2, dim, keepdim=True)
    assert (l2_norms > 0).all()
    normalized_scores = scores / l2_norms
    assert ~np.isnan(normalized_scores.data.cpu().numpy()).any()
    return normalized_scores
