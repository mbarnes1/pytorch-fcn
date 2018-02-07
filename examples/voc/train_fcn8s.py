#!/usr/bin/env python

import argparse
from datetime import datetime
import os
import os.path as osp
import socket
import yaml

import torch

import torchfcn

from train_fcn32s import get_parameters, git_hash


configurations = {
    # same configuration as original work
    # https://github.com/shelhamer/fcn.berkeleyvision.org
    1: dict(
        max_iteration=100000,
        lr=1.0e-14,
        momentum=0.99,
        weight_decay=0.0005,
        interval_validate=4000,
        fcn16s_pretrained_model=torchfcn.models.FCN16s.download(),
        init_gain=1.0e-4,  # used for initiailizing network weights
        num_classes=21  # embedding dimension. must be 21 for semantic segmentation,
    )
}


def get_log_dir(model_name, config_id, cfg):
    name = '004_{}'.format(datetime.now().strftime('%b%d-%H:%M:%S'))
    #name += '_xe_norm-mse'
    name += '_instance_mse_FCN8_lr1e-5_xavier1e-4-norm_D21'
    name += '_VCS-%s' % git_hash()
    name += '_{}'.format(socket.gethostname().split('.')[0])

    # create out
    log_dir = osp.join(here, 'logs', name)
    if not osp.exists(log_dir):
        os.makedirs(log_dir)
    with open(osp.join(log_dir, 'config.yaml'), 'w') as f:
        yaml.safe_dump(cfg, f, default_flow_style=False)
    return log_dir


here = osp.dirname(osp.abspath(__file__))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=int, required=True)
    parser.add_argument('-c', '--config', type=int, default=1,
                        choices=configurations.keys())
    parser.add_argument('--instance',
                        action='store_true',
                        help='Use instance labels, else use class labels.')
    parser.add_argument('--resume', help='Checkpoint path')
    args = parser.parse_args()

    gpu = args.gpu
    cfg = configurations[args.config]
    out = get_log_dir('fcn8s', args.config, cfg)
    print 'Running experiment {}'.format(out)
    resume = args.resume

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    cuda = torch.cuda.is_available()

    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)

    # 1. dataset

    root = osp.expanduser('~/data/datasets')
    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
    if args.instance:
        print 'Beginning instance segmentation.'
        train_dataset = torchfcn.datasets.SBDInstSeg
        val_dataset = torchfcn.datasets.VOC2011InstSeg
    else:
        print 'Beginning semantic segmentation.'
        train_dataset = torchfcn.datasets.SBDClassSeg
        val_dataset = torchfcn.datasets.VOC2011ClassSeg
    train_loader = torch.utils.data.DataLoader(
        train_dataset(root, split='train', transform=True),
        batch_size=1, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        val_dataset(root, split='seg11valid', transform=True),
        batch_size=1, shuffle=False, **kwargs)

    # 2. model

    model = torchfcn.models.FCN8s(n_class=cfg['num_classes'])
    start_epoch = 0
    start_iteration = 0
    if resume:
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        start_iteration = checkpoint['iteration']
    else:
        fcn16s = torchfcn.models.FCN16s()
        fcn16s.load_state_dict(torch.load(cfg['fcn16s_pretrained_model']))
        model.copy_params_from_fcn16s(fcn16s)
    if cuda:
        model = model.cuda()

    # 3. optimizer

    optim = torch.optim.SGD(
        [
            {'params': get_parameters(model, bias=False)},
            {'params': get_parameters(model, bias=True),
             'lr': cfg['lr'] * 2, 'weight_decay': 0},
        ],
        lr=cfg['lr'],
        momentum=cfg['momentum'],
        weight_decay=cfg['weight_decay'])
    if resume:
        optim.load_state_dict(checkpoint['optim_state_dict'])

    trainer = torchfcn.Trainer(
        cuda=cuda,
        model=model,
        optimizer=optim,
        train_loader=train_loader,
        val_loader=val_loader,
        out=out,
        max_iter=cfg['max_iteration'],
        interval_validate=cfg.get('interval_validate', len(train_loader)),
        n_class=cfg['num_classes']
    )
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    trainer.train()


if __name__ == '__main__':
    main()
