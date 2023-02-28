'''
Runnable draft version

TODO:
    1. allow overwrite yaml config with args  
    1. model saving policy
    2. eval while training 
    ...
'''
import sys
sys.path.append('.')

import os
import yaml
import argparse
from addict import Dict

import cv2

import mindspore as ms
from mindspore import nn
from mindspore.communication import init, get_rank, get_group_size
from mindspore.train import LossMonitor, TimeMonitor
from mindcv.optim import create_optimizer
from mindcv.scheduler import create_scheduler

from mindocr.data import build_dataset
from mindocr.models import build_model
from mindocr.losses import build_loss
from mindocr.utils.model_wrapper import NetWithLossWrapper
from mindocr.utils.random import set_seed

def main(cfg):
    # TODO: cfg to easy dict
    # env init
    ms.set_context(mode=cfg.system.mode)
    if cfg.system.distribute:
        init()
        device_num = get_group_size()
        rank_id = get_rank()
        ms.set_auto_parallel_context(device_num=device_num,
                                     parallel_mode='data_parallel',
                                     gradients_mean=True)
    else:
        device_num = None
        rank_id = None
    
    set_seed(cfg.system.seed, rank_id)
    cv2.setNumThreads(2)

    # train pipeline
    # dataset
    loader_train = build_dataset(
            cfg['train']['dataset'], 
            cfg['train']['loader'],
            num_shards=device_num,
            shard_id=rank_id,
            is_train=True)
    num_batches = loader_train.get_dataset_size()
    
    # model
    network = build_model(cfg['model'])
    ms.amp.auto_mixed_precision(network, amp_level=cfg.system.amp_level)  

    # scheduler 
    lr_scheduler = create_scheduler(num_batches, **cfg['scheduler'])
    
    # optimizer
    optimizer = create_optimizer(network.trainable_params(), **cfg['optimizer'])
    
    # loss
    # TODO: input check for loss
    loss_fn = build_loss(cfg.loss.pop('name'), **cfg['loss'])
    
    # wrap train one step cell
    #net_with_loss = DBNetWithLossCell(network, loss_fn)
    net_with_loss = NetWithLossWrapper(network, loss_fn)

    loss_scale_manager = nn.FixedLossScaleUpdateCell(loss_scale_value=cfg.optimizer.loss_scale)
    train_net = nn.TrainOneStepWithLossScaleCell(net_with_loss,
                                                 optimizer=optimizer,
                                                 scale_sense=loss_scale_manager) 

    # log
    print('-'*30)
    print('Num batches: ', num_batches)
    print('-'*30)
    
    # training
    loss_monitor = LossMonitor(1) #(num_batches // 10)
    time_monitor = TimeMonitor()

    model = ms.Model(train_net)
    model.train(cfg.scheduler.num_epochs, loader_train, callbacks=[loss_monitor, time_monitor],
                dataset_sink_mode=config.train.dataset_sink_mode)


def parse_args():
    parser = argparse.ArgumentParser(description='Training Config', add_help=False)
    parser.add_argument('-c', '--config', type=str, default='',
                               help='YAML config file specifying default arguments (default='')')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    # argpaser
    # TODO: allow modify yaml config values by argparser
    args = parse_args()  
    yaml_fp = args.config
    with open(yaml_fp) as fp:
        config  = yaml.safe_load(fp)    
    config = Dict(config)
    
    # main train and eval
    main(config)
