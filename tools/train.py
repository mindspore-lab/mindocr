'''
Model training 

TODO:
    1. top-k model saving policy
    2. logging
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
from mindocr.postprocess import build_postprocess
from mindocr.metrics import build_metric
from mindocr.utils.model_wrapper import NetWithLossWrapper
from mindocr.utils.callbacks import EvalSaveCallback # TODO: callback in a better dir
from mindocr.utils.seed import set_seed

def main(cfg):
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
    is_main_device = rank_id in [None, 0]

    # train pipeline
    # dataset
    loader_train = build_dataset(
            cfg.train.dataset, 
            cfg.train.loader],
            num_shards=device_num,
            shard_id=rank_id,
            is_train=True)
    num_batches = loader_train.get_dataset_size()

    loader_eval = None
    # TODO: now only use device 0 to perform evaluation
    if cfg.system.val_while_train and is_main_device: 
        loader_eval = build_dataset(
                cfg.eval.dataset, 
                cfg.eval.loader,
                num_shards=None,
                shard_id=None,
                is_train=False)

    # model
    network = build_model(cfg.model)
    ms.amp.auto_mixed_precision(network, amp_level=cfg.system.amp_level)  

    # scheduler 
    lr_scheduler = create_scheduler(num_batches, **cfg['scheduler'])
    
    # optimizer
    optimizer = create_optimizer(network.trainable_params(), **cfg['optimizer'])
    
    # loss
    loss_fn = build_loss(cfg.loss.pop('name'), **cfg['loss'])
    
    # wrap train one step cell
    #net_with_loss = DBNetWithLossCell(network, loss_fn)
    net_with_loss = NetWithLossWrapper(network, loss_fn)

    loss_scale_manager = nn.FixedLossScaleUpdateCell(loss_scale_value=cfg.optimizer.loss_scale)
    train_net = nn.TrainOneStepWithLossScaleCell(net_with_loss,
                                                 optimizer=optimizer,
                                                 scale_sense=loss_scale_manager) 

    # postprocess, metric
    postprocessor = None
    if cfg.system.val_while_train:
        postprocessor = build_postprocess(cfg.postprocess)
        # postprocess network prediction
        metric = build_metric(cfg.metric) 

    # build callbacks
    eval_cb = EvalSaveCallback(
            network, 
            loader_eval, 
            postprocessor=postprocessor, 
            metrics=[metric], 
            rank_id=rank_id,
            ckpt_save_dir=cfg.system.ckpt_save_dir],
            main_indicator=cfg.metric.main_indicator)

    # log
    if is_main_device:
        print('='*40)
        print(
            f'Num devices: {device_num if device_num is not None else 1}\n'
            f'Num epochs: {cfg.scheduler.num_epochs}\n'
            f'Num batches: {num_batches}\n'
            f'Optimizer: {cfg.optimizer.opt}\n'
            f'Scheduler: {cfg.scheduler.scheduler}\n'
            f'LR: {cfg.scheduler.lr}\n'
                )
        if 'name' in cfg.model:
            print(f'Model: {cfg.model.name}')
        else:
            print(f'Model: {cfg.model.backbone.name}-{cfg.model.neck.name}-{cfg.model.head.name}')
        print('='*40)
        # save args used for training
        with open(os.path.join(cfg.system.ckpt_save_dir, 'args.yaml'), 'w') as f:
            args_text = yaml.safe_dump(cfg.to_dict(), default_flow_style=False)
            f.write(args_text)
    
    # training
    loss_monitor = LossMonitor(10) #num_batches // 10)
    time_monitor = TimeMonitor()

    model = ms.Model(train_net)
    model.train(cfg.scheduler.num_epochs, loader_train, callbacks=[loss_monitor, time_monitor, eval_cb],
                dataset_sink_mode=cfg.train.dataset_sink_mode)


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
