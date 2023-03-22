'''
Model training
'''
import sys
import os
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

from tools.arg_parser import parse_args
args = parse_args()

# modelarts
from tools.modelarts_adapter.modelarts import modelarts_setup
modelarts_setup(args) # setup env for modelarts platform if enabled

import yaml
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
from mindocr.utils.train_step_wrapper import TrainOneStepWrapper
from mindocr.utils.callbacks import EvalSaveCallback
from mindocr.utils.seed import set_seed


def main(cfg):
    # env init
    ms.set_context(mode=cfg.system.mode, device_id=4)
    if cfg.system.distribute:
        init()
        device_num = get_group_size()
        rank_id = get_rank()
        ms.set_auto_parallel_context(device_num=device_num,
                                     parallel_mode='data_parallel',
                                     gradients_mean=True,
                                     #parameter_broadcast=True,
                                     )
    else:
        device_num = None
        rank_id = None

    set_seed(cfg.system.seed)
    #cv2.setNumThreads(2) # TODO: by default, num threads = num cpu cores
    is_main_device = rank_id in [None, 0]

    # train pipeline
    # dataset
    loader_train = build_dataset(
            cfg.train.dataset,
            cfg.train.loader,
            num_shards=device_num,
            shard_id=rank_id,
            is_train=True,
            )
    num_batches = loader_train.get_dataset_size()

    loader_eval = None
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

    # optimizer and sheduler (from mindcv)
    lr_scheduler = create_scheduler(num_batches, **cfg['scheduler'])
    optimizer = create_optimizer(network.trainable_params(), lr=lr_scheduler, **cfg['optimizer'])
    loss_fn = build_loss(cfg.loss.pop('name'), **cfg['loss'])

    # wrap train-one-step cell
    net_with_loss = NetWithLossWrapper(network, loss_fn)

    loss_scale_manager = nn.FixedLossScaleUpdateCell(loss_scale_value=cfg.optimizer.loss_scale)

    train_net = TrainOneStepWrapper(net_with_loss,
                                    optimizer=optimizer,
                                    scale_sense=loss_scale_manager,
                                    drop_overflow_update=cfg.system.drop_overflow_update,
                                    verbose=True
                                    )
    # postprocess, metric
    postprocessor = None
    metric = None
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
            ckpt_save_dir=cfg.train.ckpt_save_dir,
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
            f'LR: {cfg.scheduler.lr} \n'
            f'drop_overflow_update: {cfg.system.drop_overflow_update}'
            )
        if 'name' in cfg.model:
            print(f'Model: {cfg.model.name}')
        else:
            print(f'Model: {cfg.model.backbone.name}-{cfg.model.neck.name}-{cfg.model.head.name}')
        print('='*40)
        # save args used for training
        with open(os.path.join(cfg.train.ckpt_save_dir, 'args.yaml'), 'w') as f:
            args_text = yaml.safe_dump(cfg.to_dict(), default_flow_style=False, sort_keys=False)
            f.write(args_text)

    # training
    loss_monitor = LossMonitor(min(num_batches // 10, 100))
    time_monitor = TimeMonitor()

    model = ms.Model(train_net)
    model.train(cfg.scheduler.num_epochs, loader_train, callbacks=[loss_monitor, time_monitor, eval_cb],
                dataset_sink_mode=cfg.train.dataset_sink_mode)


if __name__ == '__main__':
    yaml_fp = args.config
    with open(yaml_fp) as fp:
        config = yaml.safe_load(fp)
    config = Dict(config)

    if args.enable_modelarts:
        import moxing as mox
        from tools.modelarts_adapter.modelarts import get_device_id, sync_data
        dataset_root = '/cache/data/'
        # download dataset from server to local on device 0, other devices will wait until data sync finished.
        sync_data(args.data_url, dataset_root)
        if get_device_id() == 0:
            # mox.file.copy_parallel(src_url=args.data_url, dst_url=dataset_root)
            print(f'INFO: datasets found: {os.listdir(dataset_root)} \n'
                  f'INFO: dataset_root is changed to {dataset_root}'
                  )
        # update dataset root dir to cache
        assert 'dataset_root' in config['train']['dataset'], f'`dataset_root` must be provided in the yaml file for training on ModelArts or OpenI, but not found in {yaml_fp}. Please add `dataset_root` to `train:dataset` and `eval:dataset` in the yaml file'
        config.train.dataset.dataset_root = dataset_root
        config.eval.dataset.dataset_root = dataset_root

    # main train and eval
    main(config)

    if args.enable_modelarts:
        # upload models from local to server
        if get_device_id() == 0:
            mox.file.copy_parallel(src_url=config.train.ckpt_save_dir, dst_url=args.train_url)

