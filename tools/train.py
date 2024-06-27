"""
Model training
"""
import logging
import os
import shutil
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

from tools.arg_parser import parse_args_and_config

args, config = parse_args_and_config()

import yaml
from addict import Dict

import mindspore as ms
from mindspore.communication import get_group_size, get_rank, init

from mindocr.data import build_dataset
from mindocr.losses import build_loss
from mindocr.metrics import build_metric
from mindocr.models import build_model
from mindocr.optim import create_group_params, create_optimizer
from mindocr.postprocess import build_postprocess
from mindocr.scheduler import create_scheduler
from mindocr.utils.callbacks import EvalSaveCallback
from mindocr.utils.checkpoint import resume_train_network
from mindocr.utils.ema import EMA
from mindocr.utils.logger import set_logger
from mindocr.utils.loss_scaler import get_loss_scales
from mindocr.utils.model_wrapper import NetWithLossWrapper
from mindocr.utils.seed import set_seed
from mindocr.utils.train_step_wrapper import TrainOneStepWrapper

logger = logging.getLogger("mindocr.train")


def main(cfg):
    # init env
    ms.set_context(mode=cfg.system.mode)
    if cfg.train.max_call_depth:
        ms.set_context(max_call_depth=cfg.train.max_call_depth)
    if cfg.system.mode == 0:
        ms.set_context(jit_config={"jit_level": "O2"})
    if cfg.system.distribute:
        init()
        device_num = get_group_size()
        rank_id = get_rank()
        ms.set_auto_parallel_context(
            device_num=device_num,
            parallel_mode="data_parallel",
            gradients_mean=True,
            # parameter_broadcast=True,
        )
        # create logger, only rank0 log will be output to the screen
        set_logger(
            name="mindocr",
            output_dir=cfg.train.ckpt_save_dir,
            rank=rank_id,
            log_level=eval(cfg.system.get("log_level", "logging.INFO")),
        )
    else:
        device_num = None
        rank_id = None

        # create logger, only rank0 log will be output to the screen
        set_logger(
            name="mindocr",
            output_dir=cfg.train.ckpt_save_dir,
            rank=0,
            log_level=eval(cfg.system.get("log_level", "logging.INFO")),
        )
        if "DEVICE_ID" in os.environ:
            logger.info(
                f"Standalone training. Device id: {os.environ.get('DEVICE_ID')}, "
                f"specified by environment variable 'DEVICE_ID'."
            )
        else:
            device_id = cfg.system.get("device_id", 0)
            ms.set_context(device_id=device_id)
            logger.info(
                f"Standalone training. Device id: {device_id}, "
                f"specified by system.device_id in yaml config file or is default value 0."
            )

    set_seed(cfg.system.seed)

    # create dataset
    loader_train = build_dataset(
        cfg.train.dataset,
        cfg.train.loader,
        num_shards=device_num,
        shard_id=rank_id,
        is_train=True,
    )
    num_batches = loader_train.get_dataset_size()

    loader_eval = None
    if cfg.system.val_while_train:
        loader_eval = build_dataset(
            cfg.eval.dataset,
            cfg.eval.loader,
            num_shards=device_num,
            shard_id=rank_id,
            is_train=False,
            refine_batch_size=cfg.system.get("refine_batch_size", True),
        )

    # create model
    amp_level = cfg.system.get("amp_level", "O0")
    if ms.get_context("device_target") == "GPU" and cfg.system.val_while_train and amp_level == "O3":
        logger.warning(
            "Model evaluation does not support amp_level O3 on GPU currently. "
            "The program has switched to amp_level O2 automatically."
        )
        amp_level = "O2"
    network = build_model(cfg.model, ckpt_load_path=cfg.model.pop("pretrained", None), amp_level=amp_level)
    num_params = sum([param.size for param in network.get_parameters()])
    num_trainable_params = sum([param.size for param in network.trainable_params()])

    # create loss
    loss_fn = build_loss(cfg.loss.pop("name"), **cfg["loss"])

    net_with_loss = NetWithLossWrapper(
        network,
        loss_fn,
        input_indices=cfg.train.dataset.pop("net_input_column_index", None),
        label_indices=cfg.train.dataset.pop("label_column_index", None),
        pred_cast_fp32=cfg.train.pop("pred_cast_fp32", amp_level != "O0"),
    )  # wrap train-one-step cell

    # get loss scale setting for mixed precision training
    loss_scale_manager, optimizer_loss_scale = get_loss_scales(cfg)

    # build lr scheduler
    lr_scheduler = create_scheduler(num_batches, **cfg["scheduler"])

    # build optimizer
    cfg.optimizer.update({"lr": lr_scheduler, "loss_scale": optimizer_loss_scale})
    params = create_group_params(network.trainable_params(), **cfg.optimizer)
    optimizer = create_optimizer(params, **cfg.optimizer)

    # resume ckpt
    start_epoch = 0
    if cfg.model.resume:
        resume_ckpt = (
            os.path.join(cfg.train.ckpt_save_dir, "train_resume.ckpt")
            if isinstance(cfg.model.resume, bool)
            else cfg.model.resume
        )
        start_epoch, loss_scale, cur_iter, last_overflow_iter = resume_train_network(network, optimizer, resume_ckpt)
        loss_scale_manager.loss_scale_value = loss_scale
        if cfg.loss_scaler.type == "dynamic":
            loss_scale_manager.cur_iter = cur_iter
            loss_scale_manager.last_overflow_iter = last_overflow_iter

    # build train step cell
    gradient_accumulation_steps = cfg.train.get("gradient_accumulation_steps", 1)
    clip_grad = cfg.train.get("clip_grad", False)
    use_ema = cfg.train.get("ema", False)
    ema = EMA(network, ema_decay=cfg.train.get("ema_decay", 0.9999), updates=0) if use_ema else None

    train_net = TrainOneStepWrapper(
        net_with_loss,
        optimizer=optimizer,
        scale_sense=loss_scale_manager,
        drop_overflow_update=cfg.system.drop_overflow_update,
        gradient_accumulation_steps=gradient_accumulation_steps,
        clip_grad=clip_grad,
        clip_norm=cfg.train.get("clip_norm", 1.0),
        ema=ema,
    )

    # build postprocess and metric
    postprocessor = None
    metric = None
    if cfg.system.val_while_train:
        # postprocess network prediction
        postprocessor = build_postprocess(cfg.postprocess)
        metric = build_metric(cfg.metric, device_num=device_num, save_dir=cfg.train.ckpt_save_dir)

    # build callbacks
    eval_cb = EvalSaveCallback(
        network,
        loader_eval,
        postprocessor=postprocessor,
        metrics=[metric],
        pred_cast_fp32=cfg.eval.pop("pred_cast_fp32", amp_level != "O0"),
        rank_id=rank_id,
        device_num=device_num,
        batch_size=cfg.train.loader.batch_size,
        ckpt_save_dir=cfg.train.ckpt_save_dir,
        main_indicator=cfg.metric.main_indicator,
        ema=ema,
        loader_output_columns=cfg.eval.dataset.output_columns,
        input_indices=cfg.eval.dataset.pop("net_input_column_index", None),
        label_indices=cfg.eval.dataset.pop("label_column_index", None),
        meta_data_indices=cfg.eval.dataset.pop("meta_data_column_index", None),
        val_interval=cfg.system.get("val_interval", 1),
        val_start_epoch=cfg.system.get("val_start_epoch", 1),
        log_interval=cfg.system.get("log_interval", 100),
        ckpt_save_policy=cfg.system.get("ckpt_save_policy", "top_k"),
        ckpt_max_keep=cfg.system.get("ckpt_max_keep", 10),
        start_epoch=start_epoch,
    )

    # save args used for training
    if rank_id in [None, 0]:
        with open(os.path.join(cfg.train.ckpt_save_dir, "args.yaml"), "w") as f:
            yaml.safe_dump(cfg.to_dict(), stream=f, default_flow_style=False, sort_keys=False)

    # log
    num_devices = device_num if device_num is not None else 1
    global_batch_size = cfg.train.loader.batch_size * num_devices * gradient_accumulation_steps
    model_name = (
        cfg.model.name
        if "name" in cfg.model
        else f"{cfg.model.backbone.name}-{cfg.model.neck.name}-{cfg.model.head.name}"
    )
    info_seg = "=" * 40
    logger.info(
        f"\n{info_seg}\n"
        f"Distribute: {cfg.system.distribute}\n"
        f"Model: {model_name}\n"
        f"Total number of parameters: {num_params}\n"
        f"Total number of trainable parameters: {num_trainable_params}\n"
        f"Data root: {cfg.train.dataset.dataset_root}\n"
        f"Optimizer: {cfg.optimizer.opt}\n"
        f"Weight decay: {cfg.optimizer.weight_decay} \n"
        f"Batch size: {cfg.train.loader.batch_size}\n"
        f"Num devices: {num_devices}\n"
        f"Gradient accumulation steps: {gradient_accumulation_steps}\n"
        f"Global batch size: {cfg.train.loader.batch_size}x{num_devices}x{gradient_accumulation_steps}="
        f"{global_batch_size}\n"
        f"LR: {cfg.scheduler.lr} \n"
        f"Scheduler: {cfg.scheduler.scheduler}\n"
        f"Steps per epoch: {num_batches}\n"
        f"Num epochs: {cfg.scheduler.num_epochs}\n"
        f"Clip gradient: {clip_grad}\n"
        f"EMA: {use_ema}\n"
        f"AMP level: {amp_level}\n"
        f"Loss scaler: {cfg.loss_scaler}\n"
        f"Drop overflow update: {cfg.system.drop_overflow_update}\n"
        f"{info_seg}\n"
        f"\nStart training... (The first epoch takes longer, please wait...)\n"
    )

    # training
    model = ms.Model(train_net)
    model.train(
        cfg.scheduler.num_epochs,
        loader_train,
        callbacks=[eval_cb],
        dataset_sink_mode=cfg.train.dataset_sink_mode,
        initial_epoch=start_epoch,
    )


if __name__ == "__main__":
    # load and archive yaml config
    config = Dict(config)

    ckpt_save_dir = config.train.ckpt_save_dir
    os.makedirs(ckpt_save_dir, exist_ok=True)
    shutil.copyfile(args.config, os.path.join(ckpt_save_dir, "train_config.yaml"))

    # data sync for modelarts
    if args.enable_modelarts:
        from ast import literal_eval

        import moxing as mox

        from tools.modelarts_adapter.modelarts import (
            download_ckpt,
            get_device_id,
            sync_data,
            update_config_value_by_key,
        )

        dataset_root = "/cache/data/"
        # download dataset from server to local on device 0, other devices will wait until data sync finished.
        if args.multi_data_url:
            multi_data_url = literal_eval(args.multi_data_url)
            for x in multi_data_url:
                sync_data(x["dataset_url"], dataset_root)
        else:
            sync_data(args.data_url, dataset_root)

        if get_device_id() == 0:
            # mox.file.copy_parallel(src_url=args.data_url, dst_url=dataset_root)
            print(
                f"INFO: datasets found: {os.listdir(dataset_root)} \n"
                f"INFO: dataset_root is changed to {dataset_root}"
            )
        # update dataset root dir to cache
        assert "dataset_root" in config["train"]["dataset"], (
            f"`dataset_root` must be provided in the yaml file for training on ModelArts or OpenI, but not found in "
            f"{args.config}. Please add `dataset_root` to `train:dataset` and `eval:dataset` in the yaml file"
        )
        config.train.dataset.dataset_root = dataset_root
        config.eval.dataset.dataset_root = dataset_root

        cur_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.abspath(os.path.join(cur_dir, ".."))

        # update character_dict_path if it exists
        if config.common.character_dict_path:
            new_dict_path = os.path.join(root_dir, config.common.character_dict_path)
            update_config_value_by_key(config, "character_dict_path", new_dict_path)

        # download the pretrained checkpoint if it exists
        if args.pretrain_url:
            ckpt_root = "/cache/ckpt/"
            pretrain_url = literal_eval(args.pretrain_url)
            download_ckpt(pretrain_url[0]["model_url"], ckpt_root)
            config.model.pretrained = os.path.join(ckpt_root, os.path.basename(pretrain_url[0]["model_url"]))

    # main train and eval
    main(config)

    # model sync for modelarts
    if args.enable_modelarts:
        # upload models from local to server
        if get_device_id() == 0:
            mox.file.copy_parallel(src_url=config.train.ckpt_save_dir, dst_url=args.train_url)
