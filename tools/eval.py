"""
Model evaluation
"""
import logging
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

from addict import Dict

import mindspore as ms
from mindspore.communication import get_group_size, get_rank, init

from mindocr.data import build_dataset
from mindocr.metrics import build_metric
from mindocr.models import build_model
from mindocr.postprocess import build_postprocess
from mindocr.utils.evaluator import Evaluator
from mindocr.utils.logger import set_logger
from tools.arg_parser import parse_args_and_config

logger = logging.getLogger("mindocr.eval")


def main(cfg):
    # env init
    ms.set_context(mode=cfg.system.mode)
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
        )
        set_logger(
            name="mindocr", output_dir=cfg.train.ckpt_save_dir or "./", log_fn=f"log_eval_{rank_id}.txt", rank=rank_id
        )
    else:
        device_num = None
        rank_id = None
        set_logger(name="mindocr", output_dir=cfg.train.ckpt_save_dir or "./", log_fn=f"log_eval_{rank_id}.txt", rank=0)
        if "DEVICE_ID" in os.environ:
            logger.info(
                f"Standalone evaluation. Device id: {os.environ.get('DEVICE_ID')}, "
                f"specified by environment variable 'DEVICE_ID'."
            )
        else:
            device_id = cfg.system.get("device_id", 0)
            ms.set_context(device_id=device_id)
            logger.info(
                f"Standalone evaluation. Device id: {device_id}, "
                f"specified by system.device_id in yaml config file or is default value 0."
            )

    # load dataset
    loader_eval = build_dataset(
        cfg.eval.dataset,
        cfg.eval.loader,
        num_shards=device_num,
        shard_id=rank_id,
        is_train=False,
        refine_batch_size=True,
    )
    num_batches = loader_eval.get_dataset_size()

    # model
    cfg.model.backbone.pretrained = False
    amp_level = cfg.system.get("amp_level_infer", "O0")
    if ms.get_context("device_target") == "GPU" and amp_level == "O3":
        logger.warning(
            "Model evaluation does not support amp_level O3 on GPU currently. "
            "The program has switched to amp_level O2 automatically."
        )
        amp_level = "O2"
    network = build_model(cfg.model, ckpt_load_path=cfg.eval.ckpt_load_path, amp_level=amp_level)
    num_params = sum([param.size for param in network.get_parameters()])
    num_trainable_params = sum([param.size for param in network.trainable_params()])
    network.set_train(False)

    # postprocess, metric
    postprocessor = build_postprocess(cfg.postprocess)
    # postprocess network prediction
    metric = build_metric(cfg.metric)

    net_evaluator = Evaluator(
        network,
        loader_eval,
        loss_func=None,
        postprocessor=postprocessor,
        metrics=[metric],
        loader_output_columns=cfg.eval.dataset.output_columns,
        input_indices=cfg.eval.dataset.pop("net_input_column_index", None),
        label_indices=cfg.eval.dataset.pop("label_column_index", None),
        meta_data_indices=cfg.eval.dataset.pop("meta_data_column_index", None),
        num_epochs=1,
    )

    # log
    allow_postprocess_rescale = True
    if cfg.model.type == "det":
        if "shape_list" not in cfg.eval.dataset.output_columns:
            allow_postprocess_rescale = False
            logger.warning(
                "`shape_list` is NOT found in yaml config, which is used to rescale postprocessing result back to "
                "original image space for detection. Please add it to `eval.dataset.output_columns` for a fair "
                "evaluation. [CRITICAL!!!!!]"
            )

    model_name = (
        cfg.model.name
        if "name" in cfg.model
        else f"{cfg.model.backbone.name}-{cfg.model.neck.name}-{cfg.model.head.name}"
    )
    info_seg = "=" * 40
    det_spec = (
        f"Allow rescaling polygons for Det postprocess: {allow_postprocess_rescale}" if cfg.model.type == "det" else ""
    )
    rec_spec = (
        f"Character dict path: {cfg.common.character_dict_path}\nUse space char: {cfg.common.use_space_char}\n"
        f"Num classes: {cfg.common.num_classes}"
        if cfg.model.type == "rec"
        else ""
    )
    logger.info(
        f"\n{info_seg}\n"
        f"Model: {model_name}\n"
        f"Total number of parameters: {num_params}\n"
        f"Total number of trainable parameters: {num_trainable_params}\n"
        f"AMP level: {amp_level}\n"
        f"Num batches: {num_batches}\n"
        f"Batch size: {loader_eval.get_batch_size()}\n"
        f"{det_spec}{rec_spec}\n"
        f"{info_seg}\n"
        f"\nStart evaluating..."
    )

    measures = net_evaluator.eval()
    if rank_id in [None, 0]:
        logger.info(f"Performance: {measures}")


if __name__ == "__main__":
    args, config = parse_args_and_config()
    config = Dict(config)

    main(config)
