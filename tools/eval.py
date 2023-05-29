"""
Model evaluation
"""
import sys
import os

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

from tools.arg_parser import parse_args_and_config
import yaml
from addict import Dict

import mindspore as ms
from mindspore.communication import init, get_rank, get_group_size

from mindocr.data import build_dataset
from mindocr.models import build_model
from mindocr.postprocess import build_postprocess
from mindocr.metrics import build_metric
from mindocr.utils.evaluator import Evaluator
from mindocr.utils.logger import get_logger


def main(cfg):
    # env init
    ms.set_context(mode=cfg.system.mode)
    if cfg.system.distribute:
        init()
        device_num = get_group_size()
        rank_id = get_rank()
        ms.set_auto_parallel_context(
            device_num=device_num,
            parallel_mode="data_parallel",
            gradients_mean=True,
        )
    else:
        device_num = None
        rank_id = None

    is_main_device = rank_id in [None, 0]

    logger = get_logger(
        log_dir=cfg.train.ckpt_save_dir or "./",
        rank=rank_id,
        log_fn=f"log_eval_{rank_id}.txt",
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
    network = build_model(cfg.model, ckpt_load_path=cfg.eval.ckpt_load_path)
    network.set_train(False)

    amp_level = "O0"
    if cfg.system.amp_level_infer in ["O1", "O2", "O3"]:
        ms.amp.auto_mixed_precision(network, amp_level=cfg.system.amp_level_infer)
        amp_level = cfg.system.amp_level_infer

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
                "`shape_list` is NOT found in yaml config, which is used to rescale postprocessing result back to orginal image space for detection. Please add it to `eval.dataset.output_columns` for a fair evaluation. [CRITICAL!!!!!]"
            )

    model_name = (
        cfg.model.name
        if "name" in cfg.model
        else f"{cfg.model.backbone.name}-{cfg.model.neck.name}-{cfg.model.head.name}"
    )
    info_seg = "=" * 40
    det_spec = (
        f"Allow rescaling polygons for Det postprocess: {allow_postprocess_rescale}"
        if cfg.model.type == "det"
        else ""
    )
    rec_spec = (
        f"Character dict path: {cfg.common.character_dict_path}\nUse space char: {cfg.common.use_space_char}\nNum classes: {cfg.common.num_classes}"
        if cfg.model.type == "rec"
        else ""
    )
    logger.info(
        f"\n{info_seg}\n"
        f"Model: {model_name}\n"
        f"AMP level: {amp_level}\n"
        f"Num batches: {num_batches}\n"
        f"Batch size: {cfg.eval.loader.batch_size}\n"
        f"{det_spec}{rec_spec}\n"
        f"{info_seg}\n"
        f"\nStart evaluating..."
    )

    measures = net_evaluator.eval()
    if is_main_device:
        logger.info(f"Performance: {measures}")


if __name__ == "__main__":
    # argpaser
    args, config = parse_args_and_config()
    config = Dict(config)

    main(config)
