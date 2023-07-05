import logging
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../../../../")))

import argparse

import yaml
from addict import Dict
from tqdm import tqdm

import mindspore as ms

from mindocr.data import build_dataset
from mindocr.models import build_model
from mindocr.postprocess import build_postprocess
from mindocr.utils.logger import set_logger

logger = logging.getLogger("mindocr")


class BasePredict(object):
    def __init__(self, predict_cfg):
        # env init
        ms.set_context(mode=predict_cfg.system.mode)
        set_logger(name="mindocr")
        if predict_cfg.system.distribute:
            logger.warning("Distribut mode blocked. Evaluation only runs in standalone mode.")

        self.loader_predict = build_dataset(
            predict_cfg.eval.dataset, predict_cfg.eval.loader, num_shards=None, shard_id=None, is_train=False
        )

        self.img_path_column_idx = predict_cfg.eval.dataset.output_columns.index("img_path")
        self.image_column_idx = predict_cfg.eval.dataset.output_columns.index("image")
        if "shape_list" in predict_cfg.eval.dataset.output_columns:
            self.shape_list_column_idx = predict_cfg.eval.dataset.output_columns.index("shape_list")
        else:
            self.shape_list_column_idx = None
        self.loader_predict.get_dataset_size()

        # model
        assert (
            "ckpt_load_path" in predict_cfg.eval
        ), "Please provide \n`eval:\n\tckpt_load_path`\n in the yaml config file "
        self.network = build_model(predict_cfg.model, ckpt_load_path=predict_cfg.eval.ckpt_load_path)
        self.network.set_train(False)

        if predict_cfg.system.amp_level != "O0":
            logger.info("Evaluation will run in full-precision(fp32)")

        # TODO: check float type conversion in official Model.eval
        # ms.amp.auto_mixed_precision(network, amp_level='O0')

        self.postprocessor = build_postprocess(predict_cfg.postprocess)

    def __call__(self):
        if self.shape_list_column_idx is not None:
            pred_outputs = {"img_paths": [], "pred_images": [], "preds": [], "raw_imgs_shape": []}
        else:
            pred_outputs = {"img_paths": [], "pred_images": [], "preds": []}

        for idx, data in tqdm(enumerate(self.loader_predict)):
            img_path = data[self.img_path_column_idx]
            input_image = data[self.image_column_idx]

            pred = self.network(input_image)
            pred = self.postprocessor(pred)  # [bboxes, scores], shape=[(N, K, 4, 2), (N, K)]

            pred_outputs["img_paths"].append(img_path)
            pred_outputs["pred_images"].append(input_image)
            pred_outputs["preds"].append(pred)
            if self.shape_list_column_idx is not None:
                raw_img_shape = data[self.shape_list_column_idx][0][:2]  # [src_h, src_w]
                pred_outputs["raw_imgs_shape"].append(raw_img_shape)

        return pred_outputs


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation Config", add_help=False)
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="../../../configs/det/dbnet/db_r50_icdar15.yaml",
        help="YAML config file specifying default arguments (default=" ")",
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    yaml_fp = args.config
    with open(yaml_fp) as fp:
        config = yaml.safe_load(fp)
    config = Dict(config)
    det_predict = BasePredict(config)
    pred_result = det_predict()
    for res in pred_result:
        print(res)
