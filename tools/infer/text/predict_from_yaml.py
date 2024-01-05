"""
Inference base on custom yaml

Example:
    $ python tools/infer/text/predict_from_yaml.py  --config configs/det/dbnet/db++_r50_icdar15.yaml
    $ python tools/infer/text/predict_from_yaml.py  --config configs/rec/crnn/crnn_resnet34.yaml
"""
import logging
import os
import sys

from addict import Dict
from PIL import Image
from predict_det import save_det_res, validate_det_res
from predict_rec import save_rec_res
from tqdm import tqdm

from mindspore import get_context, set_auto_parallel_context, set_context
from mindspore.communication import get_group_size, get_rank, init

from mindocr.data import build_dataset
from mindocr.models import build_model
from mindocr.postprocess import build_postprocess
from mindocr.utils.visualize import draw_boxes, show_imgs
from tools.arg_parser import parse_args_and_config

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../../../")))

from mindocr.utils.logger import set_logger  # noqa

logger = logging.getLogger("mindocr")


def draw_det_res(det_res_list: list, img_list: list, output_save_dir: str):
    """
    Draw detection results on images and save the visualized results.

    Args:
        - det_res_list (list): List of dictionaries containing detection results.
        - img_list (list): List of image paths corresponding to the detection results.
        - output_save_dir (str): Directory to save the visualized detection results.

    Returns:
        None
    """
    for det_res, img_path in zip(det_res_list, img_list):
        img = det_res["img_ori"]
        img_name, _ = os.path.splitext(os.path.basename(img_path))
        det_vis = draw_boxes(img, det_res["polys"], is_bgr_img=False)
        show_imgs(
            [det_vis],
            show=False,
            title=img_name + "_det_res",
            save_path=os.path.join(output_save_dir, img_name + "_det_res.png"),
        )


def save_predict_result(task: str, preds_list: list, output_save_dir: str):
    """
    Save and visualize prediction results based on the specified task.

    Args:
        - task (str): Task type, either "det" (detection) or "rec" (recognition).
        - preds_list (list): List of dictionaries containing prediction results.
        - output_save_dir (str): Directory to save the results.

    Returns:
        None
    """
    if task == "det":
        det_res = []
        img_list = []
        for preds in preds_list:
            for i, single_img_polys in enumerate(preds["polys"]):
                img_path = preds["img_path"][i]
                img_list.append(img_path)
                img_shape = Image.open(img_path).size[::-1]
                single_det_res = {"polys": single_img_polys, "scores": preds["scores"][i]}
                single_det_res = validate_det_res(single_det_res, img_shape=img_shape)
                single_det_res["img_ori"] = preds["img_ori"][i]
                det_res.append(single_det_res)
        draw_det_res(det_res, img_list, output_save_dir=output_save_dir)
        save_det_res(det_res, img_list, save_path=os.path.join(output_save_dir, "det_results.txt"))
    elif task == "rec":
        rec_res = []
        img_list = []
        for preds in preds_list:
            img_path = preds["img_path"]
            rec_res.extend(list(zip(preds["texts"], preds["confs"])))
            img_list.extend(img_path)
        save_rec_res(rec_res, img_list, save_path=os.path.join(output_save_dir, "rec_results.txt"))


def main(cfg):
    # 1. Set the environment information.
    set_context(mode=cfg.system.mode)
    output_save_dir = cfg.predict.output_save_dir or "./output"
    os.makedirs(output_save_dir, exist_ok=True)
    if cfg.system.distribute:
        init()
        device_num = get_group_size()
        rank_id = get_rank()
        set_auto_parallel_context(
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
            set_context(device_id=device_id)
            logger.info(
                f"Standalone evaluation. Device id: {device_id}, "
                f"specified by system.device_id in yaml config file or is default value 0."
            )
    # 2. Configuration dataset with pre-processing transform_pipeline
    # Update the configuration for the DecodeImage transform if present
    for transform in cfg.predict.dataset.transform_pipeline:
        if "DecodeImage" in transform:
            transform["DecodeImage"].update({"keep_ori": True})
            break

    # Adjust output columns based on the model type (det or rec)
    if cfg.model.type == "det":
        cfg.predict.dataset.output_columns.extend({"img_path", "image_ori"} - set(cfg.predict.dataset.output_columns))
    elif cfg.model.type == "rec":
        cfg.predict.dataset.output_columns.extend({"img_path"} - set(cfg.predict.dataset.output_columns))

    loader_predict = build_dataset(
        cfg.predict.dataset,
        cfg.predict.loader,
        num_shards=device_num,
        shard_id=rank_id,
        is_train=False,
        refine_batch_size=True,
    )

    # 3.Build model
    amp_level = cfg.system.get("amp_level_infer", "O0")
    if get_context("device_target") == "GPU" and amp_level == "O3":
        logger.warning(
            "Model evaluation does not support amp_level O3 on GPU currently. "
            "The program has switched to amp_level O2 automatically."
        )
        amp_level = "O2"
    network = build_model(cfg.model, ckpt_load_path=cfg.predict.ckpt_load_path, amp_level=amp_level)
    network.set_train(False)

    # 4.Build postprocessor for network output
    postprocessor = build_postprocess(cfg.postprocess)

    # 5.Initialize the iterator
    iterator = loader_predict.create_tuple_iterator(num_epochs=1, output_numpy=False, do_copy=False)
    num_batches_predict = loader_predict.get_dataset_size()

    output_columns = cfg.predict.dataset.output_columns or []
    input_indices = cfg.predict.dataset.pop("net_input_column_index", None)
    meta_data_indices = cfg.predict.dataset.pop("meta_data_column_index", None)

    # 6.Start prediction
    preds_list = []
    for i, data in tqdm(enumerate(iterator), total=num_batches_predict):
        if input_indices is not None:
            inputs = [data[x] for x in input_indices]
        else:
            inputs = [data[0]]
        preds = network(*inputs)

        data_info = {"img_shape": inputs[0].shape}
        if meta_data_indices is not None:
            meta_info = [data[x] for x in meta_data_indices]
        else:
            # assume the indices not in input_indices or label_indices are all meta_data_indices
            input_indices = set(input_indices) if input_indices is not None else {0}
            meta_data_indices = sorted(set(range(len(data))) - input_indices)
            meta_info = [data[x] for x in meta_data_indices]

        data_info["meta_info"] = meta_info

        possible_keys_for_postprocess = ["shape_list", "raw_img_shape"]
        for k in possible_keys_for_postprocess:
            if k in output_columns:
                data_info[k] = data[output_columns.index(k)]

        preds = postprocessor(preds, **data_info)
        preds["img_path"] = data[output_columns.index("img_path")].numpy()
        # Add "img_ori" to preds if present, which means task is det
        if "image_ori" in output_columns:
            preds["img_ori"] = data[output_columns.index("image_ori")].numpy()
        preds_list.append(preds)

    # 7. Save the prediction results to the specified directory
    save_predict_result(cfg.model.type, preds_list, output_save_dir)


if __name__ == "__main__":
    args, config = parse_args_and_config()
    config = Dict(config)
    main(config)
