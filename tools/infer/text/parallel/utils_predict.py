import json
import os
import shutil

import numpy as np
import yaml
from addict import Dict


def check_args(args):
    if not args.raw_data_dir:
        print(
            "WARNING: The 'raw_data_dir' is empty. The detection predictor will load the data from "
            "'dataset_root/data_dir' in det yaml file. "
        )
    else:
        if not os.path.exists(args.raw_data_dir):
            raise ValueError("Invalid value of arg 'raw_data_dir'.")

    if not args.det_ckpt_path:
        print(
            "WARNING: The 'det_ckpt_path' is empty. The detection predictor will load the ckpt from 'ckpt_load_path' "
            "in det yaml file. "
        )
    else:
        if not os.path.isfile(args.det_ckpt_path):
            raise ValueError("The ckpt file of detection model does not exist. Please check the arg 'det_ckpt_path'.")

    if not args.rec_ckpt_path:
        print(
            "WARNING: The 'rec_ckpt_path' is empty. The recognition predictor will load the ckpt from 'ckpt_load_path' "
            "in rec yaml file. "
        )
    else:
        if not os.path.isfile(args.rec_ckpt_path):
            raise ValueError("The ckpt file of recognition model does not exist. Please check the arg 'rec_ckpt_path'.")

    if not os.path.isfile(args.det_config_path):
        raise ValueError("The detection model yaml config file does not exist. Please check the arg 'det_config_path'.")
    if not os.path.isfile(args.rec_config_path):
        raise ValueError(
            "The recognition model yaml config file does not exist. Please check the arg 'rec_config_path'."
        )

    if args.crop_save_dir:
        args.crop_save_dir = os.path.realpath(args.crop_save_dir)
        if os.path.exists(args.crop_save_dir):
            shutil.rmtree(args.crop_save_dir)
        os.makedirs(args.crop_save_dir)
    else:
        print(
            "WARNING: The 'crop_save_dir' is empty. The recognition predictor will load the data from "
            "'dataset_root/data_dir' in rec yaml file."
        )

    if args.result_save_dir:
        args.result_save_dir = os.path.realpath(args.result_save_dir)
        os.makedirs(os.path.dirname(args.result_save_dir), exist_ok=True)
    else:
        print("WARNING: The 'result_save_dir' is empty. The pipeline prediction result will not be saved.")

    # if args.vis_result_save_dir:
    #     if os.path.exists(args.vis_result_save_dir):
    #         shutil.rmtree(args.vis_result_save_dir)
    #     os.makedirs(args.vis_result_save_dir)
    return args


def update_config(args, cfg, model_type):
    """
    Replace some args values in yaml file with their counterparts in args parser.
    """
    if model_type == "det":
        if args.raw_data_dir:
            cfg.predict.dataset.dataset_root = args.raw_data_dir
            cfg.predict.dataset.data_dir = "."
        if args.det_ckpt_path:
            cfg.predict.ckpt_load_path = args.det_ckpt_path
    elif model_type == "rec":
        if args.crop_save_dir:
            cfg.predict.dataset.dataset_root = args.crop_save_dir
            cfg.predict.dataset.data_dir = "."
        if args.rec_ckpt_path:
            cfg.predict.ckpt_load_path = args.rec_ckpt_path
    else:
        raise ValueError("Invalid value of 'model_type'. It must be 'det' or 'rec'.")

    return cfg


def save_pipeline_results(box_dict, rec_text_dict, save_path):
    with open(save_path, "w") as f:
        for ori_img_name, crop_content in box_dict.items():
            line = []
            for crop_img_name, box in crop_content.items():
                line.append(
                    {
                        "transcription": rec_text_dict[crop_img_name],
                        "points": box.reshape(
                            -1,
                        ).tolist(),
                    }
                )
            f.write(f"{ori_img_name}\t{json.dumps(line)}\n")
    print(f"Detection and recognition prediction pipeline results are saved in '{os.path.realpath(save_path)}'.")


def rescale(det_pred_outputs):
    # assert len(det_pred_outputs['pred_images']) == len(det_pred_outputs['raw_imgs_shape'])
    # TODO: can do in BasePredict
    assert len(det_pred_outputs["pred_images"]) == len(det_pred_outputs["predicted_boxes"]), (
        "The number of images before and after detection prediction doesn't match. "
        "Please check the detection prediction results."
    )

    imgnames = [
        os.path.basename(img_path.asnumpy()[0]) for img_path in det_pred_outputs["img_paths"]
    ]  # TODO: can do in BasePredict
    pred_images_shape = np.array([np.array(img.shape[-2:]) for img in det_pred_outputs["pred_images"]])
    raw_images_shape = np.array([shape.asnumpy()[0] for shape in det_pred_outputs["raw_imgs_shape"]])
    scales = raw_images_shape / pred_images_shape
    scales = scales[:, ::-1]  # H, W -> W, H
    imgname_scale_mapping = dict(zip(imgnames, scales))
    scaled_box_dict = {}
    for ori_img_name, ori_boxes in det_pred_outputs["predicted_boxes"].items():
        scaled_box_dict[ori_img_name] = {}
        cur_scale = imgname_scale_mapping[ori_img_name]
        for crop_img_name, crop_box in ori_boxes.items():
            if crop_box.size:  # crop_box not empty
                crop_box = np.round(crop_box * cur_scale).astype("int64")
            scaled_box_dict[ori_img_name][crop_img_name] = crop_box

    det_pred_outputs["predicted_boxes"] = scaled_box_dict
    return det_pred_outputs


def load_yaml(yaml_fp):
    with open(yaml_fp) as fp:
        config = yaml.safe_load(fp)
    config = Dict(config)
    return config
