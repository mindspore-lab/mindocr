"""
Text detection inference

Example:
    $ python tools/infer/text/predict_det.py  --image_dir {path_to_img} --rec_algorithm DB++
"""

import json
import os
import sys
from typing import List

import numpy as np
from config import parse_args
from postprocess import Postprocessor
from preprocess import Preprocessor
from shapely.geometry import Polygon
from utils import get_ckpt_file, get_image_paths

import mindspore as ms

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../../../")))

from mindocr import build_model
from mindocr.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from mindocr.utils.visualize import draw_boxes, show_imgs

# map algorithm name to model name (which can be checked by `mindocr.list_models()`)
# NOTE: Modify it to add new model for inference.
algo_to_model_name = {
    "DB": "dbnet_resnet50",
    "DB++": "dbnetpp_resnet50",
    "DB_MV3": "dbnet_mobilenetv3",
    "PSE": "psenet_resnet152",
}


class TextDetector(object):
    def __init__(self, args):
        # build model for algorithm with pretrained weights or local checkpoint
        ckpt_dir = args.det_model_dir
        if ckpt_dir is None:
            pretrained = True
            ckpt_load_path = None
        else:
            ckpt_load_path = get_ckpt_file(ckpt_dir)
            pretrained = False
        assert args.det_algorithm in algo_to_model_name, (
            f"Invalid det_algorithm {args.det_algorithm}. "
            f"Supported detection algorithms are {list(algo_to_model_name.keys())}"
        )
        model_name = algo_to_model_name[args.det_algorithm]
        self.model = build_model(
            model_name, pretrained=pretrained, ckpt_load_path=ckpt_load_path, amp_level=args.det_amp_level
        )
        self.model.set_train(False)
        print(
            "INFO: Init detection model: {} --> {}. Model weights loaded from {}".format(
                args.det_algorithm, model_name, "pretrained url" if pretrained else ckpt_load_path
            )
        )

        # build preprocess and postprocess
        self.preprocess = Preprocessor(
            task="det",
            algo=args.det_algorithm,
            det_limit_side_len=args.det_limit_side_len,
            det_limit_type=args.det_limit_type,
        )

        self.postprocess = Postprocessor(task="det", algo=args.det_algorithm, box_type=args.det_box_type)

        self.vis_dir = args.draw_img_save_dir
        os.makedirs(self.vis_dir, exist_ok=True)

        self.box_type = args.det_box_type
        self.visualize_preprocess = False

    def __call__(self, img_or_path, do_visualize=True):
        """
            Args:
        img_or_path: str for img path or np.array for RGB image
        do_visualize: visualize preprocess and final result and save them

            Return:
        det_res_final (dict): detection result with keys:
                            - polys: np.array in shape [num_polygons, 4, 2] if det_box_type is 'quad'. Otherwise,
                              it is a list of np.array, each np.array is the polygon points.
                            - scores: np.array in shape [num_polygons], confidence of each detected text box.
        data (dict): input and preprocessed data with keys: (for visualization and debug)
            - image_ori (np.ndarray): original image in shape [h, w, c]
            - image (np.ndarray): preprocessed image feed for network, in shape [c, h, w]
            - shape (list): shape and scaling information [ori_h, ori_w, scale_ratio_h, scale_ratio_w]
        """
        # preprocess
        data = self.preprocess(img_or_path)
        fn = os.path.basename(data.get("img_path", "input.png")).split(".")[0]
        if do_visualize and self.visualize_preprocess:
            # show_imgs([data['image_ori']], is_bgr_img=False, title='det: '+ data['img_path'])
            # TODO: saving images increase inference time.
            show_imgs(
                [data["image"]],
                title=fn + "_det_preprocessed",
                mean_rgb=IMAGENET_DEFAULT_MEAN,
                std_rgb=IMAGENET_DEFAULT_STD,
                is_chw=True,
                show=False,
                save_path=os.path.join(self.vis_dir, fn + "_det_preproc.png"),
            )
        print("Original image shape: ", data["image_ori"].shape)
        print("After det preprocess: ", data["image"].shape)

        # infer
        input_np = data["image"]
        if len(input_np.shape) == 3:
            net_input = np.expand_dims(input_np, axis=0)

        net_output = self.model(ms.Tensor(net_input))

        # postprocess
        det_res = self.postprocess(net_output, data)

        # validate: filter polygons with too small number of points or area
        # print('Postprocess result: ', det_res)

        det_res_final = validate_det_res(det_res, data["image_ori"].shape[:2], min_poly_points=3, min_area=3)

        if do_visualize:
            det_vis = draw_boxes(data["image_ori"], det_res_final["polys"], is_bgr_img=False)
            show_imgs(
                [det_vis], show=False, title=fn + "_det_res", save_path=os.path.join(self.vis_dir, fn + "_det_res.png")
            )

        return det_res_final, data


def order_points_clockwise(points):
    rect = np.zeros((4, 2), dtype=np.float32)
    s = points.sum(axis=1)
    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]
    tmp = np.delete(points, (np.argmin(s), np.argmax(s)), axis=0)
    diff = np.diff(np.array(tmp), axis=1)
    rect[1] = tmp[np.argmin(diff)]
    rect[3] = tmp[np.argmax(diff)]

    return rect


def validate_det_res(det_res, img_shape, order_clockwise=True, min_poly_points=3, min_area=3):
    polys = det_res["polys"].copy()
    scores = det_res.get("scores", [])

    if len(polys) == 0:
        return dict(polys=[], scores=[])

    # print(polys)
    h, w = img_shape[:2]
    # clip if ouf of image
    if not isinstance(polys, list):
        polys[:, :, 0] = np.clip(polys[:, :, 0], 0, w - 1)
        polys[:, :, 1] = np.clip(polys[:, :, 1], 0, h - 1)
    else:
        for i, poly in enumerate(polys):
            polys[i][:, 0] = np.clip(polys[i][:, 0], 0, w - 1)
            polys[i][:, 1] = np.clip(polys[i][:, 1], 0, h - 1)

    # print(polys)
    new_polys = []
    if scores is not None:
        new_scores = []
    for i, poly in enumerate(polys):
        # refine points to clockwise order
        # print(poly)
        if order_clockwise:
            if len(poly) == 4:
                poly = order_points_clockwise(poly)
            else:
                print("WARNING: order_clockwise only supports quadril polygons currently")
            # print('after clockwise', poly)
        # filter
        if len(poly) < min_poly_points:
            continue

        if min_area > 0:
            p = Polygon(poly)
            # print(p.is_valid, p.is_empty)
            if p.is_valid and not p.is_empty:
                if p.area >= min_area:
                    poly_np = np.array(p.exterior.coords)[:-1, :]
                    new_polys.append(poly_np)
                    if scores is not None:
                        new_scores.append(scores[i])
        else:
            new_polys.append(poly)
            if scores is not None:
                new_scores.append(scores[i])

    if len(scores) > 0:
        new_det_res = dict(polys=np.array(new_polys, dtype=int), scores=new_scores)
    else:
        new_det_res = dict(polys=np.array(new_polys, dtype=int))

    # TODO: sort polygons from top to bottom, left to right

    return new_det_res


def save_det_res(det_res_all: List[dict], img_paths: List[str], include_score=False, save_path="./det_results.txt"):
    lines = []
    for i, det_res in enumerate(det_res_all):
        if not include_score:
            img_pred = (
                os.path.basename(img_paths[i]) + "\t" + str(json.dumps([x.tolist() for x in det_res["polys"]])) + "\n"
            )
        else:
            img_pred = (
                os.path.basename(img_paths[i])
                + "\t"
                + str(json.dumps([x.tolist() for x in det_res["polys"]]))
                + "\t"
                + str(json.dumps(det_res["scores"].tolist()))
                + "\n"
            )
        lines.append(img_pred)

    with open(save_path, "w") as f:
        f.writelines(lines)
        f.close()


if __name__ == "__main__":
    # parse args
    args = parse_args()
    save_dir = args.draw_img_save_dir
    img_paths = get_image_paths(args.image_dir)
    # uncomment it to quick test the infer FPS
    # img_paths = img_paths[:15]

    ms.set_context(mode=args.mode)

    # init detector
    text_detect = TextDetector(args)

    # run for each image
    det_res_all = []
    for i, img_path in enumerate(img_paths):
        print(f"\nINFO: Infering [{i+1}/{len(img_paths)}]: ", img_path)
        det_res, _ = text_detect(img_path, do_visualize=True)
        det_res_all.append(det_res)
        print(f"INFO: Num detected text boxes: {len(det_res['polys'])}")

    # save all results in a txt file
    save_det_res(det_res_all, img_paths, save_path=os.path.join(save_dir, "det_results.txt"))

    print("Done! Text detection results saved in ", save_dir)
