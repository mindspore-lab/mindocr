import argparse
import ast
import logging
import math
import os
import subprocess

import cv2
import numpy as np
from tqdm import tqdm

from atc_helper import DetATCConverter, RecATCConverter

logging.getLogger().setLevel(logging.INFO)

BATCH_CHOICES = [1, 4, 8, 16, 32, 64]


class BaseDatasetAnalyzer:
    def __init__(self, args):
        self.args = args
        self.interval = 32
        self.strategy = "mean_std"
        self.expand_ratio = 0.2
        self.n_std = 3

    def algorithm_min_max(self, widths):
        w_origin_min = min(widths)
        w_origin_max = max(widths)
        w_expand_min = w_origin_min - (w_origin_max - w_origin_min) * self.expand_ratio / 2
        w_expand_max = w_origin_max + (w_origin_max - w_origin_min) * self.expand_ratio / 2

        return w_expand_min, w_expand_max

    def algorithm_mean_std(self, widths):
        w_std = np.std(widths)
        w_mean = np.mean(widths)

        det_n_std = self.n_std
        w_range = (w_mean - det_n_std * w_std, w_mean + det_n_std * w_std)
        return w_range

    def clip(self, clip_range, w_range):
        w_range = (max(clip_range[0], w_range[0]), min(clip_range[1], w_range[1]))
        return w_range

    def process(self, widths, interval, width_range):
        if self.strategy == "max_min":
            w_range = self.algorithm_min_max(widths)
        else:
            w_range = self.algorithm_mean_std(widths)
        w_range = self.clip(width_range, w_range)

        w_min = math.ceil(w_range[0] / interval) * interval
        w_max = math.ceil(w_range[1] / interval) * interval

        num_w = (w_max - w_min) // interval + 1

        return w_min, w_max, num_w


class DetDatasetAnalyzer(BaseDatasetAnalyzer):
    def __init__(self, args):
        super(DetDatasetAnalyzer, self).__init__(args)
        self.args = args
        self.strategy = args.det_strategy
        self.expand_ratio = args.det_expand_ratio
        self.n_std = args.det_n_std
        self.scales = []

        image_path = args.image_path
        for path in tqdm(os.listdir(image_path)):
            img = cv2.imread(os.path.join(image_path, path))
            h, w, _ = img.shape
            self.scales.append((h, w))

    def preprocess(self):
        widths = [self.args.det_limit_side_len]
        heights = [self.args.det_limit_side_len]
        for scale in self.scales:
            h, w = scale
            det_limit_side_len = self.args.det_limit_side_len
            if max(h, w) > det_limit_side_len:
                if h > w:
                    ratio = float(det_limit_side_len) / h
                else:
                    ratio = float(det_limit_side_len) / w
            else:
                ratio = 1.
            resize_h = int(h * ratio)
            resize_w = int(w * ratio)

            resize_h = int(round(resize_h / self.interval) * self.interval)
            resize_w = int(round(resize_w / self.interval) * self.interval)

            heights.append(resize_h)
            widths.append(resize_w)
        return widths, heights

    def process(self):
        widths, heights = self.preprocess()
        w_min, w_max, num_w = super().process(widths, self.interval, self.args.det_width_range)
        h_min, h_max, num_h = super().process(heights, self.interval, self.args.det_height_range)

        product = num_h * num_w
        det_gear_limit = self.args.det_gear_limit
        if product > det_gear_limit:
            num_h = math.floor(num_h * pow(det_gear_limit / product, 0.5))
            num_w = math.floor(det_gear_limit / num_h)
        h_gear = get_gear(h_max, h_min, num_h, self.interval)
        w_gear = get_gear(w_max, w_min, num_w, self.interval)

        st = ""
        for h in h_gear:
            for w in w_gear:
                st += f"{h},{w};"
        logging.info("detection gears are as followed:")
        logging.info(st)
        return [st]


class RecDatasetAnalyzer(BaseDatasetAnalyzer):
    def __init__(self, args):
        super().__init__(args)
        self.strategy = args.rec_strategy
        self.expand_ratio = args.rec_expand_ratio
        self.n_std = args.rec_n_std

    def preprocess(self):
        gt_path = self.args.gt_path
        widths = []

        lens = []
        boxes = []
        for path in tqdm(os.listdir(gt_path)):
            with open(os.path.join(gt_path, path), 'r', encoding='utf8') as f:
                res = f.read()
                lines = res.split('\n')
                lens.append(len(lines))
                for line in lines:
                    if line:
                        if line.startswith('\ufeff'):
                            line = line.encode('utf-8').decode('utf-8-sig')
                        x = line.split(',')
                        boxes.append(x[:8])

        for box in boxes:
            points = []
            for i in range(4):
                points.append(np.array([int(box[2 * i]), int(box[2 * i + 1])]))
            img_crop_width = int(
                max(
                    np.linalg.norm(points[0] - points[1]),
                    np.linalg.norm(points[2] - points[3])))
            img_crop_height = int(
                max(
                    np.linalg.norm(points[0] - points[3]),
                    np.linalg.norm(points[1] - points[2])))

            widths.append(int(img_crop_width * self.args.rec_model_height / img_crop_height))

        if self.args.rec_multi_batch:
            if self.strategy == "max_min":
                b_range = self.algorithm_min_max(lens)
            else:
                b_range = self.algorithm_mean_std(lens)
            self.b_gear = filter(lambda x: b_range[0] <= x <= b_range[1], BATCH_CHOICES)
        else:
            self.b_gear = [1]

        return widths

    def process(self):
        widths = self.preprocess()
        w_min, w_max, num_w = super().process(widths, self.interval, self.args.rec_width_range)

        num_w = min(num_w, self.args.rec_gear_limit)

        w_gear = get_gear(w_max, w_min, num_w, self.interval)

        gears = []
        logging.info("recognition gears are as followed:")
        for b in self.b_gear:
            st = ""
            for w in w_gear:
                st += f"{b},{w};"
            logging.info(st)
            gears.append(st)
        return gears


def parse_args():
    parser = argparse.ArgumentParser()
    # detection auto gear related
    parser.add_argument('--image_path', type=str, required=False, default="")
    parser.add_argument('--det_gear_limit', type=int, required=False, default=100)
    parser.add_argument('--det_limit_side_len', type=int, required=False, default=960)
    parser.add_argument('--det_strategy', type=str, required=False, default="mean_std",
                        choices=["max_min", "mean_std"])
    parser.add_argument('--det_expand_ratio', type=float, required=False, default=0.2)
    parser.add_argument('--det_n_std', type=int, required=False, default=3)
    parser.add_argument('--det_width_range', type=str2list, required=False, default="1,8192")
    parser.add_argument('--det_height_range', type=str2list, required=False, default="1,8192")

    # recognition auto gear related
    parser.add_argument('--gt_path', type=str, required=False, default="")
    parser.add_argument('--rec_gear_limit', type=int, required=False, default=32)
    parser.add_argument('--rec_model_height', type=int, required=False)
    parser.add_argument('--rec_strategy', type=str, required=False, default="mean_std",
                        choices=["max_min", "mean_std"])
    parser.add_argument('--rec_expand_ratio', type=float, required=False, default=0.2)
    parser.add_argument('--rec_n_std', type=int, required=False, default=3)
    parser.add_argument('--rec_width_range', type=str2list, required=False, default="1,8192")
    parser.add_argument('--rec_multi_batch', type=ast.literal_eval, required=False, default=True)
    parser.add_argument('--rec_model_channel', type=ast.literal_eval, required=False, default=3, choices=[1, 3])

    # atc conversion related
    parser.add_argument('--det_onnx_path', type=str, required=False, default="")
    parser.add_argument('--rec_onnx_path', type=str, required=False, default="")
    parser.add_argument('--soc_version', type=str, required=False, default="Ascend310P3",
                        choices=["Ascend310P3", "Ascend310"])
    parser.add_argument('--output_path', type=str, required=False, default="output")
    args = parser.parse_args()
    return args


def str2list(value):
    return [int(v) for v in value.split(",")]


def get_gear(h_max, h_min, num_h, inter_h):
    gear_logical = []
    gear_true = []
    if num_h == 1:
        out_gear = math.ceil(h_max / inter_h) * inter_h
        return [out_gear - inter_h, out_gear]
    step = (h_max - h_min) / (num_h - 1)
    for i in range(num_h - 1):
        gear = h_min + step * i
        gear_logical.append(gear)
        new_gear = math.ceil(gear / inter_h) * inter_h
        if new_gear not in gear_true:
            gear_true.append(new_gear)
    gear_logical.append(h_max)
    if h_max not in gear_true:
        gear_true.append(h_max)
    return gear_true


def get_safe_name(path):
    """Remove ending path separators before retrieving the basename.

    e.g. /xxx/ -> /xxx
    """
    return os.path.basename(os.path.abspath(path))


def custom_islink(path):
    """Remove ending path separators before checking soft links.

    e.g. /xxx/ -> /xxx
    """
    return os.path.islink(os.path.abspath(path))


def check_path_valid(path):
    name = get_safe_name(path)
    if not path or not os.path.exists(path):
        raise FileExistsError(f'Error! {name} must exists!')
    if custom_islink(path):
        raise ValueError(f'Error! {name} cannot be a soft link!')
    if not os.access(path, mode=os.R_OK):
        raise ValueError(f'Error! Please check if {name} is readable.')


def args_check(opts):
    if opts.image_path:
        check_path_valid(opts.image_path)
    if opts.gt_path:
        check_path_valid(opts.gt_path)
    if opts.det_onnx_path:
        check_path_valid(opts.det_onnx_path)
    if opts.rec_onnx_path:
        check_path_valid(opts.rec_onnx_path)


if __name__ == '__main__':
    _args = parse_args()
    args_check(_args)
    subps = []
    if not os.path.exists(_args.image_path):
        logging.warning("image_path is unvalid. detection auto gear will be skipped.")
    else:
        det_analyzer = DetDatasetAnalyzer(_args)
        gears = det_analyzer.process()
        if _args.det_onnx_path and os.path.exists(_args.det_onnx_path):
            if not os.path.isfile(_args.det_onnx_path):
                raise FileNotFoundError("det_onnx_path must be a file")
            converter = DetATCConverter(_args)
            output_base = f"{_args.output_path}/dbnet/dbnet_dynamic_dims" \
                          f"_{len(gears[0].split(';'))}"
            subps.append(converter.convert_async(gears[0], _args.det_onnx_path, output_base))

    if not os.path.exists(_args.gt_path):
        logging.warning("gt_path is unvalid. recognition auto gear will be skipped.")
    else:
        if _args.rec_model_height is None:
            msg = 'Please set the height of the rec model. Please choose the right height, otherwise it will cause ' \
                  'abnormal accuracy or unpredictable errors. The height of PP-OCR 2.0 server rec model is 32, ' \
                  'and the Height of PP-OCR 3.0 rec model is 48 '
            raise ValueError(msg)
        rec_analyzer = RecDatasetAnalyzer(_args)
        gears = rec_analyzer.process()
        if _args.rec_onnx_path and os.path.exists(_args.rec_onnx_path):
            if not os.path.isfile(_args.rec_onnx_path):
                raise FileNotFoundError("rec_onnx_path must be a file")
            rec_model_name = 'rec'
            crnn_pattern = ['crnn', 'ch_ppocr_server_v2.0_rec_infer']
            svtr_pattern = ['svtr', 'ch_PP-OCRv3_rec_infer']
            if any(pattern.lower() in _args.rec_onnx_path.lower() for pattern in crnn_pattern):
                rec_model_name = 'crnn'
            elif any(pattern.lower() in _args.rec_onnx_path.lower() for pattern in svtr_pattern):
                rec_model_name = 'svtr'

            for gear in gears:
                converter = RecATCConverter(_args)
                output_base = f"{_args.output_path}/{rec_model_name}/{rec_model_name}_dynamic_dims" \
                              f"_{len(gear.split(';'))}" + f"_bs{gear.split(',')[0]}"
                subps.append(converter.convert_async(gear, _args.rec_onnx_path, output_base))

    for subp in subps:
        try:
            subp.wait(3600)
        except subprocess.TimeoutExpired:
            logging.error("Error, ONNX convert to OM more than 1 hour!")
            exit(-1)
        finally:
            subp.kill()

    logging.info("auto gear finish!")
