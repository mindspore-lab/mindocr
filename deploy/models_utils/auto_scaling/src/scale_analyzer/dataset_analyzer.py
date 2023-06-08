import json
import logging
import math
import os

import cv2
import numpy as np
from tqdm import tqdm


def get_scaling(h_max, h_min, num_h, inter_h):
    scaling_logical = []
    scaling_true = []
    if num_h == 1:
        out_scaling = math.ceil(h_max / inter_h) * inter_h
        return [out_scaling - inter_h, out_scaling]
    step = (h_max - h_min) / (num_h - 1)
    for i in range(num_h - 1):
        scaling = h_min + step * i
        scaling_logical.append(scaling)
        new_scaling = math.ceil(scaling / inter_h) * inter_h
        if new_scaling not in scaling_true:
            scaling_true.append(new_scaling)
    scaling_logical.append(h_max)
    if h_max not in scaling_true:
        scaling_true.append(h_max)
    return scaling_true


class DatasetAnalyzer:
    def __init__(self, args, cfg):
        self.args = args
        self.config = cfg.auto_scaling
        self.interval = self.config.interval
        self.limit_side_len = self.config.limit_side_len
        self.scaling_limit = self.config.max_scaling_num
        self.height_range = self.config.height_range
        self.width_range = self.config.width_range
        self.b_scaling = self.config.batch_choices

        self.strategy = self.config.strategy
        self.n_std = self.config.mean_std.n_std
        self.expand_ratio = self.config.max_min.expand_ratio

        self.batch_size, _, self.input_height, self.input_width = list(map(int, self.args.input_shape.split(",")))

    def algorithm_min_max(self, widths):
        w_origin_min = min(widths)
        w_origin_max = max(widths)
        w_expand_min = w_origin_min - (w_origin_max - w_origin_min) * self.expand_ratio / 2
        w_expand_max = w_origin_max + (w_origin_max - w_origin_min) * self.expand_ratio / 2

        return w_expand_min, w_expand_max

    def algorithm_mean_std(self, widths):
        w_std = np.std(widths)
        w_mean = np.mean(widths)

        w_range = (w_mean - self.n_std * w_std, w_mean + self.n_std * w_std)
        return w_range

    @staticmethod
    def clip(clip_range, w_range):
        w_range = (max(clip_range[0], w_range[0]), min(clip_range[1], w_range[1]))
        return w_range

    def limit_side(self, widths, heights):
        new_widths = [self.limit_side_len]
        new_heights = [self.limit_side_len]
        for h, w in zip(widths, heights):
            if max(h, w) > self.limit_side_len:
                if h > w:
                    ratio = float(self.limit_side_len) / h
                else:
                    ratio = float(self.limit_side_len) / w
            else:
                ratio = 1.0
            resize_h = int(h * ratio)
            resize_w = int(w * ratio)

            resize_h = int(round(resize_h / self.interval) * self.interval)
            resize_w = int(round(resize_w / self.interval) * self.interval)

            new_heights.append(resize_h)
            new_widths.append(resize_w)

        return new_widths, new_heights

    def read_rec_data(self, dataset_path, widths, heights):
        lens = []
        with open(dataset_path, "r", encoding="utf8") as f:
            lines = f.read().split("\n")
            for line in tqdm(lines):
                if not line:
                    continue
                split_line = line.split("\t")
                if len(split_line) < 2:
                    continue
                data_list = json.loads(split_line[1])
                lens.append(len(data_list))
                for data in data_list:
                    points = [np.array(point) for point in data["points"]]
                    img_crop_width = int(
                        max(
                            np.linalg.norm(points[0] - points[1]),
                            np.linalg.norm(points[2] - points[3]),
                        )
                    )
                    img_crop_height = int(
                        max(
                            np.linalg.norm(points[0] - points[3]),
                            np.linalg.norm(points[1] - points[2]),
                        )
                    )
                    if self.input_height == -1 and self.input_width != -1:
                        heights.append(int(img_crop_height * self.input_width / img_crop_width))
                    elif self.input_width == -1 and self.input_height != -1:
                        widths.append(int(img_crop_width * self.input_height / img_crop_height))
                    else:
                        heights.append(img_crop_height)
                        widths.append(img_crop_width)

        # limit crop image size when input h,w=-1,-1
        if self.input_width == -1 and self.input_height == -1:
            widths, heights = self.limit_side(widths, heights)

        if self.batch_size == -1:
            if self.strategy == "max_min":
                b_range = self.algorithm_min_max(lens)
            else:
                b_range = self.algorithm_mean_std(lens)
            self.b_scaling = list(filter(lambda x: b_range[0] <= x <= b_range[1], self.b_scaling))

        return widths, heights

    def start_analyzer(self):
        widths = []
        heights = []
        dataset_path = self.args.dataset_path
        if not os.path.isfile(dataset_path):
            # det
            for path in tqdm(os.listdir(dataset_path)):
                img = cv2.imread(os.path.join(dataset_path, path))
                h, w, _ = img.shape
                widths.append(w)
                heights.append(h)

            widths, heights = self.limit_side(widths, heights)
            if self.input_height == -1 and self.input_width != -1:
                heights = [h * self.input_width / w for w, h in zip(widths, heights)]
            if self.input_width == -1 and self.input_height != -1:
                widths = [w * self.input_height / h for w, h in zip(widths, heights)]
        else:
            # rec
            widths, heights = self.read_rec_data(dataset_path, widths, heights)
        w_scaling, h_scaling = self.process(widths, heights)
        b_scaling = self.b_scaling
        if self.batch_size != -1:
            b_scaling = []
        logging.info(
            f"Auto-scaling data are as followed: \nbatch_size: {b_scaling}, \nheight: {h_scaling}, "
            f"\nwidth: {w_scaling}"
        )
        scaling_data = {
            "batch_size": b_scaling,
            "height": h_scaling,
            "width": w_scaling,
        }

        return scaling_data

    def preprocess(self, side_li, side_range):
        if self.strategy == "max_min":
            w_range = self.algorithm_min_max(side_li)
        else:
            w_range = self.algorithm_mean_std(side_li)
        w_range = self.clip(side_range, w_range)

        w_min = math.ceil(w_range[0] / self.interval) * self.interval
        w_max = math.ceil(w_range[1] / self.interval) * self.interval

        num_w = (w_max - w_min) // self.interval + 1

        return w_min, w_max, num_w

    def process(self, widths, heights):
        h_scaling = []
        w_scaling = []
        if self.input_height == -1 and self.input_width != -1:
            h_min, h_max, num_h = self.preprocess(heights, self.height_range)
            num_h = min(num_h, self.scaling_limit)
            h_scaling = get_scaling(h_max, h_min, num_h, self.interval)
        elif self.input_width == -1 and self.input_height != -1:
            w_min, w_max, num_w = self.preprocess(widths, self.width_range)
            num_w = min(num_w, self.scaling_limit)
            w_scaling = get_scaling(w_max, w_min, num_w, self.interval)
        else:
            h_min, h_max, num_h = self.preprocess(heights, self.height_range)
            w_min, w_max, num_w = self.preprocess(widths, self.width_range)

            product = num_h * num_w
            if product > self.scaling_limit:
                num_h = math.floor(num_h * pow(self.scaling_limit / product, 0.5))
                num_w = math.floor(self.scaling_limit / num_h)
            h_scaling = get_scaling(h_max, h_min, num_h, self.interval)
            w_scaling = get_scaling(w_max, w_min, num_w, self.interval)

        return w_scaling, h_scaling
