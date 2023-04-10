import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon

from ...utils import log, safe_div


class DBPostProcess:
    def __init__(self):
        self.max_candidates = 1000
        self.unclip_distance = 2
        self.min_size = 3

        self.binary_thresh = 0.3
        self.box_thresh = 0.5
        self.score_thresh = 0

        self.unclip_ratio = 2
        self.score_mode = 'fast'

    def __call__(self, shrink_map, src_width, src_height):
        shrink_map = shrink_map[:, 0, :, :].reshape((shrink_map.shape[2], shrink_map.shape[3]))
        binary_map = shrink_map > self.binary_thresh

        boxes, _ = self.get_boxes_from_maps(shrink_map, binary_map, src_width, src_height)

        return boxes

    def get_boxes_from_maps(self, pred: np.ndarray, binary_map: np.ndarray, dest_width: int, dest_height: int):
        """
        get boxes and scores from feature map that output from DBNet
        :param pred: the probability map
        :param binary_map:
        :param dest_width: the width of the input image
        :param dest_height: the height of the input image
        :return:
        """
        height, width = binary_map.shape

        outs = cv2.findContours((binary_map * 255).astype(np.uint8), cv2.RETR_LIST,
                                cv2.CHAIN_APPROX_SIMPLE)
        if len(outs) == 3:
            _, contours, _ = outs[0], outs[1], outs[2]
        elif len(outs) == 2:
            contours, _ = outs[0], outs[1]

        num_contours = min(len(contours), self.max_candidates)

        boxes = []
        scores = []
        for index in range(num_contours):
            contour = contours[index]
            points, short_side = self.get_mini_boxes(contour)
            if short_side < self.min_size:
                continue
            if self.score_mode == "fast":
                score = self.box_score_fast(pred, points.reshape(-1, 2))
            else:
                score = self.box_score_slow(pred, contour)

            if self.box_thresh > score:
                continue

            if self.score_thresh > score:
                continue

            box = self.unclip(points, self.unclip_ratio)
            box, short_side = self.get_mini_boxes(box)
            if short_side < self.min_size + 2:
                continue

            box = self.construct_box(box, height, width, dest_height, dest_width)
            boxes.append(box)
            scores.append(score)

        return boxes, scores

    @staticmethod
    def box_score_fast(shrink_map: np.ndarray, input_box: np.ndarray):
        """
        using box mean score as the mean score
        :param shrink_map: the output feature map of DBNet
        :param input_box: the min boxes
        :return:
        """
        height, width = shrink_map.shape[:2]
        box = input_box.copy()
        x_min = np.clip(np.floor(box[:, 0].min()).astype(np.int32), 0, width - 1)
        x_max = np.clip(np.ceil(box[:, 0].max()).astype(np.int32), 0, width - 1)
        y_min = np.clip(np.floor(box[:, 1].min()).astype(np.int32), 0, height - 1)
        y_max = np.clip(np.ceil(box[:, 1].max()).astype(np.int32), 0, height - 1)

        mask = np.zeros((y_max - y_min + 1, x_max - x_min + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - x_min
        box[:, 1] = box[:, 1] - y_min
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(shrink_map[y_min:y_max + 1, x_min:x_max + 1], mask)[0]

    @staticmethod
    def box_score_slow(shrink_map: np.ndarray, contour: np.ndarray):
        """
        using polyon mean score as the mean score
        :param shrink_map: the output feature map of DBNet
        :param contour: the contours
        :return:
        """
        height, width = shrink_map.shape
        contour = contour.copy()
        contour = np.reshape(contour, (-1, 2))

        xmin = np.clip(np.min(contour[:, 0]), 0, width - 1)
        xmax = np.clip(np.max(contour[:, 0]), 0, width - 1)
        ymin = np.clip(np.min(contour[:, 1]), 0, height - 1)
        ymax = np.clip(np.max(contour[:, 1]), 0, height - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)

        contour[:, 0] = contour[:, 0] - xmin
        contour[:, 1] = contour[:, 1] - ymin

        cv2.fillPoly(mask, contour.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(shrink_map[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    @staticmethod
    def construct_box(box: np.ndarray, height: int, width: int, dest_height: int, dest_width: int):
        """
        resize the box to the original size.
        """
        try:
            box[:, 0] = np.clip(
                np.round(box[:, 0] / width * dest_width), 0, dest_width)
        except ZeroDivisionError as error:
            log.info(error)
        try:
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height)
        except ZeroDivisionError as error:
            log.info(error)

        return box.astype(np.int16)

    @staticmethod
    def get_mini_boxes(contour):
        """
        get the box from the contours and make the points of box orderly.
        :param contour:
        :return:
        """
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        if points[1][1] > points[0][1]:
            index_one = 0
            index_four = 1
        else:
            index_one = 1
            index_four = 0
        if points[3][1] > points[2][1]:
            index_two = 2
            index_three = 3
        else:
            index_two = 3
            index_three = 2

        box = [points[index_one], points[index_two],
               points[index_three], points[index_four]]
        return np.array(box), min(bounding_box[1])

    @staticmethod
    def unclip(box: np.ndarray, unclip_ratio: float):
        """
        expand the box by unclip ratio
        """
        poly = Polygon(box)
        distance = safe_div(poly.area * unclip_ratio, poly.length)
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance)).reshape(-1, 1, 2)
        return expanded
