from typing import Union, List
import cv2
import numpy as np
from shapely.geometry import Polygon
import pyclipper

import mindspore as ms
from mindspore import ops

__all__ = ['DBPostprocess']

# TODO: improve code, adopted from modelzoo which is adopted from WenmuZhou/DBNet.pytorch
class DBPostprocess(object):
    '''
    region_type: the text region type 'quad' or 'poly'.
    '''
    def __init__(self, 
                thresh=0.3, 
                box_thresh=0.7, 
                max_candidates=1000, 
                unclip_ratio=1.5,
                region_type='quad', 
                dest='binary',
                score_mode='fast'):

        self.min_size = 3
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.region_type = region_type
        self.dest = dest
        self.score_mode = score_mode
        assert score_mode in ['fast', 'slow'], 'Invalid value'

    def __call__(self, pred: Union[dict, List], data_samples=None):
        '''
        Args: 
            pred (dict):
                binary: text region segmentation map, with shape (N, 1, H, W)
                thresh: [if exists] threshhold prediction with shape (N, 1, H, W)
                thresh_binary: [if exists] binarized with threshold, (N, 1, H, W)
            data_samples: optional, data samples mapping
        Returns:
            result (dict) with keys: 
                polygons: np.ndarray of shape (N, K, 4, 2) for the polygons of objective regions if region_type is 'quad' 
                scores: np.ndarray of shape (N, K), score for each box
        '''
        if isinstance(pred, dict):
            pred: ms.Tensor = pred[self.dest].squeeze(1)
        else:
            dest_dict = {'binary': 0, 'thresh': 1, 'thresh_binary': 2}
            idx = dest_dict[self.dest]
            pred = pred[idx][:, 0, :, :]
        if isinstance(pred, ms.Tensor):
            pred = pred.asnumpy()
        segmentation = self.binarize(pred)
        boxes_batch = []
        scores_batch = []
        for batch_index in range(pred.shape[0]):
            height, width = pred.shape[1:]
            if self.region_type == 'poly':
                boxes, scores = self.polygons_from_bitmap(pred[batch_index], segmentation[batch_index], width, height)
            else:
                boxes, scores = self.boxes_from_bitmap(pred[batch_index], segmentation[batch_index], width, height)
            boxes_batch.append(boxes)
            scores_batch.append(scores)

        result = {} 
        result['polygons'] = boxes_batch
        result['scores'] = scores_batch
        return result

    def binarize(self, pred):
        return pred > self.thresh

    def polygons_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        '''
        _bitmap: single map with shape (H, W), whose values are binarized as {0, 1}
        '''

        assert len(_bitmap.shape) == 2
        bitmap = _bitmap.asnumpy()  # The first channel
        height, width = bitmap.shape
        boxes = []
        scores = []

        contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours[:self.max_candidates]:
            epsilon = 0.005 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape((-1, 2))
            if points.shape[0] < 4:
                continue

            score = self.box_score_fast(pred, contour.squeeze(1))
            if self.box_thresh > score:
                continue

            if points.shape[0] > 2:
                box = self.unclip(points, unclip_ratio=self.unclip_ratio)
                if len(box) > 1:
                    continue
            else:
                continue
            box = box.reshape(-1, 2)
            _, sside = self.get_mini_boxes(box.reshape((-1, 1, 2)))
            if sside < self.min_size + 2:
                continue

            if not isinstance(dest_width, int):
                dest_width = dest_width.item()
                dest_height = dest_height.item()

            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box)
            scores.append(score)
        return boxes, scores
    
    # TODO: diff from paddle? if no boxes, return [] instead of [0,0,0,0]
    def boxes_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        '''
        _bitmap: single map with shape (H, W), whose values are binarized as {0, 1}
        '''
        assert len(_bitmap.shape) == 2
        if isinstance(_bitmap, np.ndarray):
            bitmap = _bitmap
        else:
            bitmap = _bitmap.asnumpy()

        if isinstance(pred, np.ndarray):
            pred = pred
        else:
            pred = ops.stop_gradient(pred).asnumpy()

        # Contour & Constraint
        height, width = bitmap.shape
        contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        num_contours = min(len(contours), self.max_candidates)

        # Init
        boxes = np.zeros((num_contours, 4, 2), dtype=np.int16)
        scores = np.zeros((num_contours,), dtype=np.float32)

        for index in range(num_contours):
            # (4, 1, 2) -> (4, 2)
            contour = contours[index].squeeze(1)
            # Score
            score = self.box_score_fast(pred, contour)
            if self.box_thresh > score:
                continue
            # Box's points
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)
            # Box
            box = self.unclip(points, unclip_ratio=self.unclip_ratio).reshape((-1, 1, 2))
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            box = np.array(box)
            # cast
            if not isinstance(dest_width, int):
                dest_width = dest_width.item()
                dest_height = dest_height.item()
            # clip
            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes[index, :, :] = box.astype(np.int16)
            scores[index] = score
        return boxes, scores 
    # TODO: adopted from paddle
    '''
    def boxes_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        bitmap = _bitmap
        height, width = bitmap.shape

        outs = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST,
                                cv2.CHAIN_APPROX_SIMPLE)
        if len(outs) == 3:
            img, contours, _ = outs[0], outs[1], outs[2]
        elif len(outs) == 2:
            contours, _ = outs[0], outs[1]

        num_contours = min(len(contours), self.max_candidates)

        boxes = []
        scores = []
        for index in range(num_contours):
            contour = contours[index]
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)
            if self.score_mode == "fast":
                score = self.box_score_fast(pred, points.reshape(-1, 2))
            else:
                score = self.box_score_slow(pred, contour)
            if self.box_thresh > score:
                continue

            box = self.unclip(points, self.unclip_ratio).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            box = np.array(box)

            box[:, 0] = np.clip(
                np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box.astype("int32"))
            scores.append(score)
        return np.array(boxes, dtype="int32"), scores
    '''

    def unclip(self, box, unclip_ratio=1.5):
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [points[index_1], points[index_2], points[index_3], points[index_4]]
        return box, min(bounding_box[1])

    def box_score_fast(self, bitmap, _box):
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int32), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int32), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int32), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int32), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)

        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]
    
    def box_score_slow(self, bitmap, contour):
        '''
        box_score_slow: use polyon mean score as the mean score
        '''
        h, w = bitmap.shape[:2]
        contour = contour.copy()
        contour = np.reshape(contour, (-1, 2))

        xmin = np.clip(np.min(contour[:, 0]), 0, w - 1)
        xmax = np.clip(np.max(contour[:, 0]), 0, w - 1)
        ymin = np.clip(np.min(contour[:, 1]), 0, h - 1)
        ymax = np.clip(np.max(contour[:, 1]), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)

        contour[:, 0] = contour[:, 0] - xmin
        contour[:, 1] = contour[:, 1] - ymin

        cv2.fillPoly(mask, contour.reshape(1, -1, 2).astype("int32"), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]
