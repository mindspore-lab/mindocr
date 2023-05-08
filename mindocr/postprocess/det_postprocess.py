from typing import Tuple, Union
import math
import cv2
import numpy as np
from shapely.geometry import Polygon
import mindspore as ms
from mindspore import Tensor
import lanms

from ..data.transforms.det_transforms import expand_poly

__all__ = ['DBPostprocess', 'EASTPostprocess']


class DBPostprocess:
    def __init__(self, binary_thresh=0.3, box_thresh=0.7, max_candidates=1000, expand_ratio=1.5,
                 output_polygon=False, pred_name='binary'):
        self._min_size = 3
        self._binary_thresh = binary_thresh
        self._box_thresh = box_thresh
        self._max_candidates = max_candidates
        self._expand_ratio = expand_ratio
        self._out_poly = output_polygon
        self._name = pred_name
        self._names = {'binary': 0, 'thresh': 1, 'thresh_binary': 2}

    def __call__(self, pred, **kwargs):
        """
        pred (Union[Tensor, Tuple[Tensor], np.ndarray]):
            binary: text region segmentation map, with shape (N, 1, H, W)
            thresh: [if exists] threshold prediction with shape (N, 1, H, W) (optional)
            thresh_binary: [if exists] binarized with threshold, (N, 1, H, W) (optional)
        Returns:
            result (dict) with keys:
                polygons: np.ndarray of shape (N, K, 4, 2) for the polygons of objective regions if region_type is 'quad'
                scores: np.ndarray of shape (N, K), score for each box
        """
        if isinstance(pred, tuple):
            pred = pred[self._names[self._name]]
        if isinstance(pred, Tensor):
            pred = pred.asnumpy()
        pred = pred.squeeze(1)

        segmentation = pred >= self._binary_thresh

        # FIXME: dest_size is supposed to be the original image shape (pred.shape -> batch['shape'])
        dest_size = np.array(pred.shape[:0:-1])  # w, h order
        scale = dest_size / np.array(pred.shape[:0:-1])

        # TODO:
        # FIXME: output as dict, keep consistent return format to recognition
        return [self._extract_preds(pr, segm, scale, dest_size) for pr, segm in zip(pred, segmentation)]

    def _extract_preds(self, pred, bitmap, scale, dest_size):
        outs = cv2.findContours(bitmap.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(outs) == 3:  # FIXME: update to OpenCV 4.x and delete this
            _, contours, _ = outs[0], outs[1], outs[2]
        elif len(outs) == 2:
            contours, _ = outs[0], outs[1]

        polys, scores = [], []
        for contour in contours[:self._max_candidates]:
            contour = contour.squeeze(1)
            score = self._calc_score(pred, bitmap, contour)
            if score < self._box_thresh:
                continue

            if self._out_poly:
                epsilon = 0.005 * cv2.arcLength(contour, closed=True)
                points = cv2.approxPolyDP(contour, epsilon, closed=True).squeeze(1)
                if points.shape[0] < 4:
                    continue
            else:
                points, min_side = self._fit_box(contour)
                if min_side < self._min_size:
                    continue

            poly = Polygon(points)
            poly = np.array(expand_poly(points, distance=poly.area * self._expand_ratio / poly.length))
            if self._out_poly and len(poly) > 1:
                continue
            poly = poly.reshape(-1, 2)

            _box, min_side = self._fit_box(poly)
            if min_side < self._min_size + 2:
                continue
            if not self._out_poly:
                poly = _box

            # TODO: an alternative solution to avoid calling self._fit_box twice:
            # box = Polygon(points)
            # box = np.array(expand_poly(points, distance=box.area * self._expand_ratio / box.length, joint_type=pyclipper.JT_MITER))
            # assert box.shape[0] == 4, print(f'box shape is {box.shape}')

            # predictions may not be the same size as the input image => scale it
            polys.append(np.clip(np.round(poly * scale), 0, dest_size - 1).astype(np.int16))
            scores.append(score)

        if self._out_poly:
            return polys, scores
        return np.array(polys), np.array(scores).astype(np.float32)

    @staticmethod
    def _fit_box(contour):
        """
        Finds a minimum rotated rectangle enclosing the contour.
        """
        # box = cv2.minAreaRect(contour)  # returns center of a rect, size, and angle
        # # TODO: does the starting point really matter?
        # points = np.roll(cv2.boxPoints(box), -1, axis=0)  # extract box points from a rotated rectangle
        # return points, min(box[1])
        # box = cv2.minAreaRect(contour)  # returns center of a rect, size, and angle
        # # TODO: does the starting point really matter?
        # points = np.roll(cv2.boxPoints(box), -1, axis=0)  # extract box points from a rotated rectangle
        # return points, min(box[1])

        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        # index_1, index_2, index_3, index_4 = 0, 1, 2, 3
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

        box = [
            points[index_1], points[index_2], points[index_3], points[index_4]
        ]
        return box, min(bounding_box[1])

    @staticmethod
    def _calc_score(pred, mask, contour):
        """
        calculates score (mean value) of a prediction inside a given contour.
        """
        min_vals = np.clip(np.floor(np.min(contour, axis=0)), 0, np.array(pred.shape[::-1]) - 1).astype(np.int32)
        max_vals = np.clip(np.ceil(np.max(contour, axis=0)), 0, np.array(pred.shape[::-1]) - 1).astype(np.int32)
        return cv2.mean(pred[min_vals[1]:max_vals[1] + 1, min_vals[0]:max_vals[0] + 1],
                        mask[min_vals[1]:max_vals[1] + 1, min_vals[0]:max_vals[0] + 1].astype(np.uint8))[0]


class EASTPostprocess:
    def __init__(self, score_thresh=0.8, nms_thresh=0.2):
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh

    def __call__(self, pred, **kwargs):
        """
        get boxes from feature map
        Input:
                score       : score map from model <numpy.ndarray, (1,row,col)>
                geo         : geo map from model <numpy.ndarray, (5,row,col)>
                score_thresh: threshold to segment score map
                nms_thresh  : threshold in nms
        Output:
                boxes       : list of final polys and scores [(<numpy.ndarray, (n,4,2)>, numpy.ndarray, (n,1)>)]
        """
        score, geo = pred
        score, geo = np.squeeze(score.asnumpy(), axis=0), np.squeeze(geo.asnumpy(), axis=0)
        score = score[0, :, :]
        xy_text = np.argwhere(score > self.score_thresh)
        if xy_text.size == 0:
            return [(np.array([[[1, 2], [3, 4], [5, 6], [7, 8]]], 'float32'), np.array(0.))]

        xy_text = xy_text[np.argsort(xy_text[:, 0])]
        valid_pos = xy_text[:, ::-1].copy()  # n x 2, [x, y]
        valid_geo = geo[:, xy_text[:, 0], xy_text[:, 1]]  # 5 x n
        polys_restored, index = self.restore_polys(valid_pos, valid_geo, score.shape)
        if polys_restored.size == 0:
            return [(np.array([[[1, 2], [3, 4], [5, 6], [7, 8]]], 'float32'), np.array(0.))]

        boxes = np.zeros((polys_restored.shape[0], 9), dtype=np.float32)
        boxes[:, :8] = polys_restored
        boxes[:, 8] = score[xy_text[index, 0], xy_text[index, 1]]
        boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), self.nms_thresh)
        return [(boxes[:, :8].reshape(-1, 4, 2), boxes[:, 8])]

    def restore_polys(self, valid_pos, valid_geo, score_shape, scale=4):
        """
        restore polys from feature maps in given positions
        Input:
                valid_pos  : potential text positions <numpy.ndarray, (n,2)>
                valid_geo  : geometry in valid_pos <numpy.ndarray, (5,n)>
                score_shape: shape of score map
                scale      : image / feature map
        Output:
                restored polys <numpy.ndarray, (n,8)>, index
        """
        polys = []
        index = []
        valid_pos *= scale
        d = valid_geo[:4, :]  # 4 x N
        angle = valid_geo[4, :]  # N,

        for i in range(valid_pos.shape[0]):
            x = valid_pos[i, 0]
            y = valid_pos[i, 1]
            y_min = y - d[0, i]
            y_max = y + d[1, i]
            x_min = x - d[2, i]
            x_max = x + d[3, i]
            rotate_mat = self.get_rotate_mat(-angle[i])

            temp_x = np.array([[x_min, x_max, x_max, x_min]]) - x
            temp_y = np.array([[y_min, y_min, y_max, y_max]]) - y
            coordidates = np.concatenate((temp_x, temp_y), axis=0)
            res = np.dot(rotate_mat, coordidates)
            res[0, :] += x
            res[1, :] += y

            if self.is_valid_poly(res, score_shape, scale):
                index.append(i)
                polys.append([res[0, 0], res[1, 0], res[0, 1], res[1, 1],
                              res[0, 2], res[1, 2], res[0, 3], res[1, 3]])
        return np.array(polys), index

    def get_rotate_mat(self, theta):
        """positive theta value means rotate clockwise"""
        return np.array([[math.cos(theta), -math.sin(theta)],
                         [math.sin(theta), math.cos(theta)]])

    def is_valid_poly(self, res, score_shape, scale):
        """
        check if the poly in image scope
        Input:
                res        : restored poly in original image
                score_shape: score map shape
                scale      : feature map -> image
        Output:
                True if valid
        """
        cnt = 0
        for i in range(res.shape[1]):
            if res[0, i] < 0 or res[0, i] >= score_shape[1] * scale or \
                    res[1, i] < 0 or res[1, i] >= score_shape[0] * scale:
                cnt += 1
        return cnt <= 1
