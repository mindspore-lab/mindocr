import math
import numpy as np
from mindspore import Tensor
import lanms

from .det_base_postprocess import DetBasePostprocess

__all__ = ["EASTPostprocess"]


class EASTPostprocess(DetBasePostprocess):
    def __init__(self, score_thresh=0.8, nms_thresh=0.2, box_type='quad', rescale_fields=['polys']):
        super().__init__(box_type, rescale_fields)
        self._score_thresh = score_thresh
        self._nms_thresh = nms_thresh
        if rescale_fields is None:
            rescale_fields = []
        self._rescale_fields = rescale_fields

    def _postprocess(self, pred, **kwargs):
        """
        get boxes from feature map
        Input:
                pred (tuple) - (score, geo)
                    'score'       : score map from model <Tensor, (bs,1,row,col)>
                    'geo'         : geo map from model <Tensor, (bs,5,row,col)>
                shape_list (List[List[float]]: a list of shape info [raw_img_h, raw_img_w, ratio_h, ratio_w] for each sample in batch
        Output:
                boxes       : dict of polys and scores {'polys': <numpy.ndarray, (bs,n,4,2)>, 'scores': numpy.ndarray, (bs,n,1)>)}
        """
        score, geo = pred
        if isinstance(score, Tensor):
            score = score.asnumpy()
        if isinstance(geo, Tensor):
            geo = geo.asnumpy()
        img_num = score.shape[0]
        polys_list = []
        scores_list = []
        for i in range(img_num):
            score, geo = score[i], geo[i]
            score = score[0, :, :]
            xy_text = np.argwhere(score > self._score_thresh)
            if xy_text.size == 0:
                polys = np.array([[[1, 2], [3, 4], [5, 6], [7, 8]]], 'float32')
                scores = np.array([[0.]])
                polys_list.append(polys)
                scores_list.append(scores)
                continue

            xy_text = xy_text[np.argsort(xy_text[:, 0])]
            valid_pos = xy_text[:, ::-1].copy()  # n x 2, [x, y]
            valid_geo = geo[:, xy_text[:, 0], xy_text[:, 1]]  # 5 x n
            polys_restored, index = self._restore_polys(valid_pos, valid_geo, score.shape)

            if polys_restored.size == 0:
                polys = np.array([[[1, 2], [3, 4], [5, 6], [7, 8]]], 'float32')
                scores = np.array([[0.]])
                polys_list.append(polys)
                scores_list.append(scores)
                continue

            boxes = np.zeros((polys_restored.shape[0], 9), dtype=np.float32)
            boxes[:, :8] = polys_restored
            boxes[:, 8] = score[xy_text[index, 0], xy_text[index, 1]]
            boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), self._nms_thresh)
            polys = boxes[:, :8].reshape(-1, 4, 2)
            scores = boxes[:, 8].reshape(-1, 1)
            polys_list.append(polys)
            scores_list.append(scores)
        return {'polys': np.array(polys_list), 'scores': np.array(scores_list)}

    def _restore_polys(self, valid_pos, valid_geo, score_shape, scale=4):
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
            rotate_mat = self._get_rotate_mat(-angle[i])

            temp_x = np.array([[x_min, x_max, x_max, x_min]]) - x
            temp_y = np.array([[y_min, y_min, y_max, y_max]]) - y
            coordidates = np.concatenate((temp_x, temp_y), axis=0)
            res = np.dot(rotate_mat, coordidates)
            res[0, :] += x
            res[1, :] += y

            if self._is_valid_poly(res, score_shape, scale):
                index.append(i)
                polys.append([res[0, 0], res[1, 0], res[0, 1], res[1, 1],
                              res[0, 2], res[1, 2], res[0, 3], res[1, 3]])
        return np.array(polys), index

    def _get_rotate_mat(self, theta):
        """positive theta value means rotate clockwise"""
        return np.array([[math.cos(theta), -math.sin(theta)],
                         [math.sin(theta), math.cos(theta)]])

    def _is_valid_poly(self, res, score_shape, scale):
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
