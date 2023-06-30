import os
import sys
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
from numpy.fft import ifft

from mindspore import Tensor, context, ops

from ..utils import mask_utils, polygon_utils

# add mindocr root path, and import postprocess from mindocr
mindocr_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
sys.path.insert(0, mindocr_path)

from mindocr.postprocess.det_base_postprocess import DetBasePostprocess  # noqa


class FCEPostprocess(DetBasePostprocess):
    def __init__(
        self,
        fourier_degree: int,
        num_reconstr_points: int,
        scales: Sequence[int] = [8, 16, 32],
        alpha: float = 1.0,
        beta: float = 2.0,
        score_thr: float = 0.3,
        nms_thr: float = 0.1,
        box_type: str = "quad",
        rescale_fields=["polys"],
    ):
        super().__init__(rescale_fields=rescale_fields, box_type=box_type)
        self.fourier_degree = fourier_degree
        self.num_reconstr_points = num_reconstr_points
        self.scales = scales
        self.alpha = alpha
        self.beta = beta
        self.score_thr = score_thr
        self.nms_thr = nms_thr
        self.box_type = box_type

        context.set_context(device_target="CPU")

    def _postprocess(self, pred: Tuple, **kwargs) -> dict:
        batch_pred = self._split_results_from_list(pred)
        polys_list = []
        scores_list = []
        for pre_pred in batch_pred:
            polys, scores = self._process_single(pre_pred)
            polys_list.append(polys)
            scores_list.append(scores)

        return {"polys": polys_list, "scores": scores_list}

    def _process_single(self, per_pred: List[Dict]):
        assert len(per_pred) == len(self.scales)

        polys = []
        scores = []
        for idx, pred_level in enumerate(per_pred):
            level_polys, level_scores = self._get_text_instance(pred_level, self.scales[idx])
            polys += level_polys
            scores += level_scores

        polys, scores = self.poly_nms(polys, scores, self.nms_thr)
        return np.array(polys).reshape((len(polys), -1 if polys else 0, 2)), np.array(scores)

    def _get_text_instance(self, pred_level: Dict, scale: int):
        cls_pred = pred_level["cls_res"]
        tr_pred = ops.softmax(cls_pred[0:2], axis=0).asnumpy()
        tcl_pred = ops.softmax(cls_pred[2:], axis=0).asnumpy()

        reg_pred = pred_level["reg_res"].permute(1, 2, 0).asnumpy()
        x_pred = reg_pred[:, :, : 2 * self.fourier_degree + 1]
        y_pred = reg_pred[:, :, 2 * self.fourier_degree + 1 :]

        score_pred = (tr_pred[1] ** self.alpha) * (tcl_pred[1] ** self.beta)
        tr_pred_mask = (score_pred) > self.score_thr
        tr_mask = mask_utils.fill_hole(tr_pred_mask)

        tr_contours, _ = cv2.findContours(tr_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # opencv4

        mask = np.zeros_like(tr_mask)

        result_polys = []
        result_scores = []
        for cont in tr_contours:
            deal_map = mask.copy().astype(np.int8)
            cv2.drawContours(deal_map, [cont], -1, 1, -1)

            score_map = score_pred * deal_map
            score_mask = score_map > 0
            xy_text = np.argwhere(score_mask)
            dxy = xy_text[:, 1] + xy_text[:, 0] * 1j

            x, y = x_pred[score_mask], y_pred[score_mask]
            c = x + y * 1j
            c[:, self.fourier_degree] = c[:, self.fourier_degree] + dxy
            c *= scale

            polygons = self._fourier2poly(c, self.num_reconstr_points)
            scores = score_map[score_mask].reshape(-1, 1).tolist()
            polygons, scores = self.poly_nms(polygons, scores, self.nms_thr)
            result_polys += polygons
            result_scores += scores

        result_polys, result_scores = self.poly_nms(result_polys, result_scores, self.nms_thr)

        if self.box_type == "quad":
            new_polys = []
            for poly in result_polys:
                poly = np.array(poly).reshape(-1, 2).astype(np.float32)
                points = cv2.boxPoints(cv2.minAreaRect(poly))
                points = np.int0(points)
                new_polys.append(points.reshape(-1))

            return new_polys, result_scores
        return result_polys, result_scores

    def _fourier2poly(self, fourier_coeff: np.ndarray, num_reconstr_points: int = 50):
        a = np.zeros((len(fourier_coeff), num_reconstr_points), dtype="complex")
        k = (len(fourier_coeff[0]) - 1) // 2

        a[:, 0 : k + 1] = fourier_coeff[:, k:]
        a[:, -k:] = fourier_coeff[:, :k]

        poly_complex = ifft(a) * num_reconstr_points
        polygon = np.zeros((len(fourier_coeff), num_reconstr_points, 2))
        polygon[:, :, 0] = poly_complex.real
        polygon[:, :, 1] = poly_complex.imag
        return polygon.astype("int32").reshape((len(fourier_coeff), -1)).tolist()

    def _split_results_from_list(self, pred) -> List[List[Dict]]:
        assert len(pred) == 2 * len(self.scales)

        if not isinstance(pred[0], Tensor):
            pred = [Tensor(x) for x in pred]

        fields = ["cls_res", "reg_res"]
        batch_num = len(pred[0])
        level_num = len(self.scales)

        results = []
        for i in range(batch_num):
            batch_list = []
            for level in range(level_num):
                feat_dict = {}
                feat_dict[fields[0]] = pred[2 * level][i]
                feat_dict[fields[1]] = pred[2 * level + 1][i]

                batch_list.append(feat_dict)
            results.append(batch_list)

        return results

    def poly_nms(
        self, polygons: List[np.ndarray], scores: List[float], threshold: float
    ) -> Tuple[List[np.ndarray], List[float]]:
        assert isinstance(polygons, list)
        assert isinstance(scores, list)
        assert len(polygons) == len(scores)

        polygons = [np.hstack((polygon, score)) for polygon, score in zip(polygons, scores)]
        polygons = np.array(sorted(polygons, key=lambda x: x[-1]))
        keep_polys = []
        keep_scores = []
        index = [i for i in range(len(polygons))]

        while len(index) > 0:
            keep_polys.append(polygons[index[-1]][:-1].tolist())
            keep_scores.append(polygons[index[-1]][-1])
            A = polygons[index[-1]][:-1]
            index = np.delete(index, -1)

            iou_list = np.zeros((len(index),))
            for i in range(len(index)):
                B = polygons[index[i]][:-1]

                iou_list[i] = polygon_utils.poly_iou(A, B, 1)
            remove_index = np.where(iou_list > threshold)
            index = np.delete(index, remove_index)

        return keep_polys, keep_scores
