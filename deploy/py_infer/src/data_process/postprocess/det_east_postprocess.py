import os
import sys

import cv2
import lanms
import numpy as np

# add mindocr root path, and import postprocess from mindocr
mindocr_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
sys.path.insert(0, mindocr_path)

from mindocr.postprocess import det_east_postprocess  # noqa


class EASTPostprocess(det_east_postprocess.EASTPostprocess):
    """
    The post process for EAST, adapted to paddleocr.
    """

    def __init__(
        self,
        score_thresh=0.8,
        nms_thresh=0.2,
        box_type="quad",
        rescale_fields=["polys"],
        # for paddleocr east postprocess
        cover_thresh=0.1,
        from_ppocr=False,
        **kwargs
    ):
        super().__init__(score_thresh, nms_thresh, box_type, rescale_fields)

        if from_ppocr:
            self._cover_thresh = cover_thresh
            assert box_type == "quad"

        self._from_ppocr = from_ppocr

    def _postprocess(self, pred, **kwargs):
        if not self._from_ppocr:
            return super()._postprocess(pred)
        else:
            return self._postprocess_ppocr(pred)

    def _postprocess_ppocr(self, pred: tuple):
        geo_list = pred[0]
        score_list = pred[1]

        polys_list = []
        for i, (score, geo) in enumerate(zip(score_list, geo_list)):
            boxes = self.detect(
                score_map=score,
                geo_map=geo,
                score_thresh=self._score_thresh,
                cover_thresh=self._cover_thresh,
                nms_thresh=self._nms_thresh,
            )

            polys = []
            if len(boxes) > 0:
                boxes = boxes[:, :8].reshape((-1, 4, 2))
                for box in boxes:
                    box = self.sort_poly(box.astype(np.int32))
                    polys.append(box)
                polys_list.append(np.array(polys))
            else:
                polys_list.append([])

        # FIXME: scores is empty
        return {"polys": polys_list, "scores": []}

    def restore_rectangle_quad(self, origin, geometry):
        """
        Restore rectangle from quadrangle.
        """
        # quad
        origin_concat = np.concatenate((origin, origin, origin, origin), axis=1)  # (n, 8)
        pred_quads = origin_concat - geometry
        pred_quads = pred_quads.reshape((-1, 4, 2))  # (n, 4, 2)
        return pred_quads

    def detect(self, score_map, geo_map, score_thresh=0.8, cover_thresh=0.1, nms_thresh=0.2):
        """
        restore text boxes from score map and geo map
        """

        score_map = score_map[0]
        geo_map = np.swapaxes(geo_map, 1, 0)
        geo_map = np.swapaxes(geo_map, 1, 2)
        # filter the score map
        xy_text = np.argwhere(score_map > score_thresh)
        if len(xy_text) == 0:
            return []
        # sort the text boxes via the y axis
        xy_text = xy_text[np.argsort(xy_text[:, 0])]
        # restore quad proposals
        text_box_restored = self.restore_rectangle_quad(xy_text[:, ::-1] * 4, geo_map[xy_text[:, 0], xy_text[:, 1], :])
        boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
        boxes[:, :8] = text_box_restored.reshape((-1, 8))
        boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]

        boxes = lanms.merge_quadrangle_n9(boxes, nms_thresh)

        if boxes.shape[0] == 0:
            return []
        # Here we filter some low score boxes by the average score map,
        #   this is different from the orginal paper.
        for i, box in enumerate(boxes):
            mask = np.zeros_like(score_map, dtype=np.uint8)
            cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
            boxes[i, 8] = cv2.mean(score_map, mask)[0]
        boxes = boxes[boxes[:, 8] > cover_thresh]
        return boxes

    def sort_poly(self, p):
        """
        Sort polygons.
        """
        min_axis = np.argmin(np.sum(p, axis=1))
        p = p[[min_axis, (min_axis + 1) % 4, (min_axis + 2) % 4, (min_axis + 3) % 4]]
        if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
            return p
        else:
            return p[[0, 3, 2, 1]]
