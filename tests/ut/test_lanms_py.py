import json
import os

import numpy as np

from mindocr.postprocess.nms_py.lanms_py import calculate_iou, merge_quadrangle_n9, standard_nms, weighted_merge

box1 = np.array([0, 0, 0, 20, 10, 20, 10, 0, 0.8])
box2 = np.array([8, 10, 8, 50, 30, 50, 30, 10, 0.7])
box3 = np.array([9, 10, 9, 60, 30, 60, 30, 10, 1.1])

origin_boxes_test = []
expect_processed_boxes_test = []
lanms_test_jsons_path = os.path.join("tests/ut/lanms_test_jsons")
for file in os.listdir(lanms_test_jsons_path):
    with open(os.path.join(lanms_test_jsons_path, file)) as f:
        data = json.loads(f.readline())
        origin_boxes_test.append(np.array(data["origin_boxes"]))
        expect_processed_boxes_test.append(sorted(np.array(data["processed_boxes"]), key=lambda x: x[0]))


class TestLanmsPy:
    def test_calculate_iou(self):
        assert round(calculate_iou(box1, box2), 3) == 0.019

    def test_weighted_merge(self):
        expect_result = np.array([3.733, 4.667, 3.733, 34, 19.333, 34, 19.333, 4.666, 1.5])
        assert np.allclose(weighted_merge(box1, box2), expect_result, rtol=1e-2) is True

    def test_standard_nms(self):
        assert np.allclose(standard_nms([box2, box3], 0.5), box3, 1e-5) is True

    def test_lanms(self):
        expect_result = np.array([[8.611, 10, 8.611, 56.11, 30, 56.11, 30, 10, 1.8], [0, 0, 0, 20, 10, 20, 10, 0, 0.8]])
        assert np.allclose(merge_quadrangle_n9([box1, box2, box3]), expect_result, 1e-2) is True

    def test_real_situations(self):
        real_results = []
        for origin_box_test in origin_boxes_test:
            real_results.append(sorted(merge_quadrangle_n9(origin_box_test), key=lambda x: x[0]))
        for i, real_result in enumerate(real_results):
            assert np.allclose(real_result, expect_processed_boxes_test[i], 1e-2) is True
