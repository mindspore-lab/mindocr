import cv2
import numpy as np

from mx_infer.framework import ModuleBase
from mx_infer.utils import get_mini_boxes, unclip, construct_box, box_score_slow, \
    get_rotate_crop_image, get_hw_of_img, safe_div, box_score_fast


class DetPostProcess(ModuleBase):
    def __init__(self, args, msg_queue):
        super(DetPostProcess, self).__init__(args, msg_queue)
        self.without_input_queue = False
        self.thresh = 0.3
        self.max_candidates = 1000
        self.unclip_distance = 2
        self.min_size = 3
        self.box_thresh = 0.5
        self.unclip_ratio = 2
        self.score_thresh = 0
        self.score_mode = 'fast'

    def init_self_args(self):
        super().init_self_args()

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
            points, short_side = get_mini_boxes(contour)
            if short_side < self.min_size:
                continue
            if self.score_mode == "fast":
                score = box_score_fast(pred, points.reshape(-1, 2))
            else:
                score = box_score_slow(pred, contour)
            if self.box_thresh > score:
                continue

            box = unclip(points, self.unclip_ratio)
            box, short_side = get_mini_boxes(box)
            if short_side < self.min_size + 2:
                continue

            box = construct_box(box, height, width, dest_height, dest_width)
            boxes.append(box)
            scores.append(score)
        return boxes, scores

    def process(self, input_data):
        if input_data.skip:
            self.send_to_next_module(input_data)
            return
        image = input_data.frame
        shrink_map = input_data.output_array
        shrink_map = shrink_map[:, 0, :, :].reshape((shrink_map.shape[2], shrink_map.shape[3]))
        binary_map = shrink_map > self.thresh

        boxes, scores = self.get_boxes_from_maps(shrink_map, binary_map, input_data.original_width,
                                                 input_data.original_height)
        sub_image_list = []
        infer_res_list = []
        max_wh_ratio = 0
        for box, score in zip(boxes, scores):
            if score < self.score_thresh:
                continue
            points = box.flatten().tolist()
            infer_res_list.append(points[:8])
            sub_image = get_rotate_crop_image(image, np.array(box, dtype=np.float32))
            h, w = get_hw_of_img(sub_image)
            max_wh_ratio = max(max_wh_ratio, safe_div(w, h))
            sub_image_list.append(sub_image)

        input_data.max_wh_ratio = max_wh_ratio
        input_data.sub_image_list = sub_image_list
        input_data.infer_result = infer_res_list
        input_data.sub_image_total = len(sub_image_list)
        input_data.sub_image_size = len(sub_image_list)

        input_data.output_array = None

        if not (self.args.save_pipeline_crop_res
                or self.args.save_vis_det_save_dir
                or self.args.save_vis_pipeline_save_dir):
            input_data.frame = None

        if not sub_image_list:
            input_data.skip = True
        # send the ready data to post module
        self.send_to_next_module(input_data)
