import cv2
import numpy as np

from ....data_process.utils import cv_utils
from ....infer import TaskType, TextDetector
from ...framework import ModuleBase


class DetPostNode(ModuleBase):
    def __init__(self, args, msg_queue):
        super(DetPostNode, self).__init__(args, msg_queue)
        self.text_detector = None
        self.task_type = self.args.task_type
        self.is_concat = self.args.is_concat

    def init_self_args(self):
        self.text_detector = TextDetector(self.args)
        self.text_detector.init(preprocess=False, model=False, postprocess=True)
        super().init_self_args()

    def concat_crops(self, crops: list):
        """
        Concatenates the list of cropped images horizontally after resizing them to have the same height.

        Args:
            crops (list): A list of cropped images represented as numpy arrays.

        Returns:
            numpy.ndarray: A horizontally concatenated image array.
        """
        max_height = max(crop.shape[0] for crop in crops)
        resized_crops = []
        for crop in crops:
            h, w, c = crop.shape
            new_h = max_height
            new_w = int((w / h) * new_h)

            resized_img = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            resized_crops.append(resized_img)
        crops_concated = np.concatenate(resized_crops, axis=1)
        return crops_concated

    def process(self, input_data):
        if input_data.skip:
            self.send_to_next_module(input_data)
            return

        data = input_data.data
        boxes = self.text_detector.postprocess(data["pred"], data["shape_list"])
        if self.is_concat:
            boxes = sorted(boxes, key=lambda points: (points[0][1], points[0][0]))

        infer_res_list = []
        for box in boxes:
            infer_res_list.append(box.tolist())

        input_data.infer_result = infer_res_list

        if self.task_type in (TaskType.DET_REC, TaskType.DET_CLS_REC):
            input_data.sub_image_total = len(infer_res_list)
            input_data.sub_image_size = len(infer_res_list)

            image = input_data.frame[0]  # bs=1 for det
            sub_image_list = []
            for box in infer_res_list:
                sub_image = cv_utils.crop_box_from_image(image, np.array(box))
                sub_image_list.append(sub_image)
            if self.is_concat:
                sub_image_list = len(sub_image_list) * [self.concat_crops(sub_image_list)]
            input_data.sub_image_list = sub_image_list

        input_data.data = None

        if not (self.args.crop_save_dir or self.args.vis_det_save_dir or self.args.vis_pipeline_save_dir):
            input_data.frame = None

        if not infer_res_list:
            input_data.skip = True

        self.send_to_next_module(input_data)
