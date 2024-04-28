import os
import sys
import numpy as np

import cv2
import copy

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../../")))

from pipeline.framework.module_base import ModuleBase
from pipeline.tasks import TaskType
from .layout import LayoutPostProcess
from tools.infer.text.utils import crop_text_region
from pipeline.data_process.utils.cv_utils import crop_box_from_image

class LayoutPostNode(ModuleBase):
    def __init__(self, args, msg_queue, tqdm_info):
        super(LayoutPostNode, self).__init__(args, msg_queue, tqdm_info)
        self.layout_postprocess = LayoutPostProcess(args)
        self.task_type = self.args.task_type
        self.score_thres = 0.5

    def init_self_args(self):
        super().init_self_args()

    def get_layout_images(self, frame, infer_result):
        layout_images = []
        for d in infer_result:
            d['bbox'] = [int(v) for v in d['bbox']]
            x, y, dx, dy = d['bbox']
            layout_images.append(frame[0][y:(y+dy), x:(x+dx), :])
        return layout_images

    def process(self, input_data):
        """
        Input:
          - input_data.data["pred"]: [np.ndarray], shape:[1,?,?], shape e.g. [1,13294, 9] (note:[bs, N, 5+nc])
          - input_data.data["hw_ori"]: (int, int), value e.g. (792,601)
          - input_data.data["hw_scale"]: np.ndarray, shape:[1,2], value e.g. (1.0101,1.3311)
          - input_data.data["pad"]: np.ndarray, shape:[1,2], value e.g. (4,99.5)
        Output:
          - input_data.data["image_ids"]: [str]
          - input_data.infer_result: [{"image_id": str, "category_id": int, "bbox": [x:int, y:int, dx:int, dy:int]}]
          - input_data.data["layout_result"]: [{"image_id": str, "category_id": int, "bbox": [x:int, y:int, dx:int, dy:int]}]
          - input_data.data["layout_image"]: [np.ndarray], shape:[?,?,3]
        """
        if input_data.skip:
            self.send_to_next_module(input_data)
            return

        data = input_data.data
        data["image_ids"] = [input_data.image_path]

        meta_info = (data["image_ids"], [data["hw_ori"]], [data["hw_scale"]], [data["pad"]])
        # print(f"meta_info: {meta_info}")
        output = self.layout_postprocess(data["pred"][0], data["img_shape"], meta_info)
        output = [d for d in output if d["score"] > self.score_thres and d["category_id"] in (1, 2, 3)]

        if self.task_type.value == TaskType.LAYOUT.value:
            input_data_out = copy.deepcopy(input_data)
            input_data_out.infer_result = output
            self.send_to_next_module(input_data_out)
        else:
            layout_images = self.get_layout_images(input_data.frame, output)
            input_data.data["layout_collect_list"] = list(range(len(output)))
            for layout_images_id in range(len(output)):
                input_data_out = copy.deepcopy(input_data)
                # new_image_path = f"{layout_images_id}-" + os.path.basename(input_data.image_path[0])
                input_data_out.data["layout_result"] = output[layout_images_id]
                # input_data_out.data["image_ids"] = [f"{new_image_path}"]
                # input_data_out.image_path = [f"{new_image_path}"]
                input_data_out.data["layout_images"] = [layout_images[layout_images_id]]
                input_data_out.data["layout_collect_idx"] = layout_images_id
                self.send_to_next_module(input_data_out)