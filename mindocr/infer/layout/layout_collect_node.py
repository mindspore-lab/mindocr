import copy
import os
import sys

import numpy as np
import time

from collections import defaultdict

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../../")))

from pipeline.framework.module_base import ModuleBase
from pipeline.tasks import TaskType
from mindocr.infer.utils.collector import Collector
from pipeline.datatype import ProcessData, ProfilingData, StopData

class LayoutCollectNode(ModuleBase):
    def __init__(self, args, msg_queue, tqdm_info):
        super(LayoutCollectNode, self).__init__(args, msg_queue, tqdm_info)
        self.task_type = self.args.task_type
        self.collect_dict = defaultdict(Collector)

    def init_self_args(self):
        super().init_self_args()

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
        # if input_data.skip:
        #     self.send_to_next_module(input_data)
        #     return


        if isinstance(input_data, StopData):
            self.send_to_next_module(input_data)
            return

        data = input_data.data
        image_path = input_data.image_path[0]
        if image_path in self.collect_dict.keys():
            self.collect_dict[image_path].update(input_data.data["layout_collect_idx"], input_data)
        else:
            self.collect_dict[image_path].init_keys(input_data.data["layout_collect_list"])
            self.collect_dict[image_path].update(input_data.data["layout_collect_idx"], input_data)

        if self.collect_dict[image_path].complete():
            data = self.collect_dict[image_path].get_data()
            self.collect_dict.pop(image_path)
        
            input_data_out = copy.deepcopy(input_data)
            input_data_out.data = {"layout_collect_res": data}
            self.send_to_next_module(input_data_out)
