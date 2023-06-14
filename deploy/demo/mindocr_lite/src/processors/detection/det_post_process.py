import numpy as np

from ...framework import ModuleBase
from ...operators import build_postprocess
from ...utils import get_rotate_crop_image


class DetPostProcess(ModuleBase):
    def __init__(self, args, msg_queue):
        super(DetPostProcess, self).__init__(args, msg_queue)

    def init_self_args(self):
        super().init_self_args()
        self.postprocess = build_postprocess(self.args.det_algorithm)

    def process(self, input_data):
        if input_data.skip:
            self.send_to_next_module(input_data)
            return

        shrink_map = input_data.output_array
        boxes = self.postprocess(shrink_map, input_data.original_width, input_data.original_height)

        sub_image_list = []
        infer_res_list = []
        image = input_data.frame
        for box in boxes:
            infer_res_list.append(box.tolist())
            sub_image = get_rotate_crop_image(image, np.array(box, dtype=np.float32))
            sub_image_list.append(sub_image)

        input_data.sub_image_list = sub_image_list
        input_data.infer_result = infer_res_list
        input_data.sub_image_total = len(sub_image_list)
        input_data.sub_image_size = len(sub_image_list)

        input_data.output_array = None

        if not (
            self.args.save_pipeline_crop_res or self.args.save_vis_det_save_dir or self.args.save_vis_pipeline_save_dir
        ):
            input_data.frame = None

        if not sub_image_list:
            input_data.skip = True
        # send the ready data to post module
        self.send_to_next_module(input_data)
