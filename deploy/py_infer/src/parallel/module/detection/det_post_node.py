from ...framework import ModuleBase
from ....data_process.utils import cv_utils
from ....infer import TextDetector, TaskType


class DetPostNode(ModuleBase):
    def __init__(self, args, msg_queue):
        super(DetPostNode, self).__init__(args, msg_queue)
        self.text_detector = None
        self.task_type = self.args.task_type

    def init_self_args(self):
        self.text_detector = TextDetector(self.args)
        self.text_detector.init()
        self.text_detector.free_model()
        super().init_self_args()

    def process(self, input_data):
        if input_data.skip:
            self.send_to_next_module(input_data)
            return

        data = input_data.data
        boxes = self.text_detector.postprocess(data["pred"], data["shape_list"])

        infer_res_list = []
        for box in boxes:
            infer_res_list.append(box.tolist())

        input_data.infer_result = infer_res_list
        input_data.sub_image_total = len(infer_res_list)
        input_data.sub_image_size = len(infer_res_list)

        if self.task_type in (TaskType.DET_REC, TaskType.DET_CLS_REC):
            image = input_data.frame
            sub_image_list = []
            for box in boxes:
                sub_image = cv_utils.crop_box_from_image(image, box)
                sub_image_list.append(sub_image)
            input_data.sub_image_list = sub_image_list

        input_data.data = None

        if not (self.args.save_crop_res_dir
                or self.args.save_vis_det_save_dir
                or self.args.save_vis_pipeline_save_dir):
            input_data.frame = None

        if not infer_res_list:
            input_data.skip = True

        self.send_to_next_module(input_data)
