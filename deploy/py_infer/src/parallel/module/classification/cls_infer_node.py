import cv2

from ...framework import ModuleBase
from ....data_process.utils.constants import CLS_ROTATE180_THRESHOLD
from ....infer import TextClassifier, TaskType


class ClsInferNode(ModuleBase):
    def __init__(self, args, msg_queue):
        super(ClsInferNode, self).__init__(args, msg_queue)
        self.text_classifier = None
        self.cls_thresh = CLS_ROTATE180_THRESHOLD
        self.task_type = self.args.task_type

    def init_self_args(self):
        self.text_classifier = TextClassifier(self.args)
        self.text_classifier.init(warmup=True)

        super().init_self_args()

    def process(self, input_data):
        if input_data.skip:
            self.send_to_next_module(input_data)
            return

        input = input_data.input
        sub_images = input_data.sub_image_list

        batch = input_data.sub_image_size
        output = self.text_classifier.model_infer(input["image"])

        output = output[0][:batch, ...]
        output = self.text_classifier.postprocess_ops(output)

        if self.task_type == TaskType.DET_CLS_REC:
            for i in range(batch):
                label, score = output[i]
                if '180' == label and score > self.cls_thresh:
                    sub_images[i] = cv2.rotate(sub_images[i], cv2.ROTATE_180)
            input_data.sub_image_list = sub_images
        else:
            input_data.infer_result = output[0]

        input_data.input = None

        self.send_to_next_module(input_data)
