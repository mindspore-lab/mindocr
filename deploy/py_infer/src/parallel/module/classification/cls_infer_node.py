import cv2

from ....infer import TaskType, TextClassifier
from ...framework import ModuleBase


class ClsInferNode(ModuleBase):
    def __init__(self, args, msg_queue):
        super(ClsInferNode, self).__init__(args, msg_queue)
        self.text_classifier = None
        self.cls_thresh = 0.9
        self.task_type = self.args.task_type

    def init_self_args(self):
        self.text_classifier = TextClassifier(self.args)
        self.text_classifier.init(preprocess=False, model=True, postprocess=True)
        super().init_self_args()

    def process(self, input_data):
        if input_data.skip:
            self.send_to_next_module(input_data)
            return

        data = input_data.data
        batch = len(input_data.image_path) if self.task_type == TaskType.CLS else input_data.sub_image_size

        pred = self.text_classifier.model_infer(data)

        # for convenience or performance, postprocess is placed in InferNode
        output = self.text_classifier.postprocess(pred, batch)

        if self.task_type == TaskType.DET_CLS_REC:
            sub_images = input_data.sub_image_list
            for i in range(batch):
                label, score = output[i]
                if "180" == label and score > self.cls_thresh:
                    sub_images[i] = cv2.rotate(sub_images[i], cv2.ROTATE_180)
            input_data.sub_image_list = sub_images
        else:
            input_data.infer_result = output

        input_data.data = None

        self.send_to_next_module(input_data)
