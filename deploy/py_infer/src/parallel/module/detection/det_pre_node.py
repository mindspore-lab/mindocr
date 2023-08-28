from ....infer import TaskType, TextDetector
from ...framework import ModuleBase


class DetPreNode(ModuleBase):
    def __init__(self, args, msg_queue):
        super(DetPreNode, self).__init__(args, msg_queue)
        self.text_detector = None
        self.task_type = self.args.task_type

    def init_self_args(self):
        self.text_detector = TextDetector(self.args)
        self.text_detector.init(preprocess=True, model=False, postprocess=False)
        super().init_self_args()
        return self.text_detector.get_params()

    def process(self, input_data):
        if input_data.skip:
            self.send_to_next_module(input_data)
            return

        image = input_data.frame[0]  # bs = 1 for det
        data = self.text_detector.preprocess(image)

        if self.task_type == TaskType.DET and not (self.args.crop_save_dir or self.args.vis_det_save_dir):
            input_data.frame = None

        input_data.data = data

        self.send_to_next_module(input_data)
