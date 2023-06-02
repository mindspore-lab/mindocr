from ...framework import ModuleBase
from ....infer import TextDetector


class DetInferNode(ModuleBase):
    def __init__(self, args, msg_queue):
        super(DetInferNode, self).__init__(args, msg_queue)
        self.text_detector = None

    def init_self_args(self):
        self.text_detector = TextDetector(self.args)
        self.text_detector.init(warmup=True)
        super().init_self_args()

    def process(self, input_data):
        if input_data.skip:
            self.send_to_next_module(input_data)
            return

        data = input_data.data
        pred = self.text_detector.model_infer(data)

        input_data.data = {"pred": pred, "shape_list": data["shape_list"]}

        self.send_to_next_module(input_data)
