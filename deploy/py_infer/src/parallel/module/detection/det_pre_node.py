from ...framework import ModuleBase
from ....infer import TextDetector


class DetPreNode(ModuleBase):
    def __init__(self, args, msg_queue):
        super(DetPreNode, self).__init__(args, msg_queue)
        self.text_detector = None

    def init_self_args(self):
        self.text_detector = TextDetector(self.args)
        self.text_detector.init()
        self.text_detector.free_model()

        super().init_self_args()

    def process(self, input_data):
        if input_data.skip:
            self.send_to_next_module(input_data)
            return

        input = input_data.frame
        output = self.text_detector.preprocess(input)

        input_data.input = output

        self.send_to_next_module(input_data)
