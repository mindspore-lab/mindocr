from ....infer import TextRecognizer
from ...framework import ModuleBase


class RecInferNode(ModuleBase):
    def __init__(self, args, msg_queue):
        super(RecInferNode, self).__init__(args, msg_queue)
        self.text_recognizer = None

    def init_self_args(self):
        self.text_recognizer = TextRecognizer(self.args)
        self.text_recognizer.init(warmup=True)

        super().init_self_args()

    def process(self, input_data):
        if input_data.skip:
            self.send_to_next_module(input_data)
            return

        data = input_data.data

        pred = self.text_recognizer.model_infer(data)

        input_data.data = {"pred": pred}

        self.send_to_next_module(input_data)
