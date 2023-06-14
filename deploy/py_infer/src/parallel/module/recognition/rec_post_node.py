from ....infer import TaskType, TextRecognizer
from ...framework import ModuleBase


class RecPostNode(ModuleBase):
    def __init__(self, args, msg_queue):
        super(RecPostNode, self).__init__(args, msg_queue)
        self.text_recognizer = None
        self.task_type = self.args.task_type

    def init_self_args(self):
        self.text_recognizer = TextRecognizer(self.args)
        self.text_recognizer.init()
        self.text_recognizer.free_model()

        super().init_self_args()

    def process(self, input_data):
        if input_data.skip:
            self.send_to_next_module(input_data)
            return

        data = input_data.data
        batch = input_data.sub_image_size

        output = self.text_recognizer.postprocess(data["pred"], batch)

        if self.task_type == TaskType.REC:
            # TODO: only support batch=1
            input_data.infer_result = output["texts"][0]
        else:
            texts = output["texts"]
            for result, text in zip(input_data.infer_result, texts):
                result.append(text)

        input_data.data = None

        self.send_to_next_module(input_data)
