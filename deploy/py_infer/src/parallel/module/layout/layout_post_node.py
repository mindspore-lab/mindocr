from ....infer import LayoutPredictor
from ...framework import ModuleBase


class LayoutPostNode(ModuleBase):
    def __init__(self, args, msg_queue):
        super(LayoutPostNode, self).__init__(args, msg_queue)
        self.layout_predictor = None
        self.task_type = self.args.task_type

    def init_self_args(self):
        self.layout_predictor = LayoutPredictor(self.args)
        self.layout_predictor.init(preprocess=False, model=False, postprocess=True)
        super().init_self_args()

    def process(self, input_data):
        if input_data.skip:
            self.send_to_next_module(input_data)
            return

        data = input_data.data

        data["image_ids"] = input_data.image_path
        output = self.layout_predictor.postprocess(
            data["pred"][0], data["img_shape"], data["image_ids"], data["hw_ori"], data["hw_scale"], data["pad"]
        )
        input_data.infer_result = output
        input_data.data = None
        self.send_to_next_module(input_data)
