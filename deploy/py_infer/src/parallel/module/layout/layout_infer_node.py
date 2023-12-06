from ....infer import LayoutPredictor
from ...framework import ModuleBase


class LayoutInferNode(ModuleBase):
    def __init__(self, args, msg_queue):
        super(LayoutInferNode, self).__init__(args, msg_queue)
        self.layout_predictor = None

    def init_self_args(self):
        self.layout_predictor = LayoutPredictor(self.args)
        self.layout_predictor.init(preprocess=True, model=True, postprocess=True)

        super().init_self_args()

    def process(self, input_data):
        if input_data.skip:
            self.send_to_next_module(input_data)
            return

        data = input_data.data

        pred = self.layout_predictor.model_infer(data)
        net_inputs = data["net_inputs"]
        img_shape = net_inputs[0].shape
        input_data.data = {
            "pred": pred,
            "img_shape": img_shape,
            "image_ids": net_inputs[1],
            "hw_ori": net_inputs[2],
            "hw_scale": net_inputs[3],
            "pad": net_inputs[4],
        }

        self.send_to_next_module(input_data)
