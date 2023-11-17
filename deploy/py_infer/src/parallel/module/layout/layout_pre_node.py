from ....infer import LayoutPredictor
from ...datatype import ProcessData
from ...framework import ModuleBase


class LayoutPreNode(ModuleBase):
    def __init__(self, args, msg_queue):
        super(LayoutPreNode, self).__init__(args, msg_queue)
        self.layout_predictor = None
        self.task_type = self.args.task_type

    def init_self_args(self):
        self.layout_predictor = LayoutPredictor(self.args)
        self.layout_predictor.init(preprocess=True, model=False, postprocess=False)
        super().init_self_args()
        return self.layout_predictor.get_params()

    def process(self, input_data):
        """
        split the sub image list to chunks by batch size and do the preprocess.
        If use dynamic model, the batch size will be the size of whole sub images list
        """
        if input_data.skip:
            self.send_to_next_module(input_data)
            return

        images = input_data.frame
        _, split_data = self.layout_predictor.preprocess(images)

        send_data = ProcessData(
            data=split_data[0],
            image_path=input_data.image_path,
        )

        self.send_to_next_module(send_data)
