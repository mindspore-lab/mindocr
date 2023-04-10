import numpy as np

from ...framework import ModuleBase, InferModelComb
from ...operators import build_postprocess


class RecPostProcess(ModuleBase):
    def __init__(self, args, msg_queue):
        super(RecPostProcess, self).__init__(args, msg_queue)
        self.task_type = args.task_type

    def init_self_args(self):
        params = {"character_dict_path": self.args.rec_char_dict_path}
        self.postprocess = build_postprocess(self.args.rec_algorithm, init_params=params)
        super().init_self_args()

    def process(self, input_data):
        if input_data.skip:
            self.send_to_next_module(input_data)
            return

        output_array = input_data.output_array
        if len(output_array.shape) == 3:
            output_array = np.argmax(output_array, axis=2)

        batch = input_data.sub_image_size
        rec_result = self.postprocess(output_array[:batch, ...])

        if self.task_type == InferModelComb.REC:
            input_data.infer_result = rec_result
        else:
            for coord, text in zip(input_data.infer_result, rec_result):
                coord.append(text)

        self.send_to_next_module(input_data)
