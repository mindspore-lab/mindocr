import os
from collections import defaultdict

from mx_infer.framework import ModuleBase, Model, ShapeType
from mx_infer.utils import check_valid_dir


class RecInferProcess(ModuleBase):
    def __init__(self, args, msg_queue):
        super(RecInferProcess, self).__init__(args, msg_queue)
        self.model_list = defaultdict()
        self.gear_method = True

    def get_single_model(self, filename):
        model = Model(engine_type=self.args.engine_type, model_path=filename,
                      device_id=self.args.device_id)

        shape_type, shape_info = model.get_shape_info()

        if shape_type == ShapeType.DYNAMIC_SHAPE:
            self.gear_method = False
            self.model_list[-1] = model
        else:
            self.gear_method = True
            self.model_list[shape_info[0]] = model

        return model

    def init_self_args(self):
        model_path = self.args.rec_model_path

        if os.path.isfile(model_path):
            self.get_single_model(model_path)

        if os.path.isdir(model_path):
            check_valid_dir(model_path)
            for path in os.listdir(model_path):
                self.get_single_model(os.path.join(model_path, path))

        super().init_self_args()

    def process(self, input_data):
        if input_data.skip:
            self.send_to_next_module(input_data)
            return

        input_array = input_data.input_array
        batchsize, _, _, _ = input_array.shape
        if self.gear_method:
            if batchsize not in self.model_list:
                return
            outputs = self.model_list[batchsize].infer(input_array)
        else:
            outputs = self.model_list[-1].infer(input_array)
        output_array = outputs[0]
        # send the ready data to post module
        input_data.output_array = output_array
        input_data.input_array = None
        self.send_to_next_module(input_data)
