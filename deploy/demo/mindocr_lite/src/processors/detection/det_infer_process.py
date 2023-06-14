from ...framework import Model, ModuleBase


class DetInferProcess(ModuleBase):
    def __init__(self, args, msg_queue):
        super(DetInferProcess, self).__init__(args, msg_queue)
        self.model = None

    def init_self_args(self):
        self.model = Model(
            engine_type=self.args.engine_type, model_path=self.args.det_model_path, device_id=self.args.device_id
        )

        super().init_self_args()

    def process(self, input_data):
        if input_data.skip:
            self.send_to_next_module(input_data)
            return

        input_array = input_data.input_array

        outputs = self.model.infer(input_array)
        output_array = outputs[0]

        # send the ready data to post module
        input_data.output_array = output_array
        input_data.input_array = None
        self.send_to_next_module(input_data)
