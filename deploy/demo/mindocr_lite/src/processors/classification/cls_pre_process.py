import functools

from ...data_type import ProcessData
from ...framework import Model, ModuleBase, ShapeType
from ...operators import build_preprocess
from ...utils import get_batch_list_greedy, padding_batch


class CLSPreProcess(ModuleBase):
    def __init__(self, args, msg_queue):
        super(CLSPreProcess, self).__init__(args, msg_queue)
        self.batchsize_list = []

    def init_self_args(self):
        model = Model(
            engine_type=self.args.engine_type, model_path=self.args.cls_model_path, device_id=self.args.device_id
        )
        shape_type, shape_info = model.get_shape_info()
        del model

        if shape_type not in (ShapeType.DYNAMIC_BATCHSIZE, ShapeType.STATIC_SHAPE):
            raise ValueError("Input shape must be static shape or dynamic batch_size for classification model.")

        if shape_type == ShapeType.DYNAMIC_BATCHSIZE:
            self.batchsize_list, _, model_height, model_width = shape_info
        else:
            batchsize, _, model_height, model_width = shape_info
            self.batchsize_list = [batchsize]

        resized_params = {"Resize": {"dst_hw": (model_height, model_width)}}
        self.preprocess = build_preprocess(algorithm="CLS")
        self.preprocess = functools.partial(self.preprocess, extra_params=resized_params)

        super().init_self_args()

    def process(self, input_data):
        """
        split the sub image list to chunks by batch size and do the preprocess.
        :param input_data: ProcessData
        :return: None
        """
        if input_data.skip:
            self.send_to_next_module(input_data)
            return

        sub_image_list = input_data.sub_image_list
        infer_res_list = input_data.infer_result
        batch_list = get_batch_list_greedy(input_data.sub_image_size, self.batchsize_list)
        start_index = 0
        for batch in batch_list:
            upper_bound = min(start_index + batch, input_data.sub_image_size)
            split_input = sub_image_list[start_index:upper_bound]
            split_infer_res = infer_res_list[start_index:upper_bound]
            cls_model_inputs = self.preprocess(split_input)
            cls_model_inputs = padding_batch(cls_model_inputs, batch)
            send_data = ProcessData(
                sub_image_size=len(split_input),
                sub_image_list=split_input,
                image_path=input_data.image_path,
                image_total=input_data.image_total,
                infer_result=split_infer_res,
                input_array=cls_model_inputs,
                frame=input_data.frame,
                sub_image_total=input_data.sub_image_total,
                image_name=input_data.image_name,
                image_id=input_data.image_id,
            )

            start_index += batch
            self.send_to_next_module(send_data)
