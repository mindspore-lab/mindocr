import math
import os

from ...data_type import ProcessData
from ...framework import InferModelComb, Model, ModuleBase, ShapeType
from ...operators import build_preprocess
from ...utils import check_valid_dir, get_batch_list_greedy, get_hw_of_img, get_matched_gear_hw, padding_batch, safe_div


class RecPreProcess(ModuleBase):
    def __init__(self, args, msg_queue):
        super(RecPreProcess, self).__init__(args, msg_queue)
        self.gear_list = []
        self.batchsize_list = []
        self.max_dot_gear = None
        self.task_type = args.task_type
        self.shape_type = None

    def get_shape_for_single_model(self, filename):
        model = Model(engine_type=self.args.engine_type, model_path=filename, device_id=self.args.device_id)

        shape_type, shape_info = model.get_shape_info()
        del model

        self.shape_type = shape_type
        if shape_type == ShapeType.DYNAMIC_BATCHSIZE:
            raise ValueError(
                f"Input shape don't support dynamic batch_size for single recognition model, "
                f"but got dynamic batch_size={shape_info[0]} for {filename}."
            )

        if shape_type == ShapeType.STATIC_SHAPE:
            n, _, h, w = shape_info
            self.gear_list = [(h, w)]
            self.batchsize_list = [n]
        elif shape_type == ShapeType.DYNAMIC_IMAGESIZE:
            n, _, hw_list = shape_info
            self.gear_list = list(hw_list)
            self.batchsize_list.append(n)
        else:  # dynamic shape
            n, _, h, w = shape_info
            self.gear_list = [(h, w)]
            self.batchsize_list = [n]

        return shape_info

    def init_self_args(self):
        model_path = self.args.rec_model_path

        if os.path.isfile(model_path):
            self.get_shape_for_single_model(model_path)

        if os.path.isdir(model_path):
            check_valid_dir(model_path)
            chw_list = []
            for path in os.listdir(model_path):
                shape = self.get_shape_for_single_model(os.path.join(model_path, path))
                if self.shape_type in (ShapeType.STATIC_SHAPE, ShapeType.DYNAMIC_SHAPE):
                    raise FileNotFoundError(
                        f"rec_model_dir must be a file when use static or dynamic shape for recognition model, "
                        f"but got rec_model_dir={model_path} is a dir."
                    )
                chw_list.append(str((shape[2:])))

            if len(set(chw_list)) != 1 or len(set(self.batchsize_list)) != len(self.batchsize_list):
                raise ValueError(
                    f"Input shape must have same image_size and different batch_size when use the combination of "
                    f"dynamic batch_size and image_size for recognition model. "
                    f"Please check every model file in {model_path}."
                )

        self.batchsize_list.sort()
        self.max_dot_gear = self.gear_list[-1]

        self.preprocess = build_preprocess(self.args.rec_algorithm)

        super().init_self_args()

    def get_resized_hw(self, image_list):
        if self.shape_type != ShapeType.DYNAMIC_SHAPE:
            resized_hw_list = [
                get_matched_gear_hw(get_hw_of_img(image), self.gear_list, self.max_dot_gear) for image in image_list
            ]
            max_h, max_w = max(resized_hw_list, key=lambda x: x[0] * x[1])
        else:
            model_h, model_w = self.gear_list[0]
            hw_list = [get_hw_of_img(image) for image in image_list]
            max_h = model_h if model_h > 0 else math.ceil(safe_div(max([h for h, _ in hw_list]), 32)) * 32
            max_w = model_w if model_w > 0 else math.ceil(safe_div(max([w for _, w in hw_list]), 32)) * 32

        return max_h, max_w

    def process(self, input_data):
        """
        split the sub image list to chunks by batch size and do the preprocess.
        If use dynamic model, the batch size will be the size of whole sub images list
        :param input_data: ProcessData
        :return: None
        """
        if input_data.skip:
            self.send_to_next_module(input_data)
            return

        if self.task_type == InferModelComb.REC:
            self.process_without_sub_image(input_data)
        else:
            self.process_with_sub_image(input_data)

    def process_without_sub_image(self, input_data):
        split_input = [input_data.frame]
        resized_params = {"Resize": {"dst_hw": self.get_resized_hw(split_input)}}

        batch = 1 if self.batchsize_list[0] == -1 else self.batchsize_list[0]

        rec_model_inputs = self.preprocess(split_input, resized_params)
        rec_model_inputs = padding_batch(rec_model_inputs, batch)

        send_data = ProcessData(
            sub_image_size=1,
            image_path=input_data.image_path,
            image_total=input_data.image_total,
            input_array=rec_model_inputs,
            frame=input_data.frame,
            sub_image_total=1,
            image_name=input_data.image_name,
            image_id=input_data.image_id,
        )

        self.send_to_next_module(send_data)

    def process_with_sub_image(self, input_data):
        sub_image_list = input_data.sub_image_list
        infer_res_list = input_data.infer_result

        batch_list = (
            get_batch_list_greedy(input_data.sub_image_size, self.batchsize_list)
            if self.batchsize_list[0] > 0
            else [input_data.sub_image_size]
        )

        start_index = 0
        for batch in batch_list:
            upper_bound = min(start_index + batch, input_data.sub_image_size)
            split_input = sub_image_list[start_index:upper_bound]
            split_infer_res = infer_res_list[start_index:upper_bound]
            resized_params = {"Resize": {"dst_hw": self.get_resized_hw(split_input)}}
            rec_model_inputs = self.preprocess(split_input, resized_params)
            rec_model_inputs = padding_batch(rec_model_inputs, batch)

            send_data = ProcessData(
                sub_image_size=min(upper_bound - start_index, batch),
                image_path=input_data.image_path,
                image_total=input_data.image_total,
                infer_result=split_infer_res,
                input_array=rec_model_inputs,
                frame=input_data.frame,
                sub_image_total=input_data.sub_image_total,
                image_name=input_data.image_name,
                image_id=input_data.image_id,
            )

            start_index += batch
            self.send_to_next_module(send_data)
