import os.path

import numpy as np

from mx_infer.framework import ModuleBase, InferModelComb
from mx_infer.utils import array_to_texts, file_base_check, log


class RecPostProcess(ModuleBase):
    def __init__(self, args, msg_queue):
        super(RecPostProcess, self).__init__(args, msg_queue)
        self.without_input_queue = False
        self.labels = [' ']
        self.task_type = args.task_type

    def init_self_args(self):
        label_path = self.args.rec_char_dict_path
        if label_path and os.path.isfile(label_path):
            file_base_check(label_path)
        else:
            raise FileNotFoundError('rec_char_dict_path must be a file')
        with open(label_path, 'r', encoding='utf8') as f:
            for line in f.readlines():
                line = line.strip("\n").strip("\r\n")
                self.labels.append(line)
        self.labels.append(' ')
        super().init_self_args()

    def process(self, input_data):
        if input_data.skip:
            self.send_to_next_module(input_data)
            return

        output_array = input_data.output_array
        if len(output_array.shape) == 3:
            output_array = np.argmax(output_array, axis=2, keepdims=False)
            log.warn(
                f'Running argmax operator in cpu. Please use the insert_argmax script to add the argmax operator '
                f'into the model to improve the inference performance.')

        rec_result = array_to_texts(output_array, self.labels, input_data.sub_image_size)

        if self.task_type == InferModelComb.REC:
            input_data.infer_result = rec_result
        else:
            for coord, text in zip(input_data.infer_result, rec_result):
                coord.append(text)

        self.send_to_next_module(input_data)
