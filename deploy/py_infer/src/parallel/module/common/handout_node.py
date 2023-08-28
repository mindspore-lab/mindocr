import os

from ....utils import log
from ...datatype import ProcessData, StopData, StopSign
from ...framework.module_base import ModuleBase


class HandoutNode(ModuleBase):
    def __init__(self, args, msg_queue):
        super().__init__(args, msg_queue)
        self.image_total = 0

    def init_self_args(self):
        super().init_self_args()

    def process(self, input_data):
        if isinstance(input_data, (tuple, list)):
            image_path = input_data
            log.info(f"sending {', '.join([os.path.basename(x) for x in image_path])} to pipleine")
            data = ProcessData(image_path=input_data)
            self.image_total += len(image_path)
        elif isinstance(input_data, StopSign):
            data = StopData(skip=True, image_total=self.image_total)
        else:
            raise ValueError("unknown input data")

        self.send_to_next_module(data)
