import os

from ....utils import log
from ...datatype import ProcessData, StopData, StopSign
from ...framework.module_base import ModuleBase


class HandoutNode(ModuleBase):
    def __init__(self, args, msg_queue):
        super().__init__(args, msg_queue)
        self.image_id = 0
        self.image_total = 0

    def init_self_args(self):
        super().init_self_args()

    def process(self, input_data):
        if isinstance(input_data, str):
            image_path = input_data
            base_name = os.path.basename(image_path)
            log.info(f"sending {base_name} to pipleine")
            data = ProcessData(
                image_path=image_path, image_name=base_name, image_id=self.image_id, image_total=self.image_total
            )
            self.image_id += 1
        elif isinstance(input_data, StopSign):
            data = StopData(skip=True, image_total=self.image_id)
        else:
            raise ValueError("unknown input data")
        self.send_to_next_module(data)
