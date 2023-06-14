from ....data_process.utils import gear_utils
from ....infer import TaskType, TextRecognizer
from ...datatype import ProcessData
from ...framework import ModuleBase


class RecPreNode(ModuleBase):
    def __init__(self, args, msg_queue):
        super(RecPreNode, self).__init__(args, msg_queue)
        self.text_recognizer = None
        self.task_type = self.args.task_type

    def init_self_args(self):
        self.text_recognizer = TextRecognizer(self.args)
        self.text_recognizer.init()
        self.text_recognizer.free_model()

        super().init_self_args()

    def process(self, input_data):
        """
        split the sub image list to chunks by batch size and do the preprocess.
        If use dynamic model, the batch size will be the size of whole sub images list
        """
        if input_data.skip:
            self.send_to_next_module(input_data)
            return

        if self.task_type == TaskType.REC:
            self.process_with_single_rec(input_data)
        else:
            self.process_with_det_rec(input_data)

    def process_with_single_rec(self, input_data):
        images = [input_data.frame]
        _, split_data = self.text_recognizer.preprocess(images)

        send_data = ProcessData(
            sub_image_size=1,
            image_path=input_data.image_path,
            image_total=input_data.image_total,
            data=split_data[0],
            frame=input_data.frame,
            sub_image_total=1,
            image_name=input_data.image_name,
            image_id=input_data.image_id,
        )

        self.send_to_next_module(send_data)

    def process_with_det_rec(self, input_data):
        sub_images = input_data.sub_image_list
        sub_results = input_data.infer_result

        split_sub_bs, split_sub_data = self.text_recognizer.preprocess(sub_images)

        split_sub_images = gear_utils.split_by_size(sub_images, split_sub_bs)
        split_sub_results = gear_utils.split_by_size(sub_results, split_sub_bs)

        for split_image, split_data, split_result in zip(split_sub_images, split_sub_data, split_sub_results):
            send_data = ProcessData(
                sub_image_size=len(split_image),
                image_path=input_data.image_path,
                image_total=input_data.image_total,
                infer_result=split_result,
                data=split_data,
                frame=input_data.frame,
                sub_image_total=input_data.sub_image_total,
                image_name=input_data.image_name,
                image_id=input_data.image_id,
            )
            self.send_to_next_module(send_data)
