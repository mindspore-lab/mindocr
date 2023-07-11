from ....data_process.utils import gear_utils
from ....infer import TaskType, TextClassifier
from ...datatype import ProcessData
from ...framework import ModuleBase


class ClsPreNode(ModuleBase):
    def __init__(self, args, msg_queue):
        super(ClsPreNode, self).__init__(args, msg_queue)
        self.text_classifier = None
        self.task_type = self.args.task_type

    def init_self_args(self):
        self.text_classifier = TextClassifier(self.args)
        self.text_classifier.init(preprocess=True, model=False, postprocess=False)
        super().init_self_args()
        return self.text_classifier.get_params()

    def process(self, input_data):
        """
        split the sub image list to chunks by batch size and do the preprocess.
        """
        if input_data.skip:
            self.send_to_next_module(input_data)
            return

        if self.task_type == TaskType.CLS:
            self.process_with_single_cls(input_data)
        else:
            self.process_with_det_cls_rec(input_data)

    def process_with_single_cls(self, input_data):
        images = input_data.frame
        _, split_data = self.text_classifier.preprocess(images)

        # len(images) <= cls_batch_num, so len(split_data) == 1
        send_data = ProcessData(
            data=split_data[0],
            image_path=input_data.image_path,
        )

        self.send_to_next_module(send_data)

    def process_with_det_cls_rec(self, input_data):
        sub_images = input_data.sub_image_list
        sub_results = input_data.infer_result
        split_sub_bs, split_sub_data = self.text_classifier.preprocess(sub_images)

        split_sub_images = gear_utils.split_by_size(sub_images, split_sub_bs)
        split_sub_results = gear_utils.split_by_size(sub_results, split_sub_bs)

        for split_image, split_data, split_result in zip(split_sub_images, split_sub_data, split_sub_results):
            send_data = ProcessData(
                sub_image_size=len(split_image),
                sub_image_list=split_image,
                infer_result=split_result,
                data=split_data,
                image_path=input_data.image_path,
                frame=input_data.frame,
                sub_image_total=input_data.sub_image_total,
            )
            self.send_to_next_module(send_data)
