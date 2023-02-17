"""
Prediction pipeline for detection tasks.
"""
from mindspore import Tensor
from args import get_args
from mindocr.data.det_dataset import transforms_dbnet_icdar15
from mindocr.data.transforms.transform_factory import create_transforms, run_transforms

from mindocr.utils.visualize import draw_bboxes
# TODO: change mxocr dir
from mindocr.mxocr.src.demo.python.main import build_pipeline, image_sender


class Detector(object):
    def __init__(self, args) -> None:
        self.det_algorithm = args.det_algorithm
        self.config_path = args.config_path
        self.parallel_num = args.parallel_num
        self.input_queue = args.input_queue
        self.infer_res_save_path = args.infer_res_save_path

        self.transform_pipeline = transforms_dbnet_icdar15(is_train="infer")
        self.transforms = create_transforms(self.transform_pipeline)
        self.post_process = create_postprocess()

    def infer_mxocr(self, img: Tensor):
        """
        infer image using mxocr
        :param img:
        :return:
        """

        return

    # def infer_mslite(self, img):
    #     pass

    def post_processing(self, infer_res: Tensor):


    def __call__(self, img):
        img = run_transforms(img, self.transforms)
        infer_res = self.infer_mxocr(img)
        res = self.post_process(infer_res)

        return res

if __name__ == '__main__':
    # parse args
    args = get_args()
    detector = Detector(args)

    # load data


    # infer

    # visualize
    draw_bboxes(image=, bboxes=, color=(255, 0, 0), thickness=2)

    # logging
