from args import get_args
from mindocr.data.det_dataset import transforms_dbnet_icdar15
from mindocr.data.transforms.transform_factory import create_transforms
from mindocr.utils.visualize import draw_bboxes


class Detector(object):
    def __init__(self, args) -> None:
        self.det_algorithm = args.det_algorithm
        self.transform_pipeline = transforms_dbnet_icdar15(is_train="infer")
        self.transforms = create_transforms(self.transform_pipeline)

    def infer_mslite(self, img):


    def infer_mxocr(self, img):

        return


    def post_processing(self, infer_res):


    def __call__(self, img):

        self.infer_mslite(img)
        self.infer_mxocr(img)
        self.post_processing()


if __name__ == '__main__':

