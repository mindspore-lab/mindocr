import os
import sys
import time

import numpy as np

mindocr_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.insert(0, mindocr_path)

from infer_args import get_args  # noqa

from deploy.py_infer.src.data_process.utils import cv_utils  # noqa
from deploy.py_infer.src.parallel import ParallelPipeline  # noqa


class OCRServer:
    def __init__(self) -> None:
        self.args = get_args()

    def warmup(self):
        self.parallel_pipeline = ParallelPipeline(self.args)
        self.parallel_pipeline.start_pipeline()

    def infer(self, img):
        if isinstance(img, str):
            self.parallel_pipeline.infer_for_images(img)
        elif isinstance(img, np.ndarray):
            self.parallel_pipeline.infer_for_array(img)
        elif isinstance(img, (tuple, list)) and len(img) > 0 and cv_utils.check_type_in_container(img, np.ndarray):
            self.parallel_pipeline.infer_for_array(img)
        else:
            raise ValueError(f"unknown input data: {type(img)}")

    def stop(self):
        self.parallel_pipeline.stop_pipeline()


if __name__ == "__main__":
    server = OCRServer()
    server.warmup()

    # input: one ndarray
    img_path = "deploy/py_infer/example/dataset/example1.png"
    img = cv_utils.img_read(img_path)
    server.infer(img)

    # input: ndarray list
    img_path1 = "deploy/py_infer/example/dataset/example1.png"
    img1 = cv_utils.img_read(img_path1)
    img_path2 = "deploy/py_infer/example/dataset/example2.png"
    img2 = cv_utils.img_read(img_path2)
    img_path3 = "deploy/py_infer/example/dataset/example3.png"
    img3 = cv_utils.img_read(img_path3)
    server.infer([img1, img2, img3])

    # input: image path
    img_path = "deploy/py_infer/example/dataset/example1.png"
    server.infer(img_path)

    # input: image folder
    img_path = "deploy/py_infer/example/dataset"
    server.infer(img_path)

    # simulate waiting
    time.sleep(100)

    # input: image path
    img_path = "deploy/py_infer/example/dataset/example2.png"
    server.infer(img_path)

    server.stop()
