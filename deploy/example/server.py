import os
import time
import sys
import numpy as np

mindocr_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, mindocr_path)

from infer_args import get_args
from deploy.py_infer.src.parallel import ParallelPipeline  # noqa

from deploy.py_infer.src.data_process.utils import cv_utils 


class OCRServer:
    def __init__(self) -> None:
        self.args = get_args()

    def warmup(self):
        self.parallel_pipeline = ParallelPipeline(self.args)
        self.parallel_pipeline.start_pipeline()
    
    # 用户传入img，路径或图片array
    def infer(self, img):
        if isinstance(img, str):
            self.parallel_pipeline.infer_for_images(img)
        elif isinstance(img, np.ndarray):
            self.parallel_pipeline.infer_for_array(img)
        elif isinstance(img, (tuple, list)) and len(img) > 0 and \
            cv_utils.check_type_in_container(img, np.ndarray):
            self.parallel_pipeline.infer_for_array(img)
        else:
            raise ValueError(f"unknown input data: {type(img)}")

    def stop(self):
        self.parallel_pipeline.stop_pipeline()

if __name__ == "__main__":
    server = OCRServer()
    server.warmup()

    # 输入为ndarray
    img_path = "/home/zhq/datasets/id_cards/101.jpg"
    img = cv_utils.img_read(img_path)
    server.infer(img)
    print(f"infering ndarray: {img_path}")

    # 输入为多ndarray
    img_path1 = "/home/zhq/datasets/id_cards/101.jpg"
    img1 = cv_utils.img_read(img_path1)
    img_path2 = "/home/zhq/datasets/id_cards/102.jpg"
    img2 = cv_utils.img_read(img_path2)
    server.infer([img1, img2])

    # # 输入为图片path
    # img_path = "/home/zhq/datasets/id_cards/100.jpg"
    # server.infer(img_path)
    # print(f"infering img path: {img_path}")

    time.sleep(30)
    # 输入为图片文件夹
    img_path = "/home/zhq/datasets/other_lang"
    server.infer(img_path)
    print(f"infering img fold: {img_path}")

    # 模拟用户暂停输入
    time.sleep(100)
    img_path = "/home/zhq/datasets/id_cards/10.jpg"
    server.infer(img_path)




    server.stop()


