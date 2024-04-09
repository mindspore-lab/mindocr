import copy
import os
import sys
import time
from ctypes import c_uint64
from multiprocessing import Lock, Manager, Process

import numpy as np

mindocr_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.insert(0, mindocr_path)

from infer_args import get_args  # noqa

from deploy.py_infer.src.data_process.utils import cv_utils  # noqa
from deploy.py_infer.src.infer import TaskType  # noqa
from deploy.py_infer.src.parallel import ParallelPipeline  # noqa


class OCRServer:
    def __init__(self) -> None:
        self.lock = Lock()
        self.update_lock = Lock()
        self.args = get_args()
        self.task_id = Manager().Value(c_uint64, 0)
        self.result = Manager().dict()

    def warmup(self):
        self.parallel_pipeline = ParallelPipeline(self.args)
        self.parallel_pipeline.start_pipeline()

    def infer(self, img):
        self.lock.acquire()
        task_id = copy.copy(self.task_id.value)
        self.task_id.value += 1
        self.lock.release()
        data_type = 0  # str or list[str]
        if isinstance(img, str):
            self.parallel_pipeline.infer_for_images(img, task_id)
        elif isinstance(img, np.ndarray):
            self.parallel_pipeline.infer_for_array(img, task_id)
            data_type = 1  # ndarray
        elif isinstance(img, (tuple, list)) and len(img) > 0 and cv_utils.check_type_in_container(img, np.ndarray):
            self.parallel_pipeline.infer_for_array(img, task_id)
            data_type = 2  # list[ndarray]
        else:
            raise ValueError(f"unknown input data: {type(img)}")
        rst = self.fetch_result(task_id)
        rst = self.produce_output(rst, data_type)
        return rst

    def fetch_result(self, task_id):
        while task_id not in self.result.keys():
            self.update_lock.acquire()
            rst = self.parallel_pipeline.fetch_result()
            if rst:
                self.result.update(rst)
            self.update_lock.release()
            time.sleep(self.args.node_fetch_interval)
        rst = self.result[task_id]
        self.update_lock.acquire()
        self.result.pop(task_id)
        self.update_lock.release()
        return rst

    def produce_output(self, rst, data_type):
        def produce_each_sample(sample):
            """
            sample: List[{"transcription": transcription, "points": points, "score": score}]
            """
            output = []

            if self.parallel_pipeline.args.task_type in [TaskType.DET, TaskType.CLS, TaskType.REC]:
                return sample
            if self.parallel_pipeline.args.task_type in [TaskType.DET_REC, TaskType.DET_CLS_REC]:
                for data in sample:
                    sub_data = []
                    if "points" in data.keys():
                        sub_data.append(data["points"])
                    rec_out = []
                    if "score" in data.keys():
                        rec_out.append(data["score"])
                    if "transcription" in data.keys():
                        rec_out.append(data["transcription"])
                    sub_data.append(tuple(rec_out))
                    output.append(sub_data)
                return output

        if len(rst) > 0 and data_type == 2:  # list[np.ndarray]
            output_list = [None] * len(rst)
            for idx, value in rst.items():
                output_list[int(idx)] = produce_each_sample(value)
            return output_list
        else:  # list[str_path]
            output_dict = dict()
            for path, value in rst.items():
                output_dict[path] = produce_each_sample(value)
            return output_dict

    def stop(self):
        self.parallel_pipeline.stop_pipeline()


def test_multi_ocr_system():
    def call1(server):
        arg1 = "deploy/py_infer/example/dataset/det/example1.png"
        rst = server.infer(arg1)
        print(rst, "\n")

    def call2(server):
        arg2 = "deploy/py_infer/example/dataset/det/example2.png"
        rst = server.infer(arg2)
        print(rst, "\n")

    server1 = OCRServer()
    server1.warmup()
    server2 = OCRServer()
    server2.warmup()

    process_list = []
    # process 1
    process_list.append(
        Process(
            target=call1,
            args=(server1,),
            daemon=True,
        )
    )
    # process 2
    process_list.append(
        Process(
            target=call2,
            args=(server2,),
            daemon=True,
        )
    )
    for process in process_list:
        process.start()

    for process in process_list:
        process.join()

    server1.stop()
    server2.stop()


def test_multi_infer_call_one_ocr_system():
    def call1(server):
        arg1 = "deploy/py_infer/example/dataset/det/example1.png"
        rst = server.infer(arg1)
        print(rst, "\n")

    def call2(server):
        arg2 = "deploy/py_infer/example/dataset/det/example2.png"
        rst = server.infer(arg2)
        print(rst, "\n")

    server = OCRServer()
    server.warmup()

    process_list = []
    # process 1
    process_list.append(
        Process(
            target=call1,
            args=(server,),
            daemon=True,
        )
    )
    # process 2
    process_list.append(
        Process(
            target=call2,
            args=(server,),
            daemon=True,
        )
    )
    for process in process_list:
        process.start()

    for process in process_list:
        process.join()

    server.stop()


def test_ndarray_list_input():
    def call1(server):
        img_path1 = "deploy/py_infer/example/dataset/det/example1.png"
        img1 = cv_utils.img_read(img_path1)
        img_path2 = "deploy/py_infer/example/dataset/det/example2.png"
        img2 = cv_utils.img_read(img_path2)
        img_path3 = "deploy/py_infer/example/dataset/det/example3.png"
        img3 = cv_utils.img_read(img_path3)
        rst = server.infer([img1, img2, img3])
        print(rst, "\n")

    def call2(server):
        img_path1 = "deploy/py_infer/example/dataset/det/example2.png"
        img1 = cv_utils.img_read(img_path1)
        img_path2 = "deploy/py_infer/example/dataset/det/example3.png"
        img2 = cv_utils.img_read(img_path2)
        img_path3 = "deploy/py_infer/example/dataset/det/example4.png"
        img3 = cv_utils.img_read(img_path3)
        rst = server.infer([img1, img2, img3])
        print(rst, "\n")

    server = OCRServer()
    server.warmup()

    process_list = []
    # process 1
    process_list.append(
        Process(
            target=call1,
            args=(server,),
            daemon=True,
        )
    )
    # process 2
    process_list.append(
        Process(
            target=call2,
            args=(server,),
            daemon=True,
        )
    )

    for process in process_list:
        process.start()

    for process in process_list:
        process.join()

    server.stop()


def test_feed_folder():
    def call1(server):
        folder_path1 = "deploy/py_infer/example/dataset/det"
        rst = server.infer(folder_path1)
        print(rst, "\n")

    def call2(server):
        folder_path1 = "deploy/py_infer/example/dataset/det"
        rst = server.infer(folder_path1)
        print(rst, "\n")

    server = OCRServer()
    server.warmup()

    process_list = []
    # process 1
    process_list.append(
        Process(
            target=call1,
            args=(server,),
            daemon=True,
        )
    )
    # process 2
    process_list.append(
        Process(
            target=call2,
            args=(server,),
            daemon=True,
        )
    )

    for process in process_list:
        process.start()

    for process in process_list:
        process.join()

    server.stop()


def test_feed_folder_rec():
    def call1(server):
        folder_path = "deploy/py_infer/example/dataset/cls_rec"
        rst = server.infer(folder_path)
        print(rst, "\n")

    def call2(server):
        folder_path = "deploy/py_infer/example/dataset/cls_rec"
        rst = server.infer(folder_path)
        print(rst, "\n")

    server = OCRServer()
    server.warmup()

    process_list = []
    # process 1
    process_list.append(
        Process(
            target=call1,
            args=(server,),
            daemon=True,
        )
    )

    # process 2
    process_list.append(
        Process(
            target=call2,
            args=(server,),
            daemon=True,
        )
    )

    for process in process_list:
        process.start()

    for process in process_list:
        process.join()

    server.stop()


def test_feed_folder_cls():
    def call1(server):
        folder_path = "deploy/py_infer/example/dataset/cls_rec"
        rst = server.infer(folder_path)
        print(rst, "\n")

    def call2(server):
        folder_path = "deploy/py_infer/example/dataset/cls_rec"
        rst = server.infer(folder_path)
        print(rst, "\n")

    server = OCRServer()
    server.warmup()

    process_list = []
    # process 1
    process_list.append(
        Process(
            target=call1,
            args=(server,),
            daemon=True,
        )
    )

    # process 2
    process_list.append(
        Process(
            target=call2,
            args=(server,),
            daemon=True,
        )
    )

    for process in process_list:
        process.start()

    for process in process_list:
        process.join()

    server.stop()


if __name__ == "__main__":
    args = get_args()

    if args.task_type in [TaskType.DET, TaskType.DET_REC, TaskType.DET_CLS_REC]:
        test_multi_infer_call_one_ocr_system()

        test_ndarray_list_input()

        test_multi_ocr_system()

        test_feed_folder()

    if args.task_type in [TaskType.REC]:
        test_feed_folder_rec()

    if args.task_type in [TaskType.CLS]:
        test_feed_folder_cls()
