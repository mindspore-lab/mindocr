import os
from collections import defaultdict
from ctypes import c_uint64
from multiprocessing import Manager

import numpy as np

from ....data_process.utils import cv_utils
from ....infer import TaskType
from ....utils import log, safe_list_writer, visual_utils
from ...datatype import ProcessData, ProfilingData, StopData
from ...framework import ModuleBase

RESULTS_SAVE_FILENAME = {
    TaskType.DET: "det_results.txt",
    TaskType.CLS: "cls_results.txt",
    TaskType.REC: "rec_results.txt",
    TaskType.DET_REC: "pipeline_results.txt",
    TaskType.DET_CLS_REC: "pipeline_results.txt",
}


class CollectNode(ModuleBase):
    def __init__(self, args, msg_queue):
        super().__init__(args, msg_queue)
        self.image_sub_remaining = defaultdict(int)
        self.image_pipeline_res = defaultdict(list)
        self.infer_size = 0
        self.image_total = Manager().Value(c_uint64, 0)
        self.task_type = args.task_type
        self.res_save_dir = args.res_save_dir

        self.save_filename = RESULTS_SAVE_FILENAME[self.task_type]

    def init_self_args(self):
        super().init_self_args()

    def _collect_stop(self, input_data):
        self.image_total.value = input_data.image_total

    def _vis_results(self, image_name, image):
        if self.args.crop_save_dir:
            basename = os.path.basename(image_name)
            filename = os.path.join(self.args.crop_save_dir, os.path.splitext(basename)[0])
            box_list = [np.array(x["points"]).reshape(-1, 2) for x in self.image_pipeline_res[image_name]]
            crop_list = visual_utils.vis_crop(image, box_list)
            for i, crop in enumerate(crop_list):
                cv_utils.img_write(filename + "_crop_" + str(i) + ".jpg", crop)

        if self.args.vis_pipeline_save_dir:
            basename = os.path.basename(image_name)
            filename = os.path.join(self.args.vis_pipeline_save_dir, os.path.splitext(basename)[0])
            box_list = [np.array(x["points"]).reshape(-1, 2) for x in self.image_pipeline_res[image_name]]
            text_list = [x["transcription"] for x in self.image_pipeline_res[image_name]]
            box_text = visual_utils.vis_bbox_text(image, box_list, text_list, font_path=self.args.vis_font_path)
            cv_utils.img_write(filename + ".jpg", box_text)

        if self.args.vis_det_save_dir:
            basename = os.path.basename(image_name)
            filename = os.path.join(self.args.vis_det_save_dir, os.path.splitext(basename)[0])
            box_list = [np.array(x).reshape(-1, 2) for x in self.image_pipeline_res[image_name]]
            box_line = visual_utils.vis_bbox(image, box_list, [255, 255, 0], 2)
            cv_utils.img_write(filename + ".jpg", box_line)

        log.info(f"{image_name} is finished.")

    def final_text_save(self):
        save_filename = os.path.join(self.res_save_dir, self.save_filename)
        safe_list_writer(self.image_pipeline_res, save_filename)
        log.info(f"save infer result to {save_filename} successfully")

    def _collect_results(self, input_data: ProcessData):
        if self.task_type in (TaskType.DET_REC, TaskType.DET_CLS_REC):
            image_path = input_data.image_path[0]  # bs=1
            for result in input_data.infer_result:
                self.image_pipeline_res[image_path].append({"transcription": result[-1], "points": result[:-1]})
            if not input_data.infer_result:
                self.image_pipeline_res[image_path] = []
        elif self.task_type == TaskType.DET:
            image_path = input_data.image_path[0]  # bs=1
            self.image_pipeline_res[image_path] = input_data.infer_result
        elif self.task_type in (TaskType.REC, TaskType.CLS):
            for image_path, infer_result in zip(input_data.image_path, input_data.infer_result):
                self.image_pipeline_res[image_path] = infer_result
        else:
            raise NotImplementedError("Task type do not support.")

        self._update_remaining(input_data)

    def _update_remaining(self, input_data: ProcessData):
        if self.task_type in (TaskType.DET_REC, TaskType.DET_CLS_REC):  # with sub image
            for idx, image_path in enumerate(input_data.image_path):
                if image_path in self.image_sub_remaining:
                    self.image_sub_remaining[image_path] -= input_data.sub_image_size
                    if not self.image_sub_remaining[image_path]:
                        self.image_sub_remaining.pop(image_path)
                        self.infer_size += 1
                        self._vis_results(image_path, input_data.frame[idx]) if input_data.frame else ...
                else:
                    remaining = input_data.sub_image_total - input_data.sub_image_size
                    if remaining:
                        self.image_sub_remaining[image_path] = remaining
                    else:
                        self.infer_size += 1
                        self._vis_results(image_path, input_data.frame[idx]) if input_data.frame else ...
        else:  # without sub image
            for idx, image_path in enumerate(input_data.image_path):
                self.infer_size += 1
                self._vis_results(image_path, input_data.frame[idx]) if input_data.frame else ...

    def process(self, input_data):
        if isinstance(input_data, ProcessData):
            self._collect_results(input_data)
        elif isinstance(input_data, StopData):
            self._collect_stop(input_data)
            if input_data.exception:
                self.send_to_next_module("stop")
        else:
            raise ValueError("unknown input data")

        if self.image_total.value and self.infer_size == self.image_total.value:
            self.final_text_save()
            self.send_to_next_module("stop")

    def stop(self):
        profiling_data = ProfilingData(
            module_name=self.module_name,
            instance_id=self.instance_id,
            process_cost_time=self.process_cost.value,
            send_cost_time=self.send_cost.value,
            image_total=self.image_total.value,
        )
        self.msg_queue.put(profiling_data, block=False)
        self.is_stop = True
