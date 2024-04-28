import os
from collections import defaultdict
from ctypes import c_uint64
from multiprocessing import Manager

import numpy as np

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../../")))

from pipeline.data_process.utils import cv_utils
from pipeline.tasks import TaskType
from pipeline.utils import log, safe_list_writer, visual_utils
from pipeline.datatype import ProcessData, ProfilingData, StopData
from pipeline.framework.module_base import ModuleBase

RESULTS_SAVE_FILENAME = {
    TaskType.DET: "det_results.txt",
    TaskType.CLS: "cls_results.txt",
    TaskType.REC: "rec_results.txt",
    TaskType.DET_REC: "pipeline_results.txt",
    TaskType.DET_CLS_REC: "pipeline_results.txt",
    TaskType.LAYOUT: "layout_results.txt",
    TaskType.LAYOUT_DET_REC: "pipeline_results.txt",
    TaskType.LAYOUT_DET_CLS_REC: "pipeline_results.txt",
}


class CollectNode(ModuleBase):
    def __init__(self, args, msg_queue, tqdm_info):
        super().__init__(args, msg_queue, tqdm_info)
        self.image_sub_remaining = defaultdict(defaultdict)
        self.image_pipeline_res = defaultdict(defaultdict)
        self.infer_size = defaultdict(int)
        self.image_total = Manager().Value(c_uint64, 0)
        self.task_type = args.task_type
        self.res_save_dir = args.res_save_dir
        self.save_filename = RESULTS_SAVE_FILENAME[TaskType(self.task_type.value)]

    def init_self_args(self):
        super().init_self_args()

    def _collect_stop(self, input_data):
        self.image_total.value = input_data.image_total

    def _vis_results(self, image_name, image, taskid, data_type, task=None):
        if self.args.crop_save_dir and (data_type == 0 or (data_type == 1 and self.args.input_array_save_dir)):
            basename = os.path.basename(image_name)
            filename = os.path.join(self.args.crop_save_dir, os.path.splitext(basename)[0])
            box_list = [np.array(x["points"]).reshape(-1, 2) for x in self.image_pipeline_res[taskid][image_name]]
            crop_list = visual_utils.vis_crop(image, box_list)
            for i, crop in enumerate(crop_list):
                cv_utils.img_write(filename + "_crop_" + str(i) + ".jpg", crop)

        if self.args.vis_pipeline_save_dir:
            basename = os.path.basename(image_name)
            filename = os.path.join(self.args.vis_pipeline_save_dir, os.path.splitext(basename)[0])
            box_list = [np.array(x["points"]).reshape(-1, 2) for x in self.image_pipeline_res[taskid][image_name]]
            text_list = [x["transcription"] for x in self.image_pipeline_res[taskid][image_name]]
            box_text = visual_utils.vis_bbox_text(image, box_list, text_list, font_path=self.args.vis_font_path)
            cv_utils.img_write(filename + ".jpg", box_text)

        if self.args.vis_det_save_dir and (data_type == 0 or (data_type == 1 and self.args.input_array_save_dir)):
            basename = os.path.basename(image_name)
            filename = os.path.join(self.args.vis_det_save_dir, os.path.splitext(basename)[0])
            box_list = [np.array(x).reshape(-1, 2) for x in self.image_pipeline_res[taskid][image_name]]
            box_line = visual_utils.vis_bbox(image, box_list, [255, 255, 0], 2)
            cv_utils.img_write(filename + ".jpg", box_line)

        if self.args.vis_layout_save_dir and (data_type == 0 or (data_type == 1 and self.args.input_array_save_dir)):
            basename = os.path.basename(image_name)
            filename = os.path.join(self.args.vis_layout_save_dir, os.path.splitext(basename)[0])
            box_list = []
            for x in self.image_pipeline_res[taskid][image_name]:
                x, y, dx, dy = x['bbox']
                box_list.append(np.array([[x, y+dy], [x+dx, y+dy], [x+dx, y], [x, y]]))
            box_line = visual_utils.vis_bbox(image, box_list, [255, 255, 0], 2)
            cv_utils.img_write(filename + ".jpg", box_line)
        # log.info(f"{image_name} is finished.")

    def final_text_save(self):
        rst_dict = dict()
        for rst in self.image_pipeline_res.values():
            rst_dict.update(rst)
        save_filename = os.path.join(self.res_save_dir, self.save_filename)
        safe_list_writer(rst_dict, save_filename)
        # log.info(f"save infer result to {save_filename} successfully")

    def _update_layout_result(self, input_data):
        taskid = input_data.taskid
        image_path = input_data.image_path[0]
        layout_rsts = input_data.data
        
        for layout_rst in layout_rsts["layout_collect_res"]:
            # X, Y = layout_rst.data["raw_img_shape"]
            layout_bbox = layout_rst.data["layout_result"]
            lx, ly, _, _ = layout_bbox['bbox']
            for rec_rst in layout_rst.infer_result:
                bbox, transcription, score = rec_rst[:-2], rec_rst[-2], rec_rst[-1]
                bbox = [[b[0]+lx, b[1]+ly] for b in bbox]
                if score > 0.5:
                    if self.args.result_contain_score:
                        self.image_pipeline_res[taskid][image_path].append(
                            {"transcription": transcription, "points": bbox, "score": str(score)}
                        )
                    else:
                        self.image_pipeline_res[taskid][image_path].append(
                            {"transcription": transcription, "points": bbox}
                        )


    def _collect_results(self, input_data: ProcessData):
        taskid = input_data.taskid
        if self.task_type.value in (TaskType.DET_REC.value, TaskType.DET_CLS_REC.value):
            image_path = input_data.image_path[0]  # bs=1
            # print(f"input_data.infer_result:{input_data.infer_result}")
            for result in input_data.infer_result:
                # print(f"result:{result}")
                if result[-1] > 0.5:
                    if self.args.result_contain_score:
                        self.image_pipeline_res[taskid][image_path].append(
                            {"transcription": result[-2], "points": result[:-2], "score": str(result[-1])}
                        )
                    else:
                        self.image_pipeline_res[taskid][image_path].append(
                            {"transcription": result[-2], "points": result[:-2]}
                        )
            if not input_data.infer_result:
                self.image_pipeline_res[taskid][image_path] = []
        elif self.task_type.value == TaskType.DET.value:
            image_path = input_data.image_path[0]  # bs=1
            self.image_pipeline_res[taskid][image_path] = input_data.infer_result
        elif self.task_type.value in (TaskType.REC.value, TaskType.CLS.value):
            for image_path, infer_result in zip(input_data.image_path, input_data.infer_result):
                self.image_pipeline_res[taskid][image_path] = infer_result
        elif self.task_type.value == TaskType.LAYOUT.value:
            for infer_result in input_data.infer_result:
                image_path = infer_result.pop("image_id")[0]
                if image_path in self.image_pipeline_res[taskid]:
                    self.image_pipeline_res[taskid][image_path].append(infer_result)
                else:
                    self.image_pipeline_res[taskid][image_path] = [infer_result]
        elif self.task_type.value in (TaskType.LAYOUT_DET_REC.value, TaskType.LAYOUT_DET_CLS_REC.value,):
            self._update_layout_result(input_data)
        else:
            raise NotImplementedError("Task type do not support.")

        self._update_remaining(input_data)

    def _update_remaining(self, input_data: ProcessData):
        taskid = input_data.taskid
        data_type = input_data.data_type
        # if self.task_type.value in (TaskType.DET_REC.value, TaskType.DET_CLS_REC.value, TaskType.LAYOUT_DET_REC.value):  # with sub image
        #     for idx, image_path in enumerate(input_data.image_path):
        #         if image_path in self.image_sub_remaining[taskid]:
        #             self.image_sub_remaining[taskid][image_path] -= input_data.sub_image_size
        #             if not self.image_sub_remaining[taskid][image_path]:
        #                 self.image_sub_remaining[taskid].pop(image_path)
        #                 self.infer_size[taskid] += 1
        #                 if self.task_type.value in (TaskType.LAYOUT_DET_REC.value, ):
        #                     self._vis_results(image_path, input_data.data["layout_images"][idx], taskid, data_type) if input_data.frame else ...
        #                 else:
        #                     self._vis_results(
        #                         image_path, input_data.frame[idx], taskid, data_type
        #                     ) if input_data.frame else ...
        #         else:
        #             remaining = input_data.sub_image_total - input_data.sub_image_size
        #             if remaining:
        #                 self.image_sub_remaining[taskid][image_path] = remaining
        #             else:
        #                 self.infer_size[taskid] += 1
        #                 if self.task_type.value in (TaskType.LAYOUT_DET_REC.value, ):
        #                     self._vis_results(image_path, input_data.data["layout_images"][idx], taskid, data_type) if input_data.frame else ...
        #                 else:
        #                     self._vis_results(
        #                         image_path, input_data.frame[idx], taskid, data_type
        #                     ) if input_data.frame else ...
        # else:  # without sub image
        # if self.task_type.value not in (TaskType.LAYOUT_DET_REC, ):
        for idx, image_path in enumerate(input_data.image_path):
            self.infer_size[taskid] += 1
            if self.task_type.value in (TaskType.LAYOUT_DET_REC.value, ):
                self._vis_results(image_path, input_data.frame[idx], taskid, data_type) if input_data.frame else ...
            else:
                self._vis_results(image_path, input_data.frame[idx], taskid, data_type) if input_data.frame else ...


    def process(self, input_data):
        if isinstance(input_data, ProcessData):
            # print(f"ProcessData:{input_data.image_path}")
            taskid = input_data.taskid
            if input_data.taskid not in self.image_sub_remaining.keys():
                self.image_sub_remaining[input_data.taskid] = defaultdict(int)
            if input_data.taskid not in self.image_pipeline_res.keys():
                self.image_pipeline_res[input_data.taskid] = defaultdict(list)
            self._collect_results(input_data)
            if self.infer_size[taskid] == input_data.task_images_num:
                self.send_to_next_module({taskid: self.image_pipeline_res[taskid]})

        elif isinstance(input_data, StopData):
            self._collect_stop(input_data)
            if input_data.exception:
                self.stop_manager.value = True
        else:
            raise ValueError("unknown input data")

        infer_size_sum = sum(self.infer_size.values())
        if self.image_total.value and infer_size_sum == self.image_total.value:
            self.final_text_save()
            self.stop_manager.value = True

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
