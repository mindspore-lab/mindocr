import argparse
import os
import time
import sys
from collections import defaultdict
from multiprocessing import Manager, Process, Queue

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../../")))

from pipeline.utils import log, safe_div
from pipeline.datatype import ModuleConnectDesc, ModuleDesc
from pipeline.datatype import StopData, StopSign
from pipeline.framework.module_manager import ModuleManager
from pipeline.tasks import SUPPORTED_TASK_BASIC_MODULE, TaskType
# ZHQ TODO 
from mindocr.infer.node_config import MODEL_DICT_v2 as MODEL_DICT


class ParallelPipelineManager:
    TASK_QUEUE_SIZE = 32

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.input_queue = Queue(self.TASK_QUEUE_SIZE)
        self.result_queue = Queue(self.TASK_QUEUE_SIZE)
        self.process = Process(target=self._build_pipeline_kernel)
        self.module_params = Manager().dict()

    def start_pipeline(self):
        self.process.start()
        self.input_queue.get(block=True)

    def stop_pipeline(self):
        self.input_queue.put(StopSign(), block=True)
        self.process.join()
        self.process.close()

    def fetch_result(self):
        if not self.result_queue.empty():
            rst_data = self.result_queue.get(block=True)
        else:
            rst_data = None
        return rst_data
    
    def pipeline_graph(self, task_type):
        module_order = SUPPORTED_TASK_BASIC_MODULE[TaskType(task_type.value)]
        module_desc_names_set = set()
        module_desc_list = []
        module_connect_desc_list = []

        for model_name in module_order:
            model_name = model_name
            for edge in MODEL_DICT.get(model_name, []):
                # Add Node
                src_node_info, tgt_node_info = edge
                src_node_name = src_node_info[0] + src_node_info[1]
                if src_node_name not in module_desc_names_set:
                    module_desc_list.append(ModuleDesc(src_node_info[0], src_node_name, src_node_info[2]))
                    module_desc_names_set.add(src_node_name)
                tgt_node_name = tgt_node_info[0] + tgt_node_info[1]
                if tgt_node_name not in module_desc_names_set:
                    module_desc_list.append(ModuleDesc(tgt_node_info[0], tgt_node_name, tgt_node_info[2]))
                    module_desc_names_set.add(tgt_node_name)
                module_connect_desc_list.append(
                    ModuleConnectDesc(src_node_name, tgt_node_name)
                )
        module_size = sum(desc.module_count for desc in module_desc_list)
        log.info(f"module_size: {module_size}")
        return module_order, module_size, module_desc_list, module_connect_desc_list
    

    def _build_pipeline_kernel(self):
        """
        build and register pipeline
        """
        task_type = self.args.task_type
        
        module_order, module_size, module_desc_list, module_connect_desc_list = self.pipeline_graph(task_type)

        msg_queue = Queue(module_size)

        manager = ModuleManager(msg_queue, self.input_queue, self.result_queue, self.args)
        manager.register_modules(str(os.getpid()), module_desc_list, 1)
        manager.register_module_connects(str(os.getpid()), module_connect_desc_list)

        # start the pipeline, init start
        manager.run_pipeline()

        # waiting for task receive
        while not msg_queue.full() or len(manager.module_params) != len(module_order):
            time.sleep(0.1)
            continue

        for _ in range(module_size):
            msg_queue.get()

        self.module_params.update(**manager.module_params)

        # send sign for blocking input queue
        self.input_queue.put(StopSign(), block=True)

        manager.stop_manager.value = False

        start_time = time.time()

        while not manager.stop_manager.value:
            time.sleep(self.args.node_fetch_interval)

        cost_time = time.time() - start_time

        manager.deinit_pipeline_module()
        # collect the profiling data
        profiling_data = defaultdict(lambda: [0, 0])
        image_total = 0
        for _ in range(module_size):
            msg_info = msg_queue.get()
            profiling_data[msg_info.module_name][0] += msg_info.process_cost_time
            profiling_data[msg_info.module_name][1] += msg_info.send_cost_time
            if msg_info.module_name != -1:
                image_total = msg_info.image_total
        if image_total > 0:
            self.profiling(profiling_data, image_total)
            perf_info = (
                f"Number of images: {image_total}, "
                f"total cost {cost_time:.2f}s, FPS: "
                f"{safe_div(image_total, cost_time):.2f}"
            )
            print(perf_info)
            log.info(perf_info)

        msg_queue.close()
        msg_queue.join_thread()

    def profiling(self, profiling_data, image_total):
        e2e_cost_time_per_image = 0
        for module_name in profiling_data:
            data = profiling_data[module_name]
            total_time = data[0]
            process_time = data[0] - data[1]
            send_time = data[1]
            process_avg = safe_div(process_time * 1000, image_total)
            e2e_cost_time_per_image += process_avg
            log.info(
                f"{module_name} cost total {total_time:.2f} s, process avg cost {process_avg:.2f} ms, "
                f"send waiting time avg cost {safe_div(send_time * 1000, image_total):.2f} ms"
            )
            log.info("----------------------------------------------------")
        log.info(f"e2e cost time per image {e2e_cost_time_per_image}ms")

    def __del__(self):
        if hasattr(self, "process") and self.process:
            self.process.close()
