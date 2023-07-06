import argparse
import os
import time
from collections import defaultdict
from multiprocessing import Manager, Process, Queue

from ...infer import SUPPORTED_TASK_BASIC_MODULE
from ...utils import log, safe_div
from ..datatype import ModuleConnectDesc, ModuleDesc, StopSign
from ..framework import ModuleManager
from ..module import MODEL_DICT


class ParallelPipelineManager:
    TASK_QUEUE_SIZE = 32

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.input_queue = Queue(self.TASK_QUEUE_SIZE)
        self.process = Process(target=self._build_pipeline_kernel)
        self.module_params = Manager().dict()

    def start_pipeline(self):
        self.process.start()
        self.input_queue.get(block=True)

    def stop_pipeline(self):
        self.input_queue.put(StopSign(), block=True)
        self.process.join()
        self.process.close()

    def _build_pipeline_kernel(self):
        """
        build and register pipeline
        """
        task_type = self.args.task_type
        parallel_num = self.args.parallel_num
        module_desc_list = [ModuleDesc("HandoutNode", 1), ModuleDesc("DecodeNode", parallel_num)]

        module_order = SUPPORTED_TASK_BASIC_MODULE[task_type]

        for model_name in module_order:
            model_name = model_name
            for name, count in MODEL_DICT.get(model_name, []):
                module_desc_list.append(ModuleDesc(name, count * parallel_num))

        module_desc_list.append(ModuleDesc("CollectNode", 1))
        module_connect_desc_list = []
        for i in range(len(module_desc_list) - 1):
            module_connect_desc_list.append(
                ModuleConnectDesc(module_desc_list[i].module_name, module_desc_list[i + 1].module_name)
            )

        module_size = sum(desc.module_count for desc in module_desc_list)
        log.info(f"module_size: {module_size}")
        msg_queue = Queue(module_size)

        manager = ModuleManager(msg_queue, self.input_queue, self.args)
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

        # release the stop sign, infer start
        manager.stop_manager.get(block=False)

        start_time = time.time()

        # waiting for inference, and pop the sign from shared queue
        manager.stop_manager.get(block=True)

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
