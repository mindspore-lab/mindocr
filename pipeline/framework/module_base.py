import os
import tqdm
import sys
import time
from abc import abstractmethod
from ctypes import c_longdouble
from multiprocessing import Manager

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../../")))

from pipeline.datatype import ModuleInitArgs, ProfilingData
from pipeline.datatype import StopData, StopSign
from pipeline.utils import log, safe_div


class ModuleBase(object):
    def __init__(self, args, msg_queue, tqdm_info):
        self.args = args
        self.pipeline_name = ""
        self.module_name = ""
        self.without_input_queue = False
        self.instance_id = 0
        self.is_stop = False
        self.msg_queue = msg_queue
        self.input_queue = None
        self.output_queue = None
        self.send_cost = Manager().Value(typecode=c_longdouble, value=0)
        self.process_cost = Manager().Value(typecode=c_longdouble, value=0)
        self.display_id = tqdm_info["i"]
        if self.args.visual_pipeline:
            self.bar = tqdm.tqdm(total=tqdm_info["queue_len"],
                                desc=f"{self.display_id}. {self.module_name}",
                                position=self.display_id,
                                leave=False,
                                bar_format="{l_bar}{bar}|{n_fmt}/{total_fmt}",
                                ncols=150)

    def assign_init_args(self, init_args: ModuleInitArgs):
        self.pipeline_name = init_args.pipeline_name
        self.module_name = init_args.module_name
        self.instance_id = init_args.instance_id

    def process_handler(self, stop_manager, module_params, input_queue, output_queue):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.stop_manager = stop_manager
        self.queue_num = 0

        try:
            params = self.init_self_args()
            if params:
                module_params.update(**params)
        except Exception as error:
            log.error(f"{self.__class__.__name__} init failed: {error}")
            raise error

        # waiting for init sign
        while not self.msg_queue.full():
            continue

        # waiting for the release of stop sign
        while self.stop_manager.value:
            continue
        
        process_num = 0

        while True:
            time.sleep(self.args.node_fetch_interval)
            if self.stop_manager.value:
                break
            if self.input_queue.empty():
                continue
            
            process_num += 1
            data = self.input_queue.get(block=True)
            if self.args.visual_pipeline:
                qsize = self.input_queue.qsize()
                delta = qsize - self.queue_num
                self.bar.update(delta)
                self.queue_num = qsize
                info = f"{self.display_id}. Node:{self.module_name}, Has Processed:{process_num}, " + \
                    f"Process Time:{self.process_cost.value - self.send_cost.value:.2f} s, " + \
                    f"Wait Time:{self.send_cost.value:.2f} s, Queue Status:"
                info = info.ljust(85, " ")
                self.bar.set_description(info)
            self.call_process(data)
        if self.args.visual_pipeline:
            self.bar.close()

    def call_process(self, send_data=None):
        if send_data is not None or self.without_input_queue:
            start_time = time.time()
            try:
                self.process(send_data)
            except Exception as error:
                self.process(StopData(exception=True))
                image_path = [os.path.basename(filename) for filename in send_data.image_path]
                log.exception(f"ERROR occurred in {self.module_name} module for {', '.join(image_path)}: {error}.")

            cost_time = time.time() - start_time
            self.process_cost.value += cost_time

    @abstractmethod
    def process(self, input_data):
        pass

    @abstractmethod
    def init_self_args(self):
        self.msg_queue.put(f"{self.__class__.__name__} instance id {self.instance_id} init complete")
        log.info(f"{self.__class__.__name__} instance id {self.instance_id} init complete")

    def send_to_next_module(self, output_data):
        if self.is_stop:
            return
        start_time = time.time()
        self.output_queue.put(output_data, block=True)
        cost_time = time.time() - start_time
        self.send_cost.value += cost_time

    def get_module_name(self):
        return self.module_name

    def get_instance_id(self):
        return self.instance_id

    def stop(self):
        profiling_data = ProfilingData(
            module_name=self.module_name,
            instance_id=self.instance_id,
            process_cost_time=self.process_cost.value,
            send_cost_time=self.send_cost.value,
        )
        self.msg_queue.put(profiling_data, block=False)
        self.is_stop = True
