import time
from abc import abstractmethod
from ctypes import c_longdouble
from multiprocessing import Manager

from .module_data_type import ModuleInitArgs
from ..data_type import ProfilingData
from ..utils import log


class ModuleBase(object):
    def __init__(self, args, msg_queue):
        self.args = args
        self.pipeline_name = ''
        self.module_name = ''
        self.without_input_queue = False
        self.instance_id = 0
        self.device_id = -1
        self.is_stop = False
        self.msg_queue = msg_queue
        self.input_queue = None
        self.output_queue = None
        self.send_cost = Manager().Value(typecode=c_longdouble, value=0)
        self.process_cost = Manager().Value(typecode=c_longdouble, value=0)

    def assign_init_args(self, init_args: ModuleInitArgs):
        self.pipeline_name = init_args.pipeline_name
        self.module_name = init_args.module_name
        self.instance_id = init_args.instance_id

    def process_handler(self, stop_manager, input_queue, output_queue):
        self.input_queue = input_queue
        self.output_queue = output_queue
        try:
            self.init_self_args()
        except Exception as error:
            log.error(f'{self.__class__.__name__} init failed: {error}')
            raise error

        while not self.msg_queue.full() and stop_manager.full():
            continue
        time.sleep(0.5)
        while True:
            if stop_manager.full():
                break
            if self.input_queue.empty():
                continue
            else:
                data = self.input_queue.get(block=True)
            self.call_process(data)

    def call_process(self, send_data=None):
        if send_data is not None or self.without_input_queue:
            start_time = time.time()
            try:
                self.process(send_data)
            except Exception as error:
                log.exception(f'ERROR occurred in {self.module_name} module for {send_data.image_name}: {error}.')
            cost_time = time.time() - start_time
            self.process_cost.value += cost_time

    @abstractmethod
    def process(self, input_data):
        pass

    @abstractmethod
    def init_self_args(self):
        self.msg_queue.put(f'{self.__class__.__name__} instance id {self.instance_id} init complete')
        log.info(f'{self.__class__.__name__} instance id {self.instance_id} init complete')

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
        profiling_data = ProfilingData(module_name=self.module_name, instance_id=self.instance_id,
                                       device_id=self.device_id, process_cost_time=self.process_cost.value,
                                       send_cost_time=self.send_cost.value)
        self.msg_queue.put(profiling_data, block=False)
        self.is_stop = True
