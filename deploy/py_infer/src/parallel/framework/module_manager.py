from collections import defaultdict, namedtuple
from multiprocessing import Queue, Process

from ..datatype.module_data import ModulesInfo, ModuleInitArgs
from ..module import processor_initiator
from ...utils import log

OutputRegisterInfo = namedtuple('OutputRegisterInfo', ['pipeline_name', 'module_send', 'module_recv'])


class ModuleManager:
    MODULE_QUEUE_MAX_SIZE = 16

    def __init__(self, msg_queue: Queue, task_queue: Queue, args):
        self.device_id = 0
        self.pipeline_map = defaultdict(lambda: defaultdict(ModulesInfo))
        self.msg_queue = msg_queue
        self.stop_manager = Queue(1)
        self.stop_manager.put('-')
        self.args = args
        self.pipeline_name = ''
        self.process_list = []
        self.queue_list = []
        self.pipeline_queue_map = defaultdict(lambda: defaultdict(list))
        self.task_queue = task_queue

    @staticmethod
    def stop_module(module):
        module.stop()

    def init_module_instance(self, module_instance, instance_id, pipeline_name,
                             module_name):
        init_args = ModuleInitArgs(pipeline_name=pipeline_name,
                                   module_name=module_name,
                                   instance_id=instance_id)
        module_instance.assign_init_args(init_args)

    def register_modules(self, pipeline_name: str, module_desc_list: list,
                         default_count: int):
        log.info('----------------------------------------------------')
        log.info('---------------register_modules start---------------')
        modules_info_dict = self.pipeline_map[pipeline_name]

        for module_desc in module_desc_list:
            log.info('+++++++++++++++++++++++++++++++++++++')
            log.info(module_desc)
            log.info('+++++++++++++++++++++++++++++++++++++')
            module_count = default_count if module_desc.module_count == -1 else module_desc.module_count
            module_info = ModulesInfo()
            for instance_id in range(module_count):
                module_instance = processor_initiator(module_desc.module_name)(self.args, self.msg_queue)
                self.init_module_instance(module_instance, instance_id,
                                          pipeline_name,
                                          module_desc.module_name)

                module_info.module_list.append(module_instance)
            modules_info_dict[module_desc.module_name] = module_info

        self.pipeline_map[pipeline_name] = modules_info_dict

        log.info(f'----------------register_modules end---------------')
        log.info('----------------------------------------------------')

    def register_module_connects(self, pipeline_name: str,
                                 connect_desc_list: list):
        if pipeline_name not in self.pipeline_map:
            return

        log.info('----------------------------------------------------')
        log.info('-----------register_module_connects start-----------')

        modules_info_dict = self.pipeline_map[pipeline_name]
        connect_info_dict = self.pipeline_queue_map[pipeline_name]
        last_module = None
        for connect_desc in connect_desc_list:

            send_name = connect_desc.module_send_name
            recv_name = connect_desc.module_recv_name
            log.info('+++++++++++++++++++++++++++++++++++++')
            log.info(f'Add Connection Between {send_name} And {recv_name}')
            log.info('+++++++++++++++++++++++++++++++++++++')

            if send_name not in modules_info_dict:
                raise ValueError(f'cannot find send module {send_name}')

            if recv_name not in modules_info_dict:
                raise ValueError(f'cannot find receive module {recv_name}')

            queue = Queue(self.MODULE_QUEUE_MAX_SIZE)
            connect_info_dict[send_name].append(queue)
            connect_info_dict[recv_name].append(queue)
            last_module = recv_name
        connect_info_dict[last_module].append(self.stop_manager)

        log.info('------------register_module_connects end------------')
        log.info('----------------------------------------------------')

    def run_pipeline(self):

        log.info('-------------- start pipeline-----------------------')
        log.info('----------------------------------------------------')

        for pipeline_name in self.pipeline_map.keys():
            modules_info_dict = self.pipeline_map[pipeline_name]
            connect_info_dict = self.pipeline_queue_map[pipeline_name]
            for module_name in modules_info_dict.keys():
                queue_list = connect_info_dict[module_name]
                if len(queue_list) == 1:
                    input_queue = self.task_queue
                    output_queue = queue_list[0]
                else:
                    input_queue = queue_list[0]
                    output_queue = queue_list[1]

                for module in modules_info_dict[module_name].module_list:
                    self.process_list.append(
                        Process(target=module.process_handler, args=(self.stop_manager, input_queue,
                                                                     output_queue), daemon=True))

        for process in self.process_list:
            process.start()

    def deinit_pipeline_module(self):
        # wait for the stop msg
        while self.stop_manager.empty():
            continue

        # pop the sign from shared queue
        self.stop_manager.get()

        # the empty() is not reliable, double check the msg queue is empty for receive the profiling data
        while not self.msg_queue.empty():
            self.msg_queue.get()

        for queue in self.queue_list:
            while not queue.empty():
                queue.get(block=False)
            queue.close()
            queue.join_thread()

        # send the profiling data
        for pipeline_name in self.pipeline_map.keys():
            modules_info_dict = self.pipeline_map[pipeline_name]
            for module_name in modules_info_dict.keys():
                for module in modules_info_dict[module_name].module_list:
                    self.stop_module(module=module)

        # release all resource
        for process in self.process_list:
            if process.is_alive():
                process.kill()

        self.stop_manager.close()
        self.stop_manager.join_thread()
        log.info('------------------pipeline stopped------------------')
        log.info('----------------------------------------------------')
