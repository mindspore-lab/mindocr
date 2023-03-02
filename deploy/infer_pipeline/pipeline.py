import os
import time
from collections import defaultdict
from multiprocessing import Process, Queue

from src.data_type import StopSign
from src.framework import ModuleDesc, ModuleConnectDesc, ModuleManager
from src.processors import MODEL_DICT
from src.utils import log, safe_load_yaml, profiling, safe_div, TASK_QUEUE_SIZE


def image_sender(images_path, send_queue):
    input_image_list = [os.path.join(images_path, path) for path in os.listdir(images_path)]
    for image_path in input_image_list:
        send_queue.put(image_path, block=True)


def build(config_path, parallel_num, input_queue, infer_res_save_path):
    config_dict = safe_load_yaml(config_path)
    module_desc_list = [ModuleDesc('HandoutProcess', 1), ModuleDesc('DecodeProcess', parallel_num), ]

    module_order = config_dict.get('module_order', None)
    if module_order is None:
        raise ValueError('cannot find the order of module')

    for model_name in module_order:
        model_name = model_name.lower()
        if model_name not in MODEL_DICT:
            log.error(f'unsupported model {model_name}')
            raise ValueError(f'unsupported model {model_name}')
        for name, count in MODEL_DICT.get(model_name, []):
            module_desc_list.append(ModuleDesc(name, count * parallel_num))

    module_desc_list.append(ModuleDesc('CollectProcess', 1))
    module_connect_desc_list = []
    for i in range(len(module_desc_list) - 1):
        module_connect_desc_list.append(ModuleConnectDesc(module_desc_list[i].module_name,
                                                          module_desc_list[i + 1].module_name))

    module_size = sum(desc.module_count for desc in module_desc_list)
    log.info(f'module_size: {module_size}')
    msg_queue = Queue(module_size)

    manager = ModuleManager(msg_queue, input_queue, config_path, infer_res_save_path)
    manager.register_modules(str(os.getpid()), module_desc_list, 1)
    manager.register_module_connects(str(os.getpid()), module_connect_desc_list)

    # start the pipeline, init start
    manager.run_pipeline()

    # waiting for task receive
    while not msg_queue.full():
        continue

    start_time = time.time()
    # release all init sign
    for _ in range(module_size):
        msg_queue.get()

    # release the stop sign, infer start
    manager.stop_manager.get(block=False)

    manager.deinit_pipeline_module()

    cost_time = time.time() - start_time

    # collect the profiling data
    profiling_data = defaultdict(lambda: [0, 0])
    image_total = 0
    for _ in range(module_size):
        msg_info = msg_queue.get()
        profiling_data[msg_info.module_name][0] += msg_info.process_cost_time
        profiling_data[msg_info.module_name][1] += msg_info.send_cost_time
        if msg_info.module_name != -1:
            image_total = msg_info.image_total

    profiling(profiling_data, image_total)

    log.info(f'total cost {cost_time:.2f}s, FPS: {safe_div(image_total, cost_time):.2f}')
    msg_queue.close()
    msg_queue.join_thread()


def build_pipeline(args):
    task_queue = Queue(TASK_QUEUE_SIZE)
    process = Process(target=build_pipeline, args=(args.config_path, args.parallel_num, task_queue,
                                                   args.infer_res_save_path))
    process.start()
    image_sender(images_path=args.input_images_path, send_queue=task_queue)
    task_queue.put(StopSign(), block=True)
    process.join()
    process.close()
