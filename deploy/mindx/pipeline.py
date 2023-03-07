import os
import time
from collections import defaultdict
from multiprocessing import Process, Queue

from deploy.mindx.data_type import StopSign
from deploy.mindx.framework import ModuleDesc, ModuleConnectDesc, ModuleManager, SupportedTaskOrder
from deploy.mindx.processors import MODEL_DICT
from deploy.mindx.utils import log, profiling, safe_div, save_path_init, TASK_QUEUE_SIZE


def image_sender(images_path, send_queue):
    if os.path.isdir(images_path):
        input_image_list = [os.path.join(images_path, path) for path in os.listdir(images_path)]
        for image_path in input_image_list:
            send_queue.put(image_path, block=True)
    else:
        send_queue.put(images_path, block=True)


def build_pipeline_kernel(args, input_queue):
    task_type = args.task_type
    parallel_num = args.parallel_num
    module_desc_list = [ModuleDesc('HandoutProcess', 1), ModuleDesc('DecodeProcess', parallel_num), ]

    module_order = SupportedTaskOrder[task_type]

    for model_name in module_order:
        model_name = model_name
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

    manager = ModuleManager(msg_queue, input_queue, args)
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
    if args.res_save_dir:
        save_path_init(args.res_save_dir)
    if args.save_pipeline_crop_res:
        save_path_init(args.pipeline_crop_save_dir)
    if args.save_vis_pipeline_save_dir:
        save_path_init(args.vis_pipeline_save_dir)
    if args.save_vis_det_save_dir:
        save_path_init(args.vis_det_save_dir)
    if args.save_log_dir:
        save_path_init(args.save_log_dir)

    task_queue = Queue(TASK_QUEUE_SIZE)
    process = Process(target=build_pipeline_kernel, args=(args, task_queue))
    process.start()
    image_sender(images_path=args.input_images_dir, send_queue=task_queue)
    task_queue.put(StopSign(), block=True)
    process.join()
    process.close()
