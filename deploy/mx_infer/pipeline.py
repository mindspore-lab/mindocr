import os
import time
from collections import defaultdict
from multiprocessing import Process, Queue

import tqdm

from mx_infer.data_type import StopSign
from mx_infer.framework import ModuleDesc, ModuleConnectDesc, ModuleManager, SupportedTaskOrder
from mx_infer.processors import MODEL_DICT
from mx_infer.utils import log, profiling, safe_div, save_path_init, TASK_QUEUE_SIZE


def image_sender(images_path, send_queue, show_progressbar):
    """
    send image to input queue for pipeline
    """
    if os.path.isdir(images_path):
        input_image_list = [os.path.join(images_path, path) for path in os.listdir(images_path)]
        if show_progressbar:
            for image_path in tqdm.tqdm(input_image_list, desc="send image to pipeline"):
                send_queue.put(image_path, block=True)
        else:
            for image_path in input_image_list:
                send_queue.put(image_path, block=True)
    else:
        send_queue.put(images_path, block=True)


def build_pipeline_kernel(args, input_queue):
    """
    build and register pipeline
    """
    task_type = args.task_type
    parallel_num = args.parallel_num
    module_desc_list = [ModuleDesc('HandoutProcess', 1), ModuleDesc('DecodeProcess', parallel_num)]

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
    """
    Build the inference pipeline. Four modes are supported:
        Mode 1: Text detection + text direction classification + text recognition.
        Mode 2: Text detection + text recognition.
        Mode 3: Text detection.
        Mode 4: Text recognition.

    Args:
        args (argparse.Namespace):
            - input_images_dir (str): Input images dir for inference, can be dir containing multiple images or path of single image. This arg is reuqired.
            - device (str): Device type. Only 'Ascend' is supported at this stage.
            - device_id (int): Device id.
            - parallel_num (int): Number of parallel in each stage of pipeline parallelism.
            - precision_mode (str): Precision mode. Only fp32 is supported at this stage.
            - det_algorithm (str): Detection algorithm name. Please check the SUPPORT_DET_MODEL in deploy/mx_infer/processors/detection/__init__.py
            - rec_algorithm (str): Recognition algorithm name. Please check the SUPPORT_REC_MODEL in deploy/mx_infer/processors/recognition/__init__.py
            - det_model_path (str): Detection model file path.
            - cls_model_path (str): Classification model file path.
            - rec_model_path (str): Recognition model file path or directory which contains multiple recognition models.
            - rec_char_dict_path (str): Character dict file path for recognition models.
            - res_save_dir (str): Saving dir for inference results. If it's not set, the results will not be saved.
            - vis_det_save_dir (str): Saving dir for visualization of detection results. If it's not set, the results will not be saved.
            - vis_pipeline_save_dir (str): Saving dir for visualization of det+cls(optional)+rec pipeline inference results. If it's not set, the results will not be saved.
            - vis_font_path (str): Font file path for recognition model.
            - pipeline_crop_save_dir (str): Saving dir for images cropped during pipeline. If it's not set, the results will not be saved.
            - show_log (str): Whether show log when inferring. If the lower case of this arg is in ("true", "t", "1"), then True, else False.
            - save_log_dir (str): Log saving dir.

    Return:
        NoneType

    Notes:
        Args configuration guidelines for four different modes of inference pipeline:
            - Mode 1: Text detection + text direction classification + text recognition.
                These args must be set to correct dir or path: `input_images_dir`, `det_model_path`, `cls_model_path`, `rec_model_path`, `rec_char_dict_path`.
                Other args can be set if needed.
            - Mode 2: Text detection + text recognition.
                These args must be set to correct dir or path: `input_images_dir`, `det_model_path`, `rec_model_path`, `rec_char_dict_path`.
                This arg CANNOT be set: `cls_model_path`.
                Other args can be set if needed.
            - Mode 3: Text detection.
                These args must be set to correct dir or path: `input_images_dir`, `det_model_path`.
                These args CANNOT be set: `cls_model_path`, `rec_model_path`, `rec_char_dict_path`.
                Other args can be set if needed.
            - Mode 4: Text recognition.
                These args must be set to correct dir or path: `input_images_dir`, `rec_model_path`, `rec_char_dict_path`.
                These args CANNOT be set: `det_model_path`, `cls_model_path`.
                Other args can be set if needed.
        If the guidelines above are not followed, the inference pipeline cannot be built. Check your args configurations.

    Example:
    >>> from mx_infer import pipeline_args, pipeline
    >>> args = pipeline_args.get_args()
    >>> pipeline.build_pipeline(args)
    """

    if args.res_save_dir:
        save_path_init(args.res_save_dir)
    if args.save_pipeline_crop_res:
        save_path_init(args.pipeline_crop_save_dir)
    if args.save_vis_pipeline_save_dir:
        save_path_init(args.vis_pipeline_save_dir)
    if args.save_vis_det_save_dir:
        save_path_init(args.vis_det_save_dir)
    if args.save_log_dir:
        save_path_init(args.save_log_dir, exist_ok=True)

    if os.path.isdir(args.input_images_dir) and not os.listdir(args.input_images_dir):
        log.warning(f"The input_images_dir directory '{args.input_images_dir}' is empty, no image to process.")
        return

    task_queue = Queue(TASK_QUEUE_SIZE)
    process = Process(target=build_pipeline_kernel, args=(args, task_queue))
    process.start()
    image_sender(images_path=args.input_images_dir, send_queue=task_queue,
                 show_progressbar=not args.show_log)
    task_queue.put(StopSign(), block=True)
    process.join()
    process.close()
