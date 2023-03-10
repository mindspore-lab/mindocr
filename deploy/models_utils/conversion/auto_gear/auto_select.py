import argparse
import glob
import logging
import os
import shutil
import time

import numpy as np
from mindx.sdk import base, Tensor

logging.getLogger().setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    # detection auto gear related
    parser.add_argument('--rec_model_path', type=str, required=True)
    parser.add_argument('--device_id', type=int, required=False, default=0)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    model_list = []
    gear_list = []
    om_paths = glob.glob(os.path.join(args.rec_model_path, "*.om"))
    unselected_model_path = os.path.join(args.rec_model_path, "unselected")
    selected_model_path = os.path.join(args.rec_model_path, "selected")

    if not os.path.exists(unselected_model_path):
        os.makedirs(unselected_model_path, 0o750, exist_ok=True)
    if not os.path.exists(selected_model_path):
        os.makedirs(selected_model_path, 0o750, exist_ok=True)
    if not om_paths:
        raise FileNotFoundError("no om model in rec_model_path!")
    for om in om_paths:
        model = base.model(om, deviceId=args.device_id)
        if model is None:
            logging.info(f"{om} is not a valid model and will be marked not used.")
            shutil.move(om, om + "_not_used")
        else:
            gear = model.model_gear()
            gear_list.append(np.array(gear))
            model_list.append(model)
    batchsizes = [gear[0][0] for gear in gear_list]
    name_bs_dict = {batchsize: name for batchsize, name in zip(batchsizes, om_paths)}

    model_list = sorted(model_list, key=lambda x: batchsizes[model_list.index(x)])
    batchsizes = sorted(batchsizes)


    def run_model(model):
        input_tensor = Tensor(np.random.uniform(-1, 1, model.model_gear()[-1]).astype(np.float32))
        output = model.infer([input_tensor])
        infer_res = output[0]
        infer_res.to_host()


    rec_infer_times = []
    for idx, model in enumerate(model_list):
        # warm up
        for _ in range(5):
            run_model(model)

        rec_infer_start = time.time()
        for _ in range(10):
            run_model(model)
        rec_infer_time = time.time() - rec_infer_start
        rec_infer_times.append(rec_infer_time / batchsizes[idx])

    final_list = []
    best_performance = float('inf')
    for idx, t in enumerate(rec_infer_times):
        bs_str = str(batchsizes[idx])
        selected = name_bs_dict[batchsizes[idx]]
        if t < best_performance:
            best_performance = t
            shutil.move(selected, selected_model_path)
        elif idx and t >= rec_infer_times[idx - 1] * 2:
            logging.info(f"{selected} is not better performing and will be marked not used.")
            shutil.move(selected, unselected_model_path)
        else:
            shutil.move(selected, selected_model_path)

    logging.info("auto select models finish!")
