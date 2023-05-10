'''A script to evaluate multiple datasets under the same folder at once.

DATASET STRUCTURE:
    Assume you have put all benckmark datasets (e.g. CUTE80, IC03_860, IC13_1015) under evaluation/ as shown below: 

    data_lmdb_release/
    ├── evaluation
    │   ├── CUTE80
    │   │   ├── data.mdb
    │   │   └── lock.mdb
    │   ├── IC03_860
    │   │   ├── data.mdb
    │   │   └── lock.mdb
    │   ├── IC13_1015
    │   │   ├── data.mdb
    │   │   └── lock.mdb
    │   ├── ...

YAML CONFIGURATION:
    Please modify the config yaml as follows:

    ```yaml --> configs/rec/crnn/crnn_resnet34.yaml
        ...
        eval:
        dataset:
            type: LMDBDataset
            dataset_root: dir/to/data_lmdb_release/          # Root dir of evaluation dataset
            data_dir: evaluation/                            # Dir of evaluation dataset, concatenated with `dataset_root` to be the complete dir of evaluation dataset
        ...
    ```

USAGE:
    Please run the following command:
    ```
        python tools/benchmarking/multi_dataset_eval.py --config configs/rec/crnn/crnn_resnet34.yaml
    ```

    You can then get the performance of each individual dataset as well as the average score under evaluation/.
'''

import sys
import os
import copy
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../..")))

import yaml
import argparse
from addict import Dict

import mindspore as ms
from mindspore.communication import init, get_rank, get_group_size

from mindocr.data import build_dataset
from mindocr.models import build_model
from mindocr.postprocess import build_postprocess
from mindocr.metrics import build_metric
from mindocr.utils.callbacks import Evaluator

def main(cfg):
    # env init
    ms.set_context(mode=cfg.system.mode)
    if cfg.system.distribute:
        init()
        device_num = get_group_size()
        rank_id = get_rank()
        ms.set_auto_parallel_context(device_num=device_num,
                                     parallel_mode='data_parallel',
                                     gradients_mean=True,
                                     )
    else:
        device_num = None
        rank_id = None

    is_main_device = rank_id in [None, 0]

    # model
    cfg.model.backbone.pretrained = False
    network = build_model(cfg.model, ckpt_load_path=cfg.eval.ckpt_load_path)
    network.set_train(False)

    if cfg.system.amp_level != 'O0':
        print('INFO: Evaluation will run in full-precision(fp32)')

    # TODO: check float type conversion in official Model.eval
    #ms.amp.auto_mixed_precision(network, amp_level='O0')

    # postprocess, metric
    postprocessor = build_postprocess(cfg.postprocess)
    # postprocess network prediction
    metric = build_metric(cfg.metric)

    if cfg.eval.dataset["dataset_root"]:
        data_dir_root = os.path.join(cfg.eval.dataset["dataset_root"], cfg.eval.dataset["data_dir"])
    else:
        data_dir_root = cfg.eval.dataset["data_dir"]

    results = []
    acc_summary = {}
    reload_data = False
    for dirpath, dirnames, _ in os.walk(data_dir_root + '/'):
        print(dirpath)
        if not dirnames:
            dataset_config = copy.deepcopy(cfg.eval.dataset)
            dataset_config["data_dir"] = dirpath
            # dataloader
            # load dataset
            loader_eval = build_dataset(
                dataset_config,
                cfg.eval.loader,
                num_shards=device_num,
                shard_id=rank_id,
                is_train=False,
                refine_batch_size=True,
                )
            
            num_batches = loader_eval.get_dataset_size()

            if not reload_data:
                net_evaluator = Evaluator(
                    network,
                    loader_eval,
                    loss_func=None,
                    postprocessor=postprocessor,
                    metrics=[metric],
                    input_indices=dataset_config.get('net_input_column_index', None),
                    label_indices=dataset_config.get('label_column_index', None),
                    meta_data_indices=dataset_config.get('meta_data_column_index', None),
                    num_epochs=1,
                )
                reload_data = True

            else:
                net_evaluator.reload(
                    loader_eval,
                    input_indices=dataset_config.get('net_input_column_index', None),
                    label_indices=dataset_config.get('label_column_index', None),
                    meta_data_indices=dataset_config.get('meta_data_column_index', None),
                    num_epochs=1,
                )

            # log
            print('='*40)
            print(f'Num batches: {num_batches}')
            if 'name' in cfg.model:
                print(f'Model: {cfg.model.name}')
            else:
                print(f'Model: {cfg.model.backbone.name}-{cfg.model.neck.name}-{cfg.model.head.name}')
            print('='*40)

            measures = net_evaluator.eval()
            if is_main_device:
                print('Performance: ', measures)

            results.append(measures)
            acc_summary[dirpath] = measures

    if len(results) == 0:
        raise ValueError(f"Cannot find any dataset under `{data_dir_root}`. Please check the data path is correct.")

    # average
    metric_keys = results[0].keys()
    avg_dict = {}
    for metric_k in metric_keys:
        score = [res[metric_k] for res in results]
        avgscore = sum(score) / len(score)
        avg_dict[metric_k] = avgscore

    acc_summary["Average"] = avg_dict

    print("Average score:")
    print(avg_dict)
    
    print("Summary:")
    print(acc_summary)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation Config', add_help=False)
    parser.add_argument('-c', '--config', type=str, default='',
                        help='YAML config file specifying default arguments (default='')')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    # argpaser
    args = parse_args()
    yaml_fp = args.config
    with open(yaml_fp) as fp:
        config = yaml.safe_load(fp)
    config = Dict(config)

    #print(config)

    main(config)
