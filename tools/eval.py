'''
Model evaluation 
'''
import sys
sys.path.append('.')

import os
import yaml
import argparse
from addict import Dict

import mindspore as ms
from mindspore import nn

from mindocr.data import build_dataset
from mindocr.models import build_model
from mindocr.postprocess import build_postprocess
from mindocr.metrics import build_metric
from mindocr.utils.callbacks import Evaluator

def main(cfg):
    # env init
    ms.set_context(mode=cfg.system.mode)
    if cfg.system.distribute:
        print("WARNING: Distribut mode blocked. Evaluation only runs in standalone mode.")

    loader_eval = build_dataset(
            cfg.eval.dataset, 
            cfg.eval.loader,
            num_shards=None,
            shard_id=None,
            is_train=False)
    num_batches = loader_eval.get_dataset_size()

    # model
    assert 'ckpt_load_path' in cfg.eval, f'Please provide \n`eval:\n\tckpt_load_path`\n in the yaml config file '
    network = build_model(cfg.model, ckpt_load_path=cfg.eval.ckpt_load_path)
    network.set_train(False)

    if cfg.system.amp_level != 'O0':
        print('INFO: Evaluation will run in full-precision(fp32)')
    #ms.amp.auto_mixed_precision(network, amp_level='O0')  

    # postprocess, metric
    postprocessor = build_postprocess(cfg.postprocess)
    # postprocess network prediction
    metric = build_metric(cfg.metric)

    net_evaluator = Evaluator(network, None, postprocessor, [metric])

    # log
    print('='*40)
    print(
        f'Num batches: {num_batches}\n'
        )
    if 'name' in cfg.model:
        print(f'Model: {cfg.model.name}')
    else:
        print(f'Model: {cfg.model.backbone.name}-{cfg.model.neck.name}-{cfg.model.head.name}')
    print('='*40)
 
    measures = net_evaluator.eval(loader_eval)
    print('Performance: ', measures)


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
