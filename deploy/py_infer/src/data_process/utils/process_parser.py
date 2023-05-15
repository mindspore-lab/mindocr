import inspect
from collections import OrderedDict

import yaml

__all__ = ["parse_from_yaml"]


def parse_from_yaml(filename: str):
    with open(filename) as fp:
        cfg = yaml.safe_load(fp)

    preprocess_task = _parse_preprocess(cfg)
    postprocess_task = _parse_postprocess(cfg)

    return preprocess_task, postprocess_task


def _parse_preprocess(cfg):
    src_ops_pipeline = cfg['eval']['dataset']['transform_pipeline']
    tgt_ops_pipeline = OrderedDict()

    from ..preprocess import preprocess_mapping  # avoid circular import

    for src_ops in src_ops_pipeline:
        src_ops_name, src_ops_params = list(src_ops.items())[0]
        if src_ops_name in preprocess_mapping.PREPROCESS_SKIP_OPS:
            continue

        tgt_ops = preprocess_mapping.PREPROCESS_MAPPING_OPS[src_ops_name]
        tgt_params_name = list(inspect.signature(tgt_ops.__init__).parameters.keys())[1:]

        tgt_ops_params = {k: v for k, v in src_ops_params.items() if k in tgt_params_name} if src_ops_params else {}
        tgt_ops_pipeline[tgt_ops] = tgt_ops_params

    return tgt_ops_pipeline


def _parse_postprocess(cfg):
    ops_params = cfg['postprocess']

    from ..postprocess import postprocess_mapping  # avoid circular import

    ops_node = postprocess_mapping.POSTPROCESS_MAPPING_OPS[ops_params["name"]]
    ops_params.pop("name")
    ops = {ops_node: ops_params}

    return ops
