import os
import mindspore as ms
from packaging import version
from multiprocessing import cpu_count

from .det_dataset import DetDataset, SynthTextDataset
from .rec_dataset import RecDataset
from .rec_lmdb_dataset import LMDBDataset
from .transforms import create_transforms

__all__ = ['build_dataset']

supported_dataset_types = ['BaseDataset', 'DetDataset', 'RecDataset', 'LMDBDataset', 'SynthTextDataset']


def build_dataset(dataset_config: dict,
                  loader_config: dict,
                  num_shards=None,
                  shard_id=None,
                  is_train=True,
                  **kwargs) -> ms.dataset.BatchDataset:
    """
    Build dataset for training and evaluation.

    Args:
        dataset_config (dict): dataset parsing and processing configuartion containing the following keys
            - type (str): dataset class name, please choose from `supported_dataset_types`.
            - dataset_root (str): the root directory to store the (multiple) dataset(s)
            - data_dir (Union[str, List[str]]): directory to the data, which is a subfolder path related to `dataset_root`. For multiple datasets, it is a list of subfolder paths.
            - label_file (Union[str, List[str]], *optional*): file path to the annotation related to the `dataset_root`. For multiple datasets, it is a list of relative file paths. Not required if using LMDBDataset.
            - sample_ratio (float): the sampling ratio of dataset.
            - shuffle (boolean): whether to shuffle the order of data samples.
            - transform_pipeline (list[dict]): each element corresponds to a transform operation on image and/or label
            - output_columns (list[str]): list of output features for each sample.
            - num_columns_to_net (int): num inputs for network forward func in output_columns
        loader_config (dict): dataloader configuration containing keys:
            - batch_size (int): batch size for data loader
            - drop_remainder (boolean): whether to drop the data in the last batch when the total of data can not be divided by the batch_size
            - num_workers (int): number of subprocesses used to fetch the dataset in parallel.
        num_shards (int, *optional*): num of devices for distributed mode
        shard_id (int, *optional*): device id
        is_train (boolean): whether it is in training stage
        **kwargs: optional args for extension. If `refine_batch_size=True` is given in kwargs, the batch size will be refined to be divisable to avoid
            droping remainding data samples in graph model, typically used for precise evaluation.

    Return:
        data_loader (Dataset): dataloader to generate data batch

    Notes:
        - The main data process pipeline in MindSpore contains 3 parts: 1) load data files and generate source dataset, 2) perform per-data-row mapping such as image augmentation, 3) generate batch and apply batch mapping.
        - Each of the three steps supports multiprocess. Detailed mechanism can be seen in https://www.mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/mindspore.dataset.html
        - A data row is a data tuple item containing multiple elements such as (image_i, mask_i, label_i). A data column corresponds to an element in the tuple like 'image', 'label'.
        - The total number of `num_workers` used for data loading and processing should not be larger than the maximum threads of the CPU. Otherwise, it will lead to resource competing overhead. Especially for distributed training, `num_parallel_workers` should not be too large to avoid thread competition.

    Example:
        >>> # Load a DetDataset/RecDataset
        >>> from mindocr.data import build_dataset
        >>> data_config = {
        >>>     "type": "DetDataset",
        >>>     "dataset_root": "path/to/datasets/",
        >>>     "data_dir": "ic15/det/train/ch4_test_images",
        >>>     "label_file": "ic15/det/train/det_gt.txt",
        >>>     "sample_ratio": 1.0,
        >>>     "shuffle": False,
        >>>     "transform_pipeline": [
        >>>         {
        >>>             "DecodeImage": {
        >>>                 "img_mode": "RGB",
        >>>                 "to_float32": False
        >>>                 }
        >>>         },
        >>>         {
        >>>             "DetLabelEncode": {},
        >>>         },
        >>>     ],
        >>>     "output_columns": ['image', 'polys', 'ignore_tags'],
        >>>     "num_columns_to_net": 1
        >>> }
        >>> loader_config = dict(shuffle=True, batch_size=16, drop_remainder=False, num_workers=1)
        >>> data_loader = build_dataset(data_config, loader_config, num_shards=1, shard_id=0, is_train=True)
    """
    # Check dataset paths (dataset_root, data_dir, and label_file) and update to absolute format
    dataset_config = _check_dataset_paths(dataset_config)

    # prefetch_size: the length of the cache queue in the data pipeline for each worker,
    # used to reduce waiting time. Larger value leads to more memory consumption.
    ms.dataset.config.set_prefetch_size(loader_config.get("prefetch_size", 16))
    ms.dataset.config.set_enable_shared_mem(True)

    # 1. create source dataset (GeneratorDataset)
    dataset_class_name = dataset_config.pop('type')
    assert dataset_class_name in supported_dataset_types, "Invalid dataset name"
    dataset_class = eval(dataset_class_name)
    dataset_args = dict(is_train=is_train, **dataset_config)
    dataset = dataset_class(**dataset_args)

    print('==> Dataset columns: \n\t', dataset.output_columns)

    # TODO: find optimal setting automatically according to num of CPU cores
    num_workers = loader_config.get("num_workers", 8)
    cores = cpu_count()
    num_devices = 1 if num_shards is None else num_shards
    if num_workers > cores // num_devices:
        num_workers = cores // num_devices
        print(f'WARNING: num_workers is adjusted to {num_workers}, '
              f'to fit {cores} CPU cores shared for {num_devices} devices')

    # Generate source dataset (source w.r.t. the dataset.map pipeline) based on python callable numpy dataset in parallel
    ds = ms.dataset.GeneratorDataset(dataset,
                                     column_names=dataset.output_columns,
                                     num_parallel_workers=min(num_workers, 4),
                                     num_shards=num_shards,
                                     shard_id=shard_id,
                                     python_multiprocessing=False,
                                     max_rowsize=loader_config.get("max_rowsize", 64),
                                     shuffle=loader_config['shuffle'])

    # 2. data mapping using mindata C lib (optional)
    transforms, pipeline_output_cols = create_transforms(dataset_config['transform_pipeline'],
                                                         input_columns=dataset.output_columns.copy())
    # Ensure backwards compatibility with MS1.x which requires `column_order` parameter
    column_order = None if version.parse(ms.__version__) >= version.parse('2.0.0rc') else pipeline_output_cols.copy()
    ds = ds.map(operations=transforms, input_columns=dataset.output_columns, output_columns=pipeline_output_cols,
                column_order=column_order, num_parallel_workers=num_workers)

    # 3. keep the usable columns_only
    print('==> Dataset output columns: \n\t', dataset_config["output_columns"])
    ds = ds.project(dataset_config["output_columns"])

    # 4. create loader
    # get batch of dataset by collecting batch_size consecutive data rows and apply batch operations
    num_samples = ds.get_dataset_size()
    batch_size = loader_config['batch_size']
    print(f'INFO: num_samples: {num_samples}, batch_size: {batch_size}')
    if 'refine_batch_size' in kwargs:
        batch_size = _check_batch_size(num_samples, batch_size, refine=kwargs['refine_batch_size'])

    drop_remainder = loader_config.get('drop_remainder', is_train)
    if is_train and not drop_remainder:
        print('WARNING: drop_remainder should be True for training, otherwise the last batch may lead to training fail.')
    dataloader = ds.batch(batch_size,
                          drop_remainder=drop_remainder,
                          num_parallel_workers=min(num_workers, 2))    # set small workers for lite computation. TODO: increase for batch-wise mapping

    return dataloader


def _check_dataset_paths(dataset_config):
    if 'dataset_root' in dataset_config:
        if isinstance(dataset_config['data_dir'], str):
            dataset_config['data_dir'] = os.path.join(dataset_config['dataset_root'],
                                                      dataset_config['data_dir'])  # to absolute path
        else:
            dataset_config['data_dir'] = [os.path.join(dataset_config['dataset_root'], dd)
                                          for dd in dataset_config['data_dir']]

        if 'label_file' in dataset_config:
            if isinstance(dataset_config['label_file'], str):
                dataset_config['label_file'] = os.path.join(dataset_config['dataset_root'],
                                                            dataset_config['label_file'])
            else:
                dataset_config['label_file'] = [os.path.join(dataset_config['dataset_root'], lf)
                                                for lf in dataset_config['label_file']]

    return dataset_config


def _check_batch_size(num_samples, ori_batch_size=32, refine=True):
    if num_samples % ori_batch_size == 0:
        return ori_batch_size
    else:
        # search a batch size that is divisible by num samples.
        for bs in range(ori_batch_size - 1, 0, -1):
            if num_samples % bs == 0:
                print(f"WARNING: num eval samples {num_samples} can not be divided by "
                      f"the input batch size {ori_batch_size}. The batch size is refined to {bs}")
                return bs
