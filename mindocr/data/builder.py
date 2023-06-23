import os
from multiprocessing import cpu_count

import cv2
from packaging import version

import mindspore as ms

from .det_dataset import DetDataset, SynthTextDataset
from .predict_dataset import PredictDataset
from .rec_dataset import RecDataset
from .rec_lmdb_dataset import LMDBDataset
from .transforms import create_transforms

__all__ = ["build_dataset"]

supported_dataset_types = [
    "BaseDataset",
    "DetDataset",
    "RecDataset",
    "LMDBDataset",
    "SynthTextDataset",
    "PredictDataset",
]


def build_dataset(
    dataset_config: dict,
    loader_config: dict,
    num_shards=None,
    shard_id=None,
    is_train=True,
    **kwargs,
) -> ms.dataset.BatchDataset:
    """
    Build dataset for training and evaluation.

    Args:
        dataset_config (dict): dataset parsing and processing configuartion containing the following keys
            - type (str): dataset class name, please choose from `supported_dataset_types`.
            - dataset_root (str): the root directory to store the (multiple) dataset(s)
            - data_dir (Union[str, List[str]]): directory to the data, which is a subfolder path related to
              `dataset_root`. For multiple datasets, it is a list of subfolder paths.
            - label_file (Union[str, List[str]], *optional*): file path to the annotation related to the `dataset_root`.
              For multiple datasets, it is a list of relative file paths. Not required if using LMDBDataset.
            - sample_ratio (float): the sampling ratio of dataset.
            - shuffle (boolean): whether to shuffle the order of data samples.
            - transform_pipeline (list[dict]): each element corresponds to a transform operation on image and/or label
            - output_columns (list[str]): list of output features for each sample.
            - net_input_column_index (list[int]): input indices for network forward func in output_columns
        loader_config (dict): dataloader configuration containing keys:
            - batch_size (int): batch size for data loader
            - drop_remainder (boolean): whether to drop the data in the last batch when the total of data can not be
              divided by the batch_size
            - num_workers (int): number of subprocesses used to fetch the dataset in parallel.
        num_shards (int, *optional*): num of devices for distributed mode
        shard_id (int, *optional*): device id
        is_train (boolean): whether it is in training stage
        **kwargs: optional args for extension. If `refine_batch_size=True` is given in kwargs, the batch size will be
            refined to be divisable to avoid
            droping remainding data samples in graph model, typically used for precise evaluation.

    Return:
        data_loader (Dataset): dataloader to generate data batch

    Notes:
        - The main data process pipeline in MindSpore contains 3 parts: 1) load data files and generate source dataset,
            2) perform per-data-row mapping such as image augmentation, 3) generate batch and apply batch mapping.
        - Each of the three steps supports multiprocess. Detailed mechanism can be seen in
            https://www.mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/mindspore.dataset.html
        - A data row is a data tuple item containing multiple elements such as (image_i, mask_i, label_i).
            A data column corresponds to an element in the tuple like 'image', 'label'.
        - The total number of `num_workers` used for data loading and processing should not be larger than the maximum
            threads of the CPU. Otherwise, it will lead to resource competing overhead. Especially for distributed
            training, `num_parallel_workers` should not be too large to avoid thread competition.

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
        >>>     "net_input_column_index`": [0]
        >>>     "label_column_index": [1, 2]
        >>> }
        >>> loader_config = dict(shuffle=True, batch_size=16, drop_remainder=False, num_workers=1)
        >>> data_loader = build_dataset(data_config, loader_config, num_shards=1, shard_id=0, is_train=True)
    """
    # Check dataset paths (dataset_root, data_dir, and label_file) and update to absolute format
    dataset_config = _check_dataset_paths(dataset_config)

    # OpenCV spawns as many threads as there are CPU cores, thus causing multiprocessing overhead.
    # It is better to limit the number of threads per sample but increase the total number of pipeline threads
    # to process more samples simultaneously.
    cv2.setNumThreads(2)

    # prefetch_size: the length of the cache queue in the data pipeline for each worker,
    # used to reduce waiting time. Larger value leads to more memory consumption.
    ms.dataset.config.set_prefetch_size(loader_config.get("prefetch_size", 16))
    ms.dataset.config.set_enable_shared_mem(True)
    # ms.dataset.config.set_debug_mode(True)    # uncomment this to debug data pipeline

    # Set default multiprocessing params for data pipeline
    cores = cpu_count()
    num_devices = num_shards or 1
    num_workers_dataset = loader_config.get("num_workers_dataset", 4)
    num_workers_batch = loader_config.get("num_workers_batch", 2)
    # optimal num workers assuming all cpu cores are used in this job
    num_workers_map = loader_config.get("num_workers", cores // num_devices - num_workers_batch - num_workers_dataset)
    if num_workers_map > cores // num_devices:
        num_workers_map = cores // num_devices
        print(
            f"WARNING: `num_workers` is adjusted to {num_workers_map}, "
            f"to fit {cores} CPU cores shared for {num_devices} devices"
        )

    if dataset_config.get("mindrecord", False):
        # read the MR file's schema to load the stored list of columns
        reader = ms.mindrecord.FileReader(dataset_config["data_dir"])
        dataset_column_names = list(reader.schema().keys())
        reader.close()

        ds = ms.dataset.MindDataset(
            dataset_config["data_dir"],
            columns_list=dataset_column_names,
            num_parallel_workers=num_workers_dataset,
            num_shards=num_shards,
            shard_id=shard_id,
            shuffle=loader_config["shuffle"],
        )
    else:
        dataset_class_name = dataset_config.pop("type")
        assert dataset_class_name in supported_dataset_types, "Invalid dataset name"
        dataset = eval(dataset_class_name)(is_train=is_train, **dataset_config)

        ds = ms.dataset.GeneratorDataset(
            dataset,
            column_names=dataset.output_columns,
            num_parallel_workers=num_workers_dataset,
            num_shards=num_shards,
            shard_id=shard_id,
            # file reading is not CPU bounded => use multithreading for reading images and labels
            python_multiprocessing=False,
            shuffle=loader_config["shuffle"],
        )
        dataset_column_names = dataset.output_columns

    # 2. data mapping using mindata C lib
    transforms = create_transforms(
        dataset_config["transform_pipeline"],
        input_columns=dataset_column_names,
        backward_comp=version.parse(ms.__version__) < version.parse("2.0.0rc"),
    )
    using_multiprocess_for_pipeline = loader_config.get("using_multiprocess_for_pipeline", True)
    for group in transforms:
        ds = ds.map(
            **group,
            python_multiprocessing=using_multiprocess_for_pipeline,
            num_parallel_workers=num_workers_map,
            max_rowsize=loader_config.get("max_rowsize", 64),
        )

    # 3. keep the usable columns_only
    ds = ds.project(dataset_config["output_columns"])

    # 4. create loader
    # get batch of dataset by collecting batch_size consecutive data rows and apply batch operations
    num_samples = ds.get_dataset_size()
    batch_size = loader_config["batch_size"]

    rank_id = shard_id or 0
    is_main_rank = rank_id == 0
    print(
        f"INFO: Creating dataloader (training={is_train}) for rank {rank_id}. " f"Number of data samples: {num_samples}"
    )

    if "refine_batch_size" in kwargs:
        batch_size = _check_batch_size(num_samples, batch_size, refine=kwargs["refine_batch_size"])

    drop_remainder = loader_config.get("drop_remainder", is_train)
    if is_train and not drop_remainder and is_main_rank:
        print(
            "WARNING: `drop_remainder` should be True for training, "
            "otherwise the last batch may lead to training fail in Graph mode."
        )
    elif not is_train and drop_remainder:
        if is_main_rank:
            print(
                "WARNING: `drop_remainder` is forced to be False for evaluation "
                "to include the last batch for accurate evaluation."
            )
        drop_remainder = False

    dataloader = ds.batch(batch_size, drop_remainder=drop_remainder, num_parallel_workers=num_workers_batch)

    return dataloader


def _check_dataset_paths(dataset_config):
    if "dataset_root" in dataset_config:
        if isinstance(dataset_config["data_dir"], str):
            dataset_config["data_dir"] = os.path.join(
                dataset_config["dataset_root"], dataset_config["data_dir"]
            )  # to absolute path
        else:
            dataset_config["data_dir"] = [
                os.path.join(dataset_config["dataset_root"], dd) for dd in dataset_config["data_dir"]
            ]

        if "label_file" in dataset_config and dataset_config["label_file"]:
            if isinstance(dataset_config["label_file"], str):
                dataset_config["label_file"] = os.path.join(
                    dataset_config["dataset_root"], dataset_config["label_file"]
                )
            elif isinstance(dataset_config["label_file"], list):
                dataset_config["label_file"] = [
                    os.path.join(dataset_config["dataset_root"], lf) for lf in dataset_config["label_file"]
                ]

    return dataset_config


def _check_batch_size(num_samples, ori_batch_size=32, refine=True):
    if num_samples % ori_batch_size == 0:
        return ori_batch_size
    else:
        # search a batch size that is divisible by num samples.
        for bs in range(ori_batch_size - 1, 0, -1):
            if num_samples % bs == 0:
                print(
                    f"INFO: Batch size for evaluation is refined to {bs} to ensure the last batch will not be "
                    "dropped/padded in graph mode."
                )
                return bs
